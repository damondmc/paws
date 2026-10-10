"""Outlier collection of a stage, band by band.

search / injection / injection follow-up: ResultAnalysisManager.make_outlier over the stage's own Weave results.
follow-up of real candidates: the loudest candidate of each seed (seed plan, possibly reused results) must pass
the stage's excess ratio threshold and, when set, its H1/L1 excess-ratio window.
"""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from paws.analysis.outlier import loudest_rows
from paws.analysis.thresholds import (
    excess_ratio_to_mean2f_threshold,
    inside_h1_l1_window,
    search_mean2f_threshold,
)
from paws.filepaths import make_dir
from paws.pipeline import reuse
from paws.pipeline.context import N_THREADS, StageContext
from paws.pipeline.dag import dag_list_path
from paws.pipeline.followup_seeds import (
    default_seed_plan,
    load_seed_plan,
    read_seed_rows,
    result_files_per_seed,
    save_seed_plans,
    seed_outlier_file,
    selected_seed_rows,
)
from paws.pipeline.stage_thresholds import stage_thresholds, write_threshold_records

N_PROCESSES = 8
SEEDS_PER_CHUNK = 2000  # loudest rows are read and cached in chunks of this many seeds


# ------------------------ make_outlier stages -------------------------


def total_semicoherent_templates(context, freq, n_jobs):
    def templates_of_job(job_index):
        result_file = context.paths.weave_output_file(freq, context.taskname(freq), job_index, context.stage.name)
        return fits.getheader(result_file)["NSEMITPL"]

    with ThreadPoolExecutor(N_THREADS) as executor:
        return sum(executor.map(templates_of_job, range(1, n_jobs + 1)))


def make_outlier_kwargs(context):
    stage = context.stage
    return dict(
        num_toplist=stage.keep, stage=stage.name, freq_deriv_order=stage.order, n_sky=stage.n_sky,
        cluster=stage.cluster, work_in_local_dir=False, separate_saturated=stage.separate_saturated,
        is_injection=stage.is_injection, n_processes=N_PROCESSES, max_workers=N_THREADS,
    )


def outliers_search(context, freq):
    n_jobs = context.search_params(freq).size
    n_templates = total_semicoherent_templates(context, freq, n_jobs)
    mean2f_threshold = search_mean2f_threshold(n_templates, context.stage.n_seg)
    context.result_manager.make_outlier(
        context.taskname(freq), freq, mean2f_threshold, n_jobs, **make_outlier_kwargs(context)
    )


def outliers_injection(context, freq):
    search_file = context.prev_outlier_file(freq, cluster=True)
    if not search_file.is_file():
        print(f"{freq} Hz: no {context.prev_stage.name} outlier file, skipped")
        return
    mean2f_threshold = fits.getval(search_file, "mean2F_th", 0)
    context.result_manager.make_outlier(
        context.taskname(freq), freq, mean2f_threshold, context.stage.n_inj, **make_outlier_kwargs(context)
    )


def outliers_injection_followup(context, freq):
    seed_rows, _ = read_seed_rows(context.settings, context.paths, context.stage, freq)
    if seed_rows is None or len(seed_rows) == 0:
        return
    context.result_manager.make_outlier(
        context.taskname(freq), freq, np.zeros(len(seed_rows)), len(seed_rows), **make_outlier_kwargs(context)
    )


# -------------------- follow-up of real candidates --------------------


def band_seed_plan(context, freq, seed_rows):
    """The band's saved seed plan; built (and saved) when the stage's DAG step did not write one."""
    plan = load_seed_plan(context.settings, context.stage, freq)
    if plan is None:
        if context.stage.reuse:
            plan = reuse.build_seed_plan(context.settings, context.paths, context.stage, freq, seed_rows)
        else:
            plan = default_seed_plan(context.stage, seed_rows)
        save_seed_plans(context.settings, context.stage, {freq: plan})
    if len(plan.keys) != len(seed_rows):
        raise ValueError(f"{context.stage.name} {freq} Hz: seed plan has {len(plan.keys)} seeds, "
                         f"the previous stage {len(seed_rows)}; delete the stale seed_plan.npz entry")
    return plan


def loudest_cache_path(context, freq, first_seed):
    return context.config.home_dir / "results" / context.stage.name / "loudest" / f"{freq}_{first_seed}.npy"


def cache_loudest_chunk(chunk):
    """Loudest row of each seed of one chunk, saved to cache_path (skipped when already cached)."""
    cache_path, files_per_seed, label = chunk
    if cache_path.exists():
        return
    rows = loudest_rows(files_per_seed, N_THREADS)
    make_dir([cache_path])
    temporary_path = cache_path.with_name(cache_path.stem + ".tmp.npy")
    np.save(temporary_path, rows)
    temporary_path.replace(cache_path)
    print(f"{label}: done", flush=True)


def collect_thresholded_followup(context, bands):
    """Outlier files of a follow-up of real candidates: one INFO row per seed, the seeds whose loudest
    candidate passes the excess ratio threshold (and H1/L1 excess-ratio window) as outliers."""
    stage = context.stage
    thresholds = stage_thresholds(context.settings, stage)
    write_threshold_records(context.settings, stage)

    seeds_by_band, files_by_band, chunks = {}, {}, []
    for freq in bands:
        seed_rows = selected_seed_rows(context.settings, context.paths, stage, freq)
        if seed_rows is None:
            continue
        plan = band_seed_plan(context, freq, seed_rows)
        files_per_seed = result_files_per_seed(context.paths, context.settings, stage, freq, plan)
        seeds_by_band[freq], files_by_band[freq] = seed_rows, files_per_seed
        for first_seed in range(0, len(seed_rows), SEEDS_PER_CHUNK):
            stop_seed = min(first_seed + SEEDS_PER_CHUNK, len(seed_rows))
            label = f"{stage.name} {freq} Hz seeds {first_seed}-{stop_seed}"
            chunks.append((loudest_cache_path(context, freq, first_seed), files_per_seed[first_seed:stop_seed], label))
    chunks.sort(key=lambda chunk: -len(chunk[1]))  # largest first
    n_all_seeds = sum(len(rows) for rows in seeds_by_band.values())
    print(f"{stage.name}: {len(seeds_by_band)} bands, {n_all_seeds:,d} seeds, {len(chunks)} chunks", flush=True)
    with Pool(N_PROCESSES) as pool:
        for _ in pool.imap_unordered(cache_loudest_chunk, chunks):
            pass

    n_seeds = n_passed = 0
    for freq, seed_rows in seeds_by_band.items():
        loudest_per_seed = np.concatenate([
            np.load(loudest_cache_path(context, freq, first_seed))
            for first_seed in range(0, len(seed_rows), SEEDS_PER_CHUNK)
        ])
        mean2f_threshold = excess_ratio_to_mean2f_threshold(thresholds.excess_ratio_at(freq), seed_rows["mean2F"])
        passed = loudest_per_seed["mean2F"] >= mean2f_threshold
        if thresholds.h1_l1_window is not None:
            passed &= inside_h1_l1_window(
                thresholds.h1_l1_window_at(freq), loudest_per_seed["mean2F_H1"], loudest_per_seed["mean2F_L1"]
            )
        files_per_seed = files_by_band[freq]
        spacing_files = [files[0] for files in files_per_seed[:: max(1, len(files_per_seed) // 10)][:10]]
        context.result_manager.write_loudest_outliers(
            context.taskname(freq), freq, stage.name, stage.order, len(seed_rows), loudest_per_seed, passed,
            mean2f_threshold, spacing_files,
        )
        n_seeds += len(seed_rows)
        n_passed += int(passed.sum())
        print(f"{stage.name} {freq} Hz: {len(seed_rows)} seeds, {int(passed.sum())} pass", flush=True)
    print(f"{stage.name}: {n_passed:,d} of {n_seeds:,d} seeds pass")


# ------------------------------- stages -------------------------------


OUTLIER_COLLECTOR_BY_KIND = {
    "search": outliers_search,
    "injection": outliers_injection,
    "followup": outliers_injection_followup,  # follow-ups of real candidates: collect_thresholded_followup
}


def collect_stage_outliers(settings, stage_name, bands):
    """Outlier files of every band of the stage (or only `bands`; None: the stage's bands)."""
    context = StageContext(settings, stage_name)
    stage = context.stage
    bands = bands or stage.freqs
    if stage.kind == "followup" and not stage.is_injection:
        if stage.thresholds is None:
            raise SystemExit(f"{stage.name} has no thresholds in stages.yaml")
        collect_thresholded_followup(context, bands)
        return
    if stage.kind not in OUTLIER_COLLECTOR_BY_KIND:
        raise SystemExit(f"{stage.name}: {stage.kind} jobs write their own outlier files")
    for freq in tqdm(bands, desc=f"{stage.name} outliers"):
        OUTLIER_COLLECTOR_BY_KIND[stage.kind](context, freq)


def outlier_collection_dag(context, freq):
    stage = context.stage
    seed_rows, _ = read_seed_rows(context.settings, context.paths, stage, freq)
    if seed_rows is None or len(seed_rows) == 0:
        return None
    config_url, target_url = context.settings.job_config_urls()
    return context.workflow_manager.make_outlier_dag(
        config_url, target_url, context.taskname(freq), freq, stage.name, stage.order, len(seed_rows),
        seed_outlier_file(context.settings, context.paths, stage, freq), None,
        num_toplist=stage.keep, n_sky=stage.n_sky, zero_threshold=True, cluster=stage.cluster,
        separate_saturated=False, is_injection=True, max_workers=N_THREADS, request_memory="8GB",
        request_disk="8GB", request_cpu=1, image=context.settings.image_url(stage),
    )


def make_stage_outlier_dags(settings, stage_name, bands):
    """Outlier-collection Condor DAGs of an injection follow-up stage and their DAG list."""
    context = StageContext(settings, stage_name)
    stage = context.stage
    if not (stage.kind == "followup" and stage.is_injection):
        raise SystemExit("Condor outlier collection applies to injection follow-ups only (the job keeps every candidate)")
    weave_list_path = dag_list_path(context, bands)
    list_path = weave_list_path.with_name(weave_list_path.name.replace("_dag", "_outlier_dag"))
    with open(list_path, "w") as dag_list:
        for freq in tqdm(bands or stage.freqs, desc=f"{stage.name} outlier DAGs"):
            dag_file = outlier_collection_dag(context, freq)
            if dag_file is not None:
                dag_list.write(f"{dag_file}\n")
    print(f"DAG list: {list_path}")
