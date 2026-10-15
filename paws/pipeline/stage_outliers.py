"""`paws outliers`: the outlier files of a stage, band by band. The search and the injection stages collect every
candidate above the mean2F threshold (ResultAnalysisManager.make_outlier); a follow-up of real candidates keeps the
loudest candidate of each seed and applies the cuts of its threshold file."""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.io import fits
from jaxtyping import Float
from tqdm import tqdm

from paws.analysis.outlier import cached_loudest_rows
from paws.analysis.statistics import (
    excess_ratio_to_mean2f_threshold,
    inside_h1_l1_window,
    search_mean2f_threshold,
)
from paws.pipeline.context import N_THREADS, StageContext
from paws.pipeline.history import chain_history, history_column
from paws.pipeline.noise_weights import log10_noise_ratio
from paws.pipeline.seeds import (
    read_seed_rows,
    result_files_per_seed,
    selected_seed_rows,
)
from paws.pipeline.thresholds import record_threshold_file, stage_thresholds
from paws.settings import Settings

N_PROCESSES = 8
SEEDS_PER_CHUNK = 2000  # loudest rows are read and cached in chunks of this many seeds


# ------------------------ make_outlier stages -------------------------


def total_semicoherent_templates(context: StageContext, freq: int, n_jobs: int) -> int:
    def templates_of_job(job_index: int) -> int:
        result_file = context.paths.weave_output_file(
            freq, context.taskname(freq), job_index, context.stage.name
        )
        return fits.getheader(result_file)["NSEMITPL"]

    with ThreadPoolExecutor(N_THREADS) as executor:
        return sum(executor.map(templates_of_job, range(1, n_jobs + 1)))


def collect_band_outliers(
    context: StageContext,
    freq: int,
    mean2f_threshold: Union[float, Float[np.ndarray, " n_point"]],
    n_points: int,
) -> None:
    """Outlier file of the band from the stage's settings (ResultAnalysisManager.make_outlier)."""
    stage = context.stage
    context.result_manager.make_outlier(
        context.taskname(freq),
        freq,
        stage.name,
        mean2f_threshold,
        n_points,
        n_keep=stage.n_keep,
        order=stage.order,
        n_sky=stage.n_sky,
        cluster=stage.cluster,
        separate_saturated=stage.separate_saturated,
        is_search=stage.is_search,
        is_injection=stage.is_injection,
        n_processes=N_PROCESSES,
        n_threads=N_THREADS,
    )


def collect_search_outliers(context: StageContext, freq: int) -> None:
    n_points = context.search_params(freq).size
    n_templates = total_semicoherent_templates(context, freq, n_points)
    mean2f_threshold = search_mean2f_threshold(n_templates, context.stage.n_seg)
    collect_band_outliers(context, freq, mean2f_threshold, n_points)


def collect_injection_outliers(context: StageContext, freq: int) -> None:
    search_file = context.prev_outlier_file(freq, cluster=True)
    if not search_file.is_file():
        print(f"{freq} Hz: no {context.prev_stage.name} outlier file, skipped")
        return
    mean2f_threshold = fits.getval(search_file, "mean2F_th", 0)
    collect_band_outliers(
        context, freq, mean2f_threshold, context.stage.injection.n_inj
    )


def collect_injection_followup_outliers(context: StageContext, freq: int) -> None:
    seed_rows, _ = read_seed_rows(context.settings, context.paths, context.stage, freq)
    if seed_rows is None or len(seed_rows) == 0:
        return
    collect_band_outliers(context, freq, np.zeros(len(seed_rows)), len(seed_rows))


# -------------------- follow-up of real candidates --------------------


def band_result_files(
    context: StageContext, freq: int, n_seeds: int
) -> list[list[Path]]:
    """Weave result files per seed of the band."""
    stage = context.stage
    files_per_seed = result_files_per_seed(
        context.paths, context.settings, stage, freq, n_seeds + 1
    )
    if files_per_seed[-1][0].exists():
        raise ValueError(
            f"{stage.name} {freq} Hz: Weave results beyond the {n_seeds} seeds of {stage.prev}; "
            f"the run does not match the seeds"
        )
    return files_per_seed[:-1]


def loudest_cache_path(context: StageContext, freq: int, first_seed: int) -> Path:
    return (
        context.settings.environment.home_dir
        / "results"
        / context.stage.name
        / "loudest"
        / f"{freq}_{first_seed}.npy"
    )


def cache_loudest_chunk(chunk: tuple[Path, list[list[Path]], str]) -> None:
    """Caches the loudest row of each seed of one chunk."""
    cache_path, files_per_seed, label = chunk
    if not cache_path.exists():
        cached_loudest_rows(cache_path, files_per_seed, N_THREADS)
        print(f"{label}: done", flush=True)


def collect_followup_outliers(context: StageContext, bands: list[int]) -> None:
    """Outlier files of a follow-up of real candidates.
    One INFO row per seed, the seeds whose loudest candidate passes the cut.
    Each with the statistics of the earlier stages of its chain."""
    stage = context.stage
    thresholds = stage_thresholds(context.settings, stage)
    record_threshold_file(context.settings, stage)
    chain = context.settings.chain_stages(stage)

    seeds_by_band, files_by_band, chunks = {}, {}, []
    for freq in bands:
        seed_rows = selected_seed_rows(context.settings, context.paths, stage, freq)
        if seed_rows is None:
            continue
        files_per_seed = band_result_files(context, freq, len(seed_rows))
        seeds_by_band[freq], files_by_band[freq] = seed_rows, files_per_seed
        for first_seed in range(0, len(seed_rows), SEEDS_PER_CHUNK):
            stop_seed = min(first_seed + SEEDS_PER_CHUNK, len(seed_rows))
            label = f"{stage.name} {freq} Hz seeds {first_seed}-{stop_seed}"
            chunks.append(
                (
                    loudest_cache_path(context, freq, first_seed),
                    files_per_seed[first_seed:stop_seed],
                    label,
                )
            )
    chunks.sort(key=lambda chunk: -len(chunk[1]))  # largest first
    n_all_seeds = sum(len(rows) for rows in seeds_by_band.values())
    print(
        f"{stage.name}: {len(seeds_by_band)} bands, {n_all_seeds:,d} seeds, {len(chunks)} chunks",
        flush=True,
    )
    with Pool(N_PROCESSES) as pool:
        for _ in pool.imap_unordered(cache_loudest_chunk, chunks):
            pass

    n_seeds = n_passed = 0
    for freq, seed_rows in seeds_by_band.items():
        loudest_per_seed = np.concatenate(
            [
                cached_loudest_rows(
                    loudest_cache_path(context, freq, first_seed),
                    files_by_band[freq][first_seed : first_seed + SEEDS_PER_CHUNK],
                    N_THREADS,
                )
                for first_seed in range(0, len(seed_rows), SEEDS_PER_CHUNK)
            ]
        )
        history = chain_history(chain, seed_rows)
        mean2f_threshold = excess_ratio_to_mean2f_threshold(
            thresholds.in_band("excess_ratio", freq), seed_rows["mean2F"]
        )
        passed = loudest_per_seed["mean2F"] >= mean2f_threshold
        if thresholds.h1_l1_window is not None:
            passed &= inside_h1_l1_window(
                thresholds.in_band("h1_l1_window", freq),
                loudest_per_seed["mean2F_H1"],
                loudest_per_seed["mean2F_L1"],
                log10_noise_ratio(
                    context.settings, thresholds.h1_l1_noise_weights, freq
                ),
            )
        if thresholds.growth is not None:
            # the excess ratio cut over the search outlier the seed descends from
            passed &= loudest_per_seed["mean2F"] >= excess_ratio_to_mean2f_threshold(
                thresholds.in_band("growth", freq),
                history[history_column("mean2F", chain[0])],
            )
        files_per_seed = files_by_band[freq]
        spacing_files = [
            files[0]
            for files in files_per_seed[:: max(1, len(files_per_seed) // 10)][:10]
        ]
        context.result_manager.write_loudest_outliers(
            context.taskname(freq),
            freq,
            stage.name,
            stage.order,
            len(seed_rows),
            loudest_per_seed,
            passed,
            mean2f_threshold,
            history,
            {f"TCoh{s.tcoh}": s.name for s in chain},
            spacing_files,
        )
        n_seeds += len(seed_rows)
        n_passed += int(passed.sum())
        print(
            f"{stage.name} {freq} Hz: {len(seed_rows)} seeds, {int(passed.sum())} pass",
            flush=True,
        )
    print(f"{stage.name}: {n_passed:,d} of {n_seeds:,d} seeds pass")


# ------------------------------- stages -------------------------------


def collect_stage_outliers(
    settings: Settings, stage_name: str, bands: Optional[list[int]]
) -> None:
    """Outlier files of every band of the stage."""
    context = StageContext(settings, stage_name)
    stage = context.stage
    bands = bands or stage.freqs
    if stage.upperlimit:
        raise SystemExit(
            f"{stage.name}: upper-limit jobs write their own outlier files"
        )
    if stage.is_search:
        collect_band = collect_search_outliers
    elif stage.is_injection:
        collect_band = (
            collect_injection_followup_outliers
            if context.prev_stage.is_injection
            else collect_injection_outliers
        )
    elif stage.thresholds is not None:
        collect_followup_outliers(context, bands)
        return
    else:
        raise SystemExit(f"{stage.name} has no thresholds in followup.yaml")
    for freq in tqdm(bands, desc=f"{stage.name} outliers"):
        collect_band(context, freq)
