"""v3 follow-up: 0.1% (2F-4) ratio cut and 0.1% H1/L1 window at every stage, per 100 Hz band.

Stage k (STAGES, settings in stages.yaml) follows up the clustered outliers of stage k-1 (stage 0: search-0
clustered outliers inside the t5 H1/L1 window). A seed whose (freq, f1dot, f2dot) equals a seed of an earlier run
(REUSE) takes that run's Weave results; the other seeds run in followup-v3-k.

Commands (from paws/, config dir in $PAWS_CONFIG_DIR):
  uv run python scripts/followup_v3/followup_v3.py outliers 1
  uv run python scripts/followup_v3/followup_v3.py dag <k>          k = 2, 3
  uv run python scripts/followup_v3/followup_v3.py outliers <k>     k = 2, 3
  ... [--bands f1 f2 ...]
The cut files come from `paws cuts` (see README.md).
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from paws import stages
from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager
from paws.params.followup import FollowUpParamGenerator
from paws.params.models import PowerLawModel
from paws.settings import Settings
from paws.workflow.manager import WorkflowManager

settings = Settings(os.environ.get("PAWS_CONFIG_DIR"))
config, target = settings.config, settings.target
paths = PathManager(config, target)
home_dir = config.home_dir

BANDS = range(20, 400)
N_PROC, THREADS, CHUNK = 8, 32, 2000
STAGES = [settings.stage(n) for n in ("search-0", "followup-v3-1", "followup-v3-2", "followup-v3-3")]
T5_WINDOW = settings.config_path("injections-v2-0_hl_window_v3.txt")  # seeds of stage 1
# earlier runs of stage k: (stage, seeds); seed i owns files i*n_sky+1 .. (i+1)*n_sky
REUSE = {
    1: [(settings.stage("followup-1"), "search-0 clustered")],
    2: [(settings.stage("followup-2"), "followup-1 clustered"), (settings.stage("followup-v2-2"), "v2 changed seeds")],
    3: [(settings.stage("followup-v2-3"), "followup-v2-2 clustered")],
}
KEY = ("freq", "f1dot", "f2dot")


def outlier_file(stage, freq):
    return paths.outlier_file(freq, stage.taskname(target, freq), stage.name, cluster=True)


def in_window(k, freq, h1, l1):
    return stages.in_hl_window(settings.config_path(STAGES[k].hl_window) if k else T5_WINDOW, freq, h1, l1)


# ---------------------------------------------------------------- seeds and reuse
def seeds(k, freq):
    """Seeds of stage k: clustered outliers of stage k-1 (stage 0 also inside the t5 window)."""
    path = outlier_file(STAGES[k - 1], freq)
    d = stages.read_rows(path, 1) if path.is_file() else None
    if d is None:
        return None
    if k == 1:
        d = d[in_window(0, np.full(len(d), freq), d["mean2F_H1"], d["mean2F_L1"])]
    return d


def keys(rows):
    return [tuple(float(r[c]) for c in KEY) for r in rows]


def reuse_seeds(k, freq):
    """[(stage, {key: seed index})] of the earlier runs of stage k."""
    out = []
    for st, src in REUSE[k]:
        if src == "followup-1 clustered":
            fn = outlier_file(settings.stage("followup-1"), freq)
            rows = stages.read_rows(fn, 1) if fn.is_file() else None
        elif src == "v2 changed seeds":
            m = np.load(home_dir / "results/followup-v2-2/seed_map_followup-v2-1_vs_followup-1.npz")
            if f"{freq}_exact" not in m.files:
                rows = None
            else:
                fn = outlier_file(settings.stage("followup-v2-1"), freq)
                rows = stages.read_rows(fn, 1)[np.flatnonzero(m[f"{freq}_exact"] < 0)]
        elif src == "followup-v2-2 clustered":
            fn = outlier_file(settings.stage("followup-v2-2"), freq)
            rows = stages.read_rows(fn, 1) if fn.is_file() else None
        else:
            raise ValueError(src)
        out.append((st, {} if rows is None else {key: i for i, key in enumerate(keys(rows))}))
    return out


def seed_map(k, freq, rows):
    """Per seed: index into REUSE[k] (-1: new) and seed index in that run (or in followup-v3-k)."""
    src, idx = np.full(len(rows), -1), np.zeros(len(rows), int)
    for s, (_, table) in enumerate(reuse_seeds(k, freq)):
        for i, key in enumerate(keys(rows)):
            if src[i] < 0 and key in table:
                src[i], idx[i] = s, table[key]
    new = np.flatnonzero(src < 0)
    idx[new] = np.arange(new.size)
    return src, idx


def map_path(k):
    return home_dir / "results" / STAGES[k].name / "seed_map.npz"


# ---------------------------------------------------------------- dag
def cmd_dag(args):
    k = args.k
    s, p = STAGES[k], STAGES[k - 1]
    maps = dict(np.load(map_path(k))) if map_path(k).exists() else {}
    tag = "_".join(map(str, args.bands)) if args.bands else "20-400"
    dag_list = home_dir / "dagFiles" / f"{s.name}_{target.name}_dag{tag}Hz.txt"
    n_reused = n_new = 0
    with open(dag_list, "w") as f_daglist:
        for freq in args.bands or BANDS:
            rows = seeds(k, freq)
            if rows is None or len(rows) == 0:
                continue
            src, idx = seed_map(k, freq, rows)
            maps[f"{freq}_src"], maps[f"{freq}_idx"] = src, idx
            new = np.flatnonzero(src < 0)
            n_reused += len(rows) - new.size
            n_new += new.size
            if new.size == 0:
                continue
            spacing = stages.header_spacing(outlier_file(p, freq), p.order)
            params = FollowUpParamGenerator(
                PowerLawModel(nc_min=config.nc_min, nc_max=config.nc_max, tau=target.tau(freq))
            ).generate_parameter(
                alpha=target.alpha, dalpha=target.dalpha, delta=target.delta, ddelta=target.ddelta,
                data=rows[new], old_freq_deriv_order=p.order, new_freq_deriv_order=s.order,
                spacing=spacing, n_spacing=config.followup_n_spacing, sky_radius=s.sky_radius,
                spacing_alpha=s.spacing_alpha, spacing_delta=s.spacing_delta,
            )
            dag_file = manager.make_search_dag(
                s.taskname(target, freq), freq, params.data, num_top_list=s.toplist, stage=s.name,
                freq_deriv_order=s.order, n_seg=s.n_seg, sft_files=paths.sft_ensemble(freq),
                metric_file=settings.metric_url(s), request_memory=s.request_memory, request_disk=s.request_disk,
                request_cpu=s.request_cpu, use_osg=True, use_osdf=True, tasks_per_job=s.tasks_per_job,
                sky_offsets=settings.sky_offsets(s), image=settings.image_url(s),
            )
            f_daglist.write(f"{dag_file}\n")
        f_daglist.write(f"{manager.make_osdf_cleanup_dag(s.name)}\n")
    map_path(k).parent.mkdir(parents=True, exist_ok=True)
    np.savez(map_path(k), **maps)
    print(f"{s.name}: {n_reused + n_new:,d} seeds, {n_reused:,d} reuse earlier results, {n_new:,d} new "
          f"({n_new * s.n_sky:,d} Weave runs); DAG list {dag_list}")


# ---------------------------------------------------------------- outliers
def files_of(k, freq, src, idx):
    """Weave result files (n_sky per seed) of each seed of stage k."""
    s = STAGES[k]
    out = []
    for a, i in zip(src, idx):
        st = REUSE[k][a][0] if a >= 0 else s
        first = i * s.n_sky + 1
        out.append([paths.weave_output_file(freq, st.taskname(target, freq), j, st.name)
                    for j in range(first, first + s.n_sky)])
    return out


def loudest_path(k, freq, lo):
    return home_dir / "results" / STAGES[k].name / "loudest" / f"{freq}_{lo}.npy"


def loudest_chunk(task):
    """Loudest row of each seed in [lo, hi) over its files; cached per chunk."""
    k, freq, lo, hi, src, idx = task
    out = loudest_path(k, freq, lo)
    if out.exists():
        return
    stages.save_atomic(out, stages.loudest_rows(files_of(k, freq, src, idx), THREADS))
    print(f"{STAGES[k].name} {freq} Hz seeds {lo}-{hi}: done", flush=True)


def cmd_outliers(args):
    k = args.k
    s = STAGES[k]
    bands = list(args.bands or BANDS)
    if k == 1:
        maps = {}
        for freq in bands:
            rows = seeds(1, freq)
            if rows is not None and len(rows):
                # followup-1 file j <-> search-0 clustered row j-1 (all rows, before the t5 window)
                all5 = stages.read_rows(outlier_file(STAGES[0], freq), 1)
                pos = {key: i for i, key in enumerate(keys(all5))}
                maps[freq] = (np.zeros(len(rows), int), np.array([pos[key] for key in keys(rows)]))
    else:
        m = np.load(map_path(k))
        maps = {f: (m[f"{f}_src"], m[f"{f}_idx"]) for f in bands if f"{f}_src" in m.files}
    tasks = []
    for freq, (src, idx) in maps.items():
        for lo in range(0, len(src), CHUNK):
            hi = min(lo + CHUNK, len(src))
            tasks.append((k, freq, lo, hi, src[lo:hi], idx[lo:hi]))
    tasks.sort(key=lambda t: -(t[3] - t[2]))
    print(f"{s.name}: {len(maps)} bands, {sum(len(v[0]) for v in maps.values()):,d} seeds, {len(tasks)} chunks",
          flush=True)
    with Pool(N_PROC) as pool:
        for _ in pool.imap_unordered(loudest_chunk, tasks):
            pass
    n_seeds = n_pass = 0
    for freq, (src, idx) in maps.items():
        prev = seeds(k, freq)
        best = np.concatenate([np.load(loudest_path(k, freq, lo)) for lo in range(0, len(src), CHUNK)])
        th = stages.ratio_threshold(settings.config_path(s.ratio_cut), np.full(len(prev), freq), prev["mean2F"])
        passed = (best["mean2F"] >= th) & in_window(k, np.full(len(best), freq), best["mean2F_H1"], best["mean2F_L1"])
        blocks = files_of(k, freq, src, idx)
        stages.write_seed_outliers(result_manager, s, s.taskname(target, freq), freq, len(prev), best, passed, th,
                                   [b[0] for b in blocks[:: max(1, len(blocks) // 10)][:10]])
        n_seeds += len(prev)
        n_pass += int(passed.sum())
        print(f"{s.name} {freq} Hz: {len(prev)} seeds, {int(passed.sum())} pass", flush=True)
    print(f"{s.name}: {n_pass:,d} of {n_seeds:,d} seeds pass")


manager = WorkflowManager(config, target)
result_manager = ResultAnalysisManager(config, target)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dag", "outliers"):
        c = sub.add_parser(name)
        c.add_argument("k", type=int, choices=[2, 3] if name == "dag" else [1, 2, 3])
        c.add_argument("--bands", type=int, nargs="+")
    a = ap.parse_args()
    {"dag": cmd_dag, "outliers": cmd_outliers}[a.cmd](a)
