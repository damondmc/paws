"""v3 follow-up: 0.1% (2F-4) ratio cut and 0.1% H1/L1 window at every stage, per 100 Hz band.

Stage k (STAGES) follows up the clustered outliers of stage k-1 (stage 0: search-0 clustered outliers inside the
t5 H1/L1 window). A seed whose (freq, f1dot, f2dot) equals a seed of an earlier run (REUSE) takes that run's Weave
results; the other seeds run in followup-v3-k.

Commands (from paws/):
  uv run python scripts/followup_v3/followup_v3.py cuts [--both-detectors]
  uv run python scripts/followup_v3/followup_v3.py outliers 1
  uv run python scripts/followup_v3/followup_v3.py dag <k>          k = 2, 3
  uv run python scripts/followup_v3/followup_v3.py outliers <k>     k = 2, 3
  ... [--bands f1 f2 ...]
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

from paws.analysis.outlier import ResultAnalysisManager
from paws.definitions import phase_param_name
from paws.filepaths import PathManager
from paws.io import get_spacing, make_dir
from paws.params.followup import FollowUpParamGenerator
from paws.params.models import PowerLawModel
from paws.workflow.manager import WorkflowManager

with open("/home/hoitim.cheung/galacticCenter/config/config.yaml") as f:
    config = yaml.safe_load(f)
home_dir = config["home_dir"]
with open(f"{home_dir}config/gal.yaml") as f:
    target = yaml.safe_load(f)
paths = PathManager(config, target)

BANDS = range(20, 400)
EDGES = np.array([20, 100, 200, 300, 400])  # cut bands
Q = 0.1  # % of injections a cut may lose
N_PROC, THREADS, CHUNK = 8, 32, 2000
SKY_GRID = f"{home_dir}config/gc_sky_grid.txt"
INJ = ["injections-v2-0", "injections-v2-1", "injections-v2-2", "injections-v2-3"]  # injection stage of stage k
INJ_TCOH_ORDER = [(5, 2), (10, 2), (20, 2), (40, 3)]
STAGES = {
    0: dict(stage="search-0", tcoh=5, order=2),
    1: dict(stage="followup-v3-1", tcoh=10, order=2, n_sky=1),
    2: dict(stage="followup-v3-2", tcoh=20, order=2, n_sky=57, n_seg=27, tasks_per_job=114, memory="2GB"),
    3: dict(stage="followup-v3-3", tcoh=40, order=3, n_sky=57, n_seg=14, tasks_per_job=57, memory="4GB"),
}
# earlier runs of stage k: (stage, tcoh, order, seeds); seed i owns files i*n_sky+1 .. (i+1)*n_sky
REUSE = {
    1: [("followup-1", 10, 2, "search-0 clustered")],
    2: [("followup-2", 20, 2, "followup-1 clustered"), ("followup-v2-2", 20, 2, "v2 changed seeds")],
    3: [("followup-v2-3", 40, 3, "followup-v2-2 clustered")],
}
KEY = ("freq", "f1dot", "f2dot")


def tn(stage, tcoh, order, freq):
    return f"{target['name']}_{stage}_TCoh{tcoh}_O{order}_{freq}Hz"


def stage_tn(k, freq):
    s = STAGES[k]
    return tn(s["stage"], s["tcoh"], s["order"], freq)


def read_rows(path):
    with fits.open(path) as h:
        d = h[1].data
        return np.array(d) if d is not None else None


def band_index(f):
    return np.searchsorted(EDGES, np.asarray(f), side="right") - 1


def log_hl(h1, l1):
    """log10 r_HL, r_HL = (2F_H1-4)/(2F_L1-4); +-5 when a detector is at or below 4."""
    h, l = np.asarray(h1, float) - 4, np.asarray(l1, float) - 4
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(h / l)
    return np.where((h > 0) & (l > 0), x, np.where(h > l, 5.0, -5.0))


def cut_file(kind, k):
    if kind == "ratio":
        return f"{home_dir}config/{INJ[k - 1]}_vs_{INJ[k]}_threshold_v3.txt"
    return f"{home_dir}config/{INJ[k]}_hl_window_v3.txt"


def lookup(path, freq):
    """Values of the cut-file row of each frequency (10 Hz rows)."""
    t = np.loadtxt(path)
    i = ((np.asarray(freq) - t[0, 0]) // (t[0, 1] - t[0, 0])).astype(int)
    return t[i, 2:]


def in_window(k, freq, h1, l1):
    w = lookup(cut_file("hl", k), freq)
    x = log_hl(h1, l1)
    return (x >= w[:, 0]) & (x <= w[:, 1])


# ---------------------------------------------------------------- cuts
def cmd_cuts(args):
    def read(k):
        st, (t, o) = INJ[k], INJ_TCOH_ORDER[k]
        out = {}
        for f in BANDS:
            fn = paths.outlier_file(f, tn(st, t, o, f), st, cluster=True)
            if fn.is_file():
                with fits.open(fn) as h:
                    if h[1].data is not None and len(h[1].data):
                        out[f] = (np.array(h[1].data), np.array(h[2].data["Freq"]))
        return out

    def write(path, header, values):
        with open(path, "w") as fo:
            fo.write(header)
            for f0 in range(EDGES[0], EDGES[-1], 10):
                fo.write(f"{f0}\t{f0 + 10}\t" + "\t".join(f"{v:.4f}" for v in values[band_index(f0)]) + "\n")
        print("wrote", path)

    data = [read(k) for k in range(len(INJ))]
    for k in range(len(INJ)):
        f = np.concatenate([np.full(len(r), b) for b, (r, _) in data[k].items()])
        h1 = np.concatenate([r["mean2F_H1"] for r, _ in data[k].values()])
        l1 = np.concatenate([r["mean2F_L1"] for r, _ in data[k].values()])
        x = log_hl(h1, l1)
        use = (h1 > 4) & (l1 > 4) if args.both_detectors else np.ones(x.size, bool)
        win = [np.percentile(x[use & (band_index(f) == i)], [Q / 2, 100 - Q / 2]) for i in range(len(EDGES) - 1)]
        write(cut_file("hl", k), "#f_start\tf_end\tlog10 rHL low\tlog10 rHL high\n", win)
    for k in range(1, len(INJ)):
        f, r = [], []
        for band, (rb, fb) in data[k].items():
            ra, fa = data[k - 1][band]
            ia = {round(v, 9): i for i, v in enumerate(fa)}
            for j, v in enumerate(fb):
                f.append(band)
                r.append((rb["mean2F"][j] - 4) / (ra["mean2F"][ia[round(v, 9)]] - 4))
        f, r = np.array(f), np.array(r)
        th = [(np.percentile(r[band_index(f) == i], Q), r[band_index(f) == i].min()) for i in range(len(EDGES) - 1)]
        write(cut_file("ratio", k), f"#f_start\tf_end\t{Q:g} percentile\tlowest\n", th)


# ---------------------------------------------------------------- seeds and reuse
def seeds(k, freq):
    """Seeds of stage k: clustered outliers of stage k-1 (stage 0 also inside the t5 window)."""
    s = STAGES[k - 1]
    d = read_rows(paths.outlier_file(freq, stage_tn(k - 1, freq), s["stage"], cluster=True))
    if d is None:
        return None
    if k == 1:
        d = d[in_window(0, np.full(len(d), freq), d["mean2F_H1"], d["mean2F_L1"])]
    return d


def keys(rows):
    return [tuple(float(r[c]) for c in KEY) for r in rows]


def reuse_seeds(k, freq):
    """[(stage, tcoh, order, {key: seed index})] of the earlier runs of stage k."""
    out = []
    for st, t, o, src in REUSE[k]:
        if src == "followup-1 clustered":
            fn = paths.outlier_file(freq, tn("followup-1", 10, 2, freq), "followup-1", cluster=True)
            rows = read_rows(fn) if fn.is_file() else None
        elif src == "v2 changed seeds":
            m = np.load(f"{home_dir}results/followup-v2-2/seed_map_followup-v2-1_vs_followup-1.npz")
            if f"{freq}_exact" not in m.files:
                rows = None
            else:
                fn = paths.outlier_file(freq, tn("followup-v2-1", 10, 2, freq), "followup-v2-1", cluster=True)
                rows = read_rows(fn)[np.flatnonzero(m[f"{freq}_exact"] < 0)]
        elif src == "followup-v2-2 clustered":
            fn = paths.outlier_file(freq, tn("followup-v2-2", 20, 2, freq), "followup-v2-2", cluster=True)
            rows = read_rows(fn) if fn.is_file() else None
        else:
            raise ValueError(src)
        out.append((st, t, o, {} if rows is None else {key: i for i, key in enumerate(keys(rows))}))
    return out


def seed_map(k, freq, rows):
    """Per seed: index into REUSE[k] (-1: new) and seed index in that run (or in followup-v3-k)."""
    src, idx = np.full(len(rows), -1), np.zeros(len(rows), int)
    for s, (_, _, _, table) in enumerate(reuse_seeds(k, freq)):
        for i, key in enumerate(keys(rows)):
            if src[i] < 0 and key in table:
                src[i], idx[i] = s, table[key]
    new = np.flatnonzero(src < 0)
    idx[new] = np.arange(new.size)
    return src, idx


def map_path(k):
    return Path(f"{home_dir}results/{STAGES[k]['stage']}/seed_map.npz")


# ---------------------------------------------------------------- dag
def cmd_dag(args):
    k = args.k
    s, p = STAGES[k], STAGES[k - 1]
    _, df_names = phase_param_name(p["order"])
    d_alpha, d_delta = np.loadtxt(SKY_GRID, unpack=True)
    metric = f"osdf:///igwn/cit/staging/hoitim.cheung/metricSetup/o4ab_t{s['tcoh']}_s{s['order']}.fts"
    maps = dict(np.load(map_path(k))) if map_path(k).exists() else {}
    dag_list = f"{home_dir}dagFiles/{s['stage']}_{target['name']}_dag{'_'.join(map(str, args.bands)) if args.bands else '20-400'}Hz.txt"
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
            data_file = paths.outlier_file(freq, stage_tn(k - 1, freq), p["stage"], cluster=True)
            spacing = {n: fits.getval(data_file, n, 0) for n in df_names}
            tau = 86400 * 365.25 * (300 if freq < 200 else 300 + (freq - 199) * 0.5)
            params = FollowUpParamGenerator(
                PowerLawModel(nc_min=config["nc_min"], nc_max=config["nc_max"], tau=tau)
            ).generate_parameter(
                alpha=target["alpha"], dalpha=target["dalpha"], delta=target["delta"], ddelta=target["ddelta"],
                data=rows[new], old_freq_deriv_order=p["order"], new_freq_deriv_order=s["order"],
                spacing=spacing, n_spacing=config["followup_n_spacing"], sky_radius=0,
                spacing_alpha=None, spacing_delta=None,
            )
            dag_file = manager.make_search_dag(
                stage_tn(k, freq), freq, params.data, num_top_list=config["num_top_list"], stage=s["stage"],
                freq_deriv_order=s["order"], n_seg=s["n_seg"], sft_files=paths.sft_ensemble(freq),
                metric_file=metric, request_memory=s["memory"], request_disk="4GB", request_cpu=1,
                use_osg=True, use_osdf=True, tasks_per_job=s["tasks_per_job"], sky_offsets=(d_alpha, d_delta),
            )
            f_daglist.write(f"{dag_file}\n")
        f_daglist.write(f"{manager.make_osdf_cleanup_dag(s['stage'])}\n")
    make_dir([map_path(k)])
    np.savez(map_path(k), **maps)
    print(f"{s['stage']}: {n_reused + n_new:,d} seeds, {n_reused:,d} reuse earlier results, {n_new:,d} new "
          f"({n_new * s['n_sky']:,d} Weave runs); DAG list {dag_list}")


# ---------------------------------------------------------------- outliers
def read_row0(path):
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        if d is None or len(d) == 0:
            raise ValueError(f"empty Weave toplist: {path}")
        return np.array(d[:1])


def files_of(k, freq, src, idx):
    """Weave result files (n_sky per seed) of each seed of stage k."""
    s = STAGES[k]
    runs = REUSE[k]
    out = []
    for a, i in zip(src, idx):
        st, t, o = (runs[a][:3] if a >= 0 else (s["stage"], s["tcoh"], s["order"]))
        first = i * s["n_sky"] + 1
        out.append([paths.weave_output_file(freq, tn(st, t, o, freq), j, st) for j in range(first, first + s["n_sky"])])
    return out


def loudest_chunk(task):
    """Loudest row of each seed in [lo, hi) over its files; cached per chunk."""
    k, freq, lo, hi, src, idx = task
    out = Path(f"{home_dir}results/{STAGES[k]['stage']}/loudest/{freq}_{lo}.npy")
    if out.exists():
        return
    files = [f for block in files_of(k, freq, src, idx) for f in block]
    with ThreadPoolExecutor(THREADS) as ex:
        rows = np.concatenate(list(ex.map(read_row0, files))).reshape(hi - lo, -1)
    best = rows[np.arange(hi - lo), rows["mean2F"].argmax(axis=1)]
    make_dir([out])
    np.save(str(out) + ".tmp.npy", best)
    os.replace(str(out) + ".tmp.npy", out)
    print(f"{STAGES[k]['stage']} {freq} Hz seeds {lo}-{hi}: done", flush=True)


def write_band(k, freq, prev, best, passed, th, spacing_files):
    """Unclustered and clustered outlier files of stage k (INFO: one row per seed)."""
    s = STAGES[k]
    spacing = {}
    for fn in spacing_files:
        for key, v in get_spacing(fn, s["order"]).items():
            spacing[key] = max(spacing.get(key, 0), v)
    primary = fits.PrimaryHDU()
    for key, v in spacing.items():
        primary.header[f"HIERARCH {key}"] = v
    tab = Table(best[passed])
    tab.add_column(th[passed], name="mean2F threshold")
    hdu = fits.BinTableHDU(data=tab, name=f"{s['stage']}_outlier") if passed.any() else fits.BinTableHDU(name=f"{s['stage']}_outlier")
    info = np.recarray((len(prev),), dtype=[(c, ">f8") for c in ("freq", "jobIndex", "outliers", "isSaturated")])
    info["freq"], info["jobIndex"], info["outliers"], info["isSaturated"] = freq, np.arange(len(prev)), passed, 0
    path = paths.outlier_file(freq, stage_tn(k, freq), s["stage"])
    make_dir([path])
    fits.HDUList([primary, hdu, fits.BinTableHDU(data=info, name="info")]).writeto(path, overwrite=True)
    primary.header["HIERARCH cluster_n_spacing"] = config["cluster_n_spacing"]
    result_manager._write_clustered_results(freq, stage_tn(k, freq), s["stage"], hdu.data, s["order"], primary)


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
                all5 = read_rows(paths.outlier_file(freq, stage_tn(0, freq), "search-0", cluster=True))
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
    print(f"{s['stage']}: {len(maps)} bands, {sum(len(v[0]) for v in maps.values()):,d} seeds, {len(tasks)} chunks", flush=True)
    with Pool(N_PROC) as pool:
        for _ in pool.imap_unordered(loudest_chunk, tasks):
            pass
    n_seeds = n_pass = 0
    for freq, (src, idx) in maps.items():
        prev = seeds(k, freq)
        best = np.concatenate([np.load(f"{home_dir}results/{s['stage']}/loudest/{freq}_{lo}.npy")
                               for lo in range(0, len(src), CHUNK)])
        r = lookup(cut_file("ratio", k), np.full(len(prev), freq))[:, 0]
        th = (prev["mean2F"].astype(float) - 4) * r + 4
        passed = (best["mean2F"] >= th) & in_window(k, np.full(len(best), freq), best["mean2F_H1"], best["mean2F_L1"])
        blocks = files_of(k, freq, src, idx)
        write_band(k, freq, prev, best, passed, th, [b[0] for b in blocks[:: max(1, len(blocks) // 10)][:10]])
        n_seeds += len(prev)
        n_pass += int(passed.sum())
        print(f"{s['stage']} {freq} Hz: {len(prev)} seeds, {int(passed.sum())} pass", flush=True)
    print(f"{s['stage']}: {n_pass:,d} of {n_seeds:,d} seeds pass")


manager = WorkflowManager(config, target)
result_manager = ResultAnalysisManager(config, target)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cuts")
    c.add_argument("--both-detectors", action="store_true", help="H1/L1 windows from injections with both 2F > 4")
    for name in ("dag", "outliers"):
        c = sub.add_parser(name)
        c.add_argument("k", type=int, choices=[2, 3] if name == "dag" else [1, 2, 3])
        c.add_argument("--bands", type=int, nargs="+")
    a = ap.parse_args()
    {"cuts": cmd_cuts, "dag": cmd_dag, "outliers": cmd_outliers}[a.cmd](a)
