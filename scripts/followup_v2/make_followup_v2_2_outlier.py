"""t20 outliers of the regular candidates (stage followup-v2-2).

Seeds: the clustered followup-v2-1 rows (top 1 per t5 parent, v2 t5->t10 cut). Their t20 O2 57-point sky-grid
Weave results come from two stages (same Weave setup, see check_followup_v2_1_seeds.py):
  * seeds identical to an old followup-1 row j (SEED_MAP <f>_exact = j >= 0) reuse followup-2:
    files j*57+1 .. (j+1)*57;
  * changed seeds (exact < 0) ran in followup-v2-2, in row order (= changed_rows.txt):
    the k-th changed seed owns files k*57+1 .. (k+1)*57.

Cut as for the injections: the loudest candidate over a seed's 57 sky points must reach
(mean2F_t10 - 4) * r + 4, r = v2 t10->t20 1st percentile of its frequency bin; only that candidate is kept
(num_toplist = 1). Weave toplists are sorted by mean2F, so only row 0 of each file is read, which gives the
same result as ResultAnalysisManager.make_outlier(num_toplist=1, n_sky=57) at a fraction of the I/O.

Reading ~6M files from OSDF: N_PROC processes x THREADS threads (~1000 files/s). The loudest row of every seed
is cached per chunk in results/followup-v2-2/loudest/, so a rerun only redoes missing chunks.
INFO 'jobIndex' = followup-v2-1 clustered row index of the seed (not a Weave job index: two stages are mixed).
A missing Weave file raises: no silent skipping.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager
from paws.io import get_spacing, make_dir

with open("/home/hoitim.cheung/galacticCenter/config/config.yaml") as f:
    config = yaml.safe_load(f)
home_dir = config["home_dir"]
with open(f"{home_dir}config/gal.yaml") as f:
    target = yaml.safe_load(f)

paths = PathManager(config, target)
result_manager = ResultAnalysisManager(config, target)

#################################################################
fmin, fmax = 20, 400
N_PROC, THREADS = 8, 32
CHUNK = 2000  # seeds per task (114k files, ~15 min)
prev_stage, prev_tcoh = "followup-v2-1", 10
old_stage = "followup-2"  # reused t20 results of unchanged seeds
stage, tcoh, freq_deriv_order = "followup-v2-2", 20, 2
n_sky = 57
threshold_file = f"{home_dir}config/injections-v2-1_vs_injections-v2-2_threshold.txt"
# loaded into memory: a lazy NpzFile handle shared by forked workers corrupts concurrent reads
SEED_MAP = dict(np.load(f"{home_dir}results/followup-v2-2/seed_map_followup-v2-1_vs_followup-1.npz"))
LOUD_DIR = f"{home_dir}results/followup-v2-2/loudest/"
#################################################################

fs, fe, ratio_th, _ = np.loadtxt(threshold_file).T
band_step = fe[0] - fs[0]


def taskname(st, tc, freq):
    return f"{target['name']}_{st}_TCoh{tc}_O{freq_deriv_order}_{freq}Hz"


def seed_files(freq, n_seeds):
    """(stage, first file index) of every seed's block of n_sky Weave files."""
    exact = SEED_MAP[f"{freq}_exact"]
    assert len(exact) == n_seeds, f"{freq} Hz: seed map {len(exact)} vs {n_seeds} followup-v2-1 rows"
    changed = np.flatnonzero(exact < 0)
    if len(changed):
        rows = np.loadtxt(paths.dag_file(freq, "x", stage).parent / "changed_rows.txt", dtype=int, ndmin=1)
        assert np.array_equal(rows, changed), f"{freq} Hz: changed_rows.txt does not match the seed map"
    k_of = {row: k for k, row in enumerate(changed)}
    return [(old_stage, exact[i] * n_sky + 1) if exact[i] >= 0 else (stage, k_of[i] * n_sky + 1)
            for i in range(n_seeds)]


def read_row0(path):
    with fits.open(path, memmap=True) as h:
        return np.array(h[1].data[:1])


def loudest_chunk(task):
    """Loudest candidate (row 0 of the best of its n_sky files) of seeds [lo, hi) of one band, cached."""
    freq, lo, hi, n_seeds = task
    out = f"{LOUD_DIR}{freq}_{lo}.npy"
    if os.path.exists(out):
        return out
    src = seed_files(freq, n_seeds)[lo:hi]
    files = [paths.weave_output_file(freq, taskname(st, tcoh, freq), j0 + s, st)
             for st, j0 in src for s in range(n_sky)]
    with ThreadPoolExecutor(THREADS) as ex:
        rows = np.concatenate(list(ex.map(read_row0, files))).reshape(hi - lo, n_sky)
    best = rows[np.arange(hi - lo), rows["mean2F"].argmax(axis=1)]
    np.save(out + ".tmp.npy", best)
    os.replace(out + ".tmp.npy", out)
    print(f"{freq} Hz seeds {lo}-{hi}: done", flush=True)
    return out


def write_band(freq, prev, best):
    """Apply the cut and write the unclustered and clustered followup-v2-2 outlier files."""
    th = (prev["mean2F"].astype(float) - 4) * ratio_th[int((freq - fs[0]) // band_step)] + 4
    passed = best["mean2F"] >= th

    # spacing of the t20 grid (max over a sample of files from both stages), for clustering and the next stage
    src = seed_files(freq, len(prev))
    spacing = {}
    for st, j0 in src[:: max(1, len(src) // 10)][:10]:
        for k, v in get_spacing(paths.weave_output_file(freq, taskname(st, tcoh, freq), j0, st), freq_deriv_order).items():
            spacing[k] = max(spacing.get(k, 0), v)

    primary = fits.PrimaryHDU()
    for k, v in spacing.items():
        primary.header[f"HIERARCH {k}"] = v
    outliers = Table(best[passed])
    outliers.add_column(th[passed], name="mean2F threshold")
    out_hdu = (fits.BinTableHDU(data=outliers, name=f"{stage}_outlier") if passed.any()
               else fits.BinTableHDU(name=f"{stage}_outlier"))
    info = np.recarray((len(prev),), dtype=[(c, ">f8") for c in ("freq", "jobIndex", "outliers", "isSaturated")])
    info["freq"], info["jobIndex"], info["outliers"], info["isSaturated"] = freq, np.arange(len(prev)), passed, 0

    tn = taskname(stage, tcoh, freq)
    path = paths.outlier_file(freq, tn, stage, cluster=False)
    make_dir([path])
    fits.HDUList([primary, out_hdu, fits.BinTableHDU(data=info, name="info")]).writeto(path, overwrite=True)
    primary.header["HIERARCH cluster_n_spacing"] = config["cluster_n_spacing"]
    result_manager._write_clustered_results(freq, tn, stage, out_hdu.data, freq_deriv_order, primary)
    return int(passed.sum())


if __name__ == "__main__":
    # optional arguments: only these bands (e.g. a quick test), otherwise fmin..fmax
    freqs = [int(a) for a in sys.argv[1:]] or range(fmin, fmax)
    os.makedirs(LOUD_DIR, exist_ok=True)
    seeds = {}
    for freq in freqs:
        p = paths.outlier_file(freq, taskname(prev_stage, prev_tcoh, freq), prev_stage, cluster=True)
        if p.is_file():
            d = fits.getdata(p, 1)
            if d is not None and len(d):
                seeds[freq] = d
    tasks = [(f, lo, min(lo + CHUNK, len(d)), len(d)) for f, d in seeds.items() for lo in range(0, len(d), CHUNK)]
    tasks.sort(key=lambda t: -(t[2] - t[1]))
    print(f"{len(seeds)} bands, {sum(len(d) for d in seeds.values()):,d} seeds, "
          f"{sum(len(d) for d in seeds.values()) * n_sky:,d} files, {len(tasks)} chunks", flush=True)
    with Pool(N_PROC) as pool:
        for _ in pool.imap_unordered(loudest_chunk, tasks):
            pass

    n_pass = 0
    for freq, prev in seeds.items():
        best = np.concatenate([np.load(f"{LOUD_DIR}{freq}_{lo}.npy") for lo in range(0, len(prev), CHUNK)])
        n = write_band(freq, prev, best)
        n_pass += n
        print(f"Freq={freq}Hz: {len(prev)} seeds, {n} pass", flush=True)
    print(f"Done: {n_pass:,d} of {sum(len(d) for d in seeds.values()):,d} seeds pass")
