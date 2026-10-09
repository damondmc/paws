"""Loudest candidate (row 0) of every followup-1 Weave file, for postprocess/followup.ipynb.

File j of band f <-> row j-1 of the clustered search-0 outlier file.

Output: results/followup_v2/t10_loudest/<f>_<lo>.npy (mean2F, freq, f1dot, f2dot), one file per chunk of
CHUNK rows; existing chunks are skipped.
--det: results/followup_v2/t10_loudest_det/, adds mean2F_H1 and mean2F_L1; default bands include SAT_BANDS.

Usage (from paws/): uv run python scripts/followup_v2/build_followup_v2_cache.py [--det] [bands]
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import yaml
from astropy.io import fits

from paws.definitions import task_name
from paws.filepaths import PathManager

with open("/home/hoitim.cheung/galacticCenter/config/config.yaml") as f:
    config = yaml.safe_load(f)
home_dir = config["home_dir"]
with open(f"{home_dir}config/gal.yaml") as f:
    target = yaml.safe_load(f)
paths = PathManager(config, target)

#################################################################
fmin, fmax = 20, 400
SAT_BANDS = [299, 302, 303, 306, 307]  # skipped in the regular chain
N_PROC, THREADS = 8, 32
CHUNK = 5000
DET = "--det" in sys.argv
OUT_DIR = f"{home_dir}results/followup_v2/t10_loudest{'_det' if DET else ''}/"
COLS = ["mean2F", "freq", "f1dot", "f2dot"] + (["mean2F_H1", "mean2F_L1"] if DET else [])
DTYPE = [(c, "f8") for c in COLS]
#################################################################


def tn(stage, tcoh, order, freq):
    return f"{task_name(target['name'], stage, tcoh, order)}_{freq}Hz"


def read_row0(path):
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        if d is None or len(d) == 0:
            raise ValueError(f"empty Weave toplist: {path}")
        r = d[0]
        return tuple(float(r[c]) for c in COLS)


def chunk(task):
    freq, lo, hi = task
    out = f"{OUT_DIR}{freq}_{lo}.npy"
    if os.path.exists(out):
        return 0
    files = [paths.weave_output_file(freq, tn("followup-1", 10, 2, freq), j, "followup-1") for j in range(lo + 1, hi + 1)]
    with ThreadPoolExecutor(THREADS) as ex:
        rows = np.array(list(ex.map(read_row0, files)), dtype=DTYPE)
    np.save(out + ".tmp.npy", rows)
    os.replace(out + ".tmp.npy", out)
    return hi - lo


if __name__ == "__main__":
    freqs = [int(a) for a in sys.argv[1:] if a != "--det"] or [f for f in range(fmin, fmax) if DET or f not in SAT_BANDS]
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    for freq in freqs:
        n = fits.getval(paths.outlier_file(freq, tn("search-0", 5, 2, freq), "search-0", cluster=True), "NAXIS2", 1)
        tasks += [(freq, lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]
    tasks.sort(key=lambda t: -(t[2] - t[1]))
    n_tot = sum(t[2] - t[1] for t in tasks)
    print(f"{len(freqs)} bands, {n_tot:,d} files, {len(tasks)} chunks", flush=True)
    t0, done = time.time(), 0
    with Pool(N_PROC) as pool:
        for n in pool.imap_unordered(chunk, tasks):
            done += n
            print(f"{done:,d} files read, {done / max(time.time() - t0, 1e-9):.0f} files/s", flush=True)
    print(f"Done in {time.time() - t0:.0f} s")
