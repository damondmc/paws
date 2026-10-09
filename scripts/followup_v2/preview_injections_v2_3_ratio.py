"""t20->t40 injection ratio from the injections-v2-3 Weave files, excluding the PENDING jobs.

Task k (files 57(k-1)+1 .. 57k) <-> row k-1 of the clustered injections-v2-2 file.

Writes
  config/injections-v2-2_vs_injections-v2-3_threshold_preview.txt
  results/sat_followup/inj_ratio_v2-2_v2-3_preview.pkl   (freq, ratio)
  results/sat_followup/inj_t40_preview_det.npz           (band, mean2F, mean2F_H1, mean2F_L1)
"""
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from paws.filepaths import PathManager

H = "/home/hoitim.cheung/galacticCenter/"
with open(H + "config/config.yaml") as f:
    config = yaml.safe_load(f)
with open(H + "config/gal.yaml") as f:
    target = yaml.safe_load(f)
paths = PathManager(config, target)

PREV, NEXT = ("injections-v2-2", 20, 2), ("injections-v2-3", 40, 3)
N_SKY = 57
# (band, task) of unfinished injections-v2-3 jobs
PENDING = {(258, 8), (304, 8), (321, 8), (396, 4)}
EDGES = np.array([20, 100, 200, 300, 400])
N_BOOT = 2000
OUT_TH = H + "config/injections-v2-2_vs_injections-v2-3_threshold_preview.txt"
OUT_PKL = H + "results/sat_followup/inj_ratio_v2-2_v2-3_preview.pkl"
OUT_DET = H + "results/sat_followup/inj_t40_preview_det.npz"


def tn(stage, f):
    s, t, o = stage
    return f"{target['name']}_{s}_TCoh{t}_O{o}_{f}Hz"


def band(f):
    fn = paths.outlier_file(f, tn(PREV, f), PREV[0], cluster=True)
    if not fn.is_file():
        return None
    with fits.open(fn) as h:
        prev = h[1].data["mean2F"].astype(float)
        inj_freq = h[2].data["Freq"]
    if len(prev) == 0:
        return None
    tdir =f"{H}condorFiles/{NEXT[0]}/{target['name']}/{f}/tasks/"
    keep = []
    for k in range(1, len(prev) + 1):
        if (f, k) in PENDING:
            continue
        with open(f"{tdir}{tn(NEXT, f)}_task_{k}.txt") as ft:
            x = float(re.search(r";Freq=([^;]+);", ft.readline()).group(1))
        assert abs(x - inj_freq[k - 1]) < 1e-9, f"{f} Hz task {k}: Freq {x} vs v2-2 row {inj_freq[k - 1]}"
        keep.append(k)

    def one(j):  # (mean2F, mean2F_H1, mean2F_L1) of the loudest candidate
        d = fits.getdata(paths.weave_output_file(f, tn(NEXT, f), j, NEXT[0]), 1)
        i = np.argmax(d["mean2F"])
        return float(d["mean2F"][i]), float(d["mean2F_H1"][i]), float(d["mean2F_L1"][i])

    with ThreadPoolExecutor(32) as ex:
        # loudest over the 57 sky points
        nxt = np.array([max(ex.map(one, range(N_SKY * (k - 1) + 1, N_SKY * k + 1))) for k in keep])
    i = np.array(keep) - 1
    return f, (nxt[:, 0] - 4) / (prev[i] - 4), len(prev) - len(keep), nxt


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        res = [r for r in ex.map(band, range(20, 400)) if r is not None]

    freq = np.concatenate([np.full(len(r), f) for f, r, *_ in res])
    ratio = np.concatenate([r for _, r, *_ in res])
    n_left_out = sum(n for _, _, n, _ in res)
    assert n_left_out == len(PENDING), n_left_out
    pd.to_pickle((freq, ratio), OUT_PKL)
    det = np.concatenate([d for *_, d in res])
    np.savez(OUT_DET, band=freq, mean2F=det[:, 0], mean2F_H1=det[:, 1], mean2F_L1=det[:, 2])
    print(f"{len(ratio)} injections ({n_left_out} pending left out)")
    print(f"ratio median {np.median(ratio):.3f}, 1st pct {np.percentile(ratio, 1):.3f}, min {ratio.min():.3f}")

    rng = np.random.default_rng(0)
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        r = ratio[(freq >= lo) & (freq < hi)]
        boot = np.percentile(rng.choice(r, (N_BOOT, r.size)), 1, axis=1)
        rows.append((lo, hi, np.percentile(r, 1), r.min(), len(r), boot.std()))
    with open(OUT_TH, "w") as fo:
        fo.write("#f_start\tf_end\t1 percentile\tlowest\n")
        for f0 in range(20, 400, 20):
            lo, hi, p1, mn, *_ = next(r for r in rows if r[0] <= f0 < r[1])
            fo.write(f"{f0}\t{f0 + 20}\t{p1:.4f}\t{mn:.4f}\n")
    print(f"{'band':>9s} {'n':>5s} {'1%':>7s} {'±boot':>7s} {'min':>7s}")
    for lo, hi, p1, mn, n, sd in rows:
        print(f"{lo:4d}-{hi:<4d} {n:5d} {p1:7.4f} {sd:7.4f} {mn:7.4f}")
    print("saved", OUT_TH, OUT_PKL, "and", OUT_DET)
