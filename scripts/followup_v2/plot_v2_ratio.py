"""(2F-4) ratio between two v2 injection stages: histogram, 1st percentile per 20 Hz bin, threshold file."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

H = "/home/hoitim.cheung/galacticCenter/"
BASE = H + "results/{s}/GalacticCenter/C00-C01_Gated_G02_1800s/{f}/Outliers/GalacticCenter_{s}_TCoh{t}_O{o}_{f}Hz_outlier_clustered.fts"
# Usage: python plot_v2_ratio.py <pair>, pair = 0-1 (t5->t10), 1-2 (t10->t20) or 2-3 (t20->t40)
# stage: (name, tcoh, order, n_seg); old threshold file for comparison
STAGES = {0: ("injections-v2-0", 5, 2, 107), 1: ("injections-v2-1", 10, 2, 54),
          2: ("injections-v2-2", 20, 2, 27), 3: ("injections-v2-3", 40, 3, 14)}
pair = sys.argv[1] if len(sys.argv) > 1 else "0-1"
a, b = map(int, pair.split("-"))
PREV, NEXT = STAGES[a], STAGES[b]
OLD_TH = H + f"config/injections-{a}_vs_injections-{b}_threshold.txt"
# 20 inj/Hz -> 20 Hz bins hold only ~380 injections (~4 below the 1st percentile);
# 100 Hz bins (first one 80 Hz) give ~1500-1900 per bin.
EDGES = np.array([20, 100, 200, 300, 400])
N_BOOT = 2000
IDEAL = PREV[3] / NEXT[3]
tag = f"{PREV[0]}_vs_{NEXT[0]}"
OUT_PNG = H + f"plots/{tag}_ratio.png"
OUT_TH = H + f"config/{tag}_threshold.txt"

freq, ratio, m_prev, m_next = [], [], [], []
n_prev_tot = n_lost = 0
for f in range(20, 400):
    try:
        a = fits.open(BASE.format(s=PREV[0], t=PREV[1], o=PREV[2], f=f))
        b = fits.open(BASE.format(s=NEXT[0], t=NEXT[1], o=NEXT[2], f=f))
    except FileNotFoundError:
        continue
    if len(a[1].data) == 0:
        continue
    # match the followed-up injections back to their previous-stage row by injected Freq
    ia = {round(x, 9): i for i, x in enumerate(a[2].data["Freq"])}
    n_prev_tot += len(a[1].data)
    n_lost += len(a[1].data) - len(b[1].data)
    for j, x in enumerate(b[2].data["Freq"]):
        i = ia[round(x, 9)]
        p, n = a[1].data["mean2F"][i], b[1].data["mean2F"][j]
        freq.append(f); m_prev.append(p); m_next.append(n); ratio.append((n - 4) / (p - 4))
freq, ratio = np.array(freq), np.array(ratio)
print(f"{len(ratio)} injections matched ({n_prev_tot} at {PREV[0]}, {n_lost} without a {NEXT[0]} row)")
print(f"ratio median {np.median(ratio):.3f}, 1st pct {np.percentile(ratio, 1):.3f}, min {ratio.min():.3f}, ideal {IDEAL:.3f}")

edges = EDGES
rng = np.random.default_rng(0)
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    r = ratio[(freq >= lo) & (freq < hi)]
    boot = np.percentile(rng.choice(r, (N_BOOT, r.size)), 1, axis=1)
    rows.append((lo, hi, np.percentile(r, 1), r.min(), len(r), boot.std()))
# Bins are unequal (first is 80 Hz), so write the threshold file on a uniform 20 Hz
# grid (same format and band step as before), each row carrying its wide bin's value.
with open(OUT_TH, "w") as fo:
    fo.write("#f_start\tf_end\t1 percentile\tlowest\n")
    for f0 in range(20, 400, 20):
        lo, hi, p1, mn, *_ = next(r for r in rows if r[0] <= f0 < r[1])
        fo.write(f"{f0}\t{f0 + 20}\t{p1:.4f}\t{mn:.4f}\n")
print(f"{'band':>9s} {'n':>5s} {'1%':>7s} {'±boot':>7s} {'min':>7s}")
for lo, hi, p1, mn, n, sd in rows:
    print(f"{lo:4d}-{hi:<4d} {n:5d} {p1:7.4f} {sd:7.4f} {mn:7.4f}")

old = np.loadtxt(OLD_TH)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
a = ax[0]
p1 = np.percentile(ratio, 1)
a.hist(ratio, bins=np.linspace(min(0.5, ratio.min()), max(3, np.percentile(ratio, 99.9)), 60), histtype="step", lw=2, color="#1f77b4",
       label=f"median {np.median(ratio):.2f}")
a.axvline(p1, color="#d62728", ls=":", lw=2, label=f"1st percentile {p1:.3f}")
a.axvline(IDEAL, color="k", ls="--", label=f"ideal {PREV[3]}/{NEXT[3]} = {IDEAL:.2f}")
a.set_xlabel(rf"$(\overline{{2F}}_{{t{NEXT[1]}}}-4)/(\overline{{2F}}_{{t{PREV[1]}}}-4)$")
a.set_ylabel("injections")
a.set_title(f"t{PREV[1]} → t{NEXT[1]}, {len(ratio)} injections, 20–400 Hz")
a.legend(fontsize=9)

a = ax[1]
mid = np.array([(lo + hi) / 2 for lo, hi, *_ in rows])
a.step(edges, [r[2] for r in rows] + [rows[-1][2]], where="post", color="#d62728", lw=2, label="v2 1st percentile")
a.errorbar(mid, [r[2] for r in rows], yerr=[r[5] for r in rows], fmt="none", color="#d62728", capsize=4, label="bootstrap 1σ")
a.step(edges, [r[3] for r in rows] + [rows[-1][3]], where="post", color="#d62728", lw=1, ls="--", label="v2 lowest")
old_edges = np.r_[old[:, 0], old[-1, 1]]
a.step(old_edges, list(old[:, 2]) + [old[-1, 2]], where="post", color="#888888", lw=2, label="old 1st percentile (20 Hz bins)")
a.step(old_edges, list(old[:, 3]) + [old[-1, 3]], where="post", color="#888888", lw=1, ls="--", label="old lowest")
a.axhline(IDEAL, color="k", ls="--", lw=1)
a.set_xlabel("frequency [Hz]")
a.set_ylabel("ratio")
a.set_title("Per frequency bin")
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
print("saved", OUT_PNG, "and", OUT_TH)
