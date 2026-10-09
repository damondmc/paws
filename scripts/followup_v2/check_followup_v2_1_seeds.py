"""followup-v2-1 (top-1 per parent) candidates vs old followup-1 rows that drove followup-2."""
import glob, os, re, numpy as np
from collections import Counter
from astropy.io import fits
H = "/home/hoitim.cheung/galacticCenter/"
MAP_OUT = "/home/hoitim.cheung/galacticCenter/results/followup-v2-2/seed_map_followup-v2-1_vs_followup-1.npz"
NEW = H + "results/followup-v2-1/GalacticCenter/C00-C01_Gated_G02_1800s/{f}/Outliers/GalacticCenter_followup-v2-1_TCoh10_O2_{f}Hz_outlier_clustered.fts"
OLD = H + "results/followup-1/GalacticCenter/C00-C01_Gated_G02_1800s/{f}/Outliers/GalacticCenter_followup-1_TCoh10_O2_{f}Hz_outlier_clustered.fts"
RES = "/osdf/igwn/cit/staging/hoitim.cheung/o4ab/results/followup-2/GalacticCenter/C00-C01_Gated_G02_1800s/{f}/Result"
NSKY = 57; P = ("freq", "f1dot", "f2dot")
key = lambda d: list(zip(*(np.round(d[c].astype(float), 12) for c in P + ("mean2F", "alpha", "delta"))))
out, rows = {}, []
for f in range(20, 400):
    if not os.path.exists(NEW.format(f=f)): continue
    new = fits.getdata(NEW.format(f=f), 1)
    if new is None or len(new) == 0: continue
    old = fits.getdata(OLD.format(f=f), 1)
    idx = {k: j for j, k in enumerate(key(old))}
    exact = np.array([idx.get(k, -1) for k in key(new)])
    l = open(sorted(glob.glob(H + f"condorFiles/followup-2/GalacticCenter/{f}/tasks/*task_1.txt"))[0]).readline()
    w = {p: float(re.search(rf"--{p}=[^/ ]+/(\S+)", l).group(1)) for p in P}
    near = np.full(len(new), -1); dist = np.zeros(len(new))
    for i in np.where(exact < 0)[0]:
        d = np.max([np.abs(old[p] - new[p][i]) / (w[p] / 2) for p in P], axis=0)
        d[(old["alpha"] != new["alpha"][i]) | (old["delta"] != new["delta"][i])] = np.inf
        near[i] = int(np.argmin(d)); dist[i] = d.min()
    sizes = {}
    with os.scandir(RES.format(f=f)) as it:
        for e in it: sizes[int(e.name.rsplit(".", 1)[1])] = e.stat().st_size
    full = Counter(sizes.values()).most_common(1)[0][0]
    incomplete = sum(1 for j in exact[exact >= 0] if not all(sizes.get(k) == full for k in range(j * NSKY + 1, (j + 1) * NSKY + 1)))
    out[f"{f}_exact"] = exact; out[f"{f}_near"] = near; out[f"{f}_dist"] = dist
    rows.append((f, len(new), int((exact >= 0).sum()), incomplete, dist[exact < 0]))
np.savez(MAP_OUT, **out)

def summary(sel, label):
    n = sum(r[1] for r in sel); ex = sum(r[2] for r in sel); inc = sum(r[3] for r in sel)
    d = np.concatenate([r[4] for r in sel]) if sel else np.zeros(0)
    bins = [0, 0.25, 0.5, 0.75, 1.0, np.inf]
    h = np.histogram(d, bins)[0]
    print(f"{label}: {n} candidates | exact {ex} ({ex/n*100:.1f}%, incomplete {inc}) | changed seed {n-ex}: "
          f"dist<0.25 {h[0]}, 0.25-0.5 {h[1]}, 0.5-0.75 {h[2]}, 0.75-1 {h[3]}, OUTSIDE {h[4]}")
summary(rows, "ALL")
summary([r for r in rows if r[0] != 314], "without 314 Hz")
summary([r for r in rows if r[0] == 314], "314 Hz")
print("largest changed-seed counts:", sorted([(r[0], r[1] - r[2]) for r in rows], key=lambda x: -x[1])[:8])
