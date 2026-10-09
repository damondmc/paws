"""
One-off re-analysis of an existing follow-up stage, keeping only the loudest
N candidates of each parent job from the previous stage (N = 1 reproduces the
injection test, which follows up a single candidate, instead of the top 10 the
real 10->20 day follow-up used).

No Weave job is re-run. Every kept candidate of the previous stage owns n_sky
consecutive result files, so dropping a candidate just means not reading its
files -- which is the whole point, since reading them is the bottleneck.

Row order is what links a candidate back to its parent. The previous stage's
outlier collection walks its parents in order and appends one block of rows per
parent, and the block sizes are recorded in the `info` extension of the
UNclustered outlier file. The clustered table (the one that drove this stage's
DAG) is a loudness-sorted selection of those same rows, copied verbatim, so its
rows are matched back to the unclustered ones by value.

Note: this writes the same outlier file paths as the full analysis would, so it
overwrites any existing outlier file for this stage/frequency.
"""

from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from tqdm import tqdm

from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager

#################################################################
CONFIG_FILE = "/home/hoitim.cheung/galacticCenter/config/config.yaml"
TARGET_FILE = "/home/hoitim.cheung/galacticCenter/config/gal.yaml"

fmin, fmax = 100, 120
sat_band_list = [299, 302, 303, 306, 307]
THREADS = 32
cluster = True

is_injection = True
prev_stage = "injections-1"
prev_tcoh = 10
prev_freq_deriv_order = 2

stage = "injections-2-o3"
tcoh = 20
freq_deriv_order = 3

n_sky = 57

# Candidates kept per previous-stage parent job. 1 = match the injection test.
n_keep_per_parent = 1

if is_injection:
    num_toplist = 1
else:
    num_toplist = 1

# For non-injection runs, threshold on mean2F scaled by the injection-derived
# efficiency ratio between the previous and current follow-up stage.
prev_inj_stage = "injections-1"
now_inj_stage = "injections-2"

# Columns used to locate a clustered row back in the unclustered table
MATCH_COLUMNS = ["freq", "f1dot", "f2dot", "f3dot", "f4dot", "mean2F", "alpha", "delta"]
#################################################################


def resolve_outlier_file(paths, freq, taskname, stage, cluster):
    """Previous-stage outlier files land under home_dir for a local run, or
    under OSDF when written by the outlier-collection Condor job."""
    path = paths.outlier_file(freq, taskname, stage, cluster=cluster)
    if not path.exists():
        path = paths.outlier_file(freq, taskname, stage, cluster=cluster, osdf=True)
    return path


def parent_of_rows(unclustered_file, clustered_file):
    """Parent job index of every row of the table that drove this stage's jobs."""
    with fits.open(unclustered_file) as hdul:
        unclustered = hdul[1].data
        sizes = np.rint(hdul["info"].data["outliers"]).astype(int)

    n_unclustered = 0 if unclustered is None else len(unclustered)
    if sizes.sum() != n_unclustered:
        raise ValueError(
            f"{unclustered_file}: info extension accounts for {sizes.sum()} rows "
            f"but the outlier table has {n_unclustered}, so the parent blocks "
            "cannot be recovered (written with separate_saturated=True?)."
        )
    parent_of_unclustered = np.repeat(np.arange(len(sizes)), sizes)

    if clustered_file is None:
        return parent_of_unclustered

    with fits.open(clustered_file) as hdul:
        clustered = hdul[1].data
        # Injection-mode clustering keeps one row per input row (each replaced
        # by its cluster centre), so the two tables line up positionally.
        if any(hdu.name.upper() == "INJECTION" for hdu in hdul):
            return parent_of_unclustered

    n_clustered = 0 if clustered is None else len(clustered)
    if n_clustered == 0:
        return np.zeros(0, dtype=int)

    cols = [
        name
        for name in MATCH_COLUMNS
        if name in clustered.columns.names and name in unclustered.columns.names
    ]
    stacked = np.vstack(
        [
            np.column_stack([np.asarray(unclustered[c], dtype=float) for c in cols]),
            np.column_stack([np.asarray(clustered[c], dtype=float) for c in cols]),
        ]
    )
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    # First (lowest-index) unclustered row carrying each distinct key
    first_row = np.full(inverse.max() + 1, -1, dtype=int)
    first_row[inverse[:n_unclustered][::-1]] = np.arange(n_unclustered)[::-1]

    source = first_row[inverse[n_unclustered:]]
    if (source < 0).any():
        raise ValueError(
            f"{(source < 0).sum()} clustered rows have no counterpart in "
            f"{unclustered_file}; the two files are not from the same run."
        )

    return parent_of_unclustered[source]


def top_per_parent(parents, loudness, n_keep):
    """0-based indices of the n_keep loudest rows of each parent, sorted."""
    n_rows = len(parents)
    if n_rows == 0:
        return np.zeros(0, dtype=int)

    order = np.lexsort((-np.asarray(loudness, dtype=float), parents))
    ranked = parents[order]
    is_block_start = np.concatenate(([True], ranked[1:] != ranked[:-1]))
    block_start = np.maximum.accumulate(np.where(is_block_start, np.arange(n_rows), 0))
    rank_in_parent = np.arange(n_rows) - block_start

    return np.sort(order[rank_in_parent < n_keep])


with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

with open(TARGET_FILE, "r") as f:
    target = yaml.safe_load(f)

paths = PathManager(config, target)
result_manager = ResultAnalysisManager(config, target)

if not is_injection:
    threshold_filename = (
        f"{config['home_dir']}config/{prev_inj_stage}_vs_{now_inj_stage}_threshold.txt"
    )
    fs, fe, ratio_th, _ = np.loadtxt(threshold_filename).T
    band_step = fe[0] - fs[0]

n_before = 0
n_after = 0

for freq in tqdm(range(fmin, fmax), total=(fmax - fmin)):
    if freq in sat_band_list:
        print(f"Skipping saturated band {freq} Hz")
        continue

    prev_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"
    taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

    data_file = resolve_outlier_file(paths, freq, prev_taskname, prev_stage, cluster)
    unclustered_file = resolve_outlier_file(
        paths, freq, prev_taskname, prev_stage, cluster=False
    )

    if not Path(data_file).is_file() or fits.getdata(data_file).size == 0:
        print(f"No outlier for {freq}Hz, skip.")
        continue

    if not Path(unclustered_file).is_file():
        print(f"No unclustered {prev_stage} file for {freq}Hz, skip.")
        continue

    prev_data = fits.getdata(data_file)

    if is_injection:
        mean2f_th = np.zeros_like(prev_data["mean2F threshold"])
    else:
        idx = int((freq - fs[0]) // band_step)
        mean2f_th = (prev_data["mean2F"] - 4) * ratio_th[idx] + 4
        print(f"Freq={freq}Hz: th={ratio_th[idx]}")

    n_jobs = mean2f_th.size

    parents = parent_of_rows(unclustered_file, data_file if cluster else None)
    if len(parents) != n_jobs:
        raise ValueError(
            f"{freq}Hz: recovered {len(parents)} parent labels for {n_jobs} "
            "parameter points."
        )

    keep = top_per_parent(parents, prev_data["mean2F"], n_keep_per_parent)
    n_before += n_jobs * n_sky
    n_after += len(keep) * n_sky
    print(
        f"Freq={freq}Hz: {n_jobs} candidates over {len(np.unique(parents))} "
        f"parent jobs -> keeping {len(keep)} "
        f"({len(keep) * n_sky} of {n_jobs * n_sky} result files)"
    )

    result_file = result_manager.make_outlier(
        taskname,
        freq,
        mean2f_th,
        n_jobs,
        num_toplist=num_toplist,
        stage=stage,
        freq_deriv_order=freq_deriv_order,
        n_sky=n_sky,
        cluster=cluster,
        work_in_local_dir=False,
        separate_saturated=False,
        is_injection=is_injection,
        max_workers=THREADS,
        param_indices=keep,
    )

print(f"Read {n_after} result files instead of {n_before}.")
