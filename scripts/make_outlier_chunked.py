"""
One-off outlier collection for bands whose raw Weave results are too large to
analyse in one pass (~200 GB in a single 1 Hz band).

Nothing about the search changes -- only how its results are read. The band's
parameter points are cut into N_CHUNKS contiguous slices; each slice is
analysed on its own with make_outlier(param_indices=...), which reads only that
slice's result files, and is written to a part file. The parts are then
concatenated into the ordinary outlier file (and clustered) exactly as one
unsplit run would have produced it, because the slices are contiguous and are
combined in order -- so the row order that links a candidate back to its parent
job is preserved.

Part files are kept until the merge succeeds and finished chunks are skipped on
a re-run, so an interrupted pass resumes instead of re-reading everything.

Only follow-up stages can be chunked: a search stage's non_sat_band extension
comes from reshaping the info table into the band's f0 sub-bands, which a
partial info table cannot fill.
"""

import os
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table, vstack

from paws.analysis.outlier import ResultAnalysisManager
from paws.definitions import phase_param_name
from paws.filepaths import PathManager
from paws.io import make_dir

#################################################################
CONFIG_FILE = "/home/hoitim.cheung/galacticCenter/config/config.yaml"
TARGET_FILE = "/home/hoitim.cheung/galacticCenter/config/gal.yaml"

# Bands to analyse this way, and how many slices each is cut into.
big_bands = [314]
n_chunks = 30

# Which slices this run works on, 1-based: None does all of them in one go, or
# name a subset (e.g. [1] then [2] ...) to spread the band over several runs.
# The merge happens as soon as every part is on disk, whichever run that is.
chunks_to_run = None

THREADS = 32
cluster = True
keep_parts = True  # part files are deleted after a successful merge when False

is_injection = False
prev_stage = "followup-1"
prev_tcoh = 10
prev_freq_deriv_order = 2

stage = "followup-2"
tcoh = 20
freq_deriv_order = 2

n_sky = 57

if is_injection:
    num_toplist = 1
else:
    num_toplist = 10

# For non-injection runs, threshold on mean2F scaled by the injection-derived
# efficiency ratio between the previous and current follow-up stage.
prev_inj_stage = "injections-1"
now_inj_stage = "injections-2"
#################################################################


def resolve_outlier_file(paths, freq, taskname, stage, cluster):
    """Previous-stage outlier files land under home_dir for a local run, or
    under OSDF when written by the outlier-collection Condor job."""
    path = paths.outlier_file(freq, taskname, stage, cluster=cluster)
    if not path.exists():
        path = paths.outlier_file(freq, taskname, stage, cluster=cluster, osdf=True)
    return path


def part_file(paths, freq, taskname, stage, chunk):
    """Where slice `chunk` of a split collection is parked until the merge."""
    path = paths.outlier_file(freq, taskname, stage, cluster=False)
    return path.with_name(f"{path.stem}_part{chunk}{path.suffix}")


def merge_parts(result_manager, part_files, freq, taskname, stage, freq_deriv_order):
    """Concatenates the part files into the band's outlier file, and clusters
    it, reproducing what a single unsplit make_outlier call would have written.
    `part_files` must be in ascending slice order."""
    _, dfn = phase_param_name(freq_deriv_order)

    outlier_tables = []
    inj_tables = []
    info_arrays = []
    max_spacing = {}
    scalar_th = None

    for path in part_files:
        with fits.open(path) as hdul:
            header = hdul[0].header
            # Each part reports the coarsest grid of the files it read; the
            # merged file has to report the coarsest over the whole band.
            for key in dfn:
                if f"HIERARCH {key}" in header:
                    max_spacing[key] = max(
                        max_spacing.get(key, 0), header[f"HIERARCH {key}"]
                    )
            if "HIERARCH mean2F_th" in header:
                scalar_th = header["HIERARCH mean2F_th"]

            outliers = hdul[f"{stage}_outlier"].data
            if outliers is not None and len(outliers) > 0:
                outlier_tables.append(Table(outliers))

            if any(hdu.name.upper() == "INJECTION" for hdu in hdul):
                injections = hdul["INJECTION"].data
                if injections is not None and len(injections) > 0:
                    inj_tables.append(Table(injections))

            # Copy, so the merge outlives the closed (memory-mapped) file.
            info_arrays.append(np.array(hdul["info"].data))

    primary_hdu = fits.PrimaryHDU()
    if scalar_th is not None:
        primary_hdu.header["HIERARCH mean2F_th"] = scalar_th
    for key, val in max_spacing.items():
        primary_hdu.header[f"HIERARCH {key}"] = val

    hdus = [primary_hdu]

    if outlier_tables:
        hdus.append(
            fits.BinTableHDU(data=vstack(outlier_tables), name=f"{stage}_outlier")
        )
    else:
        hdus.append(fits.BinTableHDU(name=f"{stage}_outlier"))

    if is_injection:
        if inj_tables:
            hdus.append(fits.BinTableHDU(data=vstack(inj_tables), name="injection"))
        else:
            hdus.append(fits.BinTableHDU(name="injection"))

    hdus.append(
        fits.BinTableHDU(
            data=np.concatenate(info_arrays).view(np.recarray), name="info"
        )
    )

    outlier_file_path = result_manager.paths.outlier_file(
        freq, taskname, stage, cluster=False
    )
    make_dir([outlier_file_path])
    fits.HDUList(hdus).writeto(outlier_file_path, overwrite=True)

    if cluster and hdus[1].data is not None:
        primary_hdu.header["HIERARCH cluster_n_spacing"] = result_manager.config[
            "cluster_n_spacing"
        ]
        inj_hdu = next((h for h in hdus if h.name == "INJECTION"), None)
        outlier_file_path = result_manager._write_clustered_results(
            freq,
            taskname,
            stage,
            hdus[1].data,
            freq_deriv_order,
            primary_hdu,
            inj_hdu=inj_hdu,
        )

    return outlier_file_path


if "search" in stage.lower():
    raise ValueError(
        f"Stage '{stage}' is a search stage; chunking it would break the "
        "non_sat_band extension. Only follow-up stages can be chunked."
    )

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

for freq in big_bands:
    prev_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"
    taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

    data_file = resolve_outlier_file(paths, freq, prev_taskname, prev_stage, cluster)

    if not Path(data_file).is_file() or fits.getdata(data_file).size == 0:
        print(f"No outlier for {freq}Hz, skip.")
        continue

    prev_data = fits.getdata(data_file)

    if is_injection:
        mean2f_th = np.zeros_like(prev_data["mean2F threshold"])
    else:
        idx = int((freq - fs[0]) // band_step)
        mean2f_th = (prev_data["mean2F"] - 4) * ratio_th[idx] + 4
        print(f"Freq={freq}Hz: th={ratio_th[idx]}")

    n_jobs = mean2f_th.size

    # Contiguous slices, so concatenating the parts in order reproduces the row
    # order of an unsplit run.
    bounds = np.linspace(0, n_jobs, n_chunks + 1).astype(int)
    slices = [
        (int(start), int(stop))
        for start, stop in zip(bounds[:-1], bounds[1:])
        if start != stop
    ]

    print(
        f"{freq}Hz: {n_jobs} parameter points ({n_jobs * n_sky} result files) "
        f"over {len(slices)} chunks"
    )

    part_files = []
    for chunk, (param_start, param_stop) in enumerate(slices, start=1):
        part_path = part_file(paths, freq, taskname, stage, chunk)
        part_files.append(part_path)

        if part_path.exists():
            print(f"  chunk {chunk}/{len(slices)}: already done, skipping")
            continue

        if chunks_to_run is not None and chunk not in chunks_to_run:
            print(f"  chunk {chunk}/{len(slices)}: not in this run, skipping")
            continue

        print(
            f"  chunk {chunk}/{len(slices)}: points {param_start}-{param_stop} "
            f"({(param_stop - param_start) * n_sky} result files)"
        )

        written = result_manager.make_outlier(
            taskname,
            freq,
            mean2f_th,
            n_jobs,
            num_toplist=num_toplist,
            stage=stage,
            freq_deriv_order=freq_deriv_order,
            n_sky=n_sky,
            cluster=False,
            work_in_local_dir=False,
            separate_saturated=False,
            is_injection=is_injection,
            max_workers=THREADS,
            param_indices=np.arange(param_start, param_stop),
        )

        # make_outlier names its output after the taskname, which is the same
        # for every chunk, so park this one under its own name before the next
        # chunk overwrites it.
        os.replace(written, part_path)

    missing = [p for p in part_files if not p.exists()]
    if missing:
        print(
            f"{freq}Hz: {len(missing)} of {len(part_files)} parts still to do; "
            "re-run for the remaining chunks, and the merge follows once they "
            "are all on disk."
        )
        continue

    result_file = merge_parts(
        result_manager, part_files, freq, taskname, stage, freq_deriv_order
    )
    print(f"{freq}Hz: merged {len(part_files)} parts into {result_file}")

    if not keep_parts:
        for part_path in part_files:
            part_path.unlink(missing_ok=True)
