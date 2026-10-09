"""Stage-level helpers shared by the CLI and the analysis scripts: seeds, cuts, loudest rows, outlier files."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

from paws.definitions import phase_param_name
from paws.io import get_spacing, make_dir

# ---------------------------------------------------------------- outlier files


def resolve_outlier_file(paths, freq, taskname, stage_name, cluster):
    """Outlier file of a stage: under home_dir for a local collection, under OSDF when an
    outlier-collection Condor job wrote it."""
    path = paths.outlier_file(freq, taskname, stage_name, cluster=cluster)
    if not path.exists():
        path = paths.outlier_file(freq, taskname, stage_name, cluster=cluster, osdf=True)
    return path


def read_rows(path, ext):
    """Table `ext` of a FITS file as a numpy array (None when the HDU has no data)."""
    with fits.open(path) as h:
        d = h[ext].data
        return np.array(d) if d is not None else None


def seed_file(settings, paths, stage, freq):
    """Previous-stage outlier file holding the seeds of `stage`."""
    prev = settings.prev(stage)
    return resolve_outlier_file(paths, freq, prev.taskname(settings.target, freq), prev.name, cluster=not stage.prev_sat)


def seeds(settings, paths, stage, freq):
    """(seed rows, injection rows or None) of a follow-up stage; None when the band has no seed file."""
    path = seed_file(settings, paths, stage, freq)
    if not path.is_file():
        return None, None
    if stage.prev_sat:
        rows = read_rows(path, f"{stage.prev}_sat_outlier")
    else:
        rows = read_rows(path, 1)
    inj = read_rows(path, "injection") if stage.is_injection else None
    if rows is None:
        return None, None
    if stage.n_inj_max is not None:
        rows = rows[: stage.n_inj_max]
        inj = None if inj is None else inj[: stage.n_inj_max]
    return rows, inj


def header_spacing(path, order):
    """Grid spacings (df, df1dot, ...) up to `order` from an outlier file's primary header."""
    _, names = phase_param_name(order)
    return {n: fits.getval(path, n, 0) for n in names}


# ---------------------------------------------------------------- cuts


def cut_lookup(path, freq):
    """Values (columns 2..) of the cut-file row of each frequency; rows are equal-width bands."""
    t = np.loadtxt(path, ndmin=2)
    i = ((np.asarray(freq) - t[0, 0]) // (t[0, 1] - t[0, 0])).astype(int)
    return t[i, 2:]


def ratio_threshold(path, freq, prev_mean2f):
    """mean2F threshold of each seed: (2F_prev - 4) * ratio + 4, ratio from the cut file."""
    r = cut_lookup(path, freq)[..., 0]
    return (np.asarray(prev_mean2f, float) - 4) * r + 4


def log_hl(h1, l1):
    """log10 r_HL, r_HL = (2F_H1-4)/(2F_L1-4); +-5 when a detector is at or below 4."""
    h, l = np.asarray(h1, float) - 4, np.asarray(l1, float) - 4
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(h / l)
    return np.where((h > 0) & (l > 0), x, np.where(h > l, 5.0, -5.0))


def in_hl_window(path, freq, h1, l1):
    w = cut_lookup(path, freq)
    x = log_hl(h1, l1)
    return (x >= w[:, 0]) & (x <= w[:, 1])


def band_index(edges, f):
    return np.searchsorted(edges, np.asarray(f), side="right") - 1


def write_cut_file(path, header, edges, values, step):
    """One row per `step` Hz from edges[0] to edges[-1], holding values[band of the row]."""
    with open(path, "w") as fo:
        fo.write(header)
        for f0 in range(edges[0], edges[-1], step):
            fo.write(f"{f0}\t{f0 + step}\t" + "\t".join(f"{v:.4f}" for v in values[band_index(edges, f0)]) + "\n")
    print("wrote", path)


def injection_outliers(settings, paths, stage, freqs):
    """{band: (outlier rows, injected Freq)} of an injection stage (clustered files)."""
    out = {}
    for f in freqs:
        fn = paths.outlier_file(f, stage.taskname(settings.target, f), stage.name, cluster=True)
        if fn.is_file():
            with fits.open(fn) as h:
                if h[1].data is not None and len(h[1].data):
                    out[f] = (np.array(h[1].data), np.array(h[2].data["Freq"]))
    return out


def injection_ratios(prev, now):
    """(band, (2F_now - 4) / (2F_prev - 4)) of every injection in `now`, matched to `prev` by injected Freq."""
    f, r = [], []
    for band, (rb, fb) in now.items():
        ra, fa = prev[band]
        ia = {round(v, 9): i for i, v in enumerate(fa)}
        for j, v in enumerate(fb):
            f.append(band)
            r.append((rb["mean2F"][j] - 4) / (ra["mean2F"][ia[round(v, 9)]] - 4))
    return np.array(f), np.array(r)


# ---------------------------------------------------------------- loudest rows


def read_row0(path):
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        if d is None or len(d) == 0:
            raise ValueError(f"empty Weave toplist: {path}")
        return np.array(d[:1])


def loudest_rows(files, threads):
    """Loudest row of each seed over its result files; `files` is one equal-length list per seed."""
    n = len(files)
    with ThreadPoolExecutor(threads) as ex:
        rows = np.concatenate(list(ex.map(read_row0, [f for block in files for f in block]))).reshape(n, -1)
    return rows[np.arange(n), rows["mean2F"].argmax(axis=1)]


def save_atomic(path, array):
    make_dir([path])
    np.save(str(path) + ".tmp.npy", array)
    os.replace(str(path) + ".tmp.npy", path)


def write_seed_outliers(result_manager, stage, taskname, freq, n_seeds, best, passed, th, spacing_files):
    """Unclustered and clustered outlier files of a stage collected one loudest row per seed
    (INFO: one row per seed)."""
    spacing = {}
    for fn in spacing_files:
        for key, v in get_spacing(fn, stage.order).items():
            spacing[key] = max(spacing.get(key, 0), v)
    primary = fits.PrimaryHDU()
    for key, v in spacing.items():
        primary.header[f"HIERARCH {key}"] = v
    tab = Table(best[passed])
    tab.add_column(th[passed], name="mean2F threshold")
    name = f"{stage.name}_outlier"
    hdu = fits.BinTableHDU(data=tab, name=name) if passed.any() else fits.BinTableHDU(name=name)
    info = np.recarray((n_seeds,), dtype=[(c, ">f8") for c in ("freq", "jobIndex", "outliers", "isSaturated")])
    info["freq"], info["jobIndex"], info["outliers"], info["isSaturated"] = freq, np.arange(n_seeds), passed, 0
    path = result_manager.paths.outlier_file(freq, taskname, stage.name)
    make_dir([path])
    fits.HDUList([primary, hdu, fits.BinTableHDU(data=info, name="info")]).writeto(path, overwrite=True)
    primary.header["HIERARCH cluster_n_spacing"] = result_manager.config.cluster_n_spacing
    result_manager._write_clustered_results(freq, taskname, stage.name, hdu.data, stage.order, primary)


# ---------------------------------------------------------------- chunked collection


def part_file(paths, freq, taskname, stage_name, chunk):
    """Where slice `chunk` of a split collection is parked until the merge."""
    path = paths.outlier_file(freq, taskname, stage_name, cluster=False)
    return path.with_name(f"{path.stem}_part{chunk}{path.suffix}")


def merge_parts(result_manager, part_files, freq, taskname, stage_name, freq_deriv_order, is_injection, cluster):
    """Concatenates the part files (ascending slice order) into the band's outlier file and clusters it,
    reproducing what a single unsplit make_outlier call writes."""
    _, dfn = phase_param_name(freq_deriv_order)
    outlier_tables, inj_tables, info_arrays = [], [], []
    max_spacing = {}
    scalar_th = None

    for path in part_files:
        with fits.open(path) as hdul:
            header = hdul[0].header
            # the merged file reports the coarsest grid over the whole band
            for key in dfn:
                if f"HIERARCH {key}" in header:
                    max_spacing[key] = max(max_spacing.get(key, 0), header[f"HIERARCH {key}"])
            if "HIERARCH mean2F_th" in header:
                scalar_th = header["HIERARCH mean2F_th"]

            outliers = hdul[f"{stage_name}_outlier"].data
            if outliers is not None and len(outliers) > 0:
                outlier_tables.append(Table(outliers))
            if any(hdu.name.upper() == "INJECTION" for hdu in hdul):
                injections = hdul["INJECTION"].data
                if injections is not None and len(injections) > 0:
                    inj_tables.append(Table(injections))
            info_arrays.append(np.array(hdul["info"].data))

    primary_hdu = fits.PrimaryHDU()
    if scalar_th is not None:
        primary_hdu.header["HIERARCH mean2F_th"] = scalar_th
    for key, val in max_spacing.items():
        primary_hdu.header[f"HIERARCH {key}"] = val

    hdus = [primary_hdu]
    if outlier_tables:
        hdus.append(fits.BinTableHDU(data=vstack(outlier_tables), name=f"{stage_name}_outlier"))
    else:
        hdus.append(fits.BinTableHDU(name=f"{stage_name}_outlier"))
    if is_injection:
        if inj_tables:
            hdus.append(fits.BinTableHDU(data=vstack(inj_tables), name="injection"))
        else:
            hdus.append(fits.BinTableHDU(name="injection"))
    hdus.append(fits.BinTableHDU(data=np.concatenate(info_arrays).view(np.recarray), name="info"))

    outlier_file_path = result_manager.paths.outlier_file(freq, taskname, stage_name, cluster=False)
    make_dir([outlier_file_path])
    fits.HDUList(hdus).writeto(outlier_file_path, overwrite=True)

    if cluster and hdus[1].data is not None:
        primary_hdu.header["HIERARCH cluster_n_spacing"] = result_manager.config.cluster_n_spacing
        inj_hdu = next((h for h in hdus if h.name == "INJECTION"), None)
        outlier_file_path = result_manager._write_clustered_results(
            freq, taskname, stage_name, hdus[1].data, freq_deriv_order, primary_hdu, inj_hdu=inj_hdu
        )
    return outlier_file_path


def make_outlier_chunked(result_manager, stage, taskname, freq, mean2f_th, n_chunks, chunks_to_run, keep_parts,
                         max_workers, **make_outlier_kwargs):
    """make_outlier over contiguous slices of the seeds, one part file each, merged once all parts exist.
    Finished parts are skipped, so an interrupted collection resumes. chunks_to_run: 1-based slices to collect
    in this call (None: all)."""
    paths = result_manager.paths
    n_jobs = len(mean2f_th)
    bounds = np.linspace(0, n_jobs, n_chunks + 1).astype(int)
    slices = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if a != b]
    print(f"{freq}Hz: {n_jobs} seeds ({n_jobs * stage.n_sky} result files) over {len(slices)} chunks")

    parts = []
    for chunk, (a, b) in enumerate(slices, start=1):
        part = part_file(paths, freq, taskname, stage.name, chunk)
        parts.append(part)
        if part.exists() or (chunks_to_run is not None and chunk not in chunks_to_run):
            continue
        print(f"  chunk {chunk}/{len(slices)}: seeds {a}-{b}")
        written = result_manager.make_outlier(
            taskname, freq, mean2f_th, n_jobs, stage=stage.name, freq_deriv_order=stage.order, n_sky=stage.n_sky,
            cluster=False, max_workers=max_workers, param_indices=np.arange(a, b), **make_outlier_kwargs,
        )
        # every chunk writes under the same taskname: park it before the next chunk overwrites it
        os.replace(written, part)

    missing = [p for p in parts if not p.exists()]
    if missing:
        print(f"{freq}Hz: {len(missing)} of {len(parts)} parts still to do; re-run to finish and merge")
        return None
    path = merge_parts(result_manager, parts, freq, taskname, stage.name, stage.order,
                       make_outlier_kwargs.get("is_injection", False), stage.cluster)
    print(f"{freq}Hz: merged {len(parts)} parts into {path}")
    if not keep_parts:
        for p in parts:
            Path(p).unlink(missing_ok=True)
    return path
