"""Noise-weights file of the noise-normalized H1/L1 excess-ratio window (paws.analysis.noise_weights).

`paws noise-weights <file>` writes <config-dir>/<file> (columns band, W_H1, W_L1, n_H1, n_L1); a calibration uses it
through `h1_l1_noise_weights: <file>` (followup.yaml), recorded in the threshold file it writes.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache

import numpy as np
from tqdm import tqdm

from paws.analysis.noise_weights import band_noise_weight
from paws.filepaths import PathManager

DETECTORS = ("H1", "L1")
COLUMNS = ("band", "W_H1", "W_L1", "n_H1", "n_L1")


def write_noise_weights(settings, file_name, bands, n_workers):
    """Noise weights of the given 1 Hz bands into <config-dir>/<file_name>; bands already in the file are kept, and
    the file is rewritten every 20 bands."""
    path = settings.config_dir / file_name
    paths = PathManager(settings)
    rows = {int(row[0]): list(row) for row in _read_rows(path)} if path.exists() else {}
    todo = [band for band in bands if band not in rows]
    with ProcessPoolExecutor(n_workers) as pool:
        futures = [pool.submit(_band_row, paths, band) for band in todo]
        for k, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), unit="band")
        ):
            row = future.result()
            rows[row[0]] = row
            if (k + 1) % 20 == 0 or k + 1 == len(futures):
                _write_rows(path, [rows[band] for band in sorted(rows)])
    print("wrote", path)


def _band_row(paths, band):
    (w_h1, n_h1), (w_l1, n_l1) = (
        band_noise_weight(paths.sft_file_path(band, detector), band)
        for detector in DETECTORS
    )
    return [band, w_h1, w_l1, n_h1, n_l1]


def log10_noise_ratio(settings, file_name, freq):
    """log10(W_L1 / W_H1) of the 1 Hz band(s) freq from the noise-weights file `file_name` of the config directory;
    0 when file_name is None (window on r_HL)."""
    if file_name is None:
        return np.zeros(np.shape(freq))
    ratios = _log10_noise_ratios(settings.config_dir / file_name)
    return np.vectorize(lambda band: ratios[int(band)], otypes=[float])(freq)


@cache
def _log10_noise_ratios(path):
    rows = _read_rows(path)
    return dict(zip(rows[:, 0].astype(int), np.log10(rows[:, 2] / rows[:, 1])))


def _read_rows(path):
    return np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))


def _write_rows(path, rows):
    tmp = f"{path}.tmp"
    np.savetxt(
        tmp,
        np.array(rows, float),
        delimiter=",",
        header=",".join(COLUMNS),
        comments="",
        fmt=["%d", "%.10e", "%.10e", "%d", "%d"],
    )
    os.replace(tmp, path)
