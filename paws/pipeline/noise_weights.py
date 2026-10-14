"""Noise-weights file of the noise-normalized H1/L1 excess-ratio window (paws.analysis.noise_weights).

`paws noise-weights <file>` writes <config-dir>/<file> (columns band, W_H1, W_L1, n_H1, n_L1); a calibration uses it
through `h1_l1_noise_weights: <file>` (followup.yaml), recorded in the threshold file it writes.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
from jaxtyping import Float, Int
from tqdm import tqdm

from paws.analysis.noise_weights import band_noise_weight
from paws.filepaths import PathManager
from paws.settings import Settings

DETECTORS = ("H1", "L1")
COLUMNS = ("band", "W_H1", "W_L1", "n_H1", "n_L1")


def write_noise_weights(
    settings: Settings, file_name: str, bands: list[int], n_workers: int
) -> None:
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


def _band_row(paths: PathManager, band: int) -> list[Union[int, float]]:
    """[band, W_H1, W_L1, n_H1, n_L1] of one 1 Hz band."""
    (w_h1, n_h1), (w_l1, n_l1) = (
        band_noise_weight(paths.sft_file_path(band, detector), band)
        for detector in DETECTORS
    )
    return [band, w_h1, w_l1, n_h1, n_l1]


def log10_noise_ratio(
    settings: Settings,
    file_name: Optional[str],
    freq: Union[int, Int[np.ndarray, " n_candidate"]],
) -> Union[float, Float[np.ndarray, " n_candidate"]]:
    """log10(W_L1 / W_H1) of the 1 Hz band freq (or of each band of an array) from the noise-weights file `file_name`
    of the config directory; 0 when file_name is None (window on r_HL)."""
    if file_name is None:
        return 0.0 if np.ndim(freq) == 0 else np.zeros(np.shape(freq))
    ratios = _log10_noise_ratios(settings.config_dir / file_name)
    if np.ndim(freq) == 0:
        return ratios[int(freq)]
    return np.vectorize(lambda band: ratios[int(band)], otypes=[float])(freq)


@cache
def _log10_noise_ratios(path: Path) -> dict[int, float]:
    """{band: log10(W_L1 / W_H1)} of a noise-weights file."""
    rows = _read_rows(path)
    return {
        int(band): float(ratio)
        for band, ratio in zip(rows[:, 0], np.log10(rows[:, 2] / rows[:, 1]))
    }


def _read_rows(path: Path) -> Float[np.ndarray, "n_band 5"]:
    return np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))


def _write_rows(path: Path, rows: list[list[Union[int, float]]]) -> None:
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
