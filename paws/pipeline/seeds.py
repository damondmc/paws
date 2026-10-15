"""Seeds of a follow-up stage: the previous stage's outliers of a band and their Weave result files (seed i runs as
jobs i * n_sky + 1 to (i + 1) * n_sky)."""

from pathlib import Path
from typing import Optional

import numpy as np
from jaxtyping import Shaped

from paws.analysis.statistics import inside_h1_l1_window
from paws.filepaths import PathManager
from paws.fits_io import read_fits_table
from paws.pipeline.noise_weights import log10_noise_ratio
from paws.pipeline.thresholds import stage_thresholds
from paws.settings import Settings, Stage


def seed_outlier_file(
    settings: Settings, paths: PathManager, stage: Stage, freq: int
) -> Path:
    """Previous-stage outlier file holding the seeds of `stage`."""
    prev_stage = settings.prev(stage)
    prev_taskname = prev_stage.taskname(settings.target, freq)
    return paths.outlier_file(
        freq,
        prev_taskname,
        prev_stage.name,
        cluster=not stage.prev_sat,
    )


def read_seed_rows(
    settings: Settings, paths: PathManager, stage: Stage, freq: int
) -> tuple[
    Optional[Shaped[np.ndarray, " n_seed"]], Optional[Shaped[np.ndarray, " n_seed"]]
]:
    """(seed rows, their injection rows or None) of a follow-up stage; (None, None) when the band has no seeds."""
    path = seed_outlier_file(settings, paths, stage, freq)
    if not path.is_file():
        return None, None
    if stage.prev_sat:
        seed_rows = read_fits_table(path, f"{stage.prev}_sat_outlier")
    else:
        seed_rows = read_fits_table(path, 1)
    injection_rows = read_fits_table(path, "injection") if stage.is_injection else None
    if seed_rows is None:
        return None, None
    return seed_rows, injection_rows


def selected_seed_rows(
    settings: Settings, paths: PathManager, stage: Stage, freq: int
) -> Optional[Shaped[np.ndarray, " n_seed"]]:
    """Seed rows of a follow-up stage in one band: the previous stage's outliers

    seeds from the initial search are kept only inside the stage's seed H1/L1 excess-ratio window.
    None when the band has no seeds.
    """
    seed_rows, _ = read_seed_rows(settings, paths, stage, freq)
    if seed_rows is None or len(seed_rows) == 0:
        return None
    if settings.prev(stage).is_search and stage.thresholds is not None:
        thresholds = stage_thresholds(settings, stage)
        if thresholds.seed_h1_l1_window is not None:
            seed_rows = seed_rows[
                inside_h1_l1_window(
                    thresholds.in_band("seed_h1_l1_window", freq),
                    seed_rows["mean2F_H1"],
                    seed_rows["mean2F_L1"],
                    log10_noise_ratio(settings, thresholds.h1_l1_noise_weights, freq),
                )
            ]
    return seed_rows if len(seed_rows) else None


def result_files_per_seed(
    paths: PathManager, settings: Settings, stage: Stage, freq: int, n_seeds: int
) -> list[list[Path]]:
    """Weave result files (n_sky per seed) of each of the band's n_seeds seeds, in seed order."""
    taskname = stage.taskname(settings.target, freq)
    return [
        [
            paths.weave_output_file(freq, taskname, job, stage.name)
            for job in range(seed * stage.n_sky + 1, (seed + 1) * stage.n_sky + 1)
        ]
        for seed in range(n_seeds)
    ]
