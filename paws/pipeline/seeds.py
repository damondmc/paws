"""Seeds of a follow-up stage: the previous stage's outliers in one band (with their injections for an injection
follow-up); seeds from a search stage are kept only inside the stage's seed H1/L1 excess-ratio window, when it has one.
"""

from paws.analysis.statistics import inside_h1_l1_window
from paws.fits_io import read_fits_table
from paws.pipeline.noise_weights import log10_noise_ratio
from paws.pipeline.stage_thresholds import stage_thresholds


def seed_outlier_file(settings, paths, stage, freq):
    """Previous-stage outlier file holding the seeds of `stage`."""
    prev_stage = settings.prev(stage)
    prev_taskname = prev_stage.taskname(settings.target, freq)
    return paths.outlier_file(
        freq,
        prev_taskname,
        prev_stage.name,
        cluster=not stage.prev_sat,
        location="existing",
    )


def read_seed_rows(settings, paths, stage, freq):
    """(seed rows, injection rows or None) of a follow-up stage; (None, None) when the band has no seeds."""
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
    if stage.n_inj_max is not None:
        seed_rows = seed_rows[: stage.n_inj_max]
        injection_rows = (
            None if injection_rows is None else injection_rows[: stage.n_inj_max]
        )
    return seed_rows, injection_rows


def selected_seed_rows(settings, paths, stage, freq):
    """Seed rows of a follow-up stage in one band: the previous stage's outliers; seeds from a search stage are
    kept only inside the stage's seed H1/L1 excess-ratio window (when it has one). None when the band has no seeds."""
    seed_rows, _ = read_seed_rows(settings, paths, stage, freq)
    if seed_rows is None or len(seed_rows) == 0:
        return None
    if (
        settings.prev(stage).kind == "search"
        and stage.thresholds is not None
        and stage.thresholds.h1_l1_percentile is not None
    ):
        window = stage_thresholds(settings, stage).at("seed_h1_l1_window", freq)
        seed_rows = seed_rows[
            inside_h1_l1_window(
                window,
                seed_rows["mean2F_H1"],
                seed_rows["mean2F_L1"],
                log10_noise_ratio(settings, stage, freq),
            )
        ]
    return seed_rows if len(seed_rows) else None
