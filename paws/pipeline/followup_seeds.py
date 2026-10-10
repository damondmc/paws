"""Seeds of a follow-up stage: reading them from the previous stage, and the seed plan.

Seeds: the previous stage's outliers in one band (with their injections for an injection follow-up); seeds from a
search stage are kept only inside the stage's seed H1/L1 excess-ratio window, when it has one.

Seed plan: per band, each seed's key and the Weave run holding its results. Seed i of a plan has its n_sky result
files in stage source_stage[i], jobs first_job[i] .. first_job[i] + n_sky - 1. Without reuse every seed is the
stage's own run: seed i -> jobs i * n_sky + 1 ... A stage's plans are saved in results/<stage>/seed_plan.npz; a band
missing there has the default plan.
"""

from dataclasses import dataclass

import numpy as np

from paws.analysis.thresholds import inside_h1_l1_window
from paws.fits_io import read_fits_table
from paws.pipeline.stage_thresholds import stage_thresholds

# ---------------------------- reading seeds ----------------------------


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


# ----------------------------- seed plans ------------------------------

SEED_KEY_COLUMNS = ("freq", "f1dot", "f2dot")


@dataclass(frozen=True)
class SeedPlan:
    keys: np.ndarray  # (n_seeds, 3): freq, f1dot, f2dot of each seed
    # (n_seeds,) stage name whose Weave run holds the seed's results
    source_stage: np.ndarray
    # (n_seeds,) first Weave job (result file index) of the seed in that run
    first_job: np.ndarray

    def own_runs(self, stage_name):
        """Indices of the seeds that run in stage_name itself."""
        return np.flatnonzero(self.source_stage == stage_name)


def seed_keys(rows):
    return np.array(
        [[float(row[column]) for column in SEED_KEY_COLUMNS] for row in rows],
        dtype=float,
    ).reshape(-1, 3)


def default_seed_plan(stage, seed_rows):
    n_seeds = len(seed_rows)
    return SeedPlan(
        seed_keys(seed_rows),
        np.full(n_seeds, stage.name),
        np.arange(n_seeds) * stage.n_sky + 1,
    )


def selected_seed_rows(settings, paths, stage, freq):
    """Seed rows of a follow-up stage in one band: the previous stage's outliers; seeds from a search stage are
    kept only inside the stage's seed H1/L1 excess-ratio window (when it has one). None when the band has no seeds."""
    seed_rows, _ = read_seed_rows(settings, paths, stage, freq)
    if seed_rows is None or len(seed_rows) == 0:
        return None
    if stage.thresholds is not None and stage.thresholds.h1_l1_percentile is not None:
        thresholds = stage_thresholds(settings, stage)
        if thresholds.seed_h1_l1_window is not None:
            window = thresholds.seed_h1_l1_window_at(freq)
            seed_rows = seed_rows[
                inside_h1_l1_window(
                    window, seed_rows["mean2F_H1"], seed_rows["mean2F_L1"]
                )
            ]
    return seed_rows if len(seed_rows) else None


def seed_plan_path(settings, stage):
    return settings.config.home_dir / "results" / stage.name / "seed_plan.npz"


def load_seed_plan(settings, stage, freq):
    """Saved plan of one band, or None when none was saved."""
    path = seed_plan_path(settings, stage)
    if not path.exists():
        return None
    with np.load(path) as saved:
        if f"{freq}_key" not in saved.files:
            return None
        return SeedPlan(
            saved[f"{freq}_key"],
            saved[f"{freq}_source_stage"],
            saved[f"{freq}_first_job"],
        )


def save_seed_plans(settings, stage, plan_by_band):
    """Adds or replaces the plans of these bands in the stage's seed_plan.npz."""
    path = seed_plan_path(settings, stage)
    arrays = dict(np.load(path)) if path.exists() else {}
    for freq, plan in plan_by_band.items():
        arrays[f"{freq}_key"] = plan.keys
        arrays[f"{freq}_source_stage"] = plan.source_stage.astype(str)
        arrays[f"{freq}_first_job"] = plan.first_job
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def result_files_per_seed(paths, settings, stage, freq, plan):
    """Weave result files (n_sky per seed) of each seed of a plan."""
    files_per_seed = []
    for source_name, first_job in zip(plan.source_stage, plan.first_job):
        source = settings.stage(str(source_name))
        taskname = source.taskname(settings.target, freq)
        files_per_seed.append(
            [
                paths.weave_output_file(freq, taskname, job, source.name)
                for job in range(int(first_job), int(first_job) + stage.n_sky)
            ]
        )
    return files_per_seed
