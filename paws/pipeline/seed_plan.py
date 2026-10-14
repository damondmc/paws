"""Seed plan of a follow-up stage: per band, each seed's key and the Weave run holding its results.

Seed i of a plan has its n_sky result files in stage source_stage[i], jobs first_job[i] .. first_job[i] + n_sky - 1.
Without reuse every seed is the stage's own run: seed i -> jobs i * n_sky + 1 ... A stage's plans are saved in
results/<stage>/seed_plan.npz; a band missing there has the default plan.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from jaxtyping import Float, Int, Shaped

from paws.filepaths import PathManager
from paws.settings import Settings, Stage

SEED_KEY_COLUMNS = ("freq", "f1dot", "f2dot")


@dataclass(frozen=True)
class SeedPlan:
    keys: Float[np.ndarray, "n_seed 3"]  # freq, f1dot, f2dot of each seed
    # stage name whose Weave run holds the seed's results
    source_stage: Shaped[np.ndarray, " n_seed"]
    # first Weave job (result file index) of the seed in that run
    first_job: Int[np.ndarray, " n_seed"]

    def own_runs(self, stage_name: str) -> Int[np.ndarray, " n_own"]:
        """Indices of the seeds that run in stage_name itself."""
        return np.flatnonzero(self.source_stage == stage_name)


def seed_keys(rows: Shaped[np.ndarray, " n_seed"]) -> Float[np.ndarray, "n_seed 3"]:
    return np.array(
        [[float(row[column]) for column in SEED_KEY_COLUMNS] for row in rows],
        dtype=float,
    ).reshape(-1, 3)


def default_seed_plan(
    stage: Stage, seed_rows: Shaped[np.ndarray, " n_seed"]
) -> SeedPlan:
    n_seeds = len(seed_rows)
    return SeedPlan(
        seed_keys(seed_rows),
        np.full(n_seeds, stage.name),
        np.arange(n_seeds) * stage.n_sky + 1,
    )


def seed_plan_path(settings: Settings, stage: Stage) -> Path:
    return settings.environment.home_dir / "results" / stage.name / "seed_plan.npz"


def load_seed_plan(settings: Settings, stage: Stage, freq: int) -> Optional[SeedPlan]:
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


def save_seed_plans(
    settings: Settings, stage: Stage, plan_by_band: dict[int, SeedPlan]
) -> None:
    """Adds or replaces the plans of these bands in the stage's seed_plan.npz."""
    path = seed_plan_path(settings, stage)
    arrays = dict(np.load(path)) if path.exists() else {}
    for freq, plan in plan_by_band.items():
        arrays[f"{freq}_key"] = plan.keys
        arrays[f"{freq}_source_stage"] = plan.source_stage.astype(str)
        arrays[f"{freq}_first_job"] = plan.first_job
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def result_files_per_seed(
    paths: PathManager, settings: Settings, stage: Stage, freq: int, plan: SeedPlan
) -> list[list[Path]]:
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
