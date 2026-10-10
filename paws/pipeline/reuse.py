"""Reuse of earlier Weave runs: a seed equal (freq, f1dot, f2dot) to a seed that an earlier stage (stage.reuse) ran
itself takes that run's result files instead of running again.

Only used to build a stage's seed plan; everything downstream reads the plan and does not know about reuse.
"""

import numpy as np

from paws.pipeline.followup_seeds import (
    SeedPlan,
    default_seed_plan,
    load_seed_plan,
    seed_keys,
    selected_seed_rows,
)


def own_run_seeds(settings, paths, stage, freq):
    """{seed key: first job} of the seeds `stage` ran itself in one band."""
    plan = load_seed_plan(settings, stage, freq)
    if plan is None:
        seed_rows = selected_seed_rows(settings, paths, stage, freq)
        if seed_rows is None:
            return {}
        plan = default_seed_plan(stage, seed_rows)
    own = plan.own_runs(stage.name)
    return {tuple(key): int(first_job) for key, first_job in zip(plan.keys[own], plan.first_job[own])}


def build_seed_plan(settings, paths, stage, freq, seed_rows):
    """Plan of one band: each seed from the first stage in stage.reuse that ran it, else a new run of this stage."""
    keys = seed_keys(seed_rows)
    source_stage = np.full(len(keys), stage.name, dtype=object)
    first_job = np.zeros(len(keys), int)
    reused = np.zeros(len(keys), bool)
    for earlier_name in stage.reuse:
        earlier_runs = own_run_seeds(settings, paths, settings.stage(earlier_name), freq)
        for seed, key in enumerate(map(tuple, keys)):
            if not reused[seed] and key in earlier_runs:
                source_stage[seed], first_job[seed], reused[seed] = earlier_name, earlier_runs[key], True
    new_runs = np.flatnonzero(~reused)
    first_job[new_runs] = np.arange(new_runs.size) * stage.n_sky + 1
    return SeedPlan(keys, source_stage.astype(str), first_job)
