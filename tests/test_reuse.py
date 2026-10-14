import numpy as np

from paws.pipeline import reuse
from paws.pipeline.seed_plan import default_seed_plan


def rows(keys):
    table = np.zeros(
        len(keys),
        dtype=[("freq", float), ("f1dot", float), ("f2dot", float), ("mean2F", float)],
    )
    for i, key in enumerate(keys):
        table[i]["freq"], table[i]["f1dot"], table[i]["f2dot"] = key
    return table


def test_default_plan_runs_every_seed_here(settings):
    stage = settings.stage("followup-2")  # 3 sky points
    plan = default_seed_plan(stage, rows([(1, 2, 3), (4, 5, 6)]))
    assert list(plan.source_stage) == ["followup-2", "followup-2"]
    assert list(plan.first_job) == [1, 4]


def test_reuse_takes_first_matching_earlier_run(settings, paths, monkeypatch):
    # followup-2 (3 sky points) taking the runs of followup-old-1 before those of followup-1
    stage = settings.stage("followup-2").model_copy(
        update={"reuse": ["followup-old-1", "followup-1"]}
    )
    runs = {
        "followup-old-1": {(1.0, 2.0, 3.0): 115},
        "followup-1": {(1.0, 2.0, 3.0): 1, (4.0, 5.0, 6.0): 58},
    }
    monkeypatch.setattr(
        reuse,
        "own_run_seeds",
        lambda settings, paths, earlier, freq: runs[earlier.name],
    )

    seed_rows = rows([(1, 2, 3), (7, 8, 9), (4, 5, 6), (0, 0, 0)])
    plan = reuse.build_seed_plan(settings, paths, stage, 100, seed_rows)

    assert list(plan.source_stage) == [
        "followup-old-1",
        "followup-2",
        "followup-1",
        "followup-2",
    ]
    # new runs are numbered in seed order, 3 jobs each: 1, 4
    assert list(plan.first_job) == [115, 1, 58, 4]
    assert list(plan.own_runs("followup-2")) == [1, 3]
