import numpy as np

from paws.pipeline import reuse
from paws.pipeline.followup_seeds import default_seed_plan


class FakeStage:
    def __init__(self, name, reuse_from, n_sky):
        self.name, self.reuse, self.n_sky = name, reuse_from, n_sky


class FakeSettings:
    def __init__(self, stages):
        self.stages = {stage.name: stage for stage in stages}

    def stage(self, name):
        return self.stages[name]


def rows(keys):
    table = np.zeros(
        len(keys),
        dtype=[("freq", float), ("f1dot", float), ("f2dot", float), ("mean2F", float)],
    )
    for i, key in enumerate(keys):
        table[i]["freq"], table[i]["f1dot"], table[i]["f2dot"] = key
    return table


def test_default_plan_runs_every_seed_here():
    plan = default_seed_plan(
        FakeStage("now", [], n_sky=57), rows([(1, 2, 3), (4, 5, 6)])
    )
    assert list(plan.source_stage) == ["now", "now"]
    assert list(plan.first_job) == [1, 58]


def test_reuse_takes_first_matching_earlier_run(monkeypatch):
    earlier_a, earlier_b = FakeStage("a", [], 57), FakeStage("b", [], 57)
    stage = FakeStage("now", ["a", "b"], 57)
    runs = {"a": {(1.0, 2.0, 3.0): 115}, "b": {(1.0, 2.0, 3.0): 1, (4.0, 5.0, 6.0): 58}}
    monkeypatch.setattr(
        reuse,
        "own_run_seeds",
        lambda settings, paths, earlier, freq: runs[earlier.name],
    )

    seed_rows = rows([(1, 2, 3), (7, 8, 9), (4, 5, 6), (0, 0, 0)])
    plan = reuse.build_seed_plan(
        FakeSettings([earlier_a, earlier_b, stage]), None, stage, 100, seed_rows
    )

    assert list(plan.source_stage) == ["a", "now", "b", "now"]
    # new runs are numbered in seed order: 1, 58
    assert list(plan.first_job) == [115, 1, 58, 58]
    assert list(plan.own_runs("now")) == [1, 3]
