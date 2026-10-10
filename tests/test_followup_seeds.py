import numpy as np

from paws.pipeline.followup_seeds import (
    SeedPlan,
    default_seed_plan,
    load_seed_plan,
    result_files_per_seed,
    save_seed_plans,
    seed_keys,
)


def plan(sources, first_jobs):
    keys = np.arange(3 * len(sources), dtype=float).reshape(-1, 3)
    return SeedPlan(keys, np.array(sources), np.array(first_jobs))


def test_seed_keys():
    rows = np.array([(50.0, -1e-10, 2e-19, 7.0)], dtype=[("freq", ">f8"), ("f1dot", ">f8"), ("f2dot", ">f8"),
                                                       ("mean2F", ">f8")])
    assert seed_keys(rows).tolist() == [[50.0, -1e-10, 2e-19]]


def test_save_load_and_merge(settings):
    stage = settings.stage("followup-2")
    assert load_seed_plan(settings, stage, 50) is None
    save_seed_plans(settings, stage, {50: plan(["followup-2", "followup-2"], [1, 4])})
    save_seed_plans(settings, stage, {60: plan(["followup-2"], [1])})  # adds a band, keeps band 50
    loaded = load_seed_plan(settings, stage, 50)
    assert list(loaded.source_stage) == ["followup-2", "followup-2"] and list(loaded.first_job) == [1, 4]
    assert load_seed_plan(settings, stage, 60) is not None
    assert load_seed_plan(settings, stage, 70) is None  # band not in the file: default plan


def test_default_plan_numbers_jobs_by_n_sky(settings):
    stage = settings.stage("followup-2")  # 3 sky points
    rows = np.zeros(3, dtype=[("freq", float), ("f1dot", float), ("f2dot", float)])
    default = default_seed_plan(stage, rows)
    assert list(default.first_job) == [1, 4, 7] and set(default.source_stage) == {"followup-2"}


def test_result_files_per_seed(settings, paths):
    stage = settings.stage("followup-1")  # 1 sky point, reuses followup-old-1
    seed_plan = plan(["followup-old-1", "followup-1"], [5, 1])
    files = result_files_per_seed(paths, settings, stage, 50, seed_plan)
    assert [path.name for path in files[0]] == ["TestTarget_followup-old-1_TCoh10_O2_50Hz.fts.5"]
    assert files[1][0].parent.parent.parent.parent.parent.name == "followup-1"
    assert [path.name for path in files[1]] == ["TestTarget_followup-1_TCoh10_O2_50Hz.fts.1"]


def test_own_runs():
    assert list(plan(["a", "now", "now"], [1, 1, 4]).own_runs("now")) == [1, 2]
