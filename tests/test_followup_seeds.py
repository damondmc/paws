import numpy as np
import pytest

from paws.pipeline.followup_seeds import (
    CARRIED_COLUMNS,
    SeedPlan,
    carried_history,
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
    rows = np.array(
        [(50.0, -1e-10, 2e-19, 7.0)],
        dtype=[("freq", ">f8"), ("f1dot", ">f8"), ("f2dot", ">f8"), ("mean2F", ">f8")],
    )
    assert seed_keys(rows).tolist() == [[50.0, -1e-10, 2e-19]]


def test_save_load_and_merge(settings):
    stage = settings.stage("followup-2")
    assert load_seed_plan(settings, stage, 50) is None
    save_seed_plans(settings, stage, {50: plan(["followup-2", "followup-2"], [1, 4])})
    # adds a band, keeps band 50
    save_seed_plans(settings, stage, {60: plan(["followup-2"], [1])})
    loaded = load_seed_plan(settings, stage, 50)
    assert list(loaded.source_stage) == ["followup-2", "followup-2"] and list(
        loaded.first_job
    ) == [1, 4]
    assert load_seed_plan(settings, stage, 60) is not None
    # band not in the file: default plan
    assert load_seed_plan(settings, stage, 70) is None


def test_default_plan_numbers_jobs_by_n_sky(settings):
    stage = settings.stage("followup-2")  # 3 sky points
    rows = np.zeros(3, dtype=[("freq", float), ("f1dot", float), ("f2dot", float)])
    default = default_seed_plan(stage, rows)
    assert list(default.first_job) == [1, 4, 7] and set(default.source_stage) == {
        "followup-2"
    }


def test_result_files_per_seed(settings, paths):
    stage = settings.stage("followup-1")  # 1 sky point, reuses followup-old-1
    seed_plan = plan(["followup-old-1", "followup-1"], [5, 1])
    files = result_files_per_seed(paths, settings, stage, 50, seed_plan)
    assert [path.name for path in files[0]] == [
        "TestTarget_followup-old-1_TCoh10_O2_50Hz.fts.5"
    ]
    assert files[1][0].parent.parent.parent.parent.parent.name == "followup-1"
    assert [path.name for path in files[1]] == [
        "TestTarget_followup-1_TCoh10_O2_50Hz.fts.1"
    ]


def test_own_runs():
    assert list(plan(["a", "now", "now"], [1, 1, 4]).own_runs("now")) == [1, 2]


def history_rows(names_and_values):
    return np.array(
        [tuple(v for _, v in names_and_values)],
        dtype=[
            (n, ">f8") if np.ndim(v) == 0 else (n, ">f8", np.shape(v))
            for n, v in names_and_values
        ],
    )


def test_chain_stages(settings):
    assert [s.name for s in settings.chain_stages(settings.stage("followup-1"))] == [
        "search-0"
    ]
    assert [s.name for s in settings.chain_stages(settings.stage("followup-2"))] == [
        "search-0",
        "followup-1",
    ]


def test_carried_history(settings):
    # seeds of followup-1 are search-0 rows (5 days): their own values become the _TCoh5 columns
    search_rows = history_rows(
        [
            ("mean2F", 7.0),
            ("mean2F_H1", 6.0),
            ("mean2F_L1", 5.5),
            ("coh2F_H1", [1.0, 2.0]),
            ("coh2F_L1", [3.0, 4.0]),
        ]
    )
    stage1 = settings.stage("followup-1")
    history = carried_history(settings.chain_stages(stage1), search_rows)
    assert sorted(history) == sorted(f"{c}_TCoh5" for c in CARRIED_COLUMNS)
    assert history["coh2F_L1_TCoh5"].tolist() == [[3.0, 4.0]]
    # seeds of followup-2 are followup-1 outliers (10 days) carrying the 5-day columns
    followup1_rows = history_rows(
        [
            ("mean2F", 9.0),
            ("mean2F_H1", 8.0),
            ("mean2F_L1", 7.0),
            ("coh2F_H1", [5.0]),
            ("coh2F_L1", [6.0]),
        ]
        + [(name, history[name][0]) for name in history]
    )
    stage2 = settings.stage("followup-2")
    history2 = carried_history(settings.chain_stages(stage2), followup1_rows)
    assert history2["mean2F_TCoh5"].tolist() == [7.0] and history2[
        "mean2F_TCoh10"
    ].tolist() == [9.0]
    assert len(history2) == 2 * len(CARRIED_COLUMNS)
    # outliers written before the history was carried
    with pytest.raises(ValueError, match="no mean2F_TCoh5"):
        carried_history(settings.chain_stages(stage2), search_rows)
