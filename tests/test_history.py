import numpy as np
import pytest

from paws.pipeline.history import HISTORY_COLUMNS, chain_history


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


def test_chain_history(settings):
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
    history = chain_history(settings.chain_stages(stage1), search_rows)
    assert sorted(history) == sorted(f"{c}_TCoh5" for c in HISTORY_COLUMNS)
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
    history2 = chain_history(settings.chain_stages(stage2), followup1_rows)
    assert history2["mean2F_TCoh5"].tolist() == [7.0] and history2[
        "mean2F_TCoh10"
    ].tolist() == [9.0]
    assert len(history2) == 2 * len(HISTORY_COLUMNS)
    # outliers written before the history columns existed
    with pytest.raises(ValueError, match="no mean2F_TCoh5"):
        chain_history(settings.chain_stages(stage2), search_rows)
