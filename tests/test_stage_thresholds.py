import numpy as np
import yaml

from paws.pipeline.noise_weights import _write_rows, log10_noise_ratio
from paws.pipeline.stage_thresholds import (
    StageThresholds,
    read_threshold_record,
    threshold_record,
    threshold_record_path,
)


def test_threshold_record(settings, tmp_path):
    # prev is a search stage: also a seed H1/L1 window
    stage = settings.stage("followup-1")
    stage_threshold = StageThresholds(
        band_edges=np.array([20, 100, 200]),
        excess_ratio=np.array([1.2345, 1.5]),
        excess_ratio_lowest=np.array([1.1, 1.4]),
        h1_l1_window=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        seed_h1_l1_window=np.array([[-3.0, 3.0], [-4.0, 4.0]]),
        growth=np.array([9.5, 11.0]),
    )
    record = threshold_record(stage, stage_threshold)
    assert record["injections"] == ["injections-0", "injections-1"]
    assert (
        record["excess_ratio_percentile"] == 1.0 and record["h1_l1_percentile"] == 1.0
    )
    assert record["h1_l1_noise_weights"] is None
    assert record["bands"][1] == {
        "f_min": 100,
        "f_max": 200,
        "excess_ratio_threshold": 1.5,
        "lowest_excess_ratio": 1.4,
        "h1_l1_window": [-2.0, 2.0],
        "seed_h1_l1_window": [-4.0, 4.0],
        "growth_threshold": 11.0,
    }
    # plain Python types: the YAML dump reads back unchanged
    assert yaml.safe_load(yaml.safe_dump(record)) == record


def test_threshold_record_without_h1_l1_window(settings):
    stage = settings.stage("followup-2")
    stage_threshold = StageThresholds(
        np.array([20, 400]), np.array([1.2]), np.array([1.1]), None, None, None
    )
    band = threshold_record(stage, stage_threshold)["bands"][0]
    assert "h1_l1_window" not in band and "seed_h1_l1_window" not in band
    assert "growth_threshold" not in band


def test_read_threshold_record_round_trip(settings):
    stage = settings.stage("followup-1")
    stage_threshold = StageThresholds(
        band_edges=np.array([20, 100, 200]),
        excess_ratio=np.array([1.2345, 1.5]),
        excess_ratio_lowest=np.array([1.1, 1.4]),
        h1_l1_window=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        seed_h1_l1_window=None,
        growth=np.array([9.5, 11.0]),
    )
    path = threshold_record_path(settings, stage.name)
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(threshold_record(stage, stage_threshold)))
    read_back = read_threshold_record(settings, stage.name)
    np.testing.assert_array_equal(read_back.band_edges, [20, 100, 200])
    np.testing.assert_allclose(read_back.at("excess_ratio", [25, 150]), [1.2345, 1.5])
    np.testing.assert_allclose(read_back.at("h1_l1_window", 150), [-2.0, 2.0])
    assert read_back.seed_h1_l1_window is None
    np.testing.assert_allclose(read_back.at("growth", [50, 150]), [9.5, 11.0])


def test_log10_noise_ratio(settings):
    # no noise-weights file: the window is on r_HL itself
    stage = settings.stage("followup-1")
    assert list(log10_noise_ratio(settings, stage, [50, 51])) == [0.0, 0.0]
    # with one: log10(W_L1 / W_H1) of each candidate's 1 Hz band
    _write_rows(
        settings.config_dir / "noise.csv", [[50, 1.0, 10.0, 5, 5], [51, 4.0, 1.0, 5, 5]]
    )
    normalized = stage.model_copy(
        update={
            "thresholds": stage.thresholds.model_copy(
                update={"h1_l1_noise_weights": "noise.csv"}
            )
        }
    )
    np.testing.assert_allclose(
        log10_noise_ratio(settings, normalized, [51, 50, 50]),
        [np.log10(0.25), 1.0, 1.0],
    )
