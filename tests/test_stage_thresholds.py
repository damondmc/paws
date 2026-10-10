import numpy as np
import yaml

from paws.pipeline.stage_thresholds import StageThresholds, threshold_record


def test_threshold_record(settings, tmp_path):
    stage = settings.stage("followup-1")  # prev is a search stage: also a seed H1/L1 window
    stage_threshold = StageThresholds(
        band_edges=np.array([20, 100, 200]),
        excess_ratio=np.array([1.2345, 1.5]),
        excess_ratio_lowest=np.array([1.1, 1.4]),
        h1_l1_window=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        seed_h1_l1_window=np.array([[-3.0, 3.0], [-4.0, 4.0]]),
    )
    record = threshold_record(stage, stage_threshold)
    assert record["injections"] == ["injections-0", "injections-1"]
    assert record["excess_ratio_percentile"] == 1.0 and record["h1_l1_percentile"] == 1.0
    assert record["bands"][1] == {
        "f_min": 100, "f_max": 200, "excess_ratio_threshold": 1.5, "lowest_excess_ratio": 1.4,
        "h1_l1_window": [-2.0, 2.0], "seed_h1_l1_window": [-4.0, 4.0],
    }
    # plain Python types: the YAML dump reads back unchanged
    assert yaml.safe_load(yaml.safe_dump(record)) == record


def test_threshold_record_without_h1_l1_window(settings):
    stage = settings.stage("followup-2")
    stage_threshold = StageThresholds(np.array([20, 400]), np.array([1.2]), np.array([1.1]), None, None)
    band = threshold_record(stage, stage_threshold)["bands"][0]
    assert "h1_l1_window" not in band and "seed_h1_l1_window" not in band
