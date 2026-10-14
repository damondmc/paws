import numpy as np
import pytest

from paws.analysis import calibration


def test_injection_excess_ratios_matched_by_injected_freq():
    # 2F - 4 = 10, 20
    prev_rows = np.array([(14.0,), (24.0,)], dtype=[("mean2F", float)])
    # 2F - 4 = 40, 20
    now_rows = np.array([(44.0,), (24.0,)], dtype=[("mean2F", float)])
    ratios = calibration.injection_excess_ratios(
        prev_rows, np.array([50.1, 50.2]), now_rows, np.array([50.2, 50.1])
    )
    assert list(ratios) == [2.0, 2.0]  # 40 / 20 and 20 / 10


def test_excess_ratio_thresholds_per_band():
    band_edges = np.array([20, 100, 200])
    band_of_injection = np.array([50] * 101 + [150] * 3)
    ratios = np.r_[np.linspace(1.0, 2.0, 101), [3.0, 4.0, 5.0]]
    values = calibration.excess_ratio_thresholds(
        band_of_injection, ratios, band_edges, percentile=1.0
    )
    assert values[0] == pytest.approx([1.01, 1.0])  # 1st percentile, lowest
    assert values[1][1] == 3.0


def test_h1_l1_windows_hold_central_range():
    band_edges = np.array([20, 100])
    log_ratio = np.linspace(-1.0, 1.0, 201)
    window = calibration.h1_l1_windows(
        np.full(201, 50), log_ratio, band_edges, percentile=10.0
    )
    assert window[0] == pytest.approx([-0.9, 0.9])


def test_threshold_band_index():
    assert list(
        calibration.threshold_band_index(
            np.array([20, 100, 200]), np.array([20, 99, 100, 199])
        )
    ) == [0, 0, 1, 1]
