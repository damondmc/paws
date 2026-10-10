import numpy as np
import pytest
from scipy.stats.distributions import chi2

from paws.analysis import thresholds


def test_search_mean2f_threshold():
    # 2F summed over 107 segments is chi2 with 428 dof; threshold = its (1 - 1/n_templates) quantile / 107
    assert thresholds.search_mean2f_threshold(1e6, 107) == pytest.approx(
        chi2.ppf(1 - 1e-6, 428) / 107
    )
    # more templates -> higher threshold; more segments -> lower (mean2F averages over segments)
    assert thresholds.search_mean2f_threshold(
        1e9, 107
    ) > thresholds.search_mean2f_threshold(1e6, 107)
    assert thresholds.search_mean2f_threshold(
        1e6, 54
    ) > thresholds.search_mean2f_threshold(1e6, 107)


def test_log10_h1_l1_excess_ratio():
    # (2F_H1 - 4) / (2F_L1 - 4) = 10 / 1 and 1 / 10
    assert thresholds.log10_h1_l1_excess_ratio(
        [14.0, 5.0], [5.0, 14.0]
    ) == pytest.approx([1.0, -1.0])


def test_log10_h1_l1_excess_ratio_at_or_below_4_is_plus_minus_5():
    # one detector at or below 4: +5 when H1 has the larger excess, -5 otherwise
    assert list(
        thresholds.log10_h1_l1_excess_ratio([10.0, 3.0, 4.0], [4.0, 10.0, 2.0])
    ) == [5.0, -5.0, 5.0]


def test_injection_excess_ratios_matched_by_injected_freq():
    # 2F - 4 = 10, 20
    prev_rows = np.array([(14.0,), (24.0,)], dtype=[("mean2F", float)])
    # 2F - 4 = 40, 20
    now_rows = np.array([(44.0,), (24.0,)], dtype=[("mean2F", float)])
    ratios = thresholds.injection_excess_ratios(
        prev_rows, [50.1, 50.2], now_rows, [50.2, 50.1]
    )
    assert list(ratios) == [2.0, 2.0]  # 40 / 20 and 20 / 10


def test_excess_ratio_thresholds_per_band():
    band_edges = np.array([20, 100, 200])
    band_of_injection = np.array([50] * 101 + [150] * 3)
    ratios = np.r_[np.linspace(1.0, 2.0, 101), [3.0, 4.0, 5.0]]
    values = thresholds.excess_ratio_thresholds(
        band_of_injection, ratios, band_edges, percentile=1.0
    )
    assert values[0] == pytest.approx([1.01, 1.0])  # 1st percentile, lowest
    assert values[1][1] == 3.0


def test_h1_l1_windows_hold_central_range():
    band_edges = np.array([20, 100])
    log_ratio = np.linspace(-1.0, 1.0, 201)
    window = thresholds.h1_l1_windows(
        np.full(201, 50), log_ratio, band_edges, percentile=10.0
    )
    assert window[0] == pytest.approx([-0.9, 0.9])


def test_excess_ratio_to_mean2f_threshold():
    assert list(thresholds.excess_ratio_to_mean2f_threshold(1.5, [14.0, 4.0])) == [
        19.0,
        4.0,
    ]


def test_inside_h1_l1_window():
    inside = thresholds.inside_h1_l1_window(
        (-0.5, 0.5), [14.0, 104.0, 6.0], [14.0, 6.0, 104.0]
    )
    assert list(inside) == [True, False, False]


def test_threshold_band_index():
    assert list(
        thresholds.threshold_band_index(np.array([20, 100, 200]), [20, 99, 100, 199])
    ) == [0, 0, 1, 1]


def test_read_old_threshold_file(tmp_path):
    path = tmp_path / "ratio_v3.txt"
    path.write_text(
        "#f_start\tf_end\t0.1 percentile\tlowest\n"
        "20\t30\t1.2346\t1.1000\n30\t40\t1.2346\t1.1000\n40\t50\t1.5000\t1.4000\n"
    )
    np.testing.assert_allclose(
        thresholds.read_threshold_row(path, [25, 45]), [[1.2346, 1.1], [1.5, 1.4]]
    )
