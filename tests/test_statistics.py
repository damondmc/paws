import numpy as np
import pytest
from scipy.stats.distributions import chi2

from paws.analysis import statistics


def test_search_mean2f_threshold():
    # 2F summed over 107 segments is chi2 with 428 dof; threshold = its (1 - 1/n_templates) quantile / 107
    assert statistics.search_mean2f_threshold(10**6, 107) == pytest.approx(
        chi2.ppf(1 - 1e-6, 428) / 107
    )
    # more templates -> higher threshold; more segments -> lower (mean2F averages over segments)
    assert statistics.search_mean2f_threshold(
        10**9, 107
    ) > statistics.search_mean2f_threshold(10**6, 107)
    assert statistics.search_mean2f_threshold(
        10**6, 54
    ) > statistics.search_mean2f_threshold(10**6, 107)


def test_excess_ratio():
    # (44 - 4) / (24 - 4) and (24 - 4) / (14 - 4)
    assert list(
        statistics.excess_ratio(np.array([44.0, 24.0]), np.array([24.0, 14.0]))
    ) == [2.0, 2.0]


def test_log10_h1_l1_excess_ratio():
    # (2F_H1 - 4) / (2F_L1 - 4) = 10 / 1 and 1 / 10
    assert statistics.log10_h1_l1_excess_ratio(
        np.array([14.0, 5.0]), np.array([5.0, 14.0]), 0.0
    ) == pytest.approx([1.0, -1.0])


def test_log10_h1_l1_excess_ratio_noise_normalized():
    # r_HL = 10 and 1/10, times W_L1 / W_H1 = 10 per candidate
    assert statistics.log10_h1_l1_excess_ratio(
        np.array([14.0, 5.0]), np.array([5.0, 14.0]), np.array([1.0, 1.0])
    ) == pytest.approx([2.0, 0.0])


def test_log10_h1_l1_excess_ratio_at_or_below_4_is_plus_minus_5():
    # one detector at or below 4: +5 when H1 has the larger excess, -5 otherwise
    assert list(
        statistics.log10_h1_l1_excess_ratio(
            np.array([10.0, 3.0, 4.0]), np.array([4.0, 10.0, 2.0]), 0.3
        )
    ) == [5.0, -5.0, 5.0]


def test_excess_ratio_to_mean2f_threshold():
    assert list(
        statistics.excess_ratio_to_mean2f_threshold(1.5, np.array([14.0, 4.0]))
    ) == [19.0, 4.0]


def test_inside_h1_l1_window():
    inside = statistics.inside_h1_l1_window(
        np.array([-0.5, 0.5]),
        np.array([14.0, 104.0, 6.0]),
        np.array([14.0, 6.0, 104.0]),
        0.0,
    )
    assert list(inside) == [True, False, False]
    # r_HL = 50 inside once normalized by W_L1 / W_H1 = 1/50
    inside = statistics.inside_h1_l1_window(
        np.array([-0.5, 0.5]), np.array([104.0]), np.array([6.0]), np.log10(1 / 50)
    )
    assert list(inside) == [True]
    # one window per candidate
    inside = statistics.inside_h1_l1_window(
        np.array([[-0.5, 0.5], [0.9, 1.1]]),
        np.array([14.0, 104.0]),
        np.array([14.0, 14.0]),
        0.0,
    )
    assert list(inside) == [True, True]
