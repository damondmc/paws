import numpy as np
import pytest

from paws.analysis.sigmoid import (
    BRACKET_FACTORS,
    DRAW_WINDOW,
    SigmoidFitter,
    find_h95,
)
from paws.settings import Target

TARGET = Target(name="T", alpha=0.0, delta=0.0, dalpha=0.0, ddelta=0.0)
H95 = 2e-25


def true_efficiency(h0):
    """A sigmoid in h0 with 95 % efficiency at H95."""
    return float(SigmoidFitter.sigmoid(h0 / H95, 15.0, 1.0 - np.log(19.0) / 15.0))


@pytest.fixture(autouse=True)
def seeded_strains():
    """find_h95 draws the strains with np.random, seeded as in the upper-limit job."""
    np.random.seed(0)


def drawn_detection(rng, efficiency=None):
    """detected_at for find_h95: each injection detected with the efficiency at its strain."""
    efficiency = efficiency or true_efficiency
    return lambda h0: rng.random(len(h0)) < np.array([efficiency(h) for h in h0])


def test_find_h95_converges_on_the_sigmoid():
    tried = []

    def efficiency_at(h0):
        tried.append(h0)
        return true_efficiency(h0)

    rng = np.random.default_rng(0)
    limit = find_h95(
        efficiency_at, drawn_detection(rng), 1.5e-25, SigmoidFitter(TARGET, n_inj=200)
    )
    assert limit.h95 == pytest.approx(H95, rel=0.06)
    assert abs(limit.dh95 / limit.h95) <= 0.05
    assert list(limit.h0) == tried == [1.5e-25 * f for f in BRACKET_FACTORS]
    assert list(limit.efficiency) == [true_efficiency(h0) for h0 in tried]
    assert len(limit.drawn_h0) == len(limit.drawn_detected) == 2000


def test_find_h95_draws_log_uniform_strains_around_h95():
    """The drawn strains of the accepted round span DRAW_WINDOW around the h95 they were drawn at."""
    rng = np.random.default_rng(1)
    limit = find_h95(
        true_efficiency, drawn_detection(rng), H95, SigmoidFitter(TARGET, n_inj=200)
    )
    log_h0 = np.log(limit.drawn_h0)
    width = np.log(DRAW_WINDOW[1] / DRAW_WINDOW[0])
    assert log_h0.max() - log_h0.min() == pytest.approx(width, rel=0.01)
    assert np.std(log_h0) == pytest.approx(width / np.sqrt(12), rel=0.05)


def test_fit_counts_of_single_injections_matches_the_fit_of_sets():
    """Single injections at a few strains give the same fit as their counts per strain."""
    h0 = H95 * np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    n_detected = np.array([140, 170, 186, 194, 198])
    sets = SigmoidFitter(TARGET, n_inj=200)
    assert sets.fit(h0, n_detected / 200)
    single = SigmoidFitter(TARGET, n_inj=200)
    detected = np.concatenate([np.arange(200) < k for k in n_detected])
    assert single.fit_counts(np.repeat(h0, 200), detected.astype(int), np.ones(1000))
    assert single.get_h_percentile(0.95) == pytest.approx(sets.get_h_percentile(0.95))


def test_find_h95_ignores_the_curve_far_from_h95():
    """An efficiency that follows the sigmoid only above 80 %: the starting sets give a wrong first h95, the
    drawn injections around it give H95."""

    def distorted(h0):
        p = true_efficiency(h0)
        return p if p >= 0.8 else 0.5 * p

    rng = np.random.default_rng(2)
    limit = find_h95(
        distorted,
        drawn_detection(rng, distorted),
        1.5e-25,
        SigmoidFitter(TARGET, n_inj=200),
    )
    assert limit.h95 == pytest.approx(H95, rel=0.06)


@pytest.mark.parametrize("estimate", [0.6 * H95, 1.8 * H95])
def test_find_h95_recentres_from_a_poor_first_estimate(estimate):
    rng = np.random.default_rng(3)
    limit = find_h95(
        true_efficiency,
        drawn_detection(rng),
        estimate,
        SigmoidFitter(TARGET, n_inj=200),
    )
    assert limit.h95 == pytest.approx(H95, rel=0.06)


def test_find_h95_with_drawn_injections_is_unbiased_with_matching_errors():
    """Over repeated runs, h95 scatters around H95 by about dh95."""
    rng = np.random.default_rng(4)
    pulls = []
    for _ in range(100):
        limit = find_h95(
            lambda h0: rng.binomial(200, true_efficiency(h0)) / 200,
            drawn_detection(rng),
            1.5e-25,
            SigmoidFitter(TARGET, n_inj=200),
        )
        pulls.append((limit.h95 - H95) / limit.dh95)
    assert abs(np.mean(pulls)) < 0.25
    assert 0.75 < np.std(pulls) < 1.3


def test_find_h95_stops_after_max_rounds():
    """A zero tolerance is never reached: one draw per round, then an error."""
    n_draws = []
    rng = np.random.default_rng(5)
    detect = drawn_detection(rng)

    def detected_at(h0):
        n_draws.append(len(h0))
        return detect(h0)

    with pytest.raises(RuntimeError, match="did not converge"):
        find_h95(
            true_efficiency,
            detected_at,
            1.5e-25,
            SigmoidFitter(TARGET, n_inj=200),
            n_drawn=500,
            tolerance=0.0,
            max_rounds=2,
        )
    assert n_draws == [500, 500]


def test_find_h95_rejects_an_unconstrained_fit():
    """A flat efficiency gives no h95."""
    with pytest.raises(RuntimeError, match="constrain"):
        find_h95(
            lambda h0: 0.5,
            lambda h0: np.zeros(len(h0), bool),
            1e-25,
            SigmoidFitter(TARGET, n_inj=200),
        )


def test_fit_of_binomial_counts_is_unbiased_with_matching_errors():
    """Over repeated draws of 200 injections per set, h95 scatters around H95 by about dh95."""
    rng = np.random.default_rng(0)
    h0 = H95 * np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.5])
    p_true = np.array([true_efficiency(h) for h in h0])
    pulls = []
    for _ in range(300):
        fitter = SigmoidFitter(TARGET, n_inj=200)
        assert fitter.fit(h0, rng.binomial(200, p_true) / 200)
        h95, dh95 = fitter.get_h_percentile(0.95)
        pulls.append((h95 - H95) / dh95)
    assert abs(np.mean(pulls)) < 0.2
    assert 0.8 < np.std(pulls) < 1.2


def test_fit_accepts_sets_at_zero_and_full_efficiency():
    fitter = SigmoidFitter(TARGET, n_inj=200)
    assert fitter.fit(
        H95 * np.array([0.3, 0.8, 1.0, 2.0]), np.array([0.0, 0.6, 0.94, 1.0])
    )
    assert np.all(np.isfinite(fitter.pcov))
