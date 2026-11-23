import numpy as np
import pytest

from paws.analysis.sigmoid import SigmoidFitter
from paws.settings import Target

TARGET = Target(name="T", alpha=0.0, delta=0.0, dalpha=0.0, ddelta=0.0)
H95 = 2e-25


def true_efficiency(h0):
    """A sigmoid in h0 with 95 % efficiency at H95."""
    return float(SigmoidFitter.sigmoid(h0 / H95, 15.0, 1.0 - np.log(19.0) / 15.0))


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
