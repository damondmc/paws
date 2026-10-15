import numpy as np
import pytest

from paws.analysis.sigmoid import BRACKET_FACTORS, SigmoidFitter, find_h95
from paws.settings import Target

TARGET = Target(name="T", alpha=0.0, delta=0.0, dalpha=0.0, ddelta=0.0)
H95 = 2e-25


def true_efficiency(h0):
    """A sigmoid in h0 with 95 % efficiency at H95."""
    return float(SigmoidFitter.sigmoid(h0 / H95, 15.0, 1.0 - np.log(19.0) / 15.0))


def test_find_h95_converges_on_the_sigmoid():
    tried = []

    def efficiency_at(h0):
        tried.append(h0)
        return true_efficiency(h0)

    limit = find_h95(efficiency_at, 1.5e-25, SigmoidFitter(TARGET, n_inj=200))
    assert limit.h95 == pytest.approx(H95, rel=0.02)
    assert abs(limit.dh95 / limit.h95) <= 0.05
    assert list(limit.h0) == tried and len(tried) >= len(BRACKET_FACTORS)
    assert list(limit.h0[: len(BRACKET_FACTORS)]) == [
        1.5e-25 * factor for factor in BRACKET_FACTORS
    ]
    assert list(limit.efficiency) == [true_efficiency(h0) for h0 in tried]


def test_find_h95_stops_after_max_rounds():
    """A zero tolerance is never reached: two refinement strains per round, then an error."""
    tried = []

    def efficiency_at(h0):
        tried.append(h0)
        return true_efficiency(h0)

    with pytest.raises(RuntimeError, match="did not converge"):
        find_h95(
            efficiency_at,
            1.5e-25,
            SigmoidFitter(TARGET, n_inj=200),
            tolerance=0.0,
            max_rounds=2,
        )
    assert len(tried) == len(BRACKET_FACTORS) + 2 * 2


def test_find_h95_rejects_an_unconstrained_fit():
    """A flat efficiency gives no h95."""
    with pytest.raises(RuntimeError, match="constrain"):
        find_h95(lambda h0: 0.5, 1e-25, SigmoidFitter(TARGET, n_inj=200))
