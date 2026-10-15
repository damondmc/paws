"""Test spin-down ranges: the power law with an age of 300 years up to 199 Hz, then 0.5 years more per Hz."""

import numpy as np

from paws.definitions import SECONDS_PER_YEAR
from paws.params.models import PowerLawModel, SpindownModel


def spindown_model(freq: float) -> SpindownModel:
    age = np.interp(freq, [199, 399], [300, 400])
    return PowerLawModel(nc_min=2.0, nc_max=7.0, tau=age * SECONDS_PER_YEAR)
