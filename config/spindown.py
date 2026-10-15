"""Example spin-down ranges: the power law with braking index 2 to 7 and the target's age of 600 years."""

from paws.params.models import SECONDS_PER_YEAR, PowerLawModel, SpindownModel


def spindown_model(freq: float) -> SpindownModel:
    return PowerLawModel(nc_min=2.0, nc_max=7.0, tau=600 * SECONDS_PER_YEAR)


# Alternative: an age that rises with frequency, to cap the computing cost of the high bands.
# The age is 300 years up to 199 Hz and then 0.5 years more per Hz.
#
# def spindown_model(freq: float) -> SpindownModel:
#     age = 300 + 0.5 * max(freq - 199, 0)  # years
#     return PowerLawModel(nc_min=2.0, nc_max=7.0, tau=age * SECONDS_PER_YEAR)
