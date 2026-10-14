"""Strain, ellipticity and braking-index relations of a continuous-wave source (scalar or per frequency).

Izz: principal moment of inertia [kg m^2], I0: its reference value, r: distance [kpc], freq: GW frequency [Hz].
"""

from typing import Union

import numpy as np
from jaxtyping import Float


def age_strainLimit(
    tau: Union[float, Float[np.ndarray, " n_value"]], Izz: float, I0: float, r: float
) -> Union[float, Float[np.ndarray, " n_value"]]:
    """Age-based strain upper limit h0 of a source of characteristic age tau [years]."""
    return 2.3e-24 * (1.0 / r) * np.sqrt((1000 / tau) * (Izz / I0))


def h0_from_ellipticity(
    freq: Union[float, Float[np.ndarray, " n_value"]],
    e: Union[float, Float[np.ndarray, " n_value"]],
    Izz: float,
    I0: float,
    r: float,
) -> Union[float, Float[np.ndarray, " n_value"]]:
    """Strain h0 of a source of ellipticity e."""
    return 1.1e-24 * (e / 1.0e-6) * (Izz / I0) * (freq / 1.0e3) ** 2 * (1.0 / r)


def ellipticity_from_h0(
    freq: Union[float, Float[np.ndarray, " n_value"]],
    h0: Union[float, Float[np.ndarray, " n_value"]],
    Izz: float,
    I0: float,
    r: float,
) -> Union[float, Float[np.ndarray, " n_value"]]:
    """Ellipticity e of a source of strain h0."""
    return 9.46e-5 * (h0 / 1e-24) * (I0 / Izz) * (r / 1.0) * (1e2 / freq) ** 2


def h0_from_alpha(
    freq: Union[float, Float[np.ndarray, " n_value"]],
    alpha: Union[float, Float[np.ndarray, " n_value"]],
    Izz: float,
    I0: float,
    r: float,
) -> Union[float, Float[np.ndarray, " n_value"]]:
    """Strain h0 of a source with r-mode amplitude alpha."""
    return 3.6e-23 * (alpha / 1.0e-3) * (freq / 1.0e3) ** 3 * (1.0 / r)


def alpha_from_h0(
    freq: Union[float, Float[np.ndarray, " n_value"]],
    h0: Union[float, Float[np.ndarray, " n_value"]],
    Izz: float,
    I0: float,
    r: float,
) -> Union[float, Float[np.ndarray, " n_value"]]:
    """r-mode amplitude alpha of a source of strain h0."""
    return 0.028 * (h0 / 1e-24) * (r / 1.0) * (1e2 / freq) ** 3
