"""Spin-down ranges of the search: from the braking index (PowerLawModel) or fixed (UniformModel)."""

import sys
from typing import Optional, Union

import numpy as np
from jaxtyping import Float

# one frequency / spin-down value, or one per candidate (outlier or injection)
PerCandidate = Union[float, Float[np.ndarray, " n_candidate"]]


class PowerLawModel:
    """
    Defines parameter ranges based on the braking index (Power Law) physics.
    Bounds are dynamically calculated using f0 and characteristic age (tau).
    """

    def __init__(
        self,
        nc_min: float = 2,
        nc_max: float = 7,
        tau: float = 300 * 365.25 * 24 * 3600,
    ) -> None:
        self.nc_min = nc_min
        self.nc_max = nc_max
        self.tau = tau

    def get_n_f1(self, freq: float, f0_band: float, df1: float = 1.5e-9) -> int:
        """Number of f1dot sub-bands of width df1 over the f1dot range at freq."""
        _, _, bandwidth = self.f1_broad_range(freq, f0_band)
        return int(np.ceil(bandwidth / df1))

    def get_n_f2(self, freq: float, f0_band: float, df2: float = 1.0e-19) -> int:
        """Number of f2dot sub-bands of width df2 over the f2dot range at freq."""
        f1_min, f1_max, _ = self.f1_broad_range(freq, f0_band)
        _, _, bandwidth = self.f2_broad_range(freq, f0_band, f1_min, f1_max)
        return int(np.ceil(bandwidth / df2))

    def f0_broad_range(self, f0: float, f0_band: float) -> tuple[float, float, float]:
        f0_min = f0
        f0_max = f0 + f0_band
        return f0_min, f0_max, f0_band

    def f1_broad_range(
        self, f0: PerCandidate, f0_band: float
    ) -> tuple[PerCandidate, PerCandidate, PerCandidate]:
        """(min, max, width) of f1dot at f0: from -(f0 + f0_band) / ((nc_min - 1) tau) to 0."""
        f1_min = -(f0 + f0_band) / ((self.nc_min - 1.0) * self.tau)
        f1_max = 0.0 if np.ndim(f1_min) == 0 else np.zeros(np.shape(f1_min))
        return f1_min, f1_max, f1_max - f1_min

    def f2_broad_range(
        self,
        f0: PerCandidate,
        f0_band: float,
        f1_min: PerCandidate,
        f1_max: PerCandidate,
    ) -> tuple[PerCandidate, PerCandidate, PerCandidate]:
        """(min, max, width) of f2dot = nc f1dot^2 / f0 over the braking index range and the f1dot range."""
        f2_min = self.nc_min * np.minimum(f1_min**2, f1_max**2) / (f0 + f0_band)
        f2_max = self.nc_max * np.maximum(f1_min**2, f1_max**2) / f0
        return f2_min, f2_max, f2_max - f2_min

    def f3_broad_range(
        self,
        f0: PerCandidate,
        f0_band: float,
        f1_min: PerCandidate,
        f1_max: PerCandidate,
    ) -> tuple[PerCandidate, PerCandidate, PerCandidate]:
        """(min, max, width) of f3dot = nc (2 nc - 1) f1dot^3 / f0^2."""
        f3_min = (
            self.nc_max
            * (2 * self.nc_max - 1)
            * np.minimum(f1_min**3, f1_max**3)
            / f0**2
        )
        f3_max = (
            self.nc_min
            * (2 * self.nc_min - 1)
            * np.maximum(f1_min**3, f1_max**3)
            / (f0 + f0_band) ** 2
        )
        return f3_min, f3_max, f3_max - f3_min

    def f4_broad_range(
        self,
        f0: PerCandidate,
        f0_band: float,
        f1_min: PerCandidate,
        f1_max: PerCandidate,
    ) -> tuple[PerCandidate, PerCandidate, PerCandidate]:
        """(min, max, width) of f4dot = nc (2 nc - 1) (3 nc - 2) f1dot^4 / f0^3."""
        f4_min = (
            self.nc_min
            * (2 * self.nc_min - 1)
            * (3 * self.nc_min - 2)
            * np.minimum(f1_min**4, f1_max**4)
            / (f0 + f0_band) ** 3
        )
        f4_max = (
            self.nc_max
            * (2 * self.nc_max - 1)
            * (3 * self.nc_max - 2)
            * np.maximum(f1_min**4, f1_max**4)
            / f0**3
        )
        return f4_min, f4_max, f4_max - f4_min

    def f3_value(
        self,
        f0: Float[np.ndarray, " n_candidate"],
        f1: Float[np.ndarray, " n_candidate"],
        f2: Float[np.ndarray, " n_candidate"],
    ) -> Float[np.ndarray, " n_candidate"]:
        """f3dot of each candidate from its braking index nc = f0 f2dot / f1dot^2."""
        with np.errstate(divide="ignore", invalid="ignore"):
            nc = f0 * f2 / f1**2
        nc = np.atleast_1d(nc)
        # We allow checking this even if inputs are arrays; handled by numpy
        # If braking index is physically impossible (<0), we error out.
        if np.any(nc < 0):
            sys.exit("braking index is negative")
        f3 = nc * (2.0 * nc - 1.0) * f1**3 / f0**2
        return f3

    def f4_value(
        self,
        f0: Float[np.ndarray, " n_candidate"],
        f1: Float[np.ndarray, " n_candidate"],
        f2: Float[np.ndarray, " n_candidate"],
    ) -> Float[np.ndarray, " n_candidate"]:
        """f4dot of each candidate from its braking index nc = f0 f2dot / f1dot^2."""
        with np.errstate(divide="ignore", invalid="ignore"):
            nc = f0 * f2 / f1**2
        nc = np.atleast_1d(nc)
        if np.any(nc < 0):
            sys.exit("braking index is negative")
        f4 = nc * (2.0 * nc - 1.0) * (3.0 * nc - 2.0) * f1**4 / f0**3
        return f4


class UniformModel:
    """
    Defines parameter ranges based on fixed rectangular bounds (Uniform).
    """

    def __init__(
        self,
        f1_lim: tuple[float, float],
        f2_lim: tuple[float, float],
        f3_lim: Optional[tuple[float, float]] = None,
        f4_lim: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        :param f1_lim: tuple/list (min, max) for f1
        :param f2_lim: tuple/list (min, max) for f2
        """
        self.f1_lim = f1_lim
        self.f2_lim = f2_lim
        self.f3_lim = f3_lim if f3_lim else (0.0, 0.0)
        self.f4_lim = f4_lim if f4_lim else (0.0, 0.0)

    def get_n_f1(self, df1: float) -> int:
        _, _, bandwidth = self.f1_broad_range()
        return int(np.ceil(bandwidth / df1))

    def get_n_f2(self, df2: float) -> int:
        _, _, bandwidth = self.f2_broad_range()
        return int(np.ceil(bandwidth / df2))

    def f0_broad_range(self, f0: float, f0_band: float) -> tuple[float, float, float]:
        return f0, f0 + f0_band, f0_band

    def f1_broad_range(self) -> tuple[float, float, float]:
        f1_min, f1_max = self.f1_lim
        return f1_min, f1_max, f1_max - f1_min

    def f2_broad_range(self) -> tuple[float, float, float]:
        f2_min, f2_max = self.f2_lim
        return f2_min, f2_max, f2_max - f2_min

    def f3_broad_range(self) -> tuple[float, float, float]:
        f3_min, f3_max = self.f3_lim
        return f3_min, f3_max, f3_max - f3_min

    def f4_broad_range(self) -> tuple[float, float, float]:
        f4_min, f4_max = self.f4_lim
        return f4_min, f4_max, f4_max - f4_min
