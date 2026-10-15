"""Spin-down models: the spin-down ranges searched at a frequency. search.yaml names a script in the config directory
defining spindown_model(freq) -> SpindownModel; a model subclasses SpindownModel."""

import sys
from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float, Int

from paws.definitions import SECONDS_PER_YEAR


class SpindownModel(ABC):
    """Spin-down ranges at each frequency f0 of an array, as (min, max), and the higher derivatives at a
    point from its lower ones (f3_value, f4_value) for injections."""

    def n_f1_ranges(
        self, f0: Float[np.ndarray, " n_point"], f0_band: float, df1: float
    ) -> Int[np.ndarray, " n_point"]:
        """Number of f1dot search ranges of width df1 over the f1dot range at each f0."""
        f1_min, f1_max = self.f1_range(f0, f0_band)
        return np.ceil((f1_max - f1_min) / df1).astype(int)

    def n_f2_ranges(
        self, f0: Float[np.ndarray, " n_point"], f0_band: float, df2: float
    ) -> Int[np.ndarray, " n_point"]:
        """Number of f2dot search ranges of width df2 over the f2dot range at each f0."""
        f1_min, f1_max = self.f1_range(f0, f0_band)
        f2_min, f2_max = self.f2_range(f0, f0_band, f1_min, f1_max)
        return np.ceil((f2_max - f2_min) / df2).astype(int)

    @abstractmethod
    def f1_range(
        self, f0: Float[np.ndarray, " n_point"], f0_band: float
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """f1dot over [f0, f0 + f0_band]."""

    @abstractmethod
    def f2_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """f2dot over [f0, f0 + f0_band] and [f1_min, f1_max]."""

    @abstractmethod
    def f3_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """f3dot over [f0, f0 + f0_band] and [f1_min, f1_max]."""

    @abstractmethod
    def f4_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """f4dot over [f0, f0 + f0_band] and [f1_min, f1_max]."""

    @abstractmethod
    def f3_value(
        self,
        f0: Float[np.ndarray, " n_point"],
        f1: Float[np.ndarray, " n_point"],
        f2: Float[np.ndarray, " n_point"],
    ) -> Float[np.ndarray, " n_point"]:
        """f3dot at each point (f0, f1dot, f2dot)."""

    @abstractmethod
    def f4_value(
        self,
        f0: Float[np.ndarray, " n_point"],
        f1: Float[np.ndarray, " n_point"],
        f2: Float[np.ndarray, " n_point"],
    ) -> Float[np.ndarray, " n_point"]:
        """f4dot at each point (f0, f1dot, f2dot)."""


class PowerLawModel(SpindownModel):
    """Ranges of the power law f1dot = -k f0^nc with braking index nc in [nc_min, nc_max] and age tau [s]:
    f1dot in [-f0 / ((nc_min - 1) tau), 0], the higher derivatives from nc and f1dot."""

    def __init__(
        self,
        nc_min: float = 2,
        nc_max: float = 7,
        tau: float = 300 * SECONDS_PER_YEAR,
    ) -> None:
        self.nc_min = nc_min
        self.nc_max = nc_max
        self.tau = tau

    def f1_range(
        self, f0: Float[np.ndarray, " n_point"], f0_band: float
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """(min, max) of f1dot at f0: from -(f0 + f0_band) / ((nc_min - 1) tau) to 0."""
        f1_min = -(f0 + f0_band) / ((self.nc_min - 1.0) * self.tau)
        f1_max = np.zeros_like(f1_min)
        return f1_min, f1_max

    def f2_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """(min, max) of f2dot = nc f1dot^2 / f0 over the braking index range and the f1dot range."""
        f2_min = self.nc_min * np.minimum(f1_min**2, f1_max**2) / (f0 + f0_band)
        f2_max = self.nc_max * np.maximum(f1_min**2, f1_max**2) / f0
        return f2_min, f2_max

    def f3_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """(min, max) of f3dot = nc (2 nc - 1) f1dot^3 / f0^2."""
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
        return f3_min, f3_max

    def f4_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """(min, max) of f4dot = nc (2 nc - 1) (3 nc - 2) f1dot^4 / f0^3."""
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
        return f4_min, f4_max

    def braking_index(
        self,
        f0: Float[np.ndarray, " n_point"],
        f1: Float[np.ndarray, " n_point"],
        f2: Float[np.ndarray, " n_point"],
    ) -> Float[np.ndarray, " n_point"]:
        """nc = f0 f2dot / f1dot^2 at each point."""
        with np.errstate(divide="ignore", invalid="ignore"):
            nc = f0 * f2 / f1**2
        if np.any(nc < 0):
            sys.exit("braking index is negative")
        return nc

    def f3_value(
        self,
        f0: Float[np.ndarray, " n_point"],
        f1: Float[np.ndarray, " n_point"],
        f2: Float[np.ndarray, " n_point"],
    ) -> Float[np.ndarray, " n_point"]:
        """f3dot at each point from its braking index."""
        nc = self.braking_index(f0, f1, f2)
        return nc * (2.0 * nc - 1.0) * f1**3 / f0**2

    def f4_value(
        self,
        f0: Float[np.ndarray, " n_point"],
        f1: Float[np.ndarray, " n_point"],
        f2: Float[np.ndarray, " n_point"],
    ) -> Float[np.ndarray, " n_point"]:
        """f4dot at each point from its braking index."""
        nc = self.braking_index(f0, f1, f2)
        return nc * (2.0 * nc - 1.0) * (3.0 * nc - 2.0) * f1**4 / f0**3


class UniformModel(SpindownModel):
    """Fixed (min, max) ranges of every spin-down, the same at every frequency; f3dot and f4dot at a point are
    the middle of their ranges."""

    def __init__(
        self,
        f1_lim: tuple[float, float],
        f2_lim: tuple[float, float],
        f3_lim: tuple[float, float] = (0.0, 0.0),
        f4_lim: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.f1_lim = f1_lim
        self.f2_lim = f2_lim
        self.f3_lim = f3_lim
        self.f4_lim = f4_lim

    @staticmethod
    def _range(
        limits: tuple[float, float], f0: Float[np.ndarray, " n_point"]
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        """The limits as (min, max) at each f0."""
        low, high = limits
        return np.full_like(f0, low), np.full_like(f0, high)

    def f1_range(
        self, f0: Float[np.ndarray, " n_point"], f0_band: float
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        return self._range(self.f1_lim, f0)

    def f2_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        return self._range(self.f2_lim, f0)

    def f3_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        return self._range(self.f3_lim, f0)

    def f4_range(
        self,
        f0: Float[np.ndarray, " n_point"],
        f0_band: float,
        f1_min: Float[np.ndarray, " n_point"],
        f1_max: Float[np.ndarray, " n_point"],
    ) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
        return self._range(self.f4_lim, f0)

    def f3_value(
        self,
        f0: Float[np.ndarray, " n_point"],
        f1: Float[np.ndarray, " n_point"],
        f2: Float[np.ndarray, " n_point"],
    ) -> Float[np.ndarray, " n_point"]:
        return np.full_like(f0, sum(self.f3_lim) / 2)

    def f4_value(
        self,
        f0: Float[np.ndarray, " n_point"],
        f1: Float[np.ndarray, " n_point"],
        f2: Float[np.ndarray, " n_point"],
    ) -> Float[np.ndarray, " n_point"]:
        return np.full_like(f0, sum(self.f4_lim) / 2)
