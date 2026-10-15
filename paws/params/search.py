"""Search parameter tables of the initial search: the f0 / f1dot / f2dot (/ f3dot / f4dot) cells of each band."""

from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from paws.definitions import phase_param_name

from .models import SpindownModel


class SearchParamGenerator:
    """Parameter tables of the initial search: the sky position and the cells of each band, from the spin-down
    ranges of `model` up to spin-down order freq_deriv_order."""

    def __init__(self, model: SpindownModel, freq_deriv_order: int = 2) -> None:
        self.freq_deriv_order = freq_deriv_order
        self.freq_param_name, self.freq_deriv_param_name = phase_param_name(
            freq_deriv_order
        )
        self.model = model

        # Initialize attributes that are set later via generate_search_ranges
        self.alpha: Optional[float] = None
        self.dalpha: Optional[float] = None
        self.delta: Optional[float] = None
        self.ddelta: Optional[float] = None

    def generate_band_search_ranges(
        self,
        freq: float,
        df: float,
        n_f1: int,
        n_f2: int,
        n_f3: int = 1,
        n_f4: int = 1,
        bandwidth: float = 1.0,
    ) -> fits.BinTableHDU:
        """The cells of [freq, freq + bandwidth): n_f0 sub-bands of width df, each split into n_f1 x n_f2 (x n_f3 x
        n_f4) spin-down cells over the model's ranges."""
        # Calculate number of steps (sub-bands) in this file
        # Use round() to avoid floating point truncation errors (e.g., 0.299999 / 0.1)
        n_f0 = int(np.round(bandwidth / df))
        # Total rows = (sub-bands) * (f1 segments) * (f2 segments) ...
        n = int(n_f0 * n_f1 * n_f2 * n_f3 * n_f4)

        dtype_list = [
            (key, ">f8") for key in (self.freq_param_name + self.freq_deriv_param_name)
        ]
        data = np.recarray((n,), dtype=dtype_list)

        for i in range(n_f0):
            f0_min = freq + i * df
            # the model takes one frequency per point: one point here
            f0 = np.array([f0_min], dtype=float)

            for j in range(n_f1):
                # 1. Get f1 range
                _f1_min, _f1_max = self.model.f1_range(f0, df)
                _f1_band = _f1_max - _f1_min

                f1_band = _f1_band / n_f1
                f1_min = _f1_min + j * f1_band
                f1_max = f1_min + f1_band

                # the last cell ends exactly at the range edge
                if j == n_f1 - 1:
                    f1_max = _f1_max
                    f1_band = f1_max - f1_min

                for k in range(n_f2):
                    # 2. Get f2 range
                    _f2_min, _f2_max = self.model.f2_range(f0, df, f1_min, f1_max)
                    _f2_band = _f2_max - _f2_min

                    f2_band = _f2_band / n_f2
                    f2_min = _f2_min + k * f2_band

                    # --- f3 Loop ---
                    for p in range(n_f3):
                        if self.freq_deriv_order >= 3:
                            _f3_min, _f3_max = self.model.f3_range(
                                f0, df, f1_min, f1_max
                            )
                            _f3_band = _f3_max - _f3_min

                            f3_band = _f3_band / n_f3
                            f3_min = _f3_min + p * f3_band

                        # --- f4 Loop ---
                        for m in range(n_f4):
                            if self.freq_deriv_order >= 4:
                                _f4_min, _f4_max = self.model.f4_range(
                                    f0, df, f1_min, f1_max
                                )
                                _f4_band = _f4_max - _f4_min

                                f4_band = _f4_band / n_f4
                                f4_min = _f4_min + m * f4_band

                            # Calculate Index
                            idx = (
                                (i * n_f1 * n_f2 * n_f3 * n_f4)
                                + (j * n_f2 * n_f3 * n_f4)
                                + (k * n_f3 * n_f4)
                                + (p * n_f4)
                                + m
                            )

                            # Assign Data
                            data[idx]["freq"], data[idx]["df"] = f0_min, df
                            data[idx]["f1dot"], data[idx]["df1dot"] = (
                                f1_min[0],
                                f1_band[0],
                            )
                            data[idx]["f2dot"], data[idx]["df2dot"] = (
                                f2_min[0],
                                f2_band[0],
                            )

                            if self.freq_deriv_order >= 3:
                                data[idx]["f3dot"], data[idx]["df3dot"] = (
                                    f3_min[0],
                                    f3_band[0],
                                )
                            if self.freq_deriv_order >= 4:
                                data[idx]["f4dot"], data[idx]["df4dot"] = (
                                    f4_min[0],
                                    f4_band[0],
                                )

        data = Table(data)

        # Sky location
        data.add_column(self.alpha * np.ones(n), name="alpha")
        data.add_column(self.dalpha * np.ones(n), name="dalpha")
        data.add_column(self.delta * np.ones(n), name="delta")
        data.add_column(self.ddelta * np.ones(n), name="ddelta")
        return fits.BinTableHDU(data)

    def generate_search_ranges(
        self,
        alpha: float,
        dalpha: float,
        delta: float,
        ddelta: float,
        f0_min: float,
        f0_max: float,
        df: float = 0.02,
        df1: float = 1e-9,
        df2: float = 1e-19,
        df3: Optional[float] = None,
        df4: Optional[float] = None,
    ) -> dict[int, fits.BinTableHDU]:
        """{1 Hz band: its cell table} over [f0_min, f0_max) at the sky position; df, df1, df2 (df3, df4) are the
        cell widths, the spin-down counts follow from the model's ranges at each band."""
        self.alpha = alpha
        self.dalpha = dalpha
        self.delta = delta
        self.ddelta = ddelta

        params = {}

        # Current start frequency
        current_freq = f0_min

        # Use a while loop to handle non-integer ranges
        pbar = tqdm(total=f0_max - f0_min)

        while current_freq < f0_max:
            next_boundary = int(current_freq + 1)

            # The actual stop frequency is the smaller of the boundary or the requested max
            next_freq = min(next_boundary, f0_max)

            # Actual bandwidth for this specific table (e.g., 31.5->32.0 is 0.5Hz)
            chunk_bw = next_freq - current_freq

            # 2. Stop if chunk is too small (handle float precision residue)
            if chunk_bw < df / 2.0:
                break

            # 3. Calculate N segments based on current_freq (the ranges may vary with freq)
            f0 = np.array([current_freq], dtype=float)
            n_f1 = int(self.model.n_f1_ranges(f0, df, df1)[0])
            n_f2 = int(self.model.n_f2_ranges(f0, df, df2)[0])
            f1_min, f1_max = self.model.f1_range(f0, df)

            # F3 Logic
            n_f3 = 1
            if self.freq_deriv_order >= 3 and df3 is not None:
                f3_min, f3_max = self.model.f3_range(f0, df, f1_min, f1_max)
                n_f3 = int(np.ceil((f3_max - f3_min)[0] / df3))

            # F4 Logic
            n_f4 = 1
            if self.freq_deriv_order >= 4 and df4 is not None:
                f4_min, f4_max = self.model.f4_range(f0, df, f1_min, f1_max)
                n_f4 = int(np.ceil((f4_max - f4_min)[0] / df4))

            # 4. Generate table with calculated chunk bandwidth
            table = self.generate_band_search_ranges(
                current_freq, df, n_f1, n_f2, n_f3, n_f4, bandwidth=chunk_bw
            )

            if table is not None:
                params[int(current_freq)] = table

            pbar.update(chunk_bw)
            current_freq = next_freq

        pbar.close()
        return params
