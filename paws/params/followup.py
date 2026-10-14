"""Follow-up parameter tables: the search range around each seed, tiled over the sky grid."""

from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from jaxtyping import Shaped

from paws.definitions import phase_param_name
from paws.params.models import PowerLawModel
from paws.params.sky import sky_hex_offsets


class FollowUpParamGenerator:
    def __init__(self, model: PowerLawModel) -> None:
        """
        Initialize the FollowUpParams generator.

        Parameters:
        - model (object): Instance of PowerLawModel (REQUIRED).
        """
        self.model = model

        # Initialize attributes that are set later via generate_parameters
        self.alpha: Optional[float] = None
        self.dalpha: Optional[float] = None
        self.delta: Optional[float] = None
        self.ddelta: Optional[float] = None

    def generate_parameter_table(
        self,
        data: Shaped[np.ndarray, " n_seed"],
        old_freq_deriv_order: int,
        new_freq_deriv_order: int,
        spacing: dict[str, float],
        n_spacing: float,
        sky_radius: float = 0.0,
        spacing_alpha: Optional[float] = None,
        spacing_delta: Optional[float] = None,
    ) -> fits.BinTableHDU:
        """
        Constructs the follow-up parameter table by expanding ranges.

        Parameters:
        - data (numpy.array): Input outlier data.
        - old_freq_deriv_order (int): The derivative order of the previous search.
        - new_freq_deriv_order (int): The derivative order for the new search.
        - spacing (dict): The template spacings (df, df1dot, ...) of the previous search.
        - n_spacing (float): The spacing multiplier for expanding ranges.

        Returns:
        - fits.BinTableHDU: The constructed follow-up parameter table.
        """
        data = Table(data)

        # Check if there are outliers
        if len(data) == 0:
            return fits.BinTableHDU(data=data)

        if "dalpha" not in data.colnames:
            data["dalpha"] = self.dalpha

        if "ddelta" not in data.colnames:
            data["ddelta"] = self.ddelta

        # Get parameter names from utils
        freq_param_names, freq_deriv_param_names = phase_param_name(
            old_freq_deriv_order
        )
        new_freq_param_names, new_freq_deriv_param_names = phase_param_name(
            new_freq_deriv_order
        )

        # Expand existing ranges centered on the outlier
        # Logic: New Start = Old Value - (nSpacing * StepSize)
        #        New Bandwidth = 2 * nSpacing * StepSize
        for freq_name, freq_deriv_name in zip(freq_param_names, freq_deriv_param_names):
            data[freq_name] = data[freq_name] - n_spacing * spacing[freq_deriv_name]
            data[freq_deriv_name] = 2 * n_spacing * spacing[freq_deriv_name]

            # Remove processed params from the "new" list so we know what's left to add
            if freq_name in new_freq_param_names:
                new_freq_param_names.remove(freq_name)
            if freq_deriv_name in new_freq_deriv_param_names:
                new_freq_deriv_param_names.remove(freq_deriv_name)

        # Add new higher-order derivatives (f3dot, f4dot) that weren't in the previous stage
        for freq_name, freq_deriv_name in zip(
            new_freq_param_names, new_freq_deriv_param_names
        ):
            if freq_name == "f3dot":
                # Calculate broad ranges based on current f1/f2 values
                f3_min, _, f3_band = self.model.f3_broad_range(
                    data["freq"],
                    spacing["df"],
                    data["f1dot"],
                    data["f1dot"] + spacing["df1dot"],
                )
                data.add_column(f3_min, name=freq_name)
                data.add_column(f3_band, name=freq_deriv_name)

            elif freq_name == "f4dot":
                f4_min, _, f4_band = self.model.f4_broad_range(
                    data["freq"],
                    spacing["df"],
                    data["f1dot"],
                    data["f1dot"] + spacing["df1dot"],
                )
                data.add_column(f4_min, name=freq_name)
                data.add_column(f4_band, name=freq_deriv_name)

        # Sky tiling: replicate each outlier row for every hex-grid sky point.
        # Ordering: outlier_0/sky_0, outlier_0/sky_1, ..., outlier_1/sky_0, ...
        # so that consecutive blocks of n_sky rows map to the same outlier.
        if sky_radius > 0 and spacing_alpha is not None and spacing_delta is not None:
            d_alpha, d_delta = sky_hex_offsets(sky_radius, spacing_alpha, spacing_delta)
            n_sky = len(d_alpha)
            n_out = len(data)
            idx = np.repeat(np.arange(n_out), n_sky)
            sky_idx = np.tile(np.arange(n_sky), n_out)
            data = data[idx]
            data["alpha"] = data["alpha"] + d_alpha[sky_idx]
            data["delta"] = data["delta"] + d_delta[sky_idx]

        return fits.BinTableHDU(data=data)

    def generate_parameter(
        self,
        alpha: float,
        dalpha: float,
        delta: float,
        ddelta: float,
        data: Shaped[np.ndarray, " n_seed"],
        old_freq_deriv_order: int,
        new_freq_deriv_order: int,
        spacing: dict[str, float],
        n_spacing: float = 1.0,
        sky_radius: float = 0.0,
        spacing_alpha: Optional[float] = None,
        spacing_delta: Optional[float] = None,
    ) -> fits.BinTableHDU:
        """
        Main entry point to generate parameters.
        """
        self.alpha = alpha
        self.dalpha = dalpha
        self.delta = delta
        self.ddelta = ddelta

        if old_freq_deriv_order > 4 or new_freq_deriv_order > 4:
            print("Error: frequency derivative order larger than 4.")
            # You might want to raise an actual Exception here:
            # raise ValueError("Frequency derivative order cannot be larger than 4")

        params = self.generate_parameter_table(
            data,
            old_freq_deriv_order,
            new_freq_deriv_order,
            spacing,
            n_spacing,
            sky_radius=sky_radius,
            spacing_alpha=spacing_alpha,
            spacing_delta=spacing_delta,
        )

        # Removed unused args: cluster, workInLocalDir are not used in logic anymore
        print("Done generation.\n")
        return params
