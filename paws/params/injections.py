"""Injection parameter tables: random signals in the non-saturated sub-bands and the search ranges around them."""

from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from jaxtyping import Float, Shaped

from paws.definitions import ext_param_name, phase_param_name
from paws.params.models import SpindownModel
from paws.params.sky import sample_sky_positions, sky_hex_offsets


class InjectionParamGenerator:
    def __init__(
        self, model: SpindownModel, ref_time: float, f0_band: float = 0.1
    ) -> None:
        self.model = model
        self.ref_time = ref_time
        self.f0_band = f0_band
        self.inj_col_names = ext_param_name()

    def draw_f0_in_non_saturated_bands(
        self, non_sat_bands: Float[np.ndarray, " n_subband"], n_inj: int
    ) -> Float[np.ndarray, " n_injection"]:
        """Randomly draws f0 values from the available non-saturated bands."""
        if non_sat_bands is None or len(non_sat_bands) == 0:
            raise ValueError("No non-saturated bands provided for injection!")

        # Uniform probability across bands
        band_idx = np.random.randint(0, len(non_sat_bands), n_inj)
        selected_bands = non_sat_bands[band_idx]

        # Uniform frequency within the specific 0.1Hz (or f0_band) chunk
        f0 = selected_bands + np.random.uniform(0, self.f0_band, n_inj)
        return f0

    def generate_injections(
        self,
        alpha: float,
        delta: float,
        non_sat_bands: Float[np.ndarray, " n_subband"],
        h0: float,
        n_inj: int,
        inj_freq_deriv_order: int,
        sky_radius: float,
    ) -> fits.BinTableHDU:
        """Generates the source injection parameters."""
        freq_params, _ = phase_param_name(inj_freq_deriv_order)

        # Create column names: Standard Inj names + Phase params (Freq, f1dot...)
        col_names = self.inj_col_names + [
            p.capitalize() if p == "freq" else p for p in freq_params
        ]

        # Pre-allocate
        inj_data = Table(np.zeros((n_inj, len(col_names))), names=col_names)

        # 1. Sky Location
        alpha, delta = sample_sky_positions(alpha, delta, sky_radius, n_inj)
        inj_data["Alpha"] = alpha
        inj_data["Delta"] = delta

        # 2. Polarization angle & reference time
        inj_data["psi"] = np.random.uniform(-np.pi / 4, np.pi / 4, n_inj)
        inj_data["refTime"] = self.ref_time

        cos_i = np.random.uniform(-1, 1, n_inj)
        inj_data["aPlus"] = h0 * (1.0 + cos_i**2) / 2.0
        inj_data["aCross"] = h0 * cos_i

        # 3. Frequency Evolution
        f0 = self.draw_f0_in_non_saturated_bands(non_sat_bands, n_inj)
        inj_data["Freq"] = f0

        if inj_freq_deriv_order >= 1:
            f1_min, f1_max = self.model.f1_range(f0, 0)
            f1 = np.random.uniform(f1_min, f1_max)
            inj_data["f1dot"] = f1

        if inj_freq_deriv_order >= 2:
            f2_min, f2_max = self.model.f2_range(f0, 0, f1, f1)
            f2 = np.random.uniform(f2_min, f2_max)
            inj_data["f2dot"] = f2

        if inj_freq_deriv_order >= 3:
            inj_data["f3dot"] = self.model.f3_value(f0, f1, f2)

        if inj_freq_deriv_order >= 4:
            inj_data["f4dot"] = self.model.f4_value(f0, f1, f2)

        return fits.BinTableHDU(inj_data)

    def generate_injection_search_ranges(
        self,
        alpha: float,
        dalpha: float,
        delta: float,
        ddelta: float,
        spacing: dict[str, float],
        inj_data: Shaped[np.ndarray, " n_injection"],
        freq_deriv_order: int,
        n_spacing: float = 1.0,
        sky_radius: float = 0.0,
        spacing_alpha: Optional[float] = None,
        spacing_delta: Optional[float] = None,
    ) -> fits.BinTableHDU:
        """
        Generates the small search windows around the injections for Weave.

        When sky_radius > 0 and spacings are provided, each injection gets n_sky rows
        (one per hex-grid sky point within sky_radius), ordered as:
        inj_0/sky_0, inj_0/sky_1, ..., inj_1/sky_0, inj_1/sky_1, ...
        Consecutive blocks of n_sky rows map to the same injection.
        """
        freq_names, deriv_names = phase_param_name(freq_deriv_order)

        if sky_radius > 0 and spacing_alpha is not None and spacing_delta is not None:
            d_alpha, d_delta = sky_hex_offsets(sky_radius, spacing_alpha, spacing_delta)
            n_sky = len(d_alpha)
        else:
            d_alpha, d_delta = np.array([0.0]), np.array([0.0])
            n_sky = 1

        n_inj = len(inj_data)
        n_rows = n_inj * n_sky
        out_cols = freq_names + deriv_names + ["alpha", "dalpha", "delta", "ddelta"]
        search_range = Table(np.zeros((n_rows, len(out_cols))), names=out_cols)

        # Expand injection data: repeat each row n_sky times
        inj_idx = np.repeat(np.arange(n_inj), n_sky)
        sky_idx = np.tile(np.arange(n_sky), n_inj)

        # 1. Frequency & Spindowns
        df = spacing["df"]
        search_range["freq"] = inj_data["Freq"][inj_idx] - n_spacing * df
        search_range["df"] = 2 * n_spacing * df

        if freq_deriv_order >= 1:
            df1 = spacing["df1dot"]
            search_range["f1dot"] = inj_data["f1dot"][inj_idx] - n_spacing * df1
            search_range["df1dot"] = 2 * n_spacing * df1

        if freq_deriv_order >= 2:
            df2 = spacing["df2dot"]
            search_range["f2dot"] = inj_data["f2dot"][inj_idx] - n_spacing * df2
            search_range["df2dot"] = 2 * n_spacing * df2

        if freq_deriv_order >= 3:
            df3 = spacing["df3dot"]
            search_range["f3dot"] = inj_data["f3dot"][inj_idx] - n_spacing * df3
            search_range["df3dot"] = 2 * n_spacing * df3

        if freq_deriv_order >= 4:
            df4 = spacing["df4dot"]
            search_range["f4dot"] = inj_data["f4dot"][inj_idx] - n_spacing * df4
            search_range["df4dot"] = 2 * n_spacing * df4

        # 2. Sky: offset each sky point from the nominal position
        search_range["alpha"] = alpha + d_alpha[sky_idx]
        search_range["dalpha"] = dalpha
        search_range["delta"] = delta + d_delta[sky_idx]
        search_range["ddelta"] = ddelta

        return fits.BinTableHDU(search_range)

    def generate_injections_and_search_ranges(
        self,
        alpha: float,
        dalpha: float,
        delta: float,
        ddelta: float,
        non_sat_bands: Float[np.ndarray, " n_subband"],
        spacing: dict[str, float],
        h0: float,
        freq: int,
        n_inj: int = 1,
        n_spacing: float = 1.0,
        freq_deriv_order: int = 2,
        inj_freq_deriv_order: int = 4,
        sky_radius: float = 0.0,
        spacing_alpha: Optional[float] = None,
        spacing_delta: Optional[float] = None,
    ) -> tuple[dict[str, fits.BinTableHDU], dict[str, fits.BinTableHDU]]:
        """
        High-level wrapper to generate both Injection Parameters and Search Ranges: ({band: search ranges},
        {band: injection parameters}) with the band as a string key.

        When sky_radius > 0 with spacings provided, each injection is searched at n_sky
        hex-grid sky points within the cap. Both returned dicts contain n_inj * n_sky rows
        (injection params replicated per sky point). Consecutive blocks of n_sky rows belong
        to the same injection, matching what ResultAnalysisManager.make_outlier expects.
        """
        if freq_deriv_order > 4:
            print("Error: frequency derivative order larger than 4.")
        if inj_freq_deriv_order > 4:
            print("Error: Injection frequency derivative order larger than 4.")

        # 1. Generate Injection Table (n_inj rows)
        ip_hdu = self.generate_injections(
            alpha, delta, non_sat_bands, h0, n_inj, inj_freq_deriv_order, sky_radius
        )

        # 2. Generate Search Range Table (n_inj * n_sky rows when sky_layers > 0)
        sp_hdu = self.generate_injection_search_ranges(
            alpha,
            dalpha,
            delta,
            ddelta,
            spacing,
            ip_hdu.data,
            freq_deriv_order,
            n_spacing,
            sky_radius=sky_radius,
            spacing_alpha=spacing_alpha,
            spacing_delta=spacing_delta,
        )

        # 3. Replicate injection params to match search range row count
        if len(sp_hdu.data) > n_inj:
            n_sky = len(sp_hdu.data) // n_inj
            inj_idx = np.repeat(np.arange(n_inj), n_sky)
            ip_hdu = fits.BinTableHDU(Table(ip_hdu.data)[inj_idx])

        searchParamDict = {str(freq): sp_hdu}
        injParamDict = {str(freq): ip_hdu}

        return searchParamDict, injParamDict
