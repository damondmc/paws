"""Thresholds of the follow-up cuts calibrated on injections, per threshold band (band_edges).

excess ratio cut: a low percentile of the injections' (2F_now - 4) / (2F_reference - 4).
H1/L1 window:     the central range of the injections' log10 H1/L1 excess ratio.
"""

import numpy as np
from jaxtyping import Float, Int, Shaped

from paws.analysis.statistics import excess_ratio


def injection_excess_ratios(
    reference_rows: Shaped[np.ndarray, " n_reference"],
    reference_injected_freqs: Float[np.ndarray, " n_reference"],
    now_rows: Shaped[np.ndarray, " n_injection"],
    now_injected_freqs: Float[np.ndarray, " n_injection"],
) -> Float[np.ndarray, " n_injection"]:
    """(2F_now - 4) / (2F_reference - 4) of every injection of one band, matched between the stages by injected Freq
    (every injection of `now` is among the reference ones)."""
    reference_row_of_freq = {
        round(f, 9): i for i, f in enumerate(reference_injected_freqs)
    }
    reference_index = np.array(
        [reference_row_of_freq[round(f, 9)] for f in now_injected_freqs], dtype=int
    )
    return excess_ratio(now_rows["mean2F"], reference_rows["mean2F"][reference_index])


def threshold_band_index(
    band_edges: Int[np.ndarray, " n_edge"], freq: Int[np.ndarray, " n_candidate"]
) -> Int[np.ndarray, " n_candidate"]:
    """Index of the threshold band (band_edges[i] <= freq < band_edges[i + 1]) of each frequency."""
    return np.searchsorted(band_edges, freq, side="right") - 1


def excess_ratio_thresholds(
    band_of_injection: Int[np.ndarray, " n_injection"],
    ratios: Float[np.ndarray, " n_injection"],
    band_edges: Int[np.ndarray, " n_edge"],
    percentile: float,
) -> Float[np.ndarray, "n_threshold_band 2"]:
    """(percentile, lowest) of the injection ratios in each threshold band."""
    band_index = threshold_band_index(band_edges, band_of_injection)
    return np.array(
        [
            (
                np.percentile(ratios[band_index == i], percentile),
                ratios[band_index == i].min(),
            )
            for i in range(len(band_edges) - 1)
        ]
    )


def h1_l1_windows(
    band_of_injection: Int[np.ndarray, " n_injection"],
    log_ratio: Float[np.ndarray, " n_injection"],
    band_edges: Int[np.ndarray, " n_edge"],
    percentile: float,
) -> Float[np.ndarray, "n_threshold_band 2"]:
    """(low, high) log10 H1/L1 excess ratio holding the central (100 - percentile)% of the injections in each threshold
    band."""
    band_index = threshold_band_index(band_edges, band_of_injection)
    return np.array(
        [
            np.percentile(
                log_ratio[band_index == i], [percentile / 2, 100 - percentile / 2]
            )
            for i in range(len(band_edges) - 1)
        ]
    )
