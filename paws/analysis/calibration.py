"""Thresholds of the follow-up cuts calibrated on injections, per threshold band (band_edges).

excess ratio cut: a low percentile of the injections' (2F_now - 4) / (2F_reference - 4).
H1/L1 window:     the central range of the injections' log10 H1/L1 excess ratio.
"""

import numpy as np

from paws.analysis.statistics import excess_ratio


def injection_excess_ratios(
    reference_rows, reference_injected_freqs, now_rows, now_injected_freqs
):
    """(2F_now - 4) / (2F_reference - 4) of every injection of one band, matched between the stages by injected Freq."""
    reference_row_of_freq = {
        round(f, 9): i for i, f in enumerate(reference_injected_freqs)
    }
    reference_index = np.array(
        [reference_row_of_freq[round(f, 9)] for f in now_injected_freqs], dtype=int
    )
    return excess_ratio(now_rows["mean2F"], reference_rows["mean2F"][reference_index])


def threshold_band_index(band_edges, freq):
    """Index of the threshold band (band_edges[i] <= freq < band_edges[i + 1]) of each frequency."""
    return np.searchsorted(band_edges, np.asarray(freq), side="right") - 1


def excess_ratio_thresholds(band_of_injection, ratios, band_edges, percentile):
    """(percentile, lowest) of the injection ratios in each threshold band, shape (n_bands, 2)."""
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


def h1_l1_windows(band_of_injection, log_ratio, band_edges, percentile):
    """(low, high) log10 H1/L1 excess ratio holding the central (100 - percentile)% of the injections in each threshold
    band, shape (n_bands, 2)."""
    band_index = threshold_band_index(band_edges, band_of_injection)
    return np.array(
        [
            np.percentile(
                log_ratio[band_index == i], [percentile / 2, 100 - percentile / 2]
            )
            for i in range(len(band_edges) - 1)
        ]
    )
