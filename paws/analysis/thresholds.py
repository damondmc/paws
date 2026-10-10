"""Thresholds of the search and of the follow-up stages.

search:    mean2F threshold of a search band from its number of semicoherent templates.
follow-up: from injections, per threshold band (band_edges): the (2F-4) excess ratio threshold between stages (a low
           percentile of the injections' (2F_now - 4) / (2F_prev - 4)) and the H1/L1 excess-ratio window (the central range
           of their log10 r_HL).
"""

import numpy as np
from scipy.stats.distributions import chi2

# ------------------------------ search -------------------------------


def search_mean2f_threshold(n_templates, n_segments):
    """mean2F threshold of a search band: the mean2F that one template exceeds with probability 1 / n_templates
    in Gaussian noise (2F summed over n_segments segments is chi2 with 4 * n_segments degrees of freedom)."""
    return chi2.ppf(1.0 - 1.0 / n_templates, 4 * n_segments) / n_segments


# --------------------- follow-up, from injections ---------------------


def injection_excess_ratios(prev_rows, prev_injected_freqs, now_rows, now_injected_freqs):
    """(2F_now - 4) / (2F_prev - 4) of every injection of one band, matched between the stages by injected Freq."""
    prev_row_of_freq = {round(f, 9): i for i, f in enumerate(prev_injected_freqs)}
    prev_index = np.array([prev_row_of_freq[round(f, 9)] for f in now_injected_freqs], dtype=int)
    return (now_rows["mean2F"] - 4) / (prev_rows["mean2F"][prev_index] - 4)


def log10_h1_l1_excess_ratio(mean2f_h1, mean2f_l1):
    """log10 r_HL, r_HL = (2F_H1-4)/(2F_L1-4); +-5 when a detector is at or below 4."""
    excess_h1 = np.asarray(mean2f_h1, float) - 4
    excess_l1 = np.asarray(mean2f_l1, float) - 4
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(excess_h1 / excess_l1)
    both_positive = (excess_h1 > 0) & (excess_l1 > 0)
    return np.where(both_positive, log_ratio, np.where(excess_h1 > excess_l1, 5.0, -5.0))


def threshold_band_index(band_edges, freq):
    """Index of the threshold band (band_edges[i] <= freq < band_edges[i + 1]) of each frequency."""
    return np.searchsorted(band_edges, np.asarray(freq), side="right") - 1


def excess_ratio_thresholds(band_of_injection, ratios, band_edges, percentile):
    """(percentile, lowest) of the injection ratios in each threshold band, shape (n_bands, 2)."""
    band_index = threshold_band_index(band_edges, band_of_injection)
    return np.array([
        (np.percentile(ratios[band_index == i], percentile), ratios[band_index == i].min())
        for i in range(len(band_edges) - 1)
    ])


def h1_l1_windows(band_of_injection, log_ratio, band_edges, percentile):
    """(low, high) log10 r_HL holding the central (100 - percentile)% of the injections in each threshold band,
    shape (n_bands, 2)."""
    band_index = threshold_band_index(band_edges, band_of_injection)
    return np.array([
        np.percentile(log_ratio[band_index == i], [percentile / 2, 100 - percentile / 2])
        for i in range(len(band_edges) - 1)
    ])


# ---------------------- follow-up, applying -----------------------


def excess_ratio_to_mean2f_threshold(excess_ratio, prev_mean2f):
    """mean2F threshold of each seed: (2F_prev - 4) * excess_ratio + 4."""
    return (np.asarray(prev_mean2f, float) - 4) * excess_ratio + 4


def inside_h1_l1_window(window, mean2f_h1, mean2f_l1):
    """True where log10 r_HL lies inside window (low, high) (one window per candidate, or one for all)."""
    window = np.asarray(window, float)
    log_ratio = log10_h1_l1_excess_ratio(mean2f_h1, mean2f_l1)
    return (log_ratio >= window[..., 0]) & (log_ratio <= window[..., 1])


# ----------------------- old threshold files ------------------------


def read_threshold_row(path, freq):
    """Values (columns 2..) of the row of each frequency in an old threshold text file (f_start f_end values,
    equal-width rows), such as config/*_v3.txt."""
    table = np.loadtxt(path, ndmin=2)
    first_row_start, row_width = table[0, 0], table[0, 1] - table[0, 0]
    row_index = ((np.asarray(freq) - first_row_start) // row_width).astype(int)
    return table[row_index, 2:]
