"""Detection statistics of candidates and the cuts on them.

search:    mean2F threshold of a search band from its number of semicoherent templates.
follow-up: the (2F-4) excess ratio of a candidate over a reference candidate (the seed it was followed up from, or
           the search outlier it descends from) and the H1/L1 excess ratio r_HL = (2F_H1-4)/(2F_L1-4), optionally
           noise-normalized to R_HL = r_HL * W_L1 / W_H1 (see paws.analysis.noise_weights).
"""

import numpy as np
from scipy.stats.distributions import chi2


def search_mean2f_threshold(n_templates, n_segments):
    """mean2F threshold of a search band: the mean2F that one template exceeds with probability 1 / n_templates
    in Gaussian noise (2F summed over n_segments segments is chi2 with 4 * n_segments degrees of freedom)."""
    return chi2.ppf(1.0 - 1.0 / n_templates, 4 * n_segments) / n_segments


def excess_ratio(mean2f, reference_mean2f):
    """(2F - 4) / (2F_reference - 4)."""
    return (np.asarray(mean2f, float) - 4) / (np.asarray(reference_mean2f, float) - 4)


def excess_ratio_to_mean2f_threshold(excess_ratio, reference_mean2f):
    """mean2F threshold of each candidate: (2F_reference - 4) * excess_ratio + 4 (2F_reference: the seed's mean2F for
    the excess ratio cut, the search outlier's for the growth cut)."""
    return (np.asarray(reference_mean2f, float) - 4) * excess_ratio + 4


def log10_h1_l1_excess_ratio(mean2f_h1, mean2f_l1, log10_noise_ratio):
    """log10 of the H1/L1 excess ratio r_HL = (2F_H1-4)/(2F_L1-4) times W_L1/W_H1, log10_noise_ratio = log10(W_L1/W_H1)
    (0: r_HL itself); +-5 when a detector is at or below 4."""
    excess_h1 = np.asarray(mean2f_h1, float) - 4
    excess_l1 = np.asarray(mean2f_l1, float) - 4
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(excess_h1 / excess_l1) + log10_noise_ratio
    both_positive = (excess_h1 > 0) & (excess_l1 > 0)
    return np.where(
        both_positive, log_ratio, np.where(excess_h1 > excess_l1, 5.0, -5.0)
    )


def inside_h1_l1_window(window, mean2f_h1, mean2f_l1, log10_noise_ratio):
    """True where the log10 H1/L1 excess ratio (normalized by log10_noise_ratio) lies inside window (low, high)
    (one window per candidate, or one for all)."""
    window = np.asarray(window, float)
    log_ratio = log10_h1_l1_excess_ratio(mean2f_h1, mean2f_l1, log10_noise_ratio)
    return (log_ratio >= window[..., 0]) & (log_ratio <= window[..., 1])
