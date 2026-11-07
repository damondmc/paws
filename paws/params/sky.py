"""Sky positions: random samples in a cap and the Weave sky-lattice offsets around a target."""

from pathlib import Path
from typing import Optional, Union

import lal
import lalpulsar
import numpy as np
from jaxtyping import Float

# widths of the freq [Hz], f1dot [Hz/s], ..., f4dot [Hz/s^4] ranges of the sky-lattice tiling
SPIN_WIDTHS = (1e-11, 1e-16, 1e-25, 1e-34, 1e-43)


def sample_sky_positions(
    target_ra: float, target_dec: float, sky_radius: float, n_samples: int
) -> tuple[Float[np.ndarray, " n_sample"], Float[np.ndarray, " n_sample"]]:
    """(alpha, delta) of n_samples sky locations uniform in a circular cap of radius sky_radius (radians) around
    (target_ra, target_dec)."""
    z_min = np.cos(sky_radius)
    z = np.random.uniform(z_min, 1.0, n_samples)
    theta = np.arccos(z)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)

    sin_d, cos_d = np.sin(target_dec), np.cos(target_dec)
    sin_a, cos_a = np.sin(target_ra), np.cos(target_ra)

    k = np.array([cos_d * cos_a, cos_d * sin_a, sin_d])

    if abs(k[2]) < 0.99:
        u = np.cross([0, 0, 1], k)
    else:
        u = np.cross([0, 1, 0], k)
    u /= np.linalg.norm(u)
    v = np.cross(k, u)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)

    vec = (
        x[None, :] * u[:, None]
        + y[None, :] * v[:, None]
        + cos_theta[None, :] * k[:, None]
    )

    delta = np.arcsin(vec[2])
    alpha = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
    return alpha, delta


def compute_lattice_points(
    setup_file: Union[str, Path],
    alpha: tuple[float, float],
    delta: tuple[float, float],
    spins: list[tuple[float, float]],
    semi_mismatch: float,
    coh_mismatch: Optional[float] = None,
) -> Float[np.ndarray, "n_point n_column"]:
    """Semicoherent lattice points, built as lalpulsar_Weave builds them (supersky metric, A_n* lattice).

    Args:
        setup_file: WeaveSetup metric file.
        alpha: (lo, hi) right ascension in rad.
        delta: (lo, hi) declination in rad.
        spins: (lo, hi) ranges of freq (Hz), f1dot (Hz/s), f2dot (Hz/s^2), ...; the first 1 + n of them are used for
            a metric of n spindowns.
        semi_mismatch: maximum semicoherent mismatch.
        coh_mismatch: maximum coherent mismatch (several segments); None for a single segment.

    Returns:
        (alpha, delta, freq, f1dot, ..., fndot) of each lattice point.
    """
    metrics = lalpulsar.FITSReadSuperskyMetrics(
        lalpulsar.FITSFileOpenRead(str(setup_file))
    )
    lalpulsar.ScaleSuperskyMetricsFiducialFreq(metrics, spins[0][1])
    lalpulsar.EqualizeReducedSuperskyMetricsFreqSpacing(
        metrics, semi_mismatch if coh_mismatch is None else coh_mismatch, semi_mismatch
    )
    metric, transf = metrics.semi_rssky_metric, metrics.semi_rssky_transf
    n_spins = metric.size1 - 2
    if len(spins) < n_spins:
        raise ValueError(
            f"{setup_file}: metric of {n_spins - 1} spindowns, {len(spins) - 1} spindown ranges given"
        )

    tiling = lalpulsar.CreateLatticeTiling(metric.size1)
    lalpulsar.SetSuperskyPhysicalSkyBounds(tiling, metric, transf, *alpha, *delta)
    for s, bounds in enumerate(spins[:n_spins]):
        lalpulsar.SetSuperskyPhysicalSpinBound(tiling, transf, s, *bounds)
    lalpulsar.SetTilingLatticeAndMetric(
        tiling, lalpulsar.TILING_LATTICE_ANSTAR, metric, semi_mismatch
    )

    iterator = lalpulsar.CreateLatticeTilingIterator(tiling, metric.size1)
    point, rows = lal.gsl_vector(metric.size1), []
    while lalpulsar.NextLatticeTilingPoint(iterator, point) > 0:
        phys = lalpulsar.PulsarDopplerParams()
        lalpulsar.ConvertSuperskyToPhysicalPoint(phys, point, None, transf)
        rows.append((phys.Alpha, phys.Delta, *phys.fkdot[:n_spins]))
    return np.array(rows)


def sky_lattice_offsets(
    setup_file: Union[str, Path],
    alpha: float,
    delta: float,
    sky_radius: float,
    freq: float,
    semi_mismatch: float,
    coh_mismatch: Optional[float] = None,
) -> tuple[Float[np.ndarray, " n_sky"], Float[np.ndarray, " n_sky"]]:
    """Offsets from (alpha, delta) of the Weave sky-lattice points, padding included, that cover the box of
    half-widths sky_radius / cos(delta) in right ascension and sky_radius in declination around it.

    Args:
        setup_file: WeaveSetup metric file.
        alpha: right ascension of the box center in rad.
        delta: declination of the box center in rad.
        sky_radius: angular half-width of the box in rad.
        freq: frequency of the lattice in Hz.
        semi_mismatch: maximum semicoherent mismatch.
        coh_mismatch: maximum coherent mismatch (several segments); None for a single segment.

    Returns:
        (d_alpha, d_delta) arrays in rad.
    """
    alpha_half_width = sky_radius / np.cos(delta)
    points = compute_lattice_points(
        setup_file,
        (alpha - alpha_half_width, alpha + alpha_half_width),
        (delta - sky_radius, delta + sky_radius),
        [(freq - SPIN_WIDTHS[0], freq)] + [(0.0, width) for width in SPIN_WIDTHS[1:]],
        semi_mismatch,
        coh_mismatch,
    )
    sky_points = np.unique(points[:, :2], axis=0)
    return sky_points[:, 0] - alpha, sky_points[:, 1] - delta
