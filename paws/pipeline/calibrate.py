"""`paws thresholds <name>`: the thresholds of a calibration in followup.yaml computed from its injection stages and
written to thresholds/<name>.yaml."""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import yaml
from jaxtyping import Float, Int, Shaped

from paws.analysis import calibration, statistics
from paws.filepaths import PathManager
from paws.fits_io import read_injection_outliers
from paws.pipeline.noise_weights import log10_noise_ratio
from paws.settings import Settings, ThresholdBand, ThresholdFile

# {1 Hz band: (outlier rows, injected Freq)} of an injection stage
InjectionOutliersByBand = dict[
    int, tuple[Shaped[np.ndarray, " n_injection"], Float[np.ndarray, " n_injection"]]
]


def round_as_recorded(
    values: Float[np.ndarray, "..."], decimals: int = 4
) -> Float[np.ndarray, "..."]:
    """The values rounded to {decimals} decimals, as written to a threshold file."""
    return np.vectorize(lambda value: float(f"{value:.{decimals}f}"))(values)


def injection_outliers_by_band(
    settings: Settings, stage_name: str, freqs: Iterable[int]
) -> InjectionOutliersByBand:
    """{band: (outlier rows, injected Freq)} of the bands of an injection stage that have outliers."""
    paths = PathManager(settings)
    stage = settings.stage(stage_name)
    outliers_by_band = {}
    for freq in freqs:
        path = paths.outlier_file(
            freq,
            stage.taskname(settings.target, freq),
            stage.name,
            cluster=True,
        )
        if path.is_file():
            band_outliers = read_injection_outliers(path)
            if band_outliers is not None:
                outliers_by_band[freq] = band_outliers
    return outliers_by_band


def excess_ratio_thresholds_from_injections(
    prev_outliers: InjectionOutliersByBand,
    now_outliers: InjectionOutliersByBand,
    band_edges: Int[np.ndarray, " n_edge"],
    percentile: float,
) -> Float[np.ndarray, "n_threshold_band 2"]:
    """(percentile, lowest) of the (2F-4) ratios of the injections, matched between two injection stages."""
    band_of_injection, ratios = [], []
    for band, (now_rows, now_injected_freqs) in now_outliers.items():
        prev_rows, prev_injected_freqs = prev_outliers[band]
        band_ratios = calibration.injection_excess_ratios(
            prev_rows, prev_injected_freqs, now_rows, now_injected_freqs
        )
        band_of_injection.append(np.full(len(band_ratios), band))
        ratios.append(band_ratios)
    return calibration.excess_ratio_thresholds(
        np.concatenate(band_of_injection),
        np.concatenate(ratios),
        band_edges,
        percentile,
    )


def h1_l1_windows_from_injections(
    settings: Settings,
    noise_weights_file: Optional[str],
    outliers_by_band: InjectionOutliersByBand,
    band_edges: Int[np.ndarray, " n_edge"],
    percentile: float,
) -> Float[np.ndarray, "n_threshold_band 2"]:
    """H1/L1 excess-ratio window holding the central (100 - percentile)% of the injections of one injection
    stage (normalized with the noise-weights file, when given)."""
    band_of_injection = np.concatenate(
        [np.full(len(rows), band) for band, (rows, _) in outliers_by_band.items()]
    )
    mean2f_h1 = np.concatenate(
        [rows["mean2F_H1"] for rows, _ in outliers_by_band.values()]
    )
    mean2f_l1 = np.concatenate(
        [rows["mean2F_L1"] for rows, _ in outliers_by_band.values()]
    )
    log_ratio = statistics.log10_h1_l1_excess_ratio(
        mean2f_h1,
        mean2f_l1,
        log10_noise_ratio(settings, noise_weights_file, band_of_injection),
    )
    return calibration.h1_l1_windows(
        band_of_injection, log_ratio, band_edges, percentile
    )


def calibrate(settings: Settings, name: str) -> ThresholdFile:
    """The ThresholdFile of calibration `name` (followup.yaml), computed from its injection stages' outliers."""
    try:
        setting = settings.followup.calibrations[name]
    except KeyError:
        raise KeyError(f"calibration {name!r} is not in followup.yaml") from None
    band_edges = np.array(setting.bands)
    freqs = range(int(band_edges[0]), int(band_edges[-1]))
    prev_injection, now_injection = setting.injections
    prev_outliers = injection_outliers_by_band(settings, prev_injection, freqs)
    now_outliers = injection_outliers_by_band(settings, now_injection, freqs)

    excess_ratio = round_as_recorded(
        excess_ratio_thresholds_from_injections(
            prev_outliers, now_outliers, band_edges, setting.excess_ratio_percentile
        )
    )
    h1_l1_window = seed_h1_l1_window = None
    if setting.h1_l1_percentile is not None:
        h1_l1_window = round_as_recorded(
            h1_l1_windows_from_injections(
                settings,
                setting.h1_l1_noise_weights,
                now_outliers,
                band_edges,
                setting.h1_l1_percentile,
            )
        )
        seed_h1_l1_window = round_as_recorded(
            h1_l1_windows_from_injections(
                settings,
                setting.h1_l1_noise_weights,
                prev_outliers,
                band_edges,
                setting.h1_l1_percentile,
            )
        )
    growth = growth_tcoh = None
    if setting.growth_percentile is not None:
        search_outliers = injection_outliers_by_band(
            settings, setting.growth_search_injection, freqs
        )
        growth = round_as_recorded(
            excess_ratio_thresholds_from_injections(
                search_outliers, now_outliers, band_edges, setting.growth_percentile
            )
        )[:, 0]
        growth_tcoh = settings.stage(setting.growth_search_injection).tcoh

    def window(
        windows: Optional[Float[np.ndarray, "n_threshold_band 2"]], i: int
    ) -> Optional[tuple[float, float]]:
        return None if windows is None else (float(windows[i, 0]), float(windows[i, 1]))

    return ThresholdFile(
        tcoh=settings.stage(now_injection).tcoh,
        seed_tcoh=settings.stage(prev_injection).tcoh,
        growth_tcoh=growth_tcoh,
        h1_l1_noise_weights=setting.h1_l1_noise_weights,
        calibration=setting,
        bands=[
            ThresholdBand(
                f_min=int(f_min),
                f_max=int(f_max),
                excess_ratio=float(excess_ratio[i, 0]),
                excess_ratio_lowest=float(excess_ratio[i, 1]),
                h1_l1_window=window(h1_l1_window, i),
                seed_h1_l1_window=window(seed_h1_l1_window, i),
                growth=None if growth is None else float(growth[i]),
            )
            for i, (f_min, f_max) in enumerate(zip(band_edges[:-1], band_edges[1:]))
        ],
    )


def write_threshold_file(settings: Settings, name: str) -> None:
    """Calibrates `name` and writes thresholds/<name>.yaml in the config directory."""
    file = calibrate(settings, name)
    path = settings.threshold_file_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as threshold_file:
        threshold_file.write(
            f"# thresholds of calibration {name}, computed from injections {list(file.calibration.injections)}\n"
        )
        yaml.safe_dump(
            file.model_dump(mode="json"),
            threshold_file,
            sort_keys=False,
            default_flow_style=None,
        )
    print("wrote", path)
