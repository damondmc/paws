"""Thresholds of a follow-up stage: thresholds/<name>.yaml of the config directory (Stage.thresholds) as arrays per
threshold band (StageThresholds), and their calibration from injection stages (`paws thresholds <name>`)."""

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
from jaxtyping import Float, Int, Shaped

from paws.analysis import calibration, statistics
from paws.filepaths import PathManager
from paws.fits_io import read_injection_outliers
from paws.pipeline.noise_weights import log10_noise_ratio
from paws.settings import Settings, Stage, ThresholdBand, ThresholdFile

# {1 Hz band: (outlier rows, injected Freq)} of an injection stage
OutliersByBand = dict[
    int, tuple[Shaped[np.ndarray, " n_injection"], Float[np.ndarray, " n_injection"]]
]


@dataclass(frozen=True)
class StageThresholds:
    """Thresholds of one follow-up stage, one value per threshold band."""

    band_edges: Int[np.ndarray, " n_edge"]  # threshold band edges [Hz]
    excess_ratio: Float[np.ndarray, " n_threshold_band"]
    # lowest injection excess ratio, for the record; None in a hand-written file
    excess_ratio_lowest: Optional[Float[np.ndarray, " n_threshold_band"]]
    # (low, high) H1/L1 excess-ratio window of the stage's candidates, in log10 r_HL or (with h1_l1_noise_weights)
    # log10 R_HL; None: no window
    h1_l1_window: Optional[Float[np.ndarray, "n_threshold_band 2"]]
    # (low, high) window of the seeds, applied when they come from the initial search
    seed_h1_l1_window: Optional[Float[np.ndarray, "n_threshold_band 2"]]
    # growth threshold on (2F - 4) / (2F_search - 4); None: no growth cut
    growth: Optional[Float[np.ndarray, " n_threshold_band"]]
    # noise-weights file of the windows; None: windows on r_HL
    h1_l1_noise_weights: Optional[str]

    @classmethod
    def from_file(cls, file: ThresholdFile) -> "StageThresholds":
        """The arrays of a ThresholdFile."""
        bands = file.bands

        def column(name: str) -> Optional[Float[np.ndarray, "n_threshold_band ..."]]:
            if getattr(bands[0], name) is None:
                return None
            return np.array([getattr(band, name) for band in bands], dtype=float)

        return cls(
            band_edges=np.array([band.f_min for band in bands] + [bands[-1].f_max]),
            excess_ratio=column("excess_ratio"),
            excess_ratio_lowest=column("excess_ratio_lowest"),
            h1_l1_window=column("h1_l1_window"),
            seed_h1_l1_window=column("seed_h1_l1_window"),
            growth=column("growth"),
            h1_l1_noise_weights=file.h1_l1_noise_weights,
        )

    def at(
        self, name: str, freq: Union[int, Int[np.ndarray, " n_candidate"]]
    ) -> Union[float, Float[np.ndarray, "..."]]:
        """Threshold `name` (a field above) of the threshold band of a frequency: one value (or one window) for an
        int freq, one per frequency for an array."""
        values = getattr(self, name)[
            calibration.threshold_band_index(self.band_edges, np.atleast_1d(freq))
        ]
        return values if np.ndim(freq) else values[0]


# ------------------------------------------------------------- calibration


def round_as_recorded(values: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
    """The values rounded to 4 decimals, as written to a threshold file."""
    return np.vectorize(lambda value: float(f"{value:.4f}"))(values)


def injection_outliers_by_band(
    settings: Settings, stage_name: str, freqs: Iterable[int]
) -> OutliersByBand:
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
            location="existing",
        )
        if path.is_file():
            band_outliers = read_injection_outliers(path)
            if band_outliers is not None:
                outliers_by_band[freq] = band_outliers
    return outliers_by_band


def excess_ratio_thresholds_from_injections(
    prev_outliers: OutliersByBand,
    now_outliers: OutliersByBand,
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
    outliers_by_band: OutliersByBand,
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
    """The ThresholdFile of calibration `name` (followup.yaml), computed from its injection stages' outliers; the
    values are rounded to 4 decimals, as written."""
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


# ------------------------------------------------------------- thresholds of a stage


@cache
def _stage_thresholds(settings: Settings, stage_name: str) -> StageThresholds:
    stage = settings.stage(stage_name)
    if stage.thresholds is None:
        raise ValueError(f"{stage_name} has no thresholds in followup.yaml")
    path = settings.threshold_file_path(stage.thresholds)
    if not path.is_file():
        raise FileNotFoundError(
            f"{stage_name}: no threshold file {path} (`paws thresholds {stage.thresholds}` writes it)"
        )
    file = ThresholdFile.from_yaml(path)
    if file.tcoh != stage.tcoh:
        raise ValueError(
            f"{stage_name} ({stage.tcoh} days): {path.name} is for {file.tcoh}-day candidates"
        )
    prev = settings.prev(stage)
    if prev.is_search and file.bands[0].seed_h1_l1_window is not None:
        if file.seed_tcoh != prev.tcoh:
            raise ValueError(
                f"{stage_name}: the seeds of {prev.name} have {prev.tcoh} days, the seed window of {path.name} "
                f"{file.seed_tcoh}"
            )
    if file.growth_tcoh is not None:
        search = settings.chain_stages(stage)[0]
        if file.growth_tcoh != search.tcoh:
            raise ValueError(
                f"{stage_name}: its chain starts from {search.name} ({search.tcoh} days), the growth cut of "
                f"{path.name} refers to {file.growth_tcoh} days"
            )
    return StageThresholds.from_file(file)


def stage_thresholds(settings: Settings, stage: Stage) -> StageThresholds:
    """Thresholds of a follow-up stage, read once per process from its threshold file and checked against the
    stage's chain."""
    return _stage_thresholds(settings, stage.name)


def threshold_record_path(settings: Settings, stage_name: str) -> Path:
    return settings.environment.home_dir / "results" / stage_name / "thresholds.yaml"


def record_threshold_file(settings: Settings, stage: Stage) -> None:
    """Copies the stage's threshold file to results/<stage>/thresholds.yaml, as applied."""
    path = threshold_record_path(settings, stage.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(settings.threshold_file_path(stage.thresholds), path)
