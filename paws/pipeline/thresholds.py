"""Thresholds of a follow-up stage: thresholds/<name>.yaml of the config directory (Stage.thresholds) as arrays per
threshold band, checked against the stage's chain, and the copy recorded under results/<stage>."""

import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
from jaxtyping import Float, Int

from paws.analysis import calibration
from paws.settings import Settings, Stage, ThresholdFile


@dataclass(frozen=True)
class StageThresholds:
    """Thresholds of one follow-up stage, one value per threshold band."""

    band_edges: Int[np.ndarray, " n_edge"]  # threshold band edges [Hz]
    excess_ratio: Float[np.ndarray, " n_threshold_band"]
    # lowest injection excess ratio, for the record; None in a hand-written file
    excess_ratio_lowest: Optional[Float[np.ndarray, " n_threshold_band"]]
    # (low, high) H1/L1 excess-ratio window of the stage's candidates, in log10 r_HL
    # or (with h1_l1_noise_weights) log10 R_HL; None: no window
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

    def in_band(
        self, name: str, freq: Union[int, Int[np.ndarray, " n_candidate"]]
    ) -> Union[float, Float[np.ndarray, "..."]]:
        """Threshold `name` (a field above) of the threshold band of a frequency: one value (or one window) for an
        int freq, one per frequency for an array."""
        values = getattr(self, name)[
            calibration.threshold_band_index(self.band_edges, np.atleast_1d(freq))
        ]
        return values if np.ndim(freq) else values[0]


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
