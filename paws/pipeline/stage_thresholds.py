"""Thresholds of a follow-up stage, computed from its injection stages as set in stages.yaml."""

from dataclasses import dataclass
from functools import cache
from typing import Optional

import numpy as np
import yaml

from paws.analysis import thresholds
from paws.filepaths import PathManager
from paws.fits_io import read_injection_outliers
from paws.pipeline.noise_weights import log10_noise_ratio


@dataclass(frozen=True)
class StageThresholds:
    """Thresholds of one follow-up stage, one value per threshold band (rounded to 4 decimals, as recorded)."""

    band_edges: np.ndarray
    excess_ratio: np.ndarray  # (n_bands,) excess ratio threshold
    # (n_bands,) lowest injection excess ratio, for the record
    excess_ratio_lowest: np.ndarray
    # (n_bands, 2) H1/L1 excess-ratio window of the stage's candidates, in log10 r_HL or (with h1_l1_noise_weights)
    # log10 R_HL; None: no window
    h1_l1_window: Optional[np.ndarray]
    # (n_bands, 2) window of the seeds when they come from a search stage
    seed_h1_l1_window: Optional[np.ndarray]
    # (n_bands,) growth threshold on (2F - 4) / (2F_search - 4); None: no growth cut
    growth: Optional[np.ndarray]

    def at(self, name, freq):
        """Threshold `name` (a field above) of the threshold band of each frequency."""
        return getattr(self, name)[
            thresholds.threshold_band_index(self.band_edges, freq)
        ]


def round_as_recorded(values):
    return np.vectorize(lambda value: float(f"{value:.4f}"))(values)


def injection_outliers_by_band(settings, stage_name, freqs):
    """{band: (outlier rows, injected Freq)} of the bands of an injection stage that have outliers."""
    paths = PathManager(settings.config, settings.target)
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
    prev_outliers, now_outliers, band_edges, percentile
):
    """(percentile, lowest) of the (2F-4) ratios of the injections, matched between two injection stages."""
    band_of_injection, ratios = [], []
    for band, (now_rows, now_injected_freqs) in now_outliers.items():
        prev_rows, prev_injected_freqs = prev_outliers[band]
        band_ratios = thresholds.injection_excess_ratios(
            prev_rows, prev_injected_freqs, now_rows, now_injected_freqs
        )
        band_of_injection.append(np.full(len(band_ratios), band))
        ratios.append(band_ratios)
    return thresholds.excess_ratio_thresholds(
        np.concatenate(band_of_injection),
        np.concatenate(ratios),
        band_edges,
        percentile,
    )


def h1_l1_windows_from_injections(
    settings, stage, outliers_by_band, band_edges, percentile
):
    """H1/L1 excess-ratio window of `stage` holding the central (100 - percentile)% of the injections of one
    injection stage."""
    band_of_injection = np.concatenate(
        [np.full(len(rows), band) for band, (rows, _) in outliers_by_band.items()]
    )
    mean2f_h1 = np.concatenate(
        [rows["mean2F_H1"] for rows, _ in outliers_by_band.values()]
    )
    mean2f_l1 = np.concatenate(
        [rows["mean2F_L1"] for rows, _ in outliers_by_band.values()]
    )
    log_ratio = thresholds.log10_h1_l1_excess_ratio(
        mean2f_h1, mean2f_l1, log10_noise_ratio(settings, stage, band_of_injection)
    )
    return thresholds.h1_l1_windows(
        band_of_injection, log_ratio, band_edges, percentile
    )


@cache
def _stage_thresholds(settings, stage_name):
    stage = settings.stage(stage_name)
    if stage.thresholds is None:
        raise ValueError(f"{stage_name} has no thresholds in stages.yaml")
    setting = stage.thresholds
    band_edges = np.array(setting.bands)
    freqs = range(band_edges[0], band_edges[-1])
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
                settings, stage, now_outliers, band_edges, setting.h1_l1_percentile
            )
        )
        if settings.prev(stage).kind == "search":
            seed_h1_l1_window = round_as_recorded(
                h1_l1_windows_from_injections(
                    settings,
                    stage,
                    prev_outliers,
                    band_edges,
                    setting.h1_l1_percentile,
                )
            )
    growth = None
    if setting.growth_percentile is not None:
        search = settings.chain_stages(stage)[0]
        if settings.stage(setting.growth_search_injection).tcoh != search.tcoh:
            raise ValueError(
                f"{stage_name}: growth_search_injection {setting.growth_search_injection} is not at the coherence "
                f"time of {search.name} ({search.tcoh} days)"
            )
        search_outliers = injection_outliers_by_band(
            settings, setting.growth_search_injection, freqs
        )
        growth = round_as_recorded(
            excess_ratio_thresholds_from_injections(
                search_outliers, now_outliers, band_edges, setting.growth_percentile
            )
        )[:, 0]
    return StageThresholds(
        band_edges,
        excess_ratio[:, 0],
        excess_ratio[:, 1],
        h1_l1_window,
        seed_h1_l1_window,
        growth,
    )


def stage_thresholds(settings, stage):
    """Thresholds of a follow-up stage (computed once per process)."""
    return _stage_thresholds(settings, stage.name)


def threshold_record(stage, stage_threshold):
    """The stage's threshold settings and per-band values as plain Python types (for the YAML record)."""
    setting = stage.thresholds
    bands = []
    for i, (f_min, f_max) in enumerate(
        zip(stage_threshold.band_edges[:-1], stage_threshold.band_edges[1:])
    ):
        band = {
            "f_min": int(f_min),
            "f_max": int(f_max),
            "excess_ratio_threshold": float(stage_threshold.excess_ratio[i]),
            "lowest_excess_ratio": float(stage_threshold.excess_ratio_lowest[i]),
        }
        if stage_threshold.h1_l1_window is not None:
            band["h1_l1_window"] = [
                float(value) for value in stage_threshold.h1_l1_window[i]
            ]
        if stage_threshold.seed_h1_l1_window is not None:
            band["seed_h1_l1_window"] = [
                float(value) for value in stage_threshold.seed_h1_l1_window[i]
            ]
        if stage_threshold.growth is not None:
            band["growth_threshold"] = float(stage_threshold.growth[i])
        bands.append(band)
    return {
        "stage": stage.name,
        "injections": list(setting.injections),
        "excess_ratio_percentile": setting.excess_ratio_percentile,
        "h1_l1_percentile": setting.h1_l1_percentile,
        "h1_l1_noise_weights": setting.h1_l1_noise_weights,
        "growth_percentile": setting.growth_percentile,
        "growth_search_injection": setting.growth_search_injection,
        "bands": bands,
    }


def threshold_record_path(settings, stage_name):
    return settings.config.home_dir / "results" / stage_name / "thresholds.yaml"


def write_threshold_records(settings, stage):
    """The stage's thresholds, as applied (rounded to 4 decimals), in results/<stage>/thresholds.yaml."""
    record = threshold_record(stage, stage_thresholds(settings, stage))
    path = threshold_record_path(settings, stage.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as record_file:
        record_file.write(
            f"# thresholds applied by {stage.name}, computed from injections {record['injections']}\n"
        )
        yaml.safe_dump(record, record_file, sort_keys=False, default_flow_style=None)
    print("wrote", path)


def read_threshold_record(settings, stage_name):
    """StageThresholds of a follow-up stage as recorded in results/<stage>/thresholds.yaml (by `paws thresholds`)."""
    with open(threshold_record_path(settings, stage_name)) as record_file:
        bands = yaml.safe_load(record_file)["bands"]

    def windows(key):
        return np.array([band[key] for band in bands]) if key in bands[0] else None

    return StageThresholds(
        band_edges=np.array([band["f_min"] for band in bands] + [bands[-1]["f_max"]]),
        excess_ratio=np.array([band["excess_ratio_threshold"] for band in bands]),
        excess_ratio_lowest=np.array([band["lowest_excess_ratio"] for band in bands]),
        h1_l1_window=windows("h1_l1_window"),
        seed_h1_l1_window=windows("seed_h1_l1_window"),
        growth=(
            np.array([band["growth_threshold"] for band in bands])
            if "growth_threshold" in bands[0]
            else None
        ),
    )
