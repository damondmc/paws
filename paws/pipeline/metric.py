"""`paws metric`: the segment list, data coverage plot and (with --metric) the Weave metric of a coherence time,
written to <home_dir>/metricSetup/."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int
from matplotlib.patches import Patch

from paws.definitions import SECONDS_PER_DAY
from paws.filepaths import PathManager
from paws.settings import FixedSettings, Settings

START_TIME = 1368970000
SFT_DURATION = 1800
DETECTOR_COLORS = ("#2a78d6", "#eb6834", "#3a9d5d", "#8e5bd6")


def read_timestamps(
    paths: PathManager,
    detectors: list[str],
    out_dir: Path,
    sft_band: int,
    refresh: bool,
) -> dict[str, Float[np.ndarray, " n_sft"]]:
    """SFT timestamps [GPS s] of each detector, read from one SFT file each (cached as text)."""
    timestamps = {}
    for detector in detectors:
        cache_path = out_dir / f"o4ab_{detector.lower()}_timestamps.txt"
        if refresh or not cache_path.exists():
            from pyfstat.utils import get_sft_as_arrays

            sft_path = sorted(paths.sft_file_path(sft_band, detector).glob("*.sft"))[0]
            np.savetxt(
                cache_path, get_sft_as_arrays(str(sft_path))[1][detector], fmt="%d"
            )
        timestamps[detector] = np.loadtxt(cache_path, ndmin=1)
    return timestamps


def make_segments(
    timestamps: dict[str, Float[np.ndarray, " n_sft"]], tcoh_day: float
) -> Int[np.ndarray, "n_segment 2"]:
    """Consecutive (start, end) segments of tcoh_day days from START_TIME; empty segments are skipped by
    restarting at the next SFT."""
    tcoh = tcoh_day * SECONDS_PER_DAY
    all_timestamps = np.unique(np.concatenate(list(timestamps.values())))
    end_time = all_timestamps.max() + SFT_DURATION
    segments, segment_start = [], START_TIME
    while segment_start < end_time:
        segment_end = segment_start + tcoh
        if any(
            np.any((times >= segment_start) & (times < segment_end))
            for times in timestamps.values()
        ):
            segments.append((segment_start, segment_end))
            segment_start = segment_end
        else:
            later_sfts = all_timestamps[all_timestamps >= segment_end]
            if later_sfts.size == 0:
                break
            segment_start = later_sfts[0]
    return np.array(segments, dtype=np.int64)


def report_coverage(
    timestamps: dict[str, Float[np.ndarray, " n_sft"]],
    segments: Int[np.ndarray, "n_segment 2"],
) -> None:
    """Number of SFTs of each detector inside the segments."""
    for detector, times in timestamps.items():
        inside = np.zeros(times.size, bool)
        for start, end in segments:
            inside |= (times >= start) & (times < end)
        print(
            f"{detector}: {inside.sum():,d} of {times.size:,d} SFTs inside the segments"
        )
        n_segments_with_data = sum(
            np.any((times >= start) & (times < end)) for start, end in segments
        )
        print(f"{detector}: data in {n_segments_with_data} of {len(segments)} segments")


def sft_runs(
    times: Float[np.ndarray, " n_sft"], origin: int, time_unit: int
) -> list[tuple[float, float]]:
    """Contiguous runs of SFTs as (start, width) in time_unit since origin, for broken_barh."""
    times = np.sort(times)
    breaks = np.flatnonzero(np.diff(times) != SFT_DURATION)
    starts = np.r_[times[0], times[breaks + 1]]
    ends = np.r_[times[breaks], times[-1]] + SFT_DURATION
    return list(zip((starts - origin) / time_unit, (ends - starts) / time_unit))


def plot_coverage(
    timestamps: dict[str, Float[np.ndarray, " n_sft"]],
    segments: Int[np.ndarray, "n_segment 2"],
    tcoh_day: float,
    path: Path,
) -> None:
    week = 7 * SECONDS_PER_DAY
    origin = int(segments[0, 0])
    n_detectors = len(timestamps)
    plt.style.use("paws")
    fig, ax = plt.subplots(figsize=(16, 5))
    segment_bars = [
        ((start - origin) / week, (end - start) / week) for start, end in segments
    ]
    ax.broken_barh(
        segment_bars,
        (0, n_detectors),
        facecolors="#dcdbd6",
        edgecolor="#52514e",
        linewidth=0.6,
    )
    for row, (detector, times) in enumerate(timestamps.items()):
        ax.broken_barh(
            sft_runs(times, origin, week),
            (row + 0.1, 0.8),
            facecolors=DETECTOR_COLORS[row % len(DETECTOR_COLORS)],
        )
    ax.set_yticks([row + 0.5 for row in range(n_detectors)], list(timestamps))
    ax.set_ylim(0, n_detectors)
    ax.set_xlim(-0.5, (segments[-1, 1] - origin) / week + 0.5)
    ax.set_xlabel(f"time since GPS {origin} [weeks]")
    ax.legend(
        handles=[
            Patch(
                facecolor="#dcdbd6",
                edgecolor="#52514e",
                label=rf"{len(segments)} segments, $T_\mathrm{{coh}}={tcoh_day:g}$ days",
            )
        ],
        loc="lower right",
        bbox_to_anchor=(1, 1),
        fontsize=16,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot: {path}")


def run_weave_setup(
    settings: FixedSettings,
    segment_file: Path,
    tcoh_day: float,
    spindowns: int,
    metric_type: str,
    out_dir: Path,
    force: bool,
) -> None:
    metric_path = out_dir / f"o4ab_t{tcoh_day:g}_s{spindowns}.fts"
    if metric_path.exists() and not force:
        raise SystemExit(f"{metric_path} exists (use --force to overwrite)")
    executable = settings.environment.executables.weave_setup or shutil.which(
        "lalpulsar_WeaveSetup"
    )
    command = [
        executable,
        f"--output-file={metric_path}",
        f"--detectors={','.join(settings.data.detectors)}",
        f"--segment-list={segment_file}",
        f"--spindowns={spindowns}",
        f"--ref-time={settings.data.ref_time}",
        f"--metric-type={metric_type}",
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"metric: {metric_path}")


def make_segments_and_metric(
    settings: Settings,
    tcoh_day: float,
    make_metric_file: bool,
    spindowns: int,
    metric_type: str,
    out_dir: Optional[Path],
    sft_band: int,
    refresh_timestamps: bool,
    force: bool,
) -> None:
    """Segment list and coverage plot for tcoh_day; with make_metric_file also the Weave metric.
    out_dir None: <home_dir>/metricSetup."""
    out_dir = out_dir or settings.environment.home_dir / "metricSetup"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = PathManager(settings)
    timestamps = read_timestamps(
        paths, settings.data.detectors, out_dir, sft_band, refresh_timestamps
    )
    segments = make_segments(timestamps, tcoh_day)
    segment_file = out_dir / f"o4_{tcoh_day:g}days_segments.txt"
    np.savetxt(segment_file, segments, fmt="%d", delimiter="\t")
    print(f"segments: {segment_file} ({len(segments)} segments of {tcoh_day:g} days)")
    report_coverage(timestamps, segments)
    plot_coverage(
        timestamps, segments, tcoh_day, out_dir / f"o4_{tcoh_day:g}days_segments.png"
    )
    if make_metric_file:
        run_weave_setup(
            settings,
            segment_file,
            tcoh_day,
            spindowns,
            metric_type,
            out_dir,
            force,
        )
