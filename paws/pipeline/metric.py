"""Segment list, data check, coverage plot and (optionally) Weave metric for a coherence time.

Writes to <home_dir>/metricSetup/ (or out_dir)
  o4ab_h1_timestamps.txt, o4ab_l1_timestamps.txt   SFT timestamps
  o4_<T>days_segments.txt                           segments of T days containing H1 or L1 data
  o4_<T>days_segments.png                           detector coverage and segments
  o4ab_t<T>_s<N>.fts                                with make_metric_file (lalpulsar_WeaveSetup)
"""

import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paws.filepaths import PathManager

START_TIME = 1368970000
SFT_DURATION = 1800
DETECTORS = ("H1", "L1")


def read_timestamps(paths, out_dir, sft_band, refresh):
    """SFT timestamps of each detector, read from one SFT file each (cached as text)."""
    timestamps = {}
    for detector in DETECTORS:
        cache_path = out_dir / f"o4ab_{detector.lower()}_timestamps.txt"
        if refresh or not cache_path.exists():
            from pyfstat.utils import get_sft_as_arrays

            sft_path = sorted(paths.sft_file_path(sft_band, detector).glob("*.sft"))[0]
            np.savetxt(
                cache_path, get_sft_as_arrays(str(sft_path))[1][detector], fmt="%d"
            )
        timestamps[detector] = np.loadtxt(cache_path, ndmin=1)
    return timestamps


def make_segments(timestamps, tcoh_day):
    """Consecutive segments of tcoh_day days from START_TIME; empty segments are skipped by
    restarting at the next SFT."""
    tcoh = tcoh_day * 86400
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


def report_coverage(timestamps, segments):
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


def sft_runs(times, origin, time_unit):
    """Contiguous runs of SFTs as (start, width) in time_unit since origin, for broken_barh."""
    times = np.sort(times)
    breaks = np.flatnonzero(np.diff(times) != SFT_DURATION)
    starts = np.r_[times[0], times[breaks + 1]]
    ends = np.r_[times[breaks], times[-1]] + SFT_DURATION
    return list(zip((starts - origin) / time_unit, (ends - starts) / time_unit))


def plot_coverage(timestamps, segments, tcoh_day, path):
    week = 7 * 86400
    origin = segments[0, 0]
    plt.style.use("paws")
    fig, ax = plt.subplots(figsize=(16, 5))
    segment_bars = [
        ((start - origin) / week, (end - start) / week) for start, end in segments
    ]
    ax.broken_barh(
        segment_bars, (0, 2), facecolors="#dcdbd6", edgecolor="#52514e", linewidth=0.6
    )
    for row, (detector, color) in enumerate((("H1", "#2a78d6"), ("L1", "#eb6834"))):
        ax.broken_barh(
            sft_runs(timestamps[detector], origin, week),
            (row + 0.1, 0.8),
            facecolors=color,
        )
    ax.set_yticks([0.5, 1.5], ["H1", "L1"])
    ax.set_ylim(0, 2)
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
    config, segment_file, tcoh_day, spindowns, metric_type, out_dir, force
):
    metric_path = out_dir / f"o4ab_t{tcoh_day:g}_s{spindowns}.fts"
    if metric_path.exists() and not force:
        raise SystemExit(f"{metric_path} exists (use --force to overwrite)")
    executable = config.executables.weave_setup or shutil.which("lalpulsar_WeaveSetup")
    command = [
        executable,
        f"--output-file={metric_path}",
        "--detectors=H1,L1",
        f"--segment-list={segment_file}",
        f"--spindowns={spindowns}",
        f"--ref-time={config.ref_time}",
        f"--metric-type={metric_type}",
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"metric: {metric_path}")


def make_segments_and_metric(
    settings,
    tcoh_day,
    make_metric_file,
    spindowns,
    metric_type,
    out_dir,
    sft_band,
    refresh_timestamps,
    force,
):
    """Segment list and coverage plot for tcoh_day; with make_metric_file also the Weave metric.
    out_dir None: <home_dir>/metricSetup."""
    out_dir = out_dir or settings.config.home_dir / "metricSetup"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = PathManager(settings.config, settings.target)
    timestamps = read_timestamps(paths, out_dir, sft_band, refresh_timestamps)
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
            settings.config,
            segment_file,
            tcoh_day,
            spindowns,
            metric_type,
            out_dir,
            force,
        )
