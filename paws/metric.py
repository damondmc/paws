"""Segment list, data check, coverage plot and (optionally) Weave metric for a coherence time.

Writes to <home_dir>/metricSetup/ (or --out-dir)
  o4ab_h1_timestamps.txt, o4ab_l1_timestamps.txt   SFT timestamps
  o4_<T>days_segments.txt                           segments of T days containing H1 or L1 data
  o4_<T>days_segments.png                           detector coverage and segments
  o4ab_t<T>_s<N>.fts                                with --metric (lalpulsar_WeaveSetup)

Usage:
  paws metric 40
  paws metric 40 --metric -s 3
"""
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paws.filepaths import PathManager

START_TIME = 1368970000
T_SFT = 1800


def read_timestamps(paths, out_dir, sft_band, refresh):
    """SFT timestamps of H1 and L1, read from one SFT file each (cached as text)."""
    ts = {}
    for det in ("H1", "L1"):
        path = out_dir / f"o4ab_{det.lower()}_timestamps.txt"
        if refresh or not path.exists():
            from pyfstat.utils import get_sft_as_arrays

            sft = sorted(paths.sft_file_path(sft_band, det).glob("*.sft"))[0]
            np.savetxt(path, get_sft_as_arrays(str(sft))[1][det], fmt="%d")
        ts[det] = np.loadtxt(path, ndmin=1)
    return ts


def make_segments(ts, tcoh_day):
    """Consecutive segments of tcoh_day days from START_TIME; empty segments are skipped by
    restarting at the next SFT."""
    tcoh = tcoh_day * 86400
    all_ts = np.unique(np.concatenate(list(ts.values())))
    end_time = all_ts.max() + T_SFT
    segs, t = [], START_TIME
    while t < end_time:
        if any(np.any((x >= t) & (x < t + tcoh)) for x in ts.values()):
            segs.append((t, t + tcoh))
            t += tcoh
        else:
            nxt = all_ts[all_ts >= t + tcoh]
            if nxt.size == 0:
                break
            t = nxt[0]
    return np.array(segs, dtype=np.int64)


def check(ts, segs):
    """Number of SFTs of each detector inside the segments."""
    for det, x in ts.items():
        inside = np.zeros(x.size, bool)
        for s, e in segs:
            inside |= (x >= s) & (x < e)
        print(f"{det}: {inside.sum():,d} of {x.size:,d} SFTs inside the segments")
        n_seg = sum(np.any((x >= s) & (x < e)) for s, e in segs)
        print(f"{det}: data in {n_seg} of {len(segs)} segments")


def bars(x, t0, scale):
    """Contiguous runs of SFTs as (start, width) for broken_barh."""
    x = np.sort(x)
    breaks = np.flatnonzero(np.diff(x) != T_SFT)
    starts = np.r_[x[0], x[breaks + 1]]
    ends = np.r_[x[breaks], x[-1]] + T_SFT
    return list(zip((starts - t0) / scale, (ends - starts) / scale))


def plot(ts, segs, tcoh_day, path):
    week = 7 * 86400
    t0 = segs[0, 0]
    plt.style.use("paws")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.broken_barh([((s - t0) / week, (e - s) / week) for s, e in segs], (0, 2), facecolors="#dcdbd6",
                   edgecolor="#52514e", linewidth=0.6)
    for y, (det, color) in enumerate((("H1", "#2a78d6"), ("L1", "#eb6834"))):
        ax.broken_barh(bars(ts[det], t0, week), (y + 0.1, 0.8), facecolors=color)
    ax.set_yticks([0.5, 1.5], ["H1", "L1"])
    ax.set_ylim(0, 2)
    ax.set_xlim(-0.5, (segs[-1, 1] - t0) / week + 0.5)
    ax.set_xlabel(f"time since GPS {t0} [weeks]")
    ax.legend(handles=[Patch(facecolor="#dcdbd6", edgecolor="#52514e",
                             label=rf"{len(segs)} segments, $T_\mathrm{{coh}}={tcoh_day:g}$ days")],
              loc="lower right", bbox_to_anchor=(1, 1), fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot: {path}")


def make_metric(config, seg_file, tcoh_day, spindowns, metric_type, out_dir, force):
    out = out_dir / f"o4ab_t{tcoh_day:g}_s{spindowns}.fts"
    if out.exists() and not force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    exe = config.executables.weave_setup or shutil.which("lalpulsar_WeaveSetup")
    cmd = [exe, f"--output-file={out}", "--detectors=H1,L1", f"--segment-list={seg_file}",
           f"--spindowns={spindowns}", f"--ref-time={config.ref_time}", f"--metric-type={metric_type}"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"metric: {out}")


def add_arguments(p):
    p.add_argument("tcoh", type=float, help="coherence time [days]")
    p.add_argument("--metric", action="store_true", help="also run lalpulsar_WeaveSetup")
    p.add_argument("-s", "--spindowns", type=int, default=2)
    p.add_argument("--metric-type", default="directed")
    p.add_argument("--out-dir", type=Path, help="default <home_dir>/metricSetup")
    p.add_argument("--sft-band", type=int, default=20, help="SFT band [Hz] the timestamps are read from")
    p.add_argument("--refresh-timestamps", action="store_true", help="re-read the SFT files")
    p.add_argument("--force", action="store_true", help="overwrite an existing metric file")


def run(settings, a):
    a.out_dir = a.out_dir or settings.config.home_dir / "metricSetup"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ts = read_timestamps(PathManager(settings.config, settings.target), a.out_dir, a.sft_band, a.refresh_timestamps)
    segs = make_segments(ts, a.tcoh)
    seg_file = a.out_dir / f"o4_{a.tcoh:g}days_segments.txt"
    np.savetxt(seg_file, segs, fmt="%d", delimiter="\t")
    print(f"segments: {seg_file} ({len(segs)} segments of {a.tcoh:g} days)")
    check(ts, segs)
    plot(ts, segs, a.tcoh, a.out_dir / f"o4_{a.tcoh:g}days_segments.png")
    if a.metric:
        make_metric(settings.config, seg_file, a.tcoh, a.spindowns, a.metric_type, a.out_dir, a.force)
