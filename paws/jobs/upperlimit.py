#!/opt/paws/.venv/bin/python
"""Upper-limit job of one band: injection sets at strains around the h0 estimate, each run through Weave, until the
sigmoid fit of their detection efficiencies gives h95; the outlier file of the confirmation set at h95 gets the fit."""

import argparse
import logging
import sys
import time
from itertools import count
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from jaxtyping import Shaped

from paws.analysis.sigmoid import SigmoidFitter, find_h95
from paws.definitions import WEAVE_EXTRA_STATISTICS, phase_param_name
from paws.jobs.weave_runner import determine_efficiency
from paws.params.injections import InjectionParamGenerator
from paws.settings import FixedSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def log_section(title: str) -> None:
    logging.info(f"\n{'=' * 20} {title.upper()} {'=' * 20}")


def print_args(args: argparse.Namespace) -> None:
    log_section("JOB PARAMETERS")
    for key, value in vars(args).items():
        logging.info(f"{key:<25} : {value}")
    logging.info("=" * 60 + "\n")


def scale_injection(
    injection_rows: Shaped[np.ndarray, " n_injection"], h0: float
) -> Shaped[np.ndarray, " n_injection"]:
    """The injection rows (drawn at h0 = 1) at strain h0."""
    rows = injection_rows.copy()
    rows["aCross"] *= h0
    rows["aPlus"] *= h0
    return rows


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """The job's command line (written by WorkflowManager.make_upperlimit_dag)."""
    parser = argparse.ArgumentParser(
        description="The 95 % upper limit of a band from injections."
    )
    add = parser.add_argument
    add("--environment_file", required=True)
    add("--data_file", required=True)
    add("--search_file", required=True)
    add("--taskname", required=True)
    add("--freq", type=int, required=True)
    add("--stage", required=True)
    add("--freq_deriv_order", type=int, required=True, help="spin-down order searched")
    add(
        "--df_grid",
        type=float,
        nargs="+",
        required=True,
        help="search spacings df df1dot ...",
    )
    add(
        "--inj_freq_deriv_order",
        type=int,
        required=True,
        help="spin-down order injected",
    )
    add("--sft_files", required=True)
    add("--metric_file", required=True, help="Weave setup file")
    add("--n_inj", type=int, required=True, help="injections per set")
    add("--num_toplist", type=int, required=True, help="candidates kept per Weave run")
    add("--h0_est", type=float, required=True, help="first h0 estimate")
    add("--mean2f_th", type=float, required=True, help="mean2F detection threshold")
    add(
        "--non_sat_bands",
        type=float,
        nargs="+",
        required=True,
        help="sub-bands the injections are drawn from",
    )
    add("--sky_radius", type=float, required=True)
    add("--spacing_alpha", type=float, default=None)
    add("--spacing_delta", type=float, default=None)
    add(
        "--n_spacing",
        type=float,
        required=True,
        help="search range around each injection: this many spacings each side",
    )
    add("--cluster", action="store_true")
    add("--work_in_local_dir", action="store_true")
    add("--save_intermediate", action="store_true")
    add("--n_cpus", type=int, required=True)
    return parser.parse_args(argv)


def upper_limit(settings: FixedSettings, args: argparse.Namespace) -> Path:
    """h95 of the band from injection sets around h0_est (see find_h95), then the confirmation set at h95; its
    outlier file gets h95, dh95, the efficiency at h95 and the fit (H0_P, POPT, PCOV tables). Returns that file."""
    freq, taskname, stage, n_inj = args.freq, args.taskname, args.stage, args.n_inj
    metric_file = (
        Path(args.metric_file).name if args.work_in_local_dir else args.metric_file
    )
    with fits.open(metric_file) as hdul:
        ref_time = float(hdul[0].header["DATE-OBS GPS"])
        n_seg = len(hdul["segments"].data)

    _, spacing_names = phase_param_name(args.freq_deriv_order)
    if len(args.df_grid) != len(spacing_names):
        raise ValueError(
            f"df_grid has {len(args.df_grid)} spacings for spin-down order {args.freq_deriv_order}"
        )
    target = settings.target
    generator = InjectionParamGenerator(
        model=settings.spindown_model(freq),
        ref_time=ref_time,
        f0_band=settings.search.f0_band,
    )
    search_data, injection_data = generator.generate_injections_and_search_ranges(
        alpha=target.alpha,
        dalpha=target.dalpha,
        delta=target.delta,
        ddelta=target.ddelta,
        non_sat_bands=np.array(args.non_sat_bands),
        spacing=dict(zip(spacing_names, args.df_grid)),
        h0=1.0,
        freq=freq,
        n_inj=n_inj,
        n_spacing=args.n_spacing,
        inj_freq_deriv_order=args.inj_freq_deriv_order,
        freq_deriv_order=args.freq_deriv_order,
        sky_radius=args.sky_radius,
        spacing_alpha=args.spacing_alpha,
        spacing_delta=args.spacing_delta,
    )
    search_rows = search_data[str(freq)].data
    injection_rows = injection_data[str(freq)].data
    n_sky = len(search_rows) // n_inj
    logging.info(f"{n_inj} injections, {n_sky} sky points per injection")

    def run_injections(run_taskname: str, h0: float) -> tuple[float, Path]:
        """(detection efficiency, outlier file) of the injection set at strain h0, run as run_taskname."""
        return determine_efficiency(
            taskname=run_taskname,
            stage=stage,
            settings=settings,
            freq=freq,
            freq_deriv_order=args.freq_deriv_order,
            n_sky=n_sky,
            num_toplist=args.num_toplist,
            n_seg=n_seg,
            sft_files=args.sft_files,
            metric_file=metric_file,
            extra_stats=WEAVE_EXTRA_STATISTICS,
            weave_exe=settings.environment.executables.weave_path,
            search_data=search_rows,
            injection_data=scale_injection(injection_rows, h0),
            mean2f_th=args.mean2f_th,
            n_cpu=args.n_cpus,
            cluster=args.cluster,
            work_in_local_dir=args.work_in_local_dir,
            save_intermediate=args.save_intermediate,
        )

    run_index = count()

    def efficiency_at(h0: float) -> float:
        logging.info(f"Testing h0 = {h0:.3e}")
        run_t0 = time.time()
        eff, _ = run_injections(f"{taskname}_{next(run_index)}", h0)
        logging.info(f"Run took {time.time() - run_t0:.2f}s | Efficiency: {eff:.2%}")
        return eff

    log_section("EFFICIENCY LOOP START")
    fitter = SigmoidFitter(target, n_inj=n_inj, n_amp=1)
    limit = find_h95(efficiency_at, args.h0_est, fitter)
    h95 = limit.h95

    logging.info(f"Performing final confirmation run at h95 = {h95:.3e}")
    eff, outlier_file_path = run_injections(taskname, h95)
    logging.info(f"Final Efficiency at h95: {eff * 100:.2f}%")

    with fits.open(outlier_file_path, mode="update") as hdul:
        hdul[0].header["HIERARCH sky_radius"] = args.sky_radius
        hdul[0].header["h95"] = h95
        hdul[0].header["dh95"] = limit.dh95
        hdul[0].header[f"p{n_inj}"] = round(eff * 100, 2)
        hdul.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="h0", format="E", array=limit.h0),
                    fits.Column(name="p", format="E", array=limit.efficiency),
                ],
                name="h0_p",
            )
        )
        hdul.append(
            fits.BinTableHDU.from_columns(
                [fits.Column(name="popt", format="E", array=fitter.popt)], name="popt"
            )
        )
        pcov = fitter.pcov.flatten()
        hdul.append(
            fits.BinTableHDU.from_columns(
                [fits.Column(name="pcov", format=f"{len(pcov)}E", array=pcov)],
                name="pcov",
            )
        )
        hdul.flush()
    return outlier_file_path


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    print_args(args)
    t0 = time.time()
    np.random.seed(0)
    settings = FixedSettings(args.environment_file, args.data_file, args.search_file)
    upper_limit(settings, args)
    logging.info(f"Job completed. Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
