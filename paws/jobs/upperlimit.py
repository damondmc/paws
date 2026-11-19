#!/opt/paws/.venv/bin/python
"""Upper-limit job of one band: injection sets at strains around the h0 estimate locate h95, injections drawn at
their own strains around it give h95 from a sigmoid fit, each run through Weave; the outlier file of the confirmation
set at h95 gets the fit."""

import argparse
import logging
import sys
import time
from itertools import count
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.io import fits
from jaxtyping import Bool, Float, Shaped

from paws.analysis.sigmoid import SigmoidFitter, find_h95
from paws.definitions import WEAVE_EXTRA_STATISTICS, phase_param_name
from paws.jobs.weave_runner import EfficiencyRun, determine_efficiency
from paws.params.injections import InjectionParamGenerator
from paws.params.sky import sky_lattice_offsets
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
    injection_rows: Shaped[np.ndarray, " n_injection"],
    h0: Union[float, Float[np.ndarray, " n_injection"]],
) -> Shaped[np.ndarray, " n_injection"]:
    """The injection rows (drawn at h0 = 1) at strain h0, one strain for all rows or one per row."""
    rows = injection_rows.copy()
    rows["aCross"] *= h0
    rows["aPlus"] *= h0
    return rows


def injection_sets_hdu(sets: list[tuple[float, EfficiencyRun]]) -> fits.BinTableHDU:
    """INJECTION_SETS table of the (strain, run) injection sets in run order, the last being the confirmation set
    at h95. Columns: h0, confirmation, efficiency, and per injection loudest_mean2F and detected."""
    h0 = np.array([h0 for h0, _ in sets])
    runs = [run for _, run in sets]
    n_inj = len(runs[0].detected)
    confirmation = np.arange(len(runs)) == len(runs) - 1
    return fits.BinTableHDU.from_columns(
        [
            fits.Column(name="h0", format="D", array=h0),
            fits.Column(name="confirmation", format="L", array=confirmation),
            fits.Column(
                name="efficiency", format="D", array=[run.efficiency for run in runs]
            ),
            fits.Column(
                name="loudest_mean2F",
                format=f"{n_inj}E",
                array=np.stack([run.loudest_mean2f for run in runs]),
            ),
            fits.Column(
                name="detected",
                format=f"{n_inj}L",
                array=np.stack([run.detected for run in runs]),
            ),
        ],
        name="injection_sets",
    )


def drawn_injections_hdu(
    draws: list[
        tuple[
            Shaped[np.ndarray, " n_drawn"],
            Float[np.ndarray, " n_drawn"],
            EfficiencyRun,
        ]
    ],
) -> fits.BinTableHDU:
    """DRAWN_INJECTIONS table of the (injections at h0 = 1, strains, run) rounds of drawn injections, one row
    per injection: the injection parameters, round, h0, detected and loudest_mean2F. h95 is fitted to the last
    round."""
    population = np.concatenate([np.asarray(rows) for rows, _, _ in draws])
    extra = [
        fits.Column(
            name="round",
            format="J",
            array=np.concatenate(
                [np.full(len(h0), i) for i, (_, h0, _) in enumerate(draws)]
            ),
        ),
        fits.Column(
            name="h0", format="D", array=np.concatenate([h0 for _, h0, _ in draws])
        ),
        fits.Column(
            name="detected",
            format="L",
            array=np.concatenate([run.detected for _, _, run in draws]),
        ),
        fits.Column(
            name="loudest_mean2F",
            format="E",
            array=np.concatenate([run.loudest_mean2f for _, _, run in draws]),
        ),
    ]
    return fits.BinTableHDU.from_columns(
        fits.BinTableHDU(data=population).columns + fits.ColDefs(extra),
        name="drawn_injections",
    )


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
    add(
        "--sky_metric_file",
        default=None,
        help="all-sky Weave setup file: search each injection at its sky lattice covering the sky_radius cap",
    )
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
    """h95 of the band (see find_h95): starting sets of n_inj injections around h0_est, then injections drawn
    at their own strains around h95, then a confirmation set of n_inj new injections at h95. Its outlier file gets
    h95, dh95, the confirmation efficiency at h95 (p<n_inj>, in %), the fit to the drawn injections (POPT, PCOV),
    the starting and confirmation sets with the loudest mean2F of each injection (INJECTION_SETS), the drawn
    injections (DRAWN_INJECTIONS) and the injections at h0 = 1 of the starting and confirmation sets
    (INJECTION_POPULATION, CONFIRMATION_POPULATION). Returns that file."""
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
    sky_offsets = None
    if args.sky_metric_file is not None:
        sky_offsets = sky_lattice_offsets(
            Path(args.sky_metric_file).name
            if args.work_in_local_dir
            else args.sky_metric_file,
            target.alpha,
            target.delta,
            args.sky_radius,
            freq + 1,
            settings.search.semi_mm,
            settings.search.coh_mm if n_seg != 1 else None,
        )

    def draw_population(
        n: int,
    ) -> tuple[Shaped[np.ndarray, " n_row"], Shaped[np.ndarray, " n_row"]]:
        """Search and injection rows (at h0 = 1) of n new injections, n_sky rows per injection."""
        search_data, injection_data = generator.generate_injections_and_search_ranges(
            alpha=target.alpha,
            dalpha=target.dalpha,
            delta=target.delta,
            ddelta=target.ddelta,
            non_sat_bands=np.array(args.non_sat_bands),
            spacing=dict(zip(spacing_names, args.df_grid)),
            h0=1.0,
            freq=freq,
            n_inj=n,
            n_spacing=args.n_spacing,
            inj_freq_deriv_order=args.inj_freq_deriv_order,
            freq_deriv_order=args.freq_deriv_order,
            sky_radius=args.sky_radius,
            sky_offsets=sky_offsets,
        )
        return search_data[str(freq)].data, injection_data[str(freq)].data

    search_rows, injection_rows = draw_population(n_inj)
    n_sky = len(search_rows) // n_inj
    logging.info(f"{n_inj} injections, {n_sky} sky points per injection")

    def run_injections(
        run_taskname: str,
        search_rows: Shaped[np.ndarray, " n_row"],
        injection_rows: Shaped[np.ndarray, " n_row"],
    ) -> EfficiencyRun:
        """The injection rows (already at their strains), run as run_taskname."""
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
            injection_data=injection_rows,
            mean2f_th=args.mean2f_th,
            n_cpu=args.n_cpus,
            cluster=args.cluster,
            work_in_local_dir=args.work_in_local_dir,
            save_intermediate=args.save_intermediate,
        )

    run_index = count()
    sets: list[tuple[float, EfficiencyRun]] = []
    draws: list[
        tuple[
            Shaped[np.ndarray, " n_drawn"], Float[np.ndarray, " n_drawn"], EfficiencyRun
        ]
    ] = []

    def efficiency_at(h0: float) -> float:
        """Efficiency of the starting injections at strain h0."""
        logging.info(f"Testing h0 = {h0:.3e}")
        run_t0 = time.time()
        run = run_injections(
            f"{taskname}_{next(run_index)}",
            search_rows,
            scale_injection(injection_rows, h0),
        )
        sets.append((h0, run))
        logging.info(
            f"Run took {time.time() - run_t0:.2f}s | Efficiency: {run.efficiency:.2%}"
        )
        return run.efficiency

    def detected_at(h0: Float[np.ndarray, " n_drawn"]) -> Bool[np.ndarray, " n_drawn"]:
        """Which of len(h0) new injections, each at its own strain, are detected."""
        logging.info(
            f"Drawing {len(h0)} injections at h0 {h0.min():.3e} - {h0.max():.3e}"
        )
        run_t0 = time.time()
        drawn_search, drawn_injection = draw_population(len(h0))
        run = run_injections(
            f"{taskname}_{next(run_index)}",
            drawn_search,
            scale_injection(drawn_injection, np.repeat(h0, n_sky)),
        )
        # drawn_injection repeats each injection over its n_sky search points
        draws.append((drawn_injection[::n_sky], h0, run))
        logging.info(
            f"Run took {time.time() - run_t0:.2f}s | Efficiency: {run.efficiency:.2%}"
        )
        return run.detected

    log_section("EFFICIENCY LOOP START")
    fitter = SigmoidFitter(target, n_inj=n_inj, n_amp=1)
    limit = find_h95(efficiency_at, detected_at, args.h0_est, fitter)
    h95 = limit.h95

    logging.info(
        f"Performing final confirmation run at h95 = {h95:.3e} with new injections"
    )
    confirmation_search, confirmation_injection = draw_population(n_inj)
    confirmation = run_injections(
        taskname, confirmation_search, scale_injection(confirmation_injection, h95)
    )
    sets.append((h95, confirmation))
    eff, outlier_file_path = confirmation.efficiency, confirmation.outlier_file
    logging.info(f"Final Efficiency at h95: {eff * 100:.2f}%")

    with fits.open(outlier_file_path, mode="update") as hdul:
        hdul[0].header["HIERARCH sky_radius"] = args.sky_radius
        hdul[0].header["h95"] = h95
        hdul[0].header["dh95"] = limit.dh95
        hdul[0].header[f"p{n_inj}"] = round(eff * 100, 2)
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
        hdul.append(injection_sets_hdu(sets))
        hdul.append(drawn_injections_hdu(draws))
        # the injection rows repeat each injection over its n_sky search points
        hdul.append(
            fits.BinTableHDU(data=injection_rows[::n_sky], name="injection_population")
        )
        hdul.append(
            fits.BinTableHDU(
                data=confirmation_injection[::n_sky], name="confirmation_population"
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
