#!/opt/paws/.venv/bin/python
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from paws.analysis.sigmoid import SigmoidFitter
from paws.definitions import phase_param_name
from paws.params.injections import InjectionParamGenerator
from paws.params.models import PowerLawModel
from paws.pipeline import determine_efficiency
from paws.settings import Config, Target

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def log_section(title):
    logging.info(f"\n{'=' * 20} {title.upper()} {'=' * 20}")


def print_args(args):
    """Prints arguments in a clean, aligned table."""
    log_section("JOB PARAMETERS")
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        logging.info(f"{key:<25} : {value}")
    logging.info("=" * 60 + "\n")


def scale_injection(ip_data, h0):
    """Scales injection parameters to a specific h0."""
    params = ip_data.copy()
    params["aCross"] *= h0
    params["aPlus"] *= h0
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 95% Upper Limit via Injection."
    )
    parser.add_argument("--config_file", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--target_file", type=str, required=True, help="Path to the target yaml")
    parser.add_argument(
        "--taskname", type=str, default="GalacticCenter_upperlimit_TCoh5_O2_97Hz"
    )
    parser.add_argument("--freq", type=int, default=97)
    parser.add_argument("--stage", type=str, default="upperlimit")
    parser.add_argument("--freq_deriv_order", type=int, default=2)
    parser.add_argument(
        "--df_grid",
        type=float,
        nargs="+",
        default=[1e-6, 1e-13, 1e-20],
        help="Frequency grid spacing for searches.",
    )
    parser.add_argument("--inj_freq_deriv_order", type=int, default=4)
    parser.add_argument("--sft_files", type=str, default="./")
    parser.add_argument(
        "--metric_file", type=str, default="./Start1368970000_TCoh432000_N107_Spin2.fts"
    )
    parser.add_argument("--n_inj", type=int, default=200)
    parser.add_argument(
        "--num_toplist",
        type=int,
        default=1,
        help="Number of top candidates to consider for efficiency calculation.",
    )
    parser.add_argument(
        "--h0_est", type=float, default=6e-26, help="Initial h0 estimate"
    )
    parser.add_argument(
        "--mean2f_th",
        type=float,
        default=6.46124359109492,
        help="Detection threshold for mean2F",
    )
    parser.add_argument(
        "--non_sat_bands",
        type=float,
        nargs="+",
        default=[20.0, 20.1],
        help="Non-saturated frequency bands to use for injections",
    )
    parser.add_argument("--sky_radius", type=float, default=1e-4)
    parser.add_argument("--spacing_alpha", type=float, default=None)
    parser.add_argument("--spacing_delta", type=float, default=None)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--work_in_local_dir", action="store_true")
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--n_cpus", type=int, default=32)

    args = parser.parse_args()
    print_args(args)

    taskname = args.taskname
    freq = args.freq
    stage = args.stage
    freq_deriv_order = args.freq_deriv_order
    df_grid = args.df_grid
    inj_freq_deriv_order = args.inj_freq_deriv_order
    sft_files = args.sft_files
    metric_file = args.metric_file
    n_inj = args.n_inj
    h0_est = args.h0_est
    mean2f_th = args.mean2f_th
    non_sat_bands = args.non_sat_bands
    sky_radius = args.sky_radius
    spacing_alpha = args.spacing_alpha
    spacing_delta = args.spacing_delta
    num_toplist = args.num_toplist

    non_sat_bands = np.array(non_sat_bands)

    extra_stats = "coh2F_det,mean2F,coh2F_det,mean2F_det"

    t0 = time.time()
    np.random.seed(0)

    # Load Configuration
    config = Config.from_yaml(args.config_file)
    target = Target.from_yaml(args.target_file)

    weave_exe = "/opt/paws/.venv/bin/lalpulsar_Weave"  # Weave of the container image

    if args.work_in_local_dir:
        metric_file = Path(metric_file).name

    # Get reference time from metric file
    ref_time = float(fits.getval(metric_file, "DATE-OBS GPS", ext=0))

    # Get spacing
    _, freq_deriv_names = phase_param_name(freq_deriv_order)
    if len(df_grid) != len(freq_deriv_names):
        raise ValueError(
            f"Length of df_grid ({len(df_grid)}) does not match number of frequency derivative names ({len(freq_deriv_names)})"
        )
    # spacing = {name: fits.getval(args.data_file, name, ext=0) for name in freq_deriv_names}
    df_grid = {name: df_grid[i] for i, name in enumerate(freq_deriv_names)}

    f0_band = config.f0_band

    # 2. Initialize Managers
    model = PowerLawModel(nc_min=config.nc_min, nc_max=config.nc_max, tau=target.tau(freq))
    injection_generator = InjectionParamGenerator(
        model=model, ref_time=ref_time, f0_band=f0_band
    )

    search_data, injection_data = injection_generator.generate_parameters(
        alpha=target.alpha,
        dalpha=target.dalpha,
        delta=target.delta,
        ddelta=target.ddelta,
        non_sat_bands=non_sat_bands,
        spacing=df_grid,
        h0=1.0,
        freq=freq,
        n_inj=n_inj,
        n_spacing=config.followup_n_spacing,
        inj_freq_deriv_order=inj_freq_deriv_order,
        freq_deriv_order=freq_deriv_order,
        sky_radius=sky_radius,
        spacing_alpha=spacing_alpha,
        spacing_delta=spacing_delta,
    )

    # n_sky: number of sky-point jobs per injection (1 if no tiling)
    n_sky = len(search_data[str(freq)].data) // n_inj

    print(
        f"Generated injection and search parameter tables for {n_inj} injections and {n_sky} sky points per injection."
    )

    # 4. Iterative Injection Loop
    h0_arr, eff_arr = [], []  # h0 and efficiency arrays
    iter_count = 0

    factors = [0.5, 0.7, 1.0, 1.5]  # Initial spread around estimate

    log_section("EFFICIENCY LOOP START")
    for factor in factors:
        current_h0 = h0_est * factor
        logging.info(f"Testing h0 = {current_h0:.3e}")

        # Scale params
        scaled_injections = scale_injection(injection_data[str(freq)].data, current_h0)

        loop_t0 = time.time()
        eff, outlier_file_path = determine_efficiency(
            taskname=f"{taskname}_{iter_count}",
            stage=stage,
            config=config,
            target=target,
            freq=freq,
            freq_deriv_order=freq_deriv_order,
            n_sky=n_sky,
            num_toplist=num_toplist,
            n_seg=107,
            sft_files=sft_files,
            metric_file=metric_file,
            extra_stats=extra_stats,
            weave_exe=weave_exe,
            search_data=search_data[str(freq)].data,
            injection_data=scaled_injections,
            mean2f_th=mean2f_th,
            n_cpu=args.n_cpus,
            cluster=args.cluster,
            work_in_local_dir=args.work_in_local_dir,
            save_intermediate=args.save_intermediate,
        )

        logging.info(
            f"Iteration took {time.time() - loop_t0:.2f}s | Efficiency: {eff:.2%}"
        )
        h0_arr.append(current_h0)
        eff_arr.append(eff)
        iter_count += 1

    # 5. Sigmoid Fitting & Refinement
    fitter = SigmoidFitter(target, n_inj=n_inj, n_amp=1)
    outlier_file_path = None  # To store final result path

    while True:
        fitter.fit(np.array(h0_arr), np.array(eff_arr))
        h95, dh95 = fitter.get_h_percentile(percentile=0.95)
        dx = dh95 / h95

        logging.info(
            f"Refinement Step | Current h95: {h95:.3e} | Uncertainty: {dx * 100:.2f}%"
        )

        # Check convergence
        if abs(dx) <= 0.05:
            logging.info(">> Convergence reached (error <= 5%).")
            break

        # Refine if uncertainty is high
        logging.warning("Uncertainty too high, adding points near h95...")

        for mult in [1 - dx, 1 + dx]:
            new_h0 = h95 * mult

            logging.info(f"Testing refinement point h0 = {new_h0:.3e}...")

            scaled_injections = scale_injection(injection_data[str(freq)].data, new_h0)

            eff, outlier_file_path = determine_efficiency(
                taskname=f"{taskname}_{iter_count}",
                stage=stage,
                config=config,
                target=target,
                freq=freq,
                freq_deriv_order=freq_deriv_order,
                n_sky=n_sky,
                num_toplist=num_toplist,
                n_seg=107,
                sft_files=sft_files,
                metric_file=metric_file,
                extra_stats=extra_stats,
                weave_exe=weave_exe,
                search_data=search_data[str(freq)].data,
                injection_data=scaled_injections,
                mean2f_th=mean2f_th,
                n_cpu=args.n_cpus,
                cluster=args.cluster,
                work_in_local_dir=args.work_in_local_dir,
                save_intermediate=args.save_intermediate,
            )

            h0_arr.append(new_h0)
            eff_arr.append(eff)
            iter_count += 1

    # 6. Final Confirmation Run
    logging.info(f"Performing final confirmation run at h95 = {h95:.3e}")
    scaled_injections = scale_injection(injection_data[str(freq)].data, h95)

    eff, outlier_file_path = determine_efficiency(
        taskname=taskname,
        stage=stage,
        config=config,
        target=target,
        freq=freq,
        freq_deriv_order=freq_deriv_order,
        n_sky=n_sky,
        num_toplist=num_toplist,
        n_seg=107,
        sft_files=sft_files,
        metric_file=metric_file,
        extra_stats=extra_stats,
        weave_exe=weave_exe,
        search_data=search_data[str(freq)].data,
        injection_data=scaled_injections,
        mean2f_th=mean2f_th,
        n_cpu=args.n_cpus,
        cluster=args.cluster,
        work_in_local_dir=args.work_in_local_dir,
        save_intermediate=args.save_intermediate,
    )

    logging.info(f"Final Efficiency at h95: {eff * 100:.2f}%")

    # 7. Update FITS Header with Results
    with fits.open(outlier_file_path, mode="update") as hdul:
        # Header info
        hdul[0].header["HIERARCH sky_radius"] = sky_radius
        hdul[0].header["h95"] = h95
        hdul[0].header["dh95"] = h95 * dx
        hdul[0].header[f"p{args.n_inj}"] = round(eff * 100, 2)

        # Append Sigmoid Data Tables
        cols_h0_p = fits.ColDefs(
            [
                fits.Column(name="h0", format="E", array=np.array(h0_arr)),
                fits.Column(name="p", format="E", array=np.array(eff_arr)),
            ]
        )
        hdul.append(fits.BinTableHDU.from_columns(cols_h0_p, name="h0_p"))

        # Append Fit Params
        col_popt = fits.Column(name="popt", format="E", array=fitter.popt)
        hdul.append(fits.BinTableHDU.from_columns([col_popt], name="popt"))

        # Flatten pcov if 2D for simple FITS storage, or use dimensions
        col_pcov = fits.Column(
            name="pcov",
            format=f"{len(fitter.pcov.flatten())}E",
            array=fitter.pcov.flatten(),
        )
        hdul.append(fits.BinTableHDU.from_columns([col_pcov], name="pcov"))

        hdul.flush()

    logging.info(f"Job completed. Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
