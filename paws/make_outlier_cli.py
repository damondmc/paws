#!/opt/paws/.venv/bin/python
import argparse
import time

import numpy as np
from astropy.io import fits

from paws.analysis.outlier import ResultAnalysisManager
from paws.settings import Config, Target


def main():
    parser = argparse.ArgumentParser(
        description="Collect Weave results for a frequency band into a filtered outlier FITS file."
    )
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--target_file", type=str, required=True)
    parser.add_argument("--taskname", type=str, required=True)
    parser.add_argument("--freq", type=int, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--freq_deriv_order", type=int, required=True)
    parser.add_argument(
        "--prev_outlier_file",
        type=str,
        required=True,
        help="Local (transferred) path to the previous stage's outlier FITS file, used to derive mean2F thresholds.",
    )
    parser.add_argument("--num_toplist", type=int, default=1000)
    parser.add_argument("--n_sky", type=int, default=1)
    parser.add_argument(
        "--zero_threshold",
        action="store_true",
        help="Ignore thresholds from the previous stage outlier file and keep all candidates up to num_toplist.",
    )
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--separate_saturated", action="store_true")
    parser.add_argument("--is_injection", action="store_true")
    parser.add_argument("--max_workers", type=int, default=32)

    args = parser.parse_args()

    t0 = time.time()

    result_manager = ResultAnalysisManager(
        Config.from_yaml(args.config_file), Target.from_yaml(args.target_file)
    )

    mean2f_th = fits.getdata(args.prev_outlier_file)["mean2F threshold"]
    if args.zero_threshold:
        mean2f_th = np.zeros_like(mean2f_th)
    n_jobs = mean2f_th.size

    result_manager.make_outlier(
        args.taskname,
        args.freq,
        mean2f_th,
        n_jobs,
        num_toplist=args.num_toplist,
        stage=args.stage,
        freq_deriv_order=args.freq_deriv_order,
        n_sky=args.n_sky,
        cluster=args.cluster,
        work_in_local_dir=True,
        separate_saturated=args.separate_saturated,
        is_injection=args.is_injection,
        max_workers=args.max_workers,
    )

    print(f"Job completed. Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
