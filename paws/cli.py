"""paws: Weave DAGs, outliers and thresholds of the stages defined in <config-dir>/search.yaml and followup.yaml."""

import argparse
import os
from pathlib import Path
from typing import Optional

from paws.pipeline.calibrate import write_threshold_file
from paws.pipeline.dag import make_stage_dags
from paws.pipeline.metric import make_segments_and_metric
from paws.pipeline.noise_weights import write_noise_weights
from paws.pipeline.stage_outliers import collect_stage_outliers
from paws.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paws",
        description=__doc__,
        epilog="The config directory (environment.yaml, data.yaml, search.yaml, followup.yaml) is --config-dir or "
        "$PAWS_CONFIG_DIR. Run `paws <command> -h` for the options of a command.",
    )
    parser.add_argument(
        "--config-dir",
        help="directory with environment.yaml, data.yaml, search.yaml and followup.yaml (default $PAWS_CONFIG_DIR)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    dag_parser = subparsers.add_parser(
        "dag",
        help="Weave DAGs of a stage -> dagFiles/ list to submit",
        description="Weave DAGs of a stage, one per 1 Hz band, and the DAG list to submit "
        "(dagFiles/<stage>_<target>_dag<bands>Hz.txt).",
    )
    dag_parser.add_argument("stage", help="stage name in search.yaml or followup.yaml")
    dag_parser.add_argument(
        "--bands", type=int, nargs="+", help="only these 1 Hz bands (own DAG list file)"
    )

    outliers_parser = subparsers.add_parser(
        "outliers",
        help="collect a stage's outliers from its Weave results",
        description="Outlier files of a stage, collected from its Weave results.",
    )
    outliers_parser.add_argument(
        "stage", help="stage name in search.yaml or followup.yaml"
    )
    outliers_parser.add_argument(
        "--bands", type=int, nargs="+", help="only these 1 Hz bands"
    )

    thresholds_parser = subparsers.add_parser(
        "thresholds",
        help="calibrate a threshold file from injection stages -> <config-dir>/thresholds/<name>.yaml",
        description="The excess ratio threshold, H1/L1 excess-ratio windows and growth cut of calibration <name> "
        "(followup.yaml), computed from its injection stages, as <config-dir>/thresholds/<name>.yaml; a follow-up "
        "stage applies the file through `thresholds: <name>`.",
    )
    thresholds_parser.add_argument("name", help="calibration name in followup.yaml")

    metric_parser = subparsers.add_parser(
        "metric",
        help="segment list (and Weave metric) for a coherence time",
        description="Segment list, data coverage check and plot for a coherence time; with --metric also the "
        "Weave metric (lalpulsar_WeaveSetup).",
    )
    metric_parser.add_argument("tcoh", type=float, help="coherence time [days]")
    metric_parser.add_argument(
        "--metric", action="store_true", help="also run lalpulsar_WeaveSetup"
    )
    metric_parser.add_argument("-s", "--spindowns", type=int, default=2)
    metric_parser.add_argument("--metric-type", default="directed")
    metric_parser.add_argument(
        "--out-dir", type=Path, help="default <home_dir>/metricSetup"
    )
    metric_parser.add_argument(
        "--sft-band",
        type=int,
        default=20,
        help="SFT band [Hz] the timestamps are read from",
    )
    metric_parser.add_argument(
        "--refresh-timestamps", action="store_true", help="re-read the SFT files"
    )
    metric_parser.add_argument(
        "--force", action="store_true", help="overwrite an existing metric file"
    )
    noise_parser = subparsers.add_parser(
        "noise-weights",
        help="per-detector noise weight of each 1 Hz band -> <config-dir>/<file>",
        description="Per-detector noise weight W = sum over SFTs of 1 / S_h of each 1 Hz band, read from the SFTs, "
        "for the noise-normalized H1/L1 window (`h1_l1_noise_weights: <file>` in a calibration of followup.yaml). "
        "Bands already in the file are kept.",
    )
    noise_parser.add_argument("file", help="file name in the config directory")
    noise_parser.add_argument(
        "--bands", type=int, nargs="+", required=True, help="1 Hz bands"
    )
    noise_parser.add_argument("--workers", type=int, default=12)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings(args.config_dir or os.environ.get("PAWS_CONFIG_DIR"))

    if args.command == "dag":
        make_stage_dags(settings, args.stage, args.bands)
    elif args.command == "outliers":
        collect_stage_outliers(settings, args.stage, args.bands)
    elif args.command == "thresholds":
        write_threshold_file(settings, args.name)
    elif args.command == "noise-weights":
        write_noise_weights(settings, args.file, args.bands, args.workers)
    elif args.command == "metric":
        make_segments_and_metric(
            settings,
            args.tcoh,
            args.metric,
            args.spindowns,
            args.metric_type,
            args.out_dir,
            args.sft_band,
            args.refresh_timestamps,
            args.force,
        )


if __name__ == "__main__":
    main()
