"""paws: Weave DAGs, outliers and thresholds of the stages defined in <config-dir>/stages.yaml."""

import argparse
import os
from pathlib import Path

from paws.pipeline.dag import make_stage_dags
from paws.pipeline.metric import make_segments_and_metric
from paws.pipeline.outliers import collect_stage_outliers, make_stage_outlier_dags
from paws.pipeline.stage_thresholds import write_threshold_records
from paws.settings import Settings


def build_parser():
    parser = argparse.ArgumentParser(
        prog="paws", description=__doc__,
        epilog="The config directory (config.yaml, stages.yaml, target yaml) is --config-dir or $PAWS_CONFIG_DIR. "
               "Run `paws <command> -h` for the options of a command.",
    )
    parser.add_argument("--config-dir", help="directory with config.yaml and stages.yaml (default $PAWS_CONFIG_DIR)")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    dag_parser = subparsers.add_parser(
        "dag", help="Weave DAGs of a stage -> dagFiles/ list to submit",
        description="Weave DAGs of a stage, one per 1 Hz band, and the DAG list to submit "
                    "(dagFiles/<stage>_<target>_dag<bands>Hz.txt). Follow-ups also save their seed plans.",
    )
    dag_parser.add_argument("stage", help="stage name in stages.yaml")
    dag_parser.add_argument("--bands", type=int, nargs="+", help="only these 1 Hz bands (own DAG list file)")

    outliers_parser = subparsers.add_parser(
        "outliers", help="collect a stage's outliers (--condor: as Condor jobs)",
        description="Outlier files of a stage, collected locally; with --condor, outlier-collection DAGs "
                    "(injection follow-ups only).",
    )
    outliers_parser.add_argument("stage", help="stage name in stages.yaml")
    outliers_parser.add_argument("--bands", type=int, nargs="+", help="only these 1 Hz bands")
    outliers_parser.add_argument("--condor", action="store_true", help="write outlier-collection DAGs instead")

    thresholds_parser = subparsers.add_parser(
        "thresholds", help="write a follow-up stage's thresholds to results/<stage>/thresholds.yaml",
        description="The excess ratio threshold and H1/L1 excess-ratio window(s) of a follow-up stage, computed from its injection "
                    "stages, as results/<stage>/thresholds.yaml.",
    )
    thresholds_parser.add_argument("stage", help="stage name in stages.yaml")

    metric_parser = subparsers.add_parser(
        "metric", help="segment list (and Weave metric) for a coherence time",
        description="Segment list, data coverage check and plot for a coherence time; with --metric also the "
                    "Weave metric (lalpulsar_WeaveSetup).",
    )
    metric_parser.add_argument("tcoh", type=float, help="coherence time [days]")
    metric_parser.add_argument("--metric", action="store_true", help="also run lalpulsar_WeaveSetup")
    metric_parser.add_argument("-s", "--spindowns", type=int, default=2)
    metric_parser.add_argument("--metric-type", default="directed")
    metric_parser.add_argument("--out-dir", type=Path, help="default <home_dir>/metricSetup")
    metric_parser.add_argument("--sft-band", type=int, default=20, help="SFT band [Hz] the timestamps are read from")
    metric_parser.add_argument("--refresh-timestamps", action="store_true", help="re-read the SFT files")
    metric_parser.add_argument("--force", action="store_true", help="overwrite an existing metric file")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings(args.config_dir or os.environ.get("PAWS_CONFIG_DIR"))

    if args.command == "dag":
        make_stage_dags(settings, args.stage, args.bands)
    elif args.command == "outliers" and args.condor:
        make_stage_outlier_dags(settings, args.stage, args.bands)
    elif args.command == "outliers":
        collect_stage_outliers(settings, args.stage, args.bands)
    elif args.command == "thresholds":
        write_threshold_records(settings, settings.stage(args.stage))
    elif args.command == "metric":
        make_segments_and_metric(settings, args.tcoh, args.metric, args.spindowns, args.metric_type, args.out_dir,
                                 args.sft_band, args.refresh_timestamps, args.force)


if __name__ == "__main__":
    main()
