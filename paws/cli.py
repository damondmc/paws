"""paws: DAGs, outliers and cuts of the stages defined in <config-dir>/stages.yaml.

  paws dag <stage> [--bands f ...]                        Weave DAGs of a stage -> dagFiles/ list to submit
  paws outliers <stage> [--bands f ...] [--chunks N]      collect the stage's outliers locally
  paws outliers <stage> --condor                          ... as Condor jobs (injection follow-ups)
  paws cuts ratio <inj-a> <inj-b> --out FILE              (2F-4) ratio cut from two injection stages
  paws cuts hl <inj> --out FILE                           H1/L1 window from an injection stage
  paws metric <tcoh> [--metric -s N]                      segment list (and Weave metric) for a coherence time

The config directory (config.yaml, stages.yaml, target yaml) is --config-dir or $PAWS_CONFIG_DIR.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from paws import metric, stages
from paws.analysis.outlier import ResultAnalysisManager
from paws.analysis.tools import detection_stat_threshold
from paws.filepaths import PathManager
from paws.params.followup import FollowUpParamGenerator
from paws.params.injections import InjectionParamGenerator
from paws.params.models import PowerLawModel
from paws.params.search import SearchParamGenerator
from paws.settings import Settings
from paws.workflow.manager import WorkflowManager

THREADS = 32
OUTLIER_CLI = "make_outlier_cli.py"  # staged in <osdf_dir>/scripts for --condor


class Run:
    """Settings of one stage plus the managers built from them."""

    def __init__(self, settings, stage_name):
        self.s = settings
        self.config, self.target = settings.config, settings.target
        self.stage = settings.stage(stage_name)
        self.prev = settings.prev(self.stage) if self.stage.prev else None
        self.paths = PathManager(self.config, self.target)
        self.manager = WorkflowManager(self.config, self.target)
        self.results = ResultAnalysisManager(self.config, self.target)

    def taskname(self, freq):
        return self.stage.taskname(self.target, freq)

    def prev_file(self, freq, cluster):
        return stages.resolve_outlier_file(
            self.paths, freq, self.prev.taskname(self.target, freq), self.prev.name, cluster=cluster
        )

    def model(self, freq):
        return PowerLawModel(nc_min=self.config.nc_min, nc_max=self.config.nc_max, tau=self.target.tau(freq))

    @cache
    def h0(self):
        freq, h0, _ = np.loadtxt(self.config.home_dir / self.stage.h0_file).T
        return dict(zip(freq.astype(int), h0))

    def search_params(self, freq):
        t = self.target
        gen = SearchParamGenerator(self.model(freq), self.stage.order)
        params = gen.generate_parameters(
            t.alpha, t.dalpha, t.delta, t.ddelta, freq, freq + 1,
            df=self.config.f0_band, df1=self.stage.search_df1, df2=self.stage.search_df2,
        )
        return params[freq].data

    def weave_dag(self, freq, params, **kwargs):
        st = self.stage
        return self.manager.make_search_dag(
            self.taskname(freq), freq, params, num_top_list=st.toplist, stage=st.name,
            freq_deriv_order=st.order, n_seg=st.n_seg, sft_files=self.paths.sft_ensemble(freq),
            metric_file=self.s.metric_url(st), request_memory=st.request_memory, request_disk=st.request_disk,
            request_cpu=st.request_cpu, use_osg=True, use_osdf=True, image=self.s.image_url(st),
            tasks_per_job=st.tasks_per_job, **kwargs,
        )


# ---------------------------------------------------------------- dag


def dag_search(r, freq):
    return r.weave_dag(freq, r.search_params(freq))


def _non_sat_seed_file(r, freq):
    """Previous (search) stage file with its non-saturated sub-bands, or None to skip the band."""
    if freq not in r.h0():
        print(f"{freq} Hz: no h0, skipped")
        return None, None
    path = r.prev_file(freq, cluster=True)
    if not path.is_file():
        print(f"{freq} Hz: no {r.prev.name} outlier file, skipped")
        return None, None
    bands = fits.getdata(path, extname="non_sat_band")["non_sat_band"]
    if len(bands) == 0:
        print(f"{freq} Hz: no non-saturated sub-band, skipped")
        return None, None
    return path, bands


def dag_injection(r, freq):
    st, t = r.stage, r.target
    path, non_sat = _non_sat_seed_file(r, freq)
    if path is None:
        return None
    gen = InjectionParamGenerator(model=r.model(freq), ref_time=r.config.ref_time, f0_band=r.config.f0_band)
    search, inj = gen.generate_parameters(
        alpha=t.alpha, dalpha=t.dalpha, delta=t.delta, ddelta=t.ddelta, non_sat_bands=non_sat,
        spacing=stages.header_spacing(path, st.order), h0=r.h0()[freq], freq=freq, n_inj=st.n_inj,
        n_spacing=r.config.followup_n_spacing, inj_freq_deriv_order=st.inj_order, freq_deriv_order=st.order,
        sky_radius=st.sky_radius, spacing_alpha=st.spacing_alpha, spacing_delta=st.spacing_delta,
    )
    return r.weave_dag(
        freq, search[str(freq)].data, inj_params=inj[str(freq)].data, inj_freq_deriv_order=st.inj_order
    )


def dag_followup(r, freq):
    st, t = r.stage, r.target
    rows, inj = stages.seeds(r.s, r.paths, st, freq)
    if rows is None or len(rows) == 0:
        return None
    path = stages.seed_file(r.s, r.paths, st, freq)
    params = FollowUpParamGenerator(r.model(freq)).generate_parameter(
        alpha=t.alpha, dalpha=t.dalpha, delta=t.delta, ddelta=t.ddelta, data=rows,
        old_freq_deriv_order=r.prev.order, new_freq_deriv_order=st.order, spacing=stages.header_spacing(path, r.prev.order),
        n_spacing=r.config.followup_n_spacing, sky_radius=st.sky_radius,
        spacing_alpha=st.spacing_alpha, spacing_delta=st.spacing_delta,
    )
    return r.weave_dag(
        freq, params.data, inj_params=inj, inj_freq_deriv_order=st.inj_order if st.is_injection else None,
        sky_offsets=r.s.sky_offsets(st) if len(params.data) else None,
    )


def dag_upperlimit(r, freq):
    st = r.stage
    path, non_sat = _non_sat_seed_file(r, freq)
    if path is None:
        return None
    config_url, target_url = r.s.job_config_urls()
    return r.manager.make_upperlimit_dag(
        config_url, target_url, r.taskname(freq), freq, st.name, st.order, r.paths.sft_ensemble(freq),
        r.s.metric_url(st), fits.getval(path, "mean2F_th", 0), non_sat, None,
        df_grid=list(stages.header_spacing(path, st.order).values()), inj_freq_deriv_order=st.inj_order,
        num_toplist=st.keep, sky_radius=st.sky_radius, spacing_alpha=st.spacing_alpha,
        spacing_delta=st.spacing_delta, h0_est=r.h0()[freq], n_inj=st.n_inj, request_memory=st.request_memory,
        request_disk=st.request_disk, request_cpu=st.request_cpu, cluster=st.cluster, work_in_local_dir=True,
        save_intermediate=False, image=r.s.image_url(st),
    )


DAG = {"search": dag_search, "injection": dag_injection, "followup": dag_followup, "upperlimit": dag_upperlimit}


def dag_list_path(r, bands):
    st = r.stage
    tag = "_".join(map(str, bands)) if bands else f"{st.bands[0]}-{st.bands[1]}"
    return r.config.home_dir / "dagFiles" / f"{st.name}_{r.target.name}_dag{tag}Hz.txt"


def cmd_dag(settings, args):
    r = Run(settings, args.stage)
    path = dag_list_path(r, args.bands)
    skipped = []
    with open(path, "w") as f:
        for freq in tqdm(args.bands or r.stage.freqs, desc=f"{r.stage.name} DAGs"):
            dag = DAG[r.stage.kind](r, freq)
            if dag is None:
                skipped.append(freq)
            else:
                f.write(f"{dag}\n")
        if r.stage.kind != "upperlimit":
            # must be the LAST entry: its ALL_NODES lines apply to every DAG in the list
            f.write(f"{r.manager.make_osdf_cleanup_dag(r.stage.name)}\n")
    if skipped:
        print(f"skipped bands: {skipped}")
    print(f"DAG list: {path}")


# ---------------------------------------------------------------- outliers


def n_templates(r, freq, n_jobs):
    def nsemitpl(job):
        return fits.getheader(r.paths.weave_output_file(freq, r.taskname(freq), job, r.stage.name))["NSEMITPL"]

    with ThreadPoolExecutor(THREADS) as ex:
        return sum(ex.map(nsemitpl, range(1, n_jobs + 1)))


def outlier_kwargs(r):
    st = r.stage
    return dict(
        num_toplist=st.keep, stage=st.name, freq_deriv_order=st.order, n_sky=st.n_sky,
        cluster=st.cluster, work_in_local_dir=False, separate_saturated=st.separate_saturated,
        is_injection=st.is_injection, max_workers=THREADS,
    )


def outliers_search(r, freq, args):
    n_jobs = r.search_params(freq).size
    th = detection_stat_threshold(n_templates(r, freq, n_jobs), r.stage.n_seg)
    r.results.make_outlier(r.taskname(freq), freq, th, n_jobs, **outlier_kwargs(r))


def outliers_injection(r, freq, args):
    path = r.prev_file(freq, cluster=True)
    if not path.is_file():
        print(f"{freq} Hz: no {r.prev.name} outlier file, skipped")
        return
    th = fits.getval(path, "mean2F_th", 0)
    r.results.make_outlier(r.taskname(freq), freq, th, r.stage.n_inj, **outlier_kwargs(r))


def followup_threshold(r, freq, rows):
    if r.stage.is_injection:
        return np.zeros(len(rows))
    return stages.ratio_threshold(r.s.config_path(r.stage.ratio_cut), freq, rows["mean2F"])


def outliers_followup(r, freq, args):
    if r.stage.hl_window:
        raise SystemExit(f"{r.stage.name} has an H1/L1 window: collect it with scripts/followup_v3/followup_v3.py")
    rows, _ = stages.seeds(r.s, r.paths, r.stage, freq)
    if rows is None or len(rows) == 0:
        return
    th = followup_threshold(r, freq, rows)
    if args.chunks:
        kw = outlier_kwargs(r)
        for k in ("stage", "freq_deriv_order", "n_sky", "cluster", "max_workers"):
            kw.pop(k)
        stages.make_outlier_chunked(r.results, r.stage, r.taskname(freq), freq, th, args.chunks,
                                    chunks_to_run=args.chunks_to_run, keep_parts=True, max_workers=THREADS, **kw)
    else:
        r.results.make_outlier(r.taskname(freq), freq, th, len(rows), **outlier_kwargs(r))


def outlier_condor_dag(r, freq):
    st = r.stage
    rows, _ = stages.seeds(r.s, r.paths, st, freq)
    if rows is None or len(rows) == 0:
        return None
    config_url, target_url = r.s.job_config_urls()
    return r.manager.make_outlier_dag(
        config_url, target_url, r.taskname(freq), freq, st.name, st.order, len(rows),
        stages.seed_file(r.s, r.paths, st, freq), r.s.osdf_url("scripts", OUTLIER_CLI),
        num_toplist=st.keep, n_sky=st.n_sky, zero_threshold=True, cluster=st.cluster,
        separate_saturated=False, is_injection=True, max_workers=THREADS, request_memory="8GB",
        request_disk="8GB", request_cpu=1, image=r.s.image_url(st),
    )


OUTLIERS = {"search": outliers_search, "injection": outliers_injection, "followup": outliers_followup}


def cmd_outliers(settings, args):
    r = Run(settings, args.stage)
    st = r.stage
    if st.kind not in OUTLIERS:
        raise SystemExit(f"{st.name}: {st.kind} jobs write their own outlier files")
    if args.chunks and st.kind != "followup":
        raise SystemExit("--chunks applies to follow-up stages only (a search needs the full info table)")
    freqs = args.bands or st.freqs
    if args.condor:
        if not (st.kind == "followup" and st.is_injection):
            raise SystemExit("--condor collects injection follow-ups only (the job keeps every candidate)")
        path = dag_list_path(r, args.bands).with_name(dag_list_path(r, args.bands).name.replace("_dag", "_outlier_dag"))
        with open(path, "w") as f:
            for freq in tqdm(freqs):
                dag = outlier_condor_dag(r, freq)
                if dag is not None:
                    f.write(f"{dag}\n")
        print(f"DAG list: {path}")
        return
    for freq in tqdm(freqs, desc=f"{st.name} outliers"):
        OUTLIERS[st.kind](r, freq, args)


# ---------------------------------------------------------------- cuts


def cmd_cuts(settings, args):
    paths = PathManager(settings.config, settings.target)
    edges = np.array(args.edges)
    freqs = range(edges[0], edges[-1])
    out = settings.config_path(args.out)
    if args.kind == "hl":
        data = stages.injection_outliers(settings, paths, settings.stage(args.stages[0]), freqs)
        f = np.concatenate([np.full(len(r), b) for b, (r, _) in data.items()])
        h1 = np.concatenate([r["mean2F_H1"] for r, _ in data.values()])
        l1 = np.concatenate([r["mean2F_L1"] for r, _ in data.values()])
        x = stages.log_hl(h1, l1)
        use = (h1 > 4) & (l1 > 4) if args.both_detectors else np.ones(x.size, bool)
        idx = stages.band_index(edges, f)
        win = [np.percentile(x[use & (idx == i)], [args.q / 2, 100 - args.q / 2]) for i in range(len(edges) - 1)]
        stages.write_cut_file(out, "#f_start\tf_end\tlog10 rHL low\tlog10 rHL high\n", edges, win, args.step)
    else:
        a, b = (stages.injection_outliers(settings, paths, settings.stage(n), freqs) for n in args.stages)
        f, ratio = stages.injection_ratios(a, b)
        idx = stages.band_index(edges, f)
        th = [(np.percentile(ratio[idx == i], args.q), ratio[idx == i].min()) for i in range(len(edges) - 1)]
        stages.write_cut_file(out, f"#f_start\tf_end\t{args.q:g} percentile\tlowest\n", edges, th, args.step)


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paws", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", help="directory with config.yaml and stages.yaml (default $PAWS_CONFIG_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dag", help="Weave DAGs of a stage")
    p.add_argument("stage")
    p.add_argument("--bands", type=int, nargs="+", help="only these 1 Hz bands (own DAG list file)")

    p = sub.add_parser("outliers", help="collect a stage's outliers")
    p.add_argument("stage")
    p.add_argument("--bands", type=int, nargs="+")
    p.add_argument("--condor", action="store_true", help="write outlier-collection DAGs instead")
    p.add_argument("--chunks", type=int, help="collect each band in N contiguous slices (very large bands)")
    p.add_argument("--chunks-to-run", type=int, nargs="+", help="with --chunks: only these 1-based slices")

    p = sub.add_parser("cuts", help="cut files from injection stages")
    p.add_argument("kind", choices=["ratio", "hl"])
    p.add_argument("stages", nargs="+", help="ratio: <inj-a> <inj-b>; hl: <inj>")
    p.add_argument("--out", required=True, help="file name in the config dir")
    p.add_argument("--q", type=float, default=0.1, help="%% of injections a cut may lose")
    p.add_argument("--edges", type=int, nargs="+", default=[20, 100, 200, 300, 400], help="cut bands [Hz]")
    p.add_argument("--step", type=int, default=10, help="row width of the cut file [Hz]")
    p.add_argument("--both-detectors", action="store_true", help="hl: only injections with both 2F > 4")

    metric.add_arguments(sub.add_parser("metric", help="segment list and Weave metric"))

    args = ap.parse_args(argv)
    if args.cmd == "cuts" and len(args.stages) != (2 if args.kind == "ratio" else 1):
        ap.error("cuts ratio takes two injection stages, cuts hl one")
    settings = Settings(args.config_dir or os.environ.get("PAWS_CONFIG_DIR"))
    {"dag": cmd_dag, "outliers": cmd_outliers, "cuts": cmd_cuts, "metric": metric.run}[args.cmd](settings, args)


if __name__ == "__main__":
    main()
