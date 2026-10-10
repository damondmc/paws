"""Weave DAGs of a stage, one per 1 Hz band, and the DAG list to submit."""

from astropy.io import fits
from tqdm import tqdm

from paws.fits_io import outlier_file_spacing
from paws.params.followup import FollowUpParamGenerator
from paws.params.injections import InjectionParamGenerator
from paws.pipeline import reuse
from paws.pipeline.context import StageContext
from paws.pipeline.followup_seeds import (
    default_seed_plan,
    read_seed_rows,
    save_seed_plans,
    seed_outlier_file,
    selected_seed_rows,
)


def dag_search(context, freq):
    return context.make_weave_dag(freq, context.search_params(freq))


def non_saturated_search_file(context, freq):
    """(previous search-stage outlier file, its non-saturated sub-bands), or (None, None) to skip the band."""
    if freq not in context.h0_by_freq():
        print(f"{freq} Hz: no h0, skipped")
        return None, None
    path = context.prev_outlier_file(freq, cluster=True)
    if not path.is_file():
        print(f"{freq} Hz: no {context.prev_stage.name} outlier file, skipped")
        return None, None
    non_saturated_bands = fits.getdata(path, extname="non_sat_band")["non_sat_band"]
    if len(non_saturated_bands) == 0:
        print(f"{freq} Hz: no non-saturated sub-band, skipped")
        return None, None
    return path, non_saturated_bands


def dag_injection(context, freq):
    stage, target = context.stage, context.target
    search_file, non_saturated_bands = non_saturated_search_file(context, freq)
    if search_file is None:
        return None
    generator = InjectionParamGenerator(
        model=context.spindown_model(freq),
        ref_time=context.config.ref_time,
        f0_band=context.config.f0_band,
    )
    search_params, injection_params = generator.generate_parameters(
        alpha=target.alpha,
        dalpha=target.dalpha,
        delta=target.delta,
        ddelta=target.ddelta,
        non_sat_bands=non_saturated_bands,
        spacing=outlier_file_spacing(search_file, stage.order),
        h0=context.h0_by_freq()[freq],
        freq=freq,
        n_inj=stage.n_inj,
        n_spacing=context.config.followup_n_spacing,
        inj_freq_deriv_order=stage.inj_order,
        freq_deriv_order=stage.order,
        sky_radius=stage.sky_radius,
        spacing_alpha=stage.spacing_alpha,
        spacing_delta=stage.spacing_delta,
    )
    return context.make_weave_dag(
        freq,
        search_params[str(freq)].data,
        inj_params=injection_params[str(freq)].data,
        inj_freq_deriv_order=stage.inj_order,
    )


def followup_params(context, freq, seed_rows):
    stage, target, prev_stage = context.stage, context.target, context.prev_stage
    seed_file = seed_outlier_file(context.settings, context.paths, stage, freq)
    return FollowUpParamGenerator(context.spindown_model(freq)).generate_parameter(
        alpha=target.alpha,
        dalpha=target.dalpha,
        delta=target.delta,
        ddelta=target.ddelta,
        data=seed_rows,
        old_freq_deriv_order=prev_stage.order,
        new_freq_deriv_order=stage.order,
        spacing=outlier_file_spacing(seed_file, prev_stage.order),
        n_spacing=context.config.followup_n_spacing,
        sky_radius=stage.sky_radius,
        spacing_alpha=stage.spacing_alpha,
        spacing_delta=stage.spacing_delta,
    )


def dag_followup(context, freq):
    """Real candidates: Weave jobs for the seeds of the band's seed plan that run in this stage."""
    if context.stage.is_injection:
        return dag_injection_followup(context, freq)
    stage = context.stage
    seed_rows = selected_seed_rows(context.settings, context.paths, stage, freq)
    if seed_rows is None:
        return None
    if stage.reuse:
        plan = reuse.build_seed_plan(
            context.settings, context.paths, stage, freq, seed_rows
        )
    else:
        plan = default_seed_plan(stage, seed_rows)
    context.seed_plans[freq] = plan
    new_seeds = plan.own_runs(stage.name)
    context.seed_counts["seeds"] += len(seed_rows)
    context.seed_counts["new"] += new_seeds.size
    if new_seeds.size == 0:
        return None
    params = followup_params(context, freq, seed_rows[new_seeds])
    return context.make_weave_dag(
        freq, params.data, sky_offsets=context.settings.sky_offsets(stage)
    )


def dag_injection_followup(context, freq):
    stage, target, prev_stage = context.stage, context.target, context.prev_stage
    seed_rows, injection_rows = read_seed_rows(
        context.settings, context.paths, stage, freq
    )
    if seed_rows is None or len(seed_rows) == 0:
        return None
    seed_file = seed_outlier_file(context.settings, context.paths, stage, freq)
    followup_params = FollowUpParamGenerator(
        context.spindown_model(freq)
    ).generate_parameter(
        alpha=target.alpha,
        dalpha=target.dalpha,
        delta=target.delta,
        ddelta=target.ddelta,
        data=seed_rows,
        old_freq_deriv_order=prev_stage.order,
        new_freq_deriv_order=stage.order,
        spacing=outlier_file_spacing(seed_file, prev_stage.order),
        n_spacing=context.config.followup_n_spacing,
        sky_radius=stage.sky_radius,
        spacing_alpha=stage.spacing_alpha,
        spacing_delta=stage.spacing_delta,
    )
    return context.make_weave_dag(
        freq,
        followup_params.data,
        inj_params=injection_rows,
        inj_freq_deriv_order=stage.inj_order if stage.is_injection else None,
        sky_offsets=context.settings.sky_offsets(stage)
        if len(followup_params.data)
        else None,
    )


def dag_upperlimit(context, freq):
    stage = context.stage
    search_file, non_saturated_bands = non_saturated_search_file(context, freq)
    if search_file is None:
        return None
    config_url, target_url = context.settings.job_config_urls()
    return context.workflow_manager.make_upperlimit_dag(
        config_url,
        target_url,
        context.taskname(freq),
        freq,
        stage.name,
        stage.order,
        context.paths.sft_ensemble(freq),
        context.settings.metric_url(stage),
        fits.getval(search_file, "mean2F_th", 0),
        non_saturated_bands,
        None,
        df_grid=list(outlier_file_spacing(search_file, stage.order).values()),
        inj_freq_deriv_order=stage.inj_order,
        num_toplist=stage.keep,
        sky_radius=stage.sky_radius,
        spacing_alpha=stage.spacing_alpha,
        spacing_delta=stage.spacing_delta,
        h0_est=context.h0_by_freq()[freq],
        n_inj=stage.n_inj,
        request_memory=stage.request_memory,
        request_disk=stage.request_disk,
        request_cpu=stage.request_cpu,
        cluster=stage.cluster,
        work_in_local_dir=True,
        save_intermediate=False,
        image=context.settings.image_url(stage),
    )


DAG_MAKER_BY_KIND = {
    "search": dag_search,
    "injection": dag_injection,
    "followup": dag_followup,
    "upperlimit": dag_upperlimit,
}


def dag_list_path(context, bands):
    stage = context.stage
    band_tag = (
        "_".join(map(str, bands)) if bands else f"{stage.bands[0]}-{stage.bands[1]}"
    )
    return (
        context.config.home_dir
        / "dagFiles"
        / f"{stage.name}_{context.target.name}_dag{band_tag}Hz.txt"
    )


def make_stage_dags(settings, stage_name, bands):
    """DAGs of every band of the stage (or only `bands`; None: the stage's bands) and their DAG list."""
    context = StageContext(settings, stage_name)
    stage = context.stage
    list_path = dag_list_path(context, bands)
    skipped_bands = []
    with open(list_path, "w") as dag_list:
        for freq in tqdm(bands or stage.freqs, desc=f"{stage.name} DAGs"):
            dag_file = DAG_MAKER_BY_KIND[stage.kind](context, freq)
            if dag_file is None:
                skipped_bands.append(freq)
            else:
                dag_list.write(f"{dag_file}\n")
        if stage.kind != "upperlimit":
            # must be the LAST entry: its ALL_NODES lines apply to every DAG in the list
            dag_list.write(
                f"{context.workflow_manager.make_osdf_cleanup_dag(stage.name)}\n"
            )
    if context.seed_plans:
        save_seed_plans(settings, stage, context.seed_plans)
        n_seeds, n_new = context.seed_counts["seeds"], context.seed_counts["new"]
        print(
            f"{stage.name}: {n_seeds:,d} seeds, {n_seeds - n_new:,d} reuse earlier results, {n_new:,d} new "
            f"({n_new * stage.n_sky:,d} Weave runs)"
        )
    elif skipped_bands:
        print(f"skipped bands: {skipped_bands}")
    print(f"DAG list: {list_path}")
