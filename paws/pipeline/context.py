"""A stage's settings plus the managers the pipeline steps build from them."""

from functools import cache

import numpy as np

from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager
from paws.params.models import PowerLawModel
from paws.params.search import SearchParamGenerator
from paws.workflow.manager import WorkflowManager

N_THREADS = 32


class StageContext:
    """One stage's settings together with the path, workflow and result managers built from them."""

    def __init__(self, settings, stage_name):
        self.settings = settings
        self.config, self.target = settings.config, settings.target
        self.stage = settings.stage(stage_name)
        self.prev_stage = settings.prev(self.stage) if self.stage.prev else None
        self.paths = PathManager(self.config, self.target)
        self.workflow_manager = WorkflowManager(self.config, self.target)
        self.result_manager = ResultAnalysisManager(self.config, self.target)
        self.seed_plans = {}  # band -> SeedPlan built while writing a follow-up stage's DAGs
        self.seed_counts = {"seeds": 0, "new": 0}

    def taskname(self, freq):
        return self.stage.taskname(self.target, freq)

    def prev_outlier_file(self, freq, cluster):
        prev_taskname = self.prev_stage.taskname(self.target, freq)
        return self.paths.outlier_file(
            freq,
            prev_taskname,
            self.prev_stage.name,
            cluster=cluster,
            location="existing",
        )

    def spindown_model(self, freq):
        return PowerLawModel(
            nc_min=self.config.nc_min,
            nc_max=self.config.nc_max,
            tau=self.target.tau(freq),
        )

    @cache
    def h0_by_freq(self):
        freqs, h0_values, _ = np.loadtxt(self.config.home_dir / self.stage.h0_file).T
        return dict(zip(freqs.astype(int), h0_values))

    def search_params(self, freq):
        target = self.target
        generator = SearchParamGenerator(self.spindown_model(freq), self.stage.order)
        params = generator.generate_parameters(
            target.alpha,
            target.dalpha,
            target.delta,
            target.ddelta,
            freq,
            freq + 1,
            df=self.config.f0_band,
            df1=self.stage.search_df1,
            df2=self.stage.search_df2,
        )
        return params[freq].data

    def make_weave_dag(self, freq, search_params, **extra_dag_kwargs):
        stage = self.stage
        return self.workflow_manager.make_search_dag(
            self.taskname(freq),
            freq,
            search_params,
            num_top_list=stage.toplist,
            stage=stage.name,
            freq_deriv_order=stage.order,
            n_seg=stage.n_seg,
            sft_files=self.paths.sft_ensemble(freq),
            metric_file=self.settings.metric_url(stage),
            request_memory=stage.request_memory,
            request_disk=stage.request_disk,
            request_cpu=stage.request_cpu,
            use_osg=True,
            use_osdf=True,
            image=self.settings.image_url(stage),
            tasks_per_job=stage.tasks_per_job,
            **extra_dag_kwargs,
        )
