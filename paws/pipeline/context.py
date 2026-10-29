"""A stage's settings plus the managers the pipeline steps build from them."""

from functools import cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from jaxtyping import Shaped

from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager
from paws.params.models import SpindownModel
from paws.params.search import SearchParamGenerator
from paws.settings import Settings, Stage
from paws.workflow.manager import WorkflowManager

N_THREADS = 32


class StageContext:
    """One stage's settings together with the path, workflow and result managers built from them."""

    def __init__(self, settings: Settings, stage_name: str) -> None:
        self.settings = settings
        self.target = settings.target
        self.stage = settings.stage(stage_name)
        self.prev_stage: Optional[Stage] = (
            settings.prev(self.stage) if self.stage.prev else None
        )
        self.paths = PathManager(settings)
        self.workflow_manager = WorkflowManager(settings)
        self.result_manager = ResultAnalysisManager(settings)
        # seeds of a follow-up stage, counted while writing its DAGs
        self.n_seeds = 0

    def taskname(self, freq: int) -> str:
        return self.stage.taskname(self.target, freq)

    def prev_outlier_file(self, freq: int, cluster: bool) -> Path:
        prev_taskname = self.prev_stage.taskname(self.target, freq)
        return self.paths.outlier_file(
            freq,
            prev_taskname,
            self.prev_stage.name,
            cluster=cluster,
        )

    def spindown_model(self, freq: int) -> SpindownModel:
        return self.settings.spindown_model(freq)

    @cache
    def h0_by_freq(self) -> dict[int, float]:
        """{1 Hz band: h0} of the stage's h0_file."""
        freqs, h0_values, _ = np.loadtxt(
            self.settings.environment.home_dir / self.stage.injection.h0_file
        ).T
        return {int(freq): float(h0) for freq, h0 in zip(freqs, h0_values)}

    def search_params(self, freq: int) -> Shaped[np.ndarray, " n_param"]:
        """Parameter rows of the initial search's Weave jobs in the band."""
        target = self.target
        generator = SearchParamGenerator(self.spindown_model(freq), self.stage.order)
        params = generator.generate_search_ranges(
            target.alpha,
            target.dalpha,
            target.delta,
            target.ddelta,
            freq,
            freq + 1,
            df=self.settings.search.f0_band,
            df1=self.stage.search_df1,
            df2=self.stage.search_df2,
        )
        return params[freq].data

    def make_weave_dag(
        self,
        freq: int,
        search_params: Shaped[np.ndarray, " n_param"],
        **extra_dag_kwargs: Any,
    ) -> Path:
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
