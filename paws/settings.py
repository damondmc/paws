"""Validated run settings: config.yaml, the target yaml and stages.yaml.

The config directory holds config.yaml and stages.yaml; stages.yaml names the target yaml.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SECONDS_PER_YEAR = 86400 * 365.25


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


class Executables(_Model):
    weave: str
    split_sfts: Optional[str]
    estimate_uls: Optional[str]
    weave_setup: Optional[str]  # lalpulsar_WeaveSetup; null: the one on PATH


class Config(_Model):
    user: str
    home_dir: Path
    osdf_dir: Path
    executables: Executables
    sft_source: str
    sft_dir: Path
    acc_group: str
    ref_time: int
    f0_band: float
    semi_mm: float
    coh_mm: float
    num_top_list: int
    cluster_n_spacing: float
    followup_n_spacing: float
    nc_min: float
    nc_max: float


class Target(_Model):
    name: str
    age: float  # years, below age_break_freq
    age_break_freq: int  # Hz; from here on the age grows by age_slope years per Hz
    age_slope: float
    alpha: float
    delta: float
    dalpha: float
    ddelta: float
    radius: float

    def tau(self, freq):
        """Spin-down age (s) used for the power-law spin-down range at freq."""
        if freq < self.age_break_freq:
            return SECONDS_PER_YEAR * self.age
        return SECONDS_PER_YEAR * (
            self.age + (freq - (self.age_break_freq - 1)) * self.age_slope
        )


class FollowupThresholds(_Model):
    """Thresholds a follow-up stage applies to the loudest candidate of each seed, computed from injections."""

    # injection stages matching (previous stage, this stage)
    injections: tuple[str, str]
    # % of injections the (2F-4) excess ratio threshold may lose
    excess_ratio_percentile: float
    # % of injections the H1/L1 excess-ratio window may lose; null: no H1/L1 excess-ratio window
    h1_l1_percentile: Optional[float]
    # threshold band edges [Hz]: one excess ratio threshold / H1/L1 excess-ratio window per band
    bands: list[int]


class Stage(_Model):
    """One stage of the pipeline: a Weave run over all bands and the outliers collected from it."""

    name: str
    kind: Literal["search", "injection", "followup", "upperlimit"]
    tcoh: int  # days
    order: int = Field(ge=2, le=4)  # spin-down order searched
    n_seg: int
    prev: Optional[str]  # stage whose outliers seed this one
    prev_sat: bool  # seeds are the previous stage's saturated table (unclustered file)
    bands: tuple[int, int]  # [f_min, f_max) in Hz
    skip_bands: list[int]

    # Weave jobs
    metric: str  # file in <osdf_dir>/metricSetup
    toplist: int  # Weave --toplist-limit
    tasks_per_job: int
    request_memory: str
    request_disk: str
    request_cpu: int
    # file in <osdf_dir>/images; null: the image set in the sub-file writer
    image: Optional[str]
    # search stage: f1dot / f2dot sub-band widths of one Weave job
    search_df1: Optional[float]
    search_df2: Optional[float]

    # sky
    # file in the config dir: (d_alpha, d_delta) offsets around each seed
    sky_grid: Optional[str]
    n_sky: int
    sky_radius: float
    spacing_alpha: Optional[float]
    spacing_delta: Optional[float]

    # injections
    # carries injections (from the previous stage when kind=followup)
    is_injection: bool
    inj_order: Optional[int]
    n_inj: Optional[int]  # injections per band (kind=injection / upperlimit)
    n_inj_max: Optional[int]  # keep only the first n_inj_max seeds per band; null: all
    h0_file: Optional[str]  # relative to home_dir: columns freq, h0, ...

    # outliers
    keep: int  # candidates kept per seed
    # follow-up of real candidates; null: injection or record-only stage
    thresholds: Optional[FollowupThresholds]
    # earlier stages whose Weave results are taken for seeds equal to theirs
    reuse: list[str]
    cluster: bool
    separate_saturated: bool

    @model_validator(mode="after")
    def _check(self):
        if self.kind != "search" and self.prev is None:
            raise ValueError(f"{self.name}: a {self.kind} stage needs prev")
        if self.kind in ("injection", "upperlimit") and None in (
            self.n_inj,
            self.h0_file,
        ):
            raise ValueError(
                f"{self.name}: a {self.kind} stage needs n_inj and h0_file"
            )
        if self.kind == "search" and None in (self.search_df1, self.search_df2):
            raise ValueError(
                f"{self.name}: a search stage needs search_df1 and search_df2"
            )
        if self.kind in ("injection", "upperlimit") and not self.is_injection:
            raise ValueError(f"{self.name}: a {self.kind} stage has is_injection: true")
        if self.is_injection and self.inj_order is None:
            raise ValueError(f"{self.name}: an injection stage needs inj_order")
        if self.sky_grid is None and self.n_sky != 1:
            raise ValueError(f"{self.name}: n_sky={self.n_sky} without a sky_grid")
        if self.is_injection and (self.thresholds is not None or self.reuse):
            raise ValueError(
                f"{self.name}: an injection stage keeps every candidate and runs its own jobs"
            )
        if self.thresholds is not None and self.keep != 1:
            raise ValueError(
                f"{self.name}: a thresholded follow-up keeps the loudest candidate per seed (keep: 1)"
            )
        return self

    def taskname(self, target, freq):
        return f"{target.name}_{self.name}_TCoh{self.tcoh}_O{self.order}_{freq}Hz"

    @property
    def freqs(self):
        return [f for f in range(*self.bands) if f not in self.skip_bands]


class Stages(_Model):
    target: str  # target yaml in the config dir
    stages: dict[str, Stage]
    chains: dict[str, list[str]]  # follow-up chains: stage names in order

    @model_validator(mode="before")
    @classmethod
    def _drop_anchors_and_name_stages(cls, data):
        data = {
            key: value for key, value in data.items() if not key.startswith("x-")
        }  # YAML anchor blocks
        data["stages"] = {
            name: {**fields, "name": name} for name, fields in data["stages"].items()
        }
        return data

    @model_validator(mode="after")
    def _references_exist(self):
        for stage in self.stages.values():
            if stage.prev is not None and stage.prev not in self.stages:
                raise ValueError(
                    f"{stage.name}: prev stage {stage.prev!r} is not defined"
                )
            for reused in stage.reuse:
                if reused not in self.stages:
                    raise ValueError(
                        f"{stage.name}: reused stage {reused!r} is not defined"
                    )
                source = self.stages[reused]
                if (source.tcoh, source.order, source.n_sky) != (
                    stage.tcoh,
                    stage.order,
                    stage.n_sky,
                ):
                    raise ValueError(
                        f"{stage.name}: cannot reuse {reused} (different tcoh, order or sky grid)"
                    )
            if stage.thresholds is not None:
                for injection_stage in stage.thresholds.injections:
                    if injection_stage not in self.stages:
                        raise ValueError(
                            f"{stage.name}: injection stage {injection_stage!r} is not defined"
                        )
        for chain, names in self.chains.items():
            for name in names:
                if name not in self.stages:
                    raise ValueError(f"chain {chain}: stage {name!r} is not defined")
        return self


class Settings:
    """config + target + stages of one config directory."""

    def __init__(self, config_dir):
        if not config_dir:
            raise RuntimeError(
                "no config directory: pass --config-dir or set PAWS_CONFIG_DIR"
            )
        self.config_dir = Path(config_dir)
        self.config = Config.from_yaml(self.config_dir / "config.yaml")
        self.stages_file = Stages.from_yaml(self.config_dir / "stages.yaml")
        self.target_file = self.config_dir / self.stages_file.target
        self.target = Target.from_yaml(self.target_file)
        self._sky_grids = {}
        for stage in self.stages_file.stages.values():
            if stage.sky_grid:
                n_grid_points = len(self.sky_offsets(stage)[0])
                if n_grid_points != stage.n_sky:
                    raise ValueError(
                        f"{stage.name}: n_sky={stage.n_sky} but {stage.sky_grid} has {n_grid_points} points"
                    )

    def stage(self, name):
        try:
            return self.stages_file.stages[name]
        except KeyError:
            raise KeyError(
                f"stage {name!r} is not in {self.config_dir / 'stages.yaml'}"
            ) from None

    def prev(self, stage):
        return self.stage(stage.prev)

    def sky_offsets(self, stage):
        """(d_alpha, d_delta) arrays of the stage's sky grid, or None for a single sky point."""
        if not stage.sky_grid:
            return None
        if stage.sky_grid not in self._sky_grids:
            grid_path = self.config_dir / stage.sky_grid
            self._sky_grids[stage.sky_grid] = tuple(
                np.loadtxt(grid_path, unpack=True, ndmin=2)
            )
        return self._sky_grids[stage.sky_grid]

    def config_path(self, name):
        return self.config_dir / name

    # files staged on OSDF for the Condor jobs
    def osdf_url(self, *parts):
        return "osdf://" + str(self.config.osdf_dir.joinpath(*parts))[len("/osdf") :]

    def metric_url(self, stage):
        return self.osdf_url("metricSetup", stage.metric)

    def image_url(self, stage):
        return self.osdf_url("images", stage.image) if stage.image else None

    def job_config_urls(self):
        """OSDF copies of config.yaml and the target yaml that outlier / upper-limit jobs read."""
        return self.osdf_url("config", "config.yaml"), self.osdf_url(
            "config", self.target_file.name
        )
