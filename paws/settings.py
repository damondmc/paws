"""Validated run settings of a config directory.

environment.yaml   user, directories, executables and accounting group (per user and cluster)
data.yaml          SFTs, detectors, reference time, f0_band and the target (fixed for the whole analysis)
search.yaml        mismatches, clustering, the spin-down range (spindown) and the initial search stage(s)
followup.yaml      n_spacing, the threshold calibrations, every later stage (injections, follow-ups, upper limits)
                   and the follow-up chains
thresholds/<name>.yaml   the per-band thresholds a follow-up stage applies (`thresholds: <name>`), written by
                   `paws thresholds <name>` from the calibration of that name in followup.yaml, or by hand

The Condor jobs (paws.jobs) read OSDF copies of environment.yaml, data.yaml and search.yaml (FixedSettings).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SECONDS_PER_YEAR = 86400 * 365.25


class YamlModel(BaseModel):
    """Frozen model with no unknown keys, loaded from a YAML file; top-level `x-` keys (YAML anchors) are dropped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    @model_validator(mode="before")
    @classmethod
    def _drop_anchor_keys(cls, data):
        if isinstance(data, dict):
            return {
                key: value
                for key, value in data.items()
                if not str(key).startswith("x-")
            }
        return data


# ------------------------------------------------------------------ environment.yaml


class Executables(YamlModel):
    weave: str
    split_sfts: Optional[str]
    estimate_uls: Optional[str]
    weave_setup: Optional[str]  # lalpulsar_WeaveSetup; null: the one on PATH


class Environment(YamlModel):
    """environment.yaml: the user's directories, executables and accounting group."""

    user: str
    home_dir: Path  # DAGs, logs, outlier files, seed plans
    osdf_dir: Path  # local mount of the OSDF directory: Weave results, metrics, images, job config copies
    executables: Executables
    acc_group: str  # Condor accounting group


# ------------------------------------------------------------------ data.yaml


class Target(YamlModel):
    """The source: its name, age and sky position."""

    name: str
    age: float  # years
    alpha: float  # rad
    delta: float  # rad
    dalpha: float  # rad, search region half-widths
    ddelta: float
    radius: float  # rad


class Data(YamlModel):
    """data.yaml: the SFTs and the target."""

    sft_source: str  # data label in result paths
    sft_dir: (
        Path  # local OSDF mount path of the SFTs: <sft_dir>/<detector>/<band>/*.sft
    )
    detectors: list[str]
    ref_time: int  # GPS reference time of the frequency parameters
    f0_band: float  # Hz, frequency sub-band of one Weave job
    target: Target


# ------------------------------------------------------------------ search.yaml


class Spindown(YamlModel):
    """Spin-down range searched at frequency f (paws.params.models.PowerLawModel): f1dot in
    [-f / ((nc_min - 1) tau(f)), 0] and the braking index in [nc_min, nc_max], with the age tau(f) interpolated
    between age_knots."""

    nc_min: float
    nc_max: float
    # (frequency [Hz], age [years]) knots: linear in between, constant outside; a single knot is a constant age
    age_knots: list[tuple[float, float]]

    @model_validator(mode="after")
    def _knots_increase(self):
        freqs = [freq for freq, _ in self.age_knots]
        if not freqs or any(f1 >= f2 for f1, f2 in zip(freqs, freqs[1:])):
            raise ValueError("age_knots need strictly increasing frequencies")
        return self

    def age(self, freq):
        """Age [years] at freq."""
        return float(
            np.interp(
                freq,
                [freq for freq, _ in self.age_knots],
                [age for _, age in self.age_knots],
            )
        )

    def tau(self, freq):
        """Age [s] at freq."""
        return SECONDS_PER_YEAR * self.age(freq)


class Injection(YamlModel):
    """Injections of a stage: new ones from h0_file when the stage follows the initial search, otherwise the
    injections of its seeds, followed up."""

    order: int = Field(ge=1, le=4)  # spin-down order of the injected signals
    n_inj: Optional[int]  # injections per band; null: the seeds' injections
    n_inj_max: Optional[int]  # keep only the first n_inj_max seeds per band; null: all
    # relative to home_dir, columns freq, h0, ...; null: the seeds' injections
    h0_file: Optional[str]


class Stage(YamlModel):
    """One Weave run over all bands and the outliers collected from it.

    No prev: the initial search. prev: a follow-up of that stage's outliers. injection: an injection stage.
    thresholds: a follow-up of real candidates that keeps the loudest candidate per seed and cuts it with
    thresholds/<thresholds>.yaml. upperlimit: the jobs run the injections themselves (paws.jobs.upperlimit).
    """

    name: str
    tcoh: int  # days
    order: int = Field(ge=2, le=4)  # spin-down order searched
    n_seg: int
    prev: Optional[str]  # stage whose outliers seed this one; null: the initial search
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
    # initial search: f1dot / f2dot sub-band widths of one Weave job
    search_df1: Optional[float]
    search_df2: Optional[float]

    # sky
    # file in the config dir: (d_alpha, d_delta) offsets around each seed
    sky_grid: Optional[str]
    n_sky: int
    sky_radius: float
    spacing_alpha: Optional[float]
    spacing_delta: Optional[float]

    injection: Optional[Injection]
    upperlimit: bool

    # outliers
    keep: int  # candidates kept per seed
    # thresholds/<name>.yaml; null: injection or record-only stage
    thresholds: Optional[str]
    # earlier stages whose Weave results are taken for seeds equal to theirs
    reuse: list[str]
    cluster: bool
    separate_saturated: bool

    @property
    def is_search(self):
        return self.prev is None

    @property
    def is_injection(self):
        return self.injection is not None

    @model_validator(mode="after")
    def _check(self):
        if self.is_search:
            if None in (self.search_df1, self.search_df2):
                raise ValueError(
                    f"{self.name}: the initial search needs search_df1 and search_df2"
                )
            if (
                self.is_injection
                or self.upperlimit
                or self.thresholds is not None
                or self.reuse
            ):
                raise ValueError(
                    f"{self.name}: the initial search (no prev) has no injection, upperlimit, thresholds or reuse"
                )
        if self.upperlimit and not self.is_injection:
            raise ValueError(f"{self.name}: an upper limit needs injection")
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


def _name_stages(data):
    """Puts each stage's key into its `name` field."""
    if isinstance(data, dict) and isinstance(data.get("stages"), dict):
        data = {
            **data,
            "stages": {
                name: {**fields, "name": name}
                for name, fields in data["stages"].items()
            },
        }
    return data


class Search(YamlModel):
    """search.yaml: the initial search."""

    semi_mm: float  # maximum mismatch of the semicoherent metric
    coh_mm: float  # maximum mismatch of the coherent metrics (each segment)
    # outliers within this many template spacings are one cluster
    cluster_n_spacing: float
    spindown: Spindown
    stages: dict[str, Stage]  # initial searches: prev null

    @model_validator(mode="before")
    @classmethod
    def _name(cls, data):
        return _name_stages(data)

    @model_validator(mode="after")
    def _initial_searches(self):
        for stage in self.stages.values():
            if not stage.is_search:
                raise ValueError(
                    f"{stage.name}: search.yaml holds the initial search (prev: null); stages with prev go in followup.yaml"
                )
        return self


# ------------------------------------------------------------------ followup.yaml


class Calibration(YamlModel):
    """How `paws thresholds <name>` computes thresholds/<name>.yaml from two injection stages matching
    (the seeds' stage, the stage the thresholds apply to)."""

    injections: tuple[str, str]
    # % of injections the (2F-4) excess ratio threshold may lose
    excess_ratio_percentile: float
    # % of injections the H1/L1 excess-ratio window may lose; null: no H1/L1 excess-ratio window
    h1_l1_percentile: Optional[float]
    # noise-weights file in the config directory (`paws noise-weights`): the window is on R_HL = r_HL * W_L1 / W_H1;
    # null: on r_HL
    h1_l1_noise_weights: Optional[str]
    # % of injections the growth cut may lose: the excess ratio of the stage over the search stage the chain starts
    # from, against the injections between growth_search_injection (the injection stage at the search stage's
    # coherence time) and injections[1]; null: no growth cut
    growth_percentile: Optional[float]
    growth_search_injection: Optional[str]
    # threshold band edges [Hz]: one excess ratio threshold / H1/L1 excess-ratio window per band
    bands: list[int]

    @model_validator(mode="after")
    def _check(self):
        if (self.growth_percentile is None) != (self.growth_search_injection is None):
            raise ValueError(
                "growth_percentile and growth_search_injection are set together"
            )
        if len(self.bands) < 2 or any(
            f1 >= f2 for f1, f2 in zip(self.bands, self.bands[1:])
        ):
            raise ValueError("bands need at least two increasing edges")
        return self


class Followup(YamlModel):
    """followup.yaml: everything after the initial search."""

    # a follow-up searches n_spacing template spacings each side of its seed
    n_spacing: float
    calibrations: dict[str, Calibration]
    stages: dict[str, Stage]  # every stage with a prev
    chains: dict[str, list[str]]  # follow-up chains: stage names in order

    @model_validator(mode="before")
    @classmethod
    def _name(cls, data):
        return _name_stages(data)

    @model_validator(mode="after")
    def _later_stages(self):
        for stage in self.stages.values():
            if stage.is_search:
                raise ValueError(
                    f"{stage.name}: followup.yaml holds the stages with a prev; the initial search goes in search.yaml"
                )
        return self


# ------------------------------------------------------------------ thresholds/<name>.yaml


class ThresholdBand(YamlModel):
    """Thresholds of one band [f_min, f_max) Hz."""

    f_min: int
    f_max: int
    # (2F - 4) / (2F_seed - 4) the loudest candidate of a seed must reach
    excess_ratio: float
    # lowest injection excess ratio, for the record
    excess_ratio_lowest: Optional[float]
    # (low, high) log10 H1/L1 excess ratio of the candidates (r_HL, or R_HL with h1_l1_noise_weights)
    h1_l1_window: Optional[tuple[float, float]]
    # (low, high) window of the seeds, applied when they come from the initial search
    seed_h1_l1_window: Optional[tuple[float, float]]
    # (2F - 4) / (2F_search - 4) over the search outlier the seed descends from
    growth: Optional[float]


class ThresholdFile(YamlModel):
    """thresholds/<name>.yaml: per-band thresholds of the follow-ups with `thresholds: <name>`.

    `paws thresholds <name>` writes it from the calibration of that name (recorded in `calibration`); a hand-written
    file (calibration: null) works the same."""

    tcoh: int  # days, of the candidates the thresholds apply to
    seed_tcoh: int  # days, of the seeds (seed_h1_l1_window)
    # days, of the search stage the growth cut refers to; null: no growth cut
    growth_tcoh: Optional[int]
    h1_l1_noise_weights: Optional[str]
    calibration: Optional[Calibration]
    bands: list[ThresholdBand]

    @model_validator(mode="after")
    def _check(self):
        if not self.bands:
            raise ValueError("a threshold file needs at least one band")
        for lower, upper in zip(self.bands, self.bands[1:]):
            if lower.f_max != upper.f_min:
                raise ValueError(
                    f"bands are not contiguous: {lower.f_max} != {upper.f_min}"
                )
        for field in (
            "excess_ratio_lowest",
            "h1_l1_window",
            "seed_h1_l1_window",
            "growth",
        ):
            present = [getattr(band, field) is not None for band in self.bands]
            if any(present) != all(present):
                raise ValueError(f"{field} is set in some bands only")
        if (self.growth_tcoh is None) != (self.bands[0].growth is None):
            raise ValueError("growth_tcoh and the bands' growth are set together")
        return self


# ------------------------------------------------------------------ settings


class FixedSettings:
    """environment + data + search: the settings fixed once the search has run, which the Condor jobs read from
    OSDF copies of the three files."""

    def __init__(self, environment_file, data_file, search_file):
        self.environment = Environment.from_yaml(environment_file)
        self.data = Data.from_yaml(data_file)
        self.search = Search.from_yaml(search_file)
        self.target = self.data.target


class Settings(FixedSettings):
    """All settings of a config directory: FixedSettings plus followup.yaml, with the stages of both files."""

    def __init__(self, config_dir):
        if not config_dir:
            raise RuntimeError(
                "no config directory: pass --config-dir or set PAWS_CONFIG_DIR"
            )
        self.config_dir = Path(config_dir)
        super().__init__(
            self.config_dir / "environment.yaml",
            self.config_dir / "data.yaml",
            self.config_dir / "search.yaml",
        )
        self.followup = Followup.from_yaml(self.config_dir / "followup.yaml")
        twice = set(self.search.stages) & set(self.followup.stages)
        if twice:
            raise ValueError(f"stages in both search.yaml and followup.yaml: {twice}")
        self.stages = {**self.search.stages, **self.followup.stages}
        self._sky_grids = {}
        self._check_references()
        for stage in self.stages.values():
            if stage.sky_grid:
                n_grid_points = len(self.sky_offsets(stage)[0])
                if n_grid_points != stage.n_sky:
                    raise ValueError(
                        f"{stage.name}: n_sky={stage.n_sky} but {stage.sky_grid} has {n_grid_points} points"
                    )

    def _check_references(self):
        """Every stage, reuse, injection and chain reference names a defined stage of the right kind."""
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
            if stage.is_injection:
                prev = self.prev(stage)
                if prev.is_injection:
                    if (stage.injection.n_inj, stage.injection.h0_file) != (None, None):
                        raise ValueError(
                            f"{stage.name}: a follow-up of {prev.name}'s injections has n_inj and h0_file null"
                        )
                elif not prev.is_search:
                    raise ValueError(
                        f"{stage.name}: new injections follow the initial search, not {prev.name}"
                    )
                elif None in (stage.injection.n_inj, stage.injection.h0_file):
                    raise ValueError(
                        f"{stage.name}: new injections need n_inj and h0_file"
                    )
        for name, calibration in self.followup.calibrations.items():
            for injection_stage in calibration.injections + (
                (calibration.growth_search_injection,)
                if calibration.growth_search_injection is not None
                else ()
            ):
                if injection_stage not in self.stages:
                    raise ValueError(
                        f"calibration {name}: injection stage {injection_stage!r} is not defined"
                    )
                if not self.stages[injection_stage].is_injection:
                    raise ValueError(
                        f"calibration {name}: {injection_stage} is not an injection stage"
                    )
        for chain, names in self.followup.chains.items():
            for name in names:
                if name not in self.stages:
                    raise ValueError(f"chain {chain}: stage {name!r} is not defined")

    def stage(self, name):
        try:
            return self.stages[name]
        except KeyError:
            raise KeyError(
                f"stage {name!r} is not in {self.config_dir}/search.yaml or followup.yaml"
            ) from None

    def prev(self, stage):
        return self.stage(stage.prev)

    def chain_stages(self, stage):
        """The earlier stages of a follow-up's chain, from the initial search it starts from to stage.prev."""
        stages = [self.prev(stage)]
        while not stages[0].is_search:
            stages.insert(0, self.prev(stages[0]))
        if len({s.tcoh for s in stages}) != len(stages):
            raise ValueError(
                f"{stage.name}: two stages of its chain have the same coherence time"
            )
        return stages

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

    def threshold_file_path(self, name):
        return self.config_dir / "thresholds" / f"{name}.yaml"

    # files staged on OSDF for the Condor jobs
    def osdf_url(self, *parts):
        return (
            "osdf://" + str(self.environment.osdf_dir.joinpath(*parts))[len("/osdf") :]
        )

    def metric_url(self, stage):
        return self.osdf_url("metricSetup", stage.metric)

    def image_url(self, stage):
        return self.osdf_url("images", stage.image) if stage.image else None

    def job_config_urls(self):
        """OSDF copies of environment.yaml, data.yaml and search.yaml that the outlier / upper-limit jobs read."""
        return tuple(
            self.osdf_url("config", name)
            for name in ("environment.yaml", "data.yaml", "search.yaml")
        )
