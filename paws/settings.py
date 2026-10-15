"""Settings of a config directory: environment.yaml (user, directories, executables), data.yaml (SFTs and target),
search.yaml (mismatches, clustering, the spindown_model script, the initial search), followup.yaml (n_spacing, the
threshold calibrations, the later stages and the chains) and thresholds/<name>.yaml (the cuts a follow-up applies,
from `paws thresholds <name>`). The Condor jobs read OSDF copies of the first three and the spindown script
(FixedSettings)."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, TypeVar, Union

import numpy as np
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, model_validator

from paws.params.models import SpindownModel

ModelT = TypeVar("ModelT", bound="YamlModel")


class YamlModel(BaseModel):
    """Frozen model with no unknown keys, loaded from a YAML file; top-level `x-` keys (YAML anchors) are dropped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_yaml(cls: type[ModelT], path: Union[str, Path]) -> ModelT:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    @model_validator(mode="before")
    @classmethod
    def _drop_anchor_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                key: value
                for key, value in data.items()
                if not str(key).startswith("x-")
            }
        return data


# -------------------------- environment.yaml --------------------------


# the venv of the container image (paws.def): paws and the lalsuite of its dependencies
CONTAINER_VENV_BIN = "/opt/paws/.venv/bin"


class Executables(YamlModel):
    """lalpulsar executables; null: the ones installed with paws (the lalsuite of its venv, CONTAINER_VENV_BIN in
    the image)."""

    weave: Optional[str]  # of the Weave jobs; null needs an image on every stage
    weave_setup: Optional[str]  # of `paws metric`; null: the one on PATH

    @property
    def weave_path(self) -> str:
        return self.weave or f"{CONTAINER_VENV_BIN}/lalpulsar_Weave"


class Environment(YamlModel):
    """environment.yaml: the user's directories, executables and accounting group."""

    user: str
    home_dir: Path  # project directory: DAGs, logs, outlier files
    osdf_dir: (
        Path  # local OSDF mount: Weave results, metrics, images, job config copies
    )
    executables: Executables
    acc_group: str  # Condor accounting group


# ----------------------------- data.yaml ------------------------------


class Target(YamlModel):
    """The source: its name and sky position."""

    name: str
    alpha: float  # rad
    delta: float  # rad
    dalpha: float  # rad, search region half-widths
    ddelta: float


class Data(YamlModel):
    """data.yaml: the SFTs and the target."""

    sft_source: str  # data label in result paths
    # local OSDF mount path of the SFTs: <sft_dir>/<detector>/<band>/*.sft
    sft_dir: Path
    detectors: list[str]
    ref_time: int  # GPS reference time of the frequency parameters
    f0_band: float  # Hz, frequency sub-band of one Weave job
    target: Target


# ---------------------------- search.yaml -----------------------------


class Injection(YamlModel):
    """Injections of a stage: new ones from h0_file when the stage follows the initial search, otherwise the
    injections of its seeds, followed up."""

    order: int = Field(ge=1, le=4)  # spin-down order of the injected signals
    n_inj: Optional[int]  # injections per band; null: the seeds' injections
    # relative to home_dir, columns freq, h0, ...; null: the seeds' injections
    h0_file: Optional[str]


class Stage(YamlModel):
    """One Weave run over all bands and the outliers collected from it.

    No prev: the initial search. prev: a follow-up of that stage's outliers. injection: an injection stage.
    thresholds: a follow-up of real candidates that keeps the loudest candidate per seed and cuts it with
    thresholds/<thresholds>.yaml. upperlimit: the jobs run the injections themselves (paws.jobs.upperlimit).
    """

    name: str
    # seeds
    prev: Optional[str]  # stage whose outliers seed this one; null: the initial search
    prev_sat: bool  # true: the seeds are the previous stage's saturated table
    injection: Optional[Injection]  # null: no injections
    upperlimit: bool  # the jobs inject and run Weave themselves (paws.jobs.upperlimit)

    # Weave run
    tcoh: int  # coherence time in days
    order: int = Field(ge=2, le=4)  # spin-down order searched
    n_seg: int  # number of segments (T_obs / tcoh)
    metric: str  # in <osdf_dir>/metricSetup (`paws metric`)
    bands: tuple[int, int]  # [f_min, f_max) Hz
    skip_bands: list[int]  # 1 Hz bands left out
    # initial search: f1dot / f2dot sub-band width of one Weave job
    search_df1: Optional[float]
    search_df2: Optional[float]
    # in the config directory: (d_alpha, d_delta) offsets around each seed
    sky_grid: Optional[str]
    n_sky: int  # sky points per seed; 1 without a sky_grid
    # injections and upper limit: sky region radius and grid spacing [rad] around the target
    sky_radius: float
    spacing_alpha: Optional[float]
    spacing_delta: Optional[float]
    toplist: int  # Weave --toplist-limit

    # Condor jobs
    tasks_per_job: int  # Weave tasks per Condor job
    request_memory: str
    request_disk: str
    request_cpu: int
    # in <osdf_dir>/images; null: no container (executables.weave must be set)
    image: Optional[str]

    # outlier collection
    n_keep: int  # candidates kept per Weave toplist (a follow-up: per seed) before the mean2F threshold
    # thresholds/<name>.yaml of a follow-up of real candidates; null: none
    thresholds: Optional[str]
    cluster: bool  # also write the clustered outlier file
    separate_saturated: bool  # saturated jobs in their own result table

    @property
    def is_search(self) -> bool:
        return self.prev is None

    @property
    def is_injection(self) -> bool:
        return self.injection is not None

    @model_validator(mode="after")
    def _check(self) -> "Stage":
        if self.is_search:
            if None in (self.search_df1, self.search_df2):
                raise ValueError(
                    f"{self.name}: the initial search needs search_df1 and search_df2"
                )
            if self.is_injection or self.upperlimit or self.thresholds is not None:
                raise ValueError(
                    f"{self.name}: the initial search (no prev) has no injection, upperlimit or thresholds"
                )
        if self.upperlimit and not self.is_injection:
            raise ValueError(f"{self.name}: an upper limit needs injection")
        if self.sky_grid is None and self.n_sky != 1:
            raise ValueError(f"{self.name}: n_sky={self.n_sky} without a sky_grid")
        if self.is_injection and self.thresholds is not None:
            raise ValueError(f"{self.name}: an injection stage keeps every candidate")
        if self.thresholds is not None and self.n_keep != 1:
            raise ValueError(
                f"{self.name}: a thresholded follow-up keeps the loudest candidate per seed (n_keep: 1)"
            )
        return self

    def taskname(self, target: Target, freq: int) -> str:
        return f"{target.name}_{self.name}_TCoh{self.tcoh}_O{self.order}_{freq}Hz"

    @property
    def freqs(self) -> list[int]:
        return [f for f in range(*self.bands) if f not in self.skip_bands]


def _name_stages(data: Any) -> Any:
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
    # Python file in the config directory defining spindown_model(freq) -> paws.params.models.SpindownModel
    spindown_model: str
    stages: dict[str, Stage]  # initial searches: prev null

    @model_validator(mode="before")
    @classmethod
    def _name(cls, data: Any) -> Any:
        return _name_stages(data)

    @model_validator(mode="after")
    def _initial_searches(self) -> "Search":
        for stage in self.stages.values():
            if not stage.is_search:
                raise ValueError(
                    f"{stage.name}: search.yaml holds the initial search (prev: null); stages with prev go in followup.yaml"
                )
        return self


# --------------------------- followup.yaml ----------------------------


class Calibration(YamlModel):
    """How `paws thresholds <name>` computes thresholds/<name>.yaml from two injection stages matching
    (the seeds' stage, the stage the thresholds apply to)."""

    # the seeds' stage and the stage the thresholds apply to
    injections: tuple[str, str]
    excess_ratio_percentile: float  # % of injections lost by the (2F - 4) excess ratio threshold over the seed
    # % of injections lost by the H1/L1 excess-ratio window; null: no window
    h1_l1_percentile: Optional[float]
    # noise-weights file in the config directory (`paws noise-weights`); null: no noise normalization
    h1_l1_noise_weights: Optional[str]
    # % of injections lost by the growth cut over the search outlier, calibrated between growth_search_injection
    # (the injection stage at the search's coherence time) and injections[1]; null: no growth cut
    growth_percentile: Optional[float]
    growth_search_injection: Optional[str]
    bands: list[int]  # threshold band edges in Hz: one threshold and window per band

    @model_validator(mode="after")
    def _check(self) -> "Calibration":
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

    n_spacing: float  # template spacings searched each side of a seed
    calibrations: dict[str, Calibration]
    stages: dict[str, Stage]  # every stage with a prev
    chains: dict[str, list[str]]  # follow-up chains: stage names in order

    @model_validator(mode="before")
    @classmethod
    def _name(cls, data: Any) -> Any:
        return _name_stages(data)

    @model_validator(mode="after")
    def _later_stages(self) -> "Followup":
        for stage in self.stages.values():
            if stage.is_search:
                raise ValueError(
                    f"{stage.name}: followup.yaml holds the stages with a prev; the initial search goes in search.yaml"
                )
        return self


# ----------------------- thresholds/<name>.yaml -----------------------


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
    def _check(self) -> "ThresholdFile":
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


# ------------------------------ settings ------------------------------


class FixedSettings:
    """environment.yaml, data.yaml and search.yaml with its spindown script: the settings fixed once the search has
    run, which the Condor jobs read from OSDF copies."""

    def __init__(
        self,
        environment_file: Union[str, Path],
        data_file: Union[str, Path],
        search_file: Union[str, Path],
    ) -> None:
        self.environment = Environment.from_yaml(environment_file)
        self.data = Data.from_yaml(data_file)
        self.search = Search.from_yaml(search_file)
        self.target = self.data.target
        self.spindown_model_file = Path(search_file).parent / self.search.spindown_model
        self._spindown_module: Optional[ModuleType] = None

    def spindown_model(self, freq: float) -> SpindownModel:
        """The spin-down ranges at freq: spindown_model(freq) of the search.yaml `spindown_model` script."""
        if self._spindown_module is None:
            if not self.spindown_model_file.is_file():
                raise ValueError(
                    f"spindown_model script not found: {self.spindown_model_file}"
                )
            spec = importlib.util.spec_from_file_location(
                self.spindown_model_file.stem, self.spindown_model_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not callable(getattr(module, "spindown_model", None)):
                raise ValueError(
                    f"{self.spindown_model_file} defines no spindown_model(freq)"
                )
            self._spindown_module = module
        model = self._spindown_module.spindown_model(freq)
        if not isinstance(model, SpindownModel):
            raise TypeError(
                f"{self.spindown_model_file}: spindown_model({freq}) is not a paws.params.models.SpindownModel"
            )
        return model


class Settings(FixedSettings):
    """All settings of a config directory: FixedSettings plus followup.yaml, with the stages of both files."""

    def __init__(self, config_dir: Optional[Union[str, Path]]) -> None:
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
        self.stages: dict[str, Stage] = {
            **self.search.stages,
            **self.followup.stages,
        }
        self._sky_grids: dict[
            str, tuple[Float[np.ndarray, " n_sky"], Float[np.ndarray, " n_sky"]]
        ] = {}
        self._check_references()
        if not self.spindown_model_file.is_file():
            raise ValueError(
                f"spindown_model script not found: {self.spindown_model_file}"
            )
        for stage in self.stages.values():
            if stage.image is None and stage.upperlimit:
                raise ValueError(
                    f"{stage.name}: upper-limit jobs run in the container image, set image"
                )
            if stage.image is None and self.environment.executables.weave is None:
                raise ValueError(
                    f"{stage.name} has no image: set executables.weave in environment.yaml"
                )
            if stage.sky_grid:
                n_grid_points = len(self.sky_offsets(stage)[0])
                if n_grid_points != stage.n_sky:
                    raise ValueError(
                        f"{stage.name}: n_sky={stage.n_sky} but {stage.sky_grid} has {n_grid_points} points"
                    )

    def _check_references(self) -> None:
        """Every stage, injection and chain reference names a defined stage of the right kind."""
        for stage in self.stages.values():
            if stage.prev is not None and stage.prev not in self.stages:
                raise ValueError(
                    f"{stage.name}: prev stage {stage.prev!r} is not defined"
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

    def stage(self, name: str) -> Stage:
        try:
            return self.stages[name]
        except KeyError:
            raise KeyError(
                f"stage {name!r} is not in {self.config_dir}/search.yaml or followup.yaml"
            ) from None

    def prev(self, stage: Stage) -> Stage:
        return self.stage(stage.prev)

    def chain_stages(self, stage: Stage) -> list[Stage]:
        """The earlier stages of a follow-up's chain, from the initial search it starts from to stage.prev."""
        stages = [self.prev(stage)]
        while not stages[0].is_search:
            stages.insert(0, self.prev(stages[0]))
        if len({s.tcoh for s in stages}) != len(stages):
            raise ValueError(
                f"{stage.name}: two stages of its chain have the same coherence time"
            )
        return stages

    def sky_offsets(
        self, stage: Stage
    ) -> Optional[tuple[Float[np.ndarray, " n_sky"], Float[np.ndarray, " n_sky"]]]:
        """(d_alpha, d_delta) arrays of the stage's sky grid, or None for a single sky point."""
        if not stage.sky_grid:
            return None
        if stage.sky_grid not in self._sky_grids:
            grid_path = self.config_dir / stage.sky_grid
            self._sky_grids[stage.sky_grid] = tuple(
                np.loadtxt(grid_path, unpack=True, ndmin=2)
            )
        return self._sky_grids[stage.sky_grid]

    def config_path(self, name: str) -> Path:
        return self.config_dir / name

    def threshold_file_path(self, name: str) -> Path:
        return self.config_dir / "thresholds" / f"{name}.yaml"

    # files staged on OSDF for the Condor jobs
    def osdf_url(self, *parts: str) -> str:
        return (
            "osdf://" + str(self.environment.osdf_dir.joinpath(*parts))[len("/osdf") :]
        )

    def metric_url(self, stage: Stage) -> str:
        return self.osdf_url("metricSetup", stage.metric)

    def image_url(self, stage: Stage) -> Optional[str]:
        return self.osdf_url("images", stage.image) if stage.image else None

    def job_config_urls(self) -> tuple[str, str, str, str]:
        """OSDF copies of environment.yaml, data.yaml, search.yaml and the spindown_model script that the
        outlier / upper-limit jobs read."""
        return tuple(
            self.osdf_url("config", name)
            for name in (
                "environment.yaml",
                "data.yaml",
                "search.yaml",
                self.search.spindown_model,
            )
        )
