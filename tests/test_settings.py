import copy
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from paws.settings import Config, Settings, Stage, Stages, Target

CONFIG_DIR = Path(os.environ.get("PAWS_CONFIG_DIR", Path(__file__).parents[2] / "config"))
needs_config = pytest.mark.skipif(not (CONFIG_DIR / "stages.yaml").exists(), reason="no config dir")


def load(name):
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


@needs_config
def test_real_files_validate():
    s = Settings(CONFIG_DIR)
    assert s.stage("followup-v3-2").n_sky == len(s.sky_offsets(s.stage("followup-v3-2"))[0])


@needs_config
def test_unknown_config_key_rejected():
    raw = load("config.yaml")
    raw["num_toplist"] = 1000
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


@needs_config
def test_missing_field_rejected():
    raw = load("stages.yaml")
    del raw["stages"]["search-0"]["toplist"]
    with pytest.raises(ValidationError):
        Stages.model_validate(raw)


@needs_config
def test_missing_prev_rejected():
    raw = load("stages.yaml")
    raw["stages"]["followup-v3-2"]["prev"] = "followup-v9-9"
    with pytest.raises(ValidationError):
        Stages.model_validate(raw)


@needs_config
def test_followup_needs_ratio_cut():
    raw = copy.deepcopy(Stages.model_validate(load("stages.yaml")).stages["followup-v3-2"].model_dump())
    raw["ratio_cut"] = None
    with pytest.raises(ValidationError):
        Stage.model_validate(raw)


@needs_config
@pytest.mark.parametrize("freq", [20, 150, 199, 200, 250, 399])
def test_tau_matches_old_formula(freq):
    old = 86400 * 365.25 * (300 if freq < 200 else 300 + (freq - 199) * 0.5)
    assert Target.model_validate(load("gal.yaml")).tau(freq) == old


@needs_config
def test_taskname():
    s = Settings(CONFIG_DIR)
    assert s.stage("injections-v2-3").taskname(s.target, 100) == "GalacticCenter_injections-v2-3_TCoh40_O3_100Hz"


@needs_config
def test_osdf_urls():
    s = Settings(CONFIG_DIR)
    assert s.metric_url(s.stage("search-0")) == (
        "osdf:///igwn/cit/staging/hoitim.cheung/metricSetup/Start1368970000_TCoh432000_N107_Spin2.fts"
    )
    assert s.job_config_urls()[1] == "osdf:///igwn/cit/staging/hoitim.cheung/config/gal.yaml"
