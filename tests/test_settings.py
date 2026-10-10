import copy
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from paws.settings import Config, Settings, Stage, Stages, Target

REAL_CONFIG_DIR = Path(os.environ.get("PAWS_CONFIG_DIR", Path(__file__).parents[2] / "config"))


def load(config_dir, name):
    with open(config_dir / name) as f:
        return yaml.safe_load(f)


def stage_fields(config_dir, name):
    return copy.deepcopy(Stages.model_validate(load(config_dir, "stages.yaml")).stages[name].model_dump())


def test_fixture_config_validates(settings):
    assert settings.stage("followup-2").n_sky == len(settings.sky_offsets(settings.stage("followup-2"))[0])
    assert settings.stages_file.chains["test"] == ["followup-1", "followup-2"]


@pytest.mark.skipif(not (REAL_CONFIG_DIR / "stages.yaml").exists(), reason="no analysis config dir")
def test_real_config_validates():
    Settings(REAL_CONFIG_DIR)


def test_unknown_config_key_rejected(config_dir):
    raw = load(config_dir, "config.yaml")
    raw["num_toplist"] = 1000
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_missing_field_rejected(config_dir):
    raw = load(config_dir, "stages.yaml")
    del raw["stages"]["search-0"]["toplist"]
    with pytest.raises(ValidationError):
        Stages.model_validate(raw)


def test_missing_prev_rejected(config_dir):
    raw = load(config_dir, "stages.yaml")
    raw["stages"]["followup-2"]["prev"] = "followup-9"
    with pytest.raises(ValidationError):
        Stages.model_validate(raw)


def test_thresholded_followup_keeps_loudest_only(config_dir):
    raw = stage_fields(config_dir, "followup-2")
    raw["keep"] = 10
    with pytest.raises(ValidationError):
        Stage.model_validate(raw)


def test_injection_stage_cannot_reuse(config_dir):
    raw = stage_fields(config_dir, "injections-1")
    Stage.model_validate(raw)
    raw["reuse"] = ["followup-old-1"]
    with pytest.raises(ValidationError):
        Stage.model_validate(raw)


def test_reuse_needs_same_grid(config_dir):
    raw = load(config_dir, "stages.yaml")
    raw["stages"]["followup-2"]["reuse"] = ["followup-old-1"]  # t10, 1 sky point for a t20, 3-point stage
    with pytest.raises(ValidationError):
        Stages.model_validate(raw)


def test_chain_stages_exist(config_dir):
    raw = load(config_dir, "stages.yaml")
    raw["chains"]["test"] = ["followup-1", "followup-9"]
    with pytest.raises(ValidationError):
        Stages.model_validate(raw)


def test_sky_grid_must_match_n_sky(config_dir):
    raw = load(config_dir, "stages.yaml")
    raw["stages"]["followup-2"]["n_sky"] = 57
    (config_dir / "stages.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        Settings(config_dir)


def test_no_config_dir():
    with pytest.raises(RuntimeError):
        Settings(None)


@pytest.mark.parametrize("freq", [20, 150, 199, 200, 250, 399])
def test_tau_matches_formula(config_dir, freq):
    expected = 86400 * 365.25 * (300 if freq < 200 else 300 + (freq - 199) * 0.5)
    assert Target.model_validate(load(config_dir, "target.yaml")).tau(freq) == expected


def test_taskname(settings):
    assert settings.stage("injections-2").taskname(settings.target, 100) == "TestTarget_injections-2_TCoh20_O2_100Hz"


def test_osdf_urls(settings):
    osdf = str(settings.config.osdf_dir)[len("/osdf"):]
    assert settings.metric_url(settings.stage("search-0")) == f"osdf://{osdf}/metricSetup/test_t5.fts"
    assert settings.job_config_urls()[1] == f"osdf://{osdf}/config/target.yaml"
