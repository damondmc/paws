import copy
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from paws.settings import Environment, Followup, Search, Settings, Spindown, Stage

REAL_CONFIG_DIR = Path(
    os.environ.get("PAWS_CONFIG_DIR", Path(__file__).parents[2] / "config")
)


def load(config_dir, name):
    with open(config_dir / name) as f:
        return yaml.safe_load(f)


def save(config_dir, name, raw):
    (config_dir / name).write_text(yaml.safe_dump(raw))


def stage_fields(config_dir, name):
    stages = Followup.model_validate(load(config_dir, "followup.yaml")).stages
    return copy.deepcopy(stages[name].model_dump())


def test_fixture_config_validates(settings):
    followup_2 = settings.stage("followup-2")
    assert followup_2.n_sky == len(settings.sky_offsets(followup_2)[0])
    assert settings.followup.chains["test"] == ["followup-1", "followup-2"]
    assert set(settings.stages) == set(settings.search.stages) | set(
        settings.followup.stages
    )
    assert settings.stage("search-0").is_search
    assert settings.stage("injections-0").is_injection
    assert not followup_2.is_injection and followup_2.thresholds == "test-t20"


@pytest.mark.skipif(
    not (REAL_CONFIG_DIR / "followup.yaml").exists(), reason="no analysis config dir"
)
def test_real_config_validates():
    Settings(REAL_CONFIG_DIR)


def test_unknown_key_rejected(config_dir):
    raw = load(config_dir, "environment.yaml")
    raw["num_toplist"] = 1000
    with pytest.raises(ValidationError):
        Environment.model_validate(raw)


def test_missing_field_rejected(config_dir):
    raw = load(config_dir, "search.yaml")
    del raw["stages"]["search-0"]["toplist"]
    with pytest.raises(ValidationError):
        Search.model_validate(raw)


def test_missing_prev_rejected(config_dir):
    raw = load(config_dir, "followup.yaml")
    raw["stages"]["followup-2"]["prev"] = "followup-9"
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="followup-9"):
        Settings(config_dir)


def test_initial_search_in_search_yaml_only(config_dir):
    raw = load(config_dir, "search.yaml")
    raw["stages"]["search-0"]["prev"] = "search-0"
    with pytest.raises(ValidationError, match="prev"):
        Search.model_validate(raw)
    raw = load(config_dir, "followup.yaml")
    raw["stages"]["followup-old-1"]["prev"] = None
    with pytest.raises(ValidationError):
        Followup.model_validate(raw)


def test_stage_in_both_files_rejected(config_dir):
    raw = load(config_dir, "search.yaml")
    raw["stages"]["followup-old-1"] = dict(raw["stages"]["search-0"])
    save(config_dir, "search.yaml", raw)
    with pytest.raises(ValueError, match="both"):
        Settings(config_dir)


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


def test_upperlimit_needs_injection(config_dir):
    raw = stage_fields(config_dir, "followup-old-1")
    raw["upperlimit"] = True
    with pytest.raises(ValidationError, match="injection"):
        Stage.model_validate(raw)


def test_new_injections_need_n_inj_and_h0_file(config_dir):
    raw = load(config_dir, "followup.yaml")
    raw["stages"]["injections-0"]["injection"]["h0_file"] = None
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="h0_file"):
        Settings(config_dir)


def test_injection_followup_takes_the_seeds_injections(config_dir):
    raw = load(config_dir, "followup.yaml")
    raw["stages"]["injections-1"]["injection"]["n_inj"] = 5
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="null"):
        Settings(config_dir)


def test_reuse_needs_same_grid(config_dir):
    raw = load(config_dir, "followup.yaml")
    # t10, 1 sky point for a t20, 3-point stage
    raw["stages"]["followup-2"]["reuse"] = ["followup-old-1"]
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="reuse"):
        Settings(config_dir)


def test_chain_stages_exist(config_dir):
    raw = load(config_dir, "followup.yaml")
    raw["chains"]["test"] = ["followup-1", "followup-9"]
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="followup-9"):
        Settings(config_dir)


def test_calibration_injection_stages_exist(config_dir):
    raw = load(config_dir, "followup.yaml")
    raw["calibrations"]["test-t10"]["injections"] = ["injections-0", "injections-9"]
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="injections-9"):
        Settings(config_dir)
    raw["calibrations"]["test-t10"]["injections"] = ["search-0", "injections-1"]
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="not an injection stage"):
        Settings(config_dir)


def test_sky_grid_must_match_n_sky(config_dir):
    raw = load(config_dir, "followup.yaml")
    raw["stages"]["followup-2"]["n_sky"] = 57
    save(config_dir, "followup.yaml", raw)
    with pytest.raises(ValueError, match="n_sky"):
        Settings(config_dir)


def test_no_config_dir():
    with pytest.raises(RuntimeError):
        Settings(None)


@pytest.mark.parametrize("freq", [20, 150, 199, 200, 250, 399])
def test_tau_matches_formula(settings, freq):
    # the knots [[199, 300], [399, 400]] reproduce 300 years below 200 Hz, then 0.5 years more per Hz
    expected = 86400 * 365.25 * (300 if freq < 200 else 300 + (freq - 199) * 0.5)
    assert settings.search.spindown.tau(freq) == expected


def test_spindown_age_knots():
    constant = Spindown(nc_min=2, nc_max=7, age_knots=[[100, 300]])
    assert constant.age(20) == constant.age(399) == 300
    with pytest.raises(ValidationError, match="increasing"):
        Spindown(nc_min=2, nc_max=7, age_knots=[[200, 300], [100, 400]])


def test_taskname(settings):
    assert (
        settings.stage("injections-2").taskname(settings.target, 100)
        == "TestTarget_injections-2_TCoh20_O2_100Hz"
    )


def test_osdf_urls(settings):
    osdf = str(settings.environment.osdf_dir)[len("/osdf") :]
    assert (
        settings.metric_url(settings.stage("search-0"))
        == f"osdf://{osdf}/metricSetup/test_t5.fts"
    )
    assert settings.job_config_urls() == tuple(
        f"osdf://{osdf}/config/{name}"
        for name in ("environment.yaml", "data.yaml", "search.yaml")
    )
