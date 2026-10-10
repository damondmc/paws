import shutil
from pathlib import Path

import pytest
import yaml

from paws.filepaths import PathManager
from paws.settings import Settings

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def config_dir(tmp_path):
    """Copy of tests/data with home_dir, osdf_dir and sft_dir pointing into tmp_path."""
    config_dir = tmp_path / "config"
    shutil.copytree(DATA_DIR, config_dir)
    config_path = config_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["home_dir"] = f"{tmp_path / 'home'}/"
    config["osdf_dir"] = f"{tmp_path / 'osdf'}/"
    config["sft_dir"] = str(tmp_path / "sfts")
    config_path.write_text(yaml.safe_dump(config))
    return config_dir


@pytest.fixture
def settings(config_dir):
    return Settings(config_dir)


@pytest.fixture
def paths(settings):
    return PathManager(settings.config, settings.target)
