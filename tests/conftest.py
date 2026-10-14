from jaxtyping import install_import_hook

# every paws function checks its argument and return types, array dtypes and shapes (tests/typecheck.py)
install_import_hook("paws", "typecheck.checker")

import shutil
from pathlib import Path

import pytest
import yaml

from paws.filepaths import PathManager
from paws.settings import Settings

DATA_DIR = Path(__file__).parent / "data"


def replace_fields(path, **fields):
    raw = yaml.safe_load(path.read_text())
    raw.update(fields)
    path.write_text(yaml.safe_dump(raw))


@pytest.fixture
def config_dir(tmp_path):
    """Copy of tests/data with home_dir, osdf_dir and sft_dir pointing into tmp_path."""
    config_dir = tmp_path / "config"
    shutil.copytree(DATA_DIR, config_dir)
    replace_fields(
        config_dir / "environment.yaml",
        home_dir=f"{tmp_path / 'home'}/",
        osdf_dir=f"{tmp_path / 'osdf'}/",
    )
    replace_fields(config_dir / "data.yaml", sft_dir=str(tmp_path / "sfts"))
    return config_dir


@pytest.fixture
def settings(config_dir):
    return Settings(config_dir)


@pytest.fixture
def paths(settings):
    return PathManager(settings)
