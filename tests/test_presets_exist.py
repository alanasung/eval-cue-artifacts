from pathlib import Path
import pytest

@pytest.fixture
def exp_dir(repo_root):
    return repo_root / "configs" / "experiment"

def test_base_experiment_exists(exp_dir):
    assert (exp_dir / "base_experiment.yaml").is_file()

def test_at_least_eight_presets(exp_dir):
    presets = list(exp_dir.glob("*.yaml"))
    assert len(presets) >= 8
