"""Dotted CLI-style overrides through the compose API.

Overrides are how a sweep varies one knob without editing YAML, so a typo in a
key has to fail at compose time rather than resolving to ``None`` three stages
later.
"""

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from hydra.errors import HydraException  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from evalaware.configs.loader import load_config  # noqa: E402


@pytest.fixture
def compose(fixture_config_dir):
    def _compose(overrides=None):
        return load_config(overrides=overrides or [], config_dir=fixture_config_dir)

    return _compose


class TestScalarOverrides:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 12345])
    def test_integer_override(self, compose, seed):
        assert compose([f"run.seed={seed}"]).run.seed == seed

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
    def test_nested_integer_override(self, compose, batch_size):
        assert compose([f"model.batch_size={batch_size}"]).model.batch_size == batch_size

    @pytest.mark.parametrize("value", ["gpt2", "Qwen/Qwen2.5-0.5B-Instruct", "sshleifer/tiny-gpt2"])
    def test_string_override_with_slashes(self, compose, value):
        assert compose([f"model.name={value}"]).model.name == value

    @pytest.mark.parametrize(("flag", "expected"), [("true", True), ("false", False)])
    def test_boolean_override(self, compose, flag, expected):
        assert compose([f"run.deterministic={flag}"]).run.deterministic is expected

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_float_override(self, compose, temperature):
        assert compose([f"model.temperature={temperature}"]).model.temperature == temperature

    def test_types_are_preserved_not_stringified(self, compose):
        cfg = compose(["run.seed=7"])
        assert isinstance(cfg.run.seed, int)
        assert not isinstance(cfg.run.seed, str)


class TestMultipleOverrides:
    def test_several_overrides_apply_together(self, compose):
        cfg = compose(["run.seed=3", "model.batch_size=16", "data.n_items=128"])
        assert cfg.run.seed == 3
        assert cfg.model.batch_size == 16
        assert cfg.data.n_items == 128

    def test_a_group_switch_composes_with_a_dotted_override(self, compose):
        cfg = compose(["experiment=beta", "run.seed=9"])
        assert cfg.experiment.name == "beta"
        assert cfg.run.seed == 9

    def test_the_last_override_of_a_key_wins(self, compose):
        assert compose(["run.seed=1", "run.seed=2"]).run.seed == 2

    def test_overrides_do_not_leak_between_composes(self, compose):
        compose(["run.seed=99"])
        assert compose().run.seed == 0

    def test_empty_override_list_is_the_default_config(self, compose):
        assert compose([]).run.seed == compose(None).run.seed


class TestRejectedOverrides:
    @pytest.mark.parametrize(
        "override", ["model.dtpye=float16", "run.sed=1", "nonexistent.key=1"]
    )
    def test_misspelled_keys_are_rejected_at_compose_time(self, compose, override):
        with pytest.raises(Exception) as excinfo:
            compose([override])
        assert not isinstance(excinfo.value, AssertionError)

    def test_an_unknown_config_group_is_rejected(self, compose):
        with pytest.raises((HydraException, Exception)):
            compose(["experiment=not_a_real_preset"])

    def test_a_malformed_override_is_rejected(self, compose):
        with pytest.raises(Exception):
            compose(["this is not an override"])


class TestOverrideEffectOnSerialization:
    def test_overridden_values_appear_in_the_resolved_yaml(self, compose):
        text = OmegaConf.to_yaml(compose(["run.seed=4321"]), resolve=True)
        assert "4321" in text

    def test_resolved_yaml_reloads(self, compose, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text(OmegaConf.to_yaml(compose(["run.seed=5"]), resolve=True), "utf-8")
        assert OmegaConf.load(path).run.seed == 5

    @pytest.mark.parametrize("seed", [0, 11, 222])
    def test_a_sweep_over_seeds_produces_distinct_configs(self, compose, seed):
        assert compose([f"run.seed={seed}"]).run.seed == seed
