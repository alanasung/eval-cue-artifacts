"""Hydra composition for every experiment preset.

This is the test the standard singles out: every preset in
``configs/experiment/`` must inherit a ``base_experiment`` that genuinely
exists, and every one of them must compose. A preset that only loads through a
hand-rolled ``OmegaConf.merge`` in some script is not composable, and the gap
shows up as a broken CLI months later.
"""

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evalaware.configs.loader import (  # noqa: E402
    CONFIG_DIR,
    available_presets,
    config_to_dict,
    load_config,
    register_schema,
    save_resolved_config,
)

PRESETS = available_presets()
NO_PRESETS = [pytest.param(None, marks=pytest.mark.skip(reason="no configs/experiment tree"))]
PRESET_PARAMS = PRESETS or NO_PRESETS

TOP_LEVEL_SECTIONS = [
    "paths",
    "model",
    "data",
    "cache",
    "eval",
    "sweep",
    "ablation",
    "reporting",
    "logging",
    "experiment",
    "run",
]


class TestPresetDiscovery:
    def test_config_dir_is_absolute(self):
        assert CONFIG_DIR.is_absolute()

    def test_config_dir_is_named_configs(self):
        assert CONFIG_DIR.name == "configs"

    def test_the_standard_asks_for_at_least_eight_presets(self, config_root):
        assert len(PRESETS) >= 8, f"only found {PRESETS}"

    def test_the_shared_base_is_excluded_from_the_preset_list(self, config_root):
        assert "base_experiment" not in PRESETS

    def test_the_shared_base_genuinely_exists(self, config_root):
        base = config_root / "experiment" / "base_experiment.yaml"
        assert base.is_file(), (
            "every preset inherits base_experiment; the file must exist rather than "
            "being a dangling defaults-list entry"
        )

    def test_preset_names_are_unique(self, config_root):
        assert len(set(PRESETS)) == len(PRESETS)

    def test_discovery_of_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert available_presets(config_dir=tmp_path) == []


class TestEveryPresetComposes:
    @pytest.mark.parametrize("preset", PRESET_PARAMS)
    def test_preset_resolves(self, preset):
        cfg = load_config(overrides=[f"experiment={preset}"])
        assert isinstance(cfg, DictConfig)

    @pytest.mark.parametrize("preset", PRESET_PARAMS)
    def test_preset_has_every_top_level_section(self, preset):
        cfg = load_config(overrides=[f"experiment={preset}"])
        for section in TOP_LEVEL_SECTIONS:
            assert section in cfg, f"{preset} is missing the {section} section"

    @pytest.mark.parametrize("preset", PRESET_PARAMS)
    def test_preset_names_itself(self, preset):
        cfg = load_config(overrides=[f"experiment={preset}"])
        assert cfg.experiment.name, f"{preset} does not set experiment.name"

    @pytest.mark.parametrize("preset", PRESET_PARAMS)
    def test_preset_carries_a_task_id(self, preset):
        cfg = load_config(overrides=[f"experiment={preset}"])
        assert cfg.experiment.task_id, f"{preset} does not set experiment.task_id"

    @pytest.mark.parametrize("preset", PRESET_PARAMS)
    def test_preset_fully_resolves_with_no_dangling_interpolations(self, preset):
        cfg = load_config(overrides=[f"experiment={preset}"])
        OmegaConf.resolve(cfg)
        assert "${" not in OmegaConf.to_yaml(cfg, resolve=True)

    @pytest.mark.parametrize("preset", PRESET_PARAMS)
    def test_preset_converts_to_a_plain_dict(self, preset):
        payload = config_to_dict(load_config(overrides=[f"experiment={preset}"]))
        assert isinstance(payload, dict)

    @pytest.mark.parametrize("preset", PRESET_PARAMS)
    def test_preset_is_json_serializable(self, preset):
        import json

        json.dumps(config_to_dict(load_config(overrides=[f"experiment={preset}"])), default=str)

    def test_presets_are_distinguishable_from_one_another(self, config_root):
        if len(PRESETS) < 2:
            pytest.skip("need at least two presets to compare")
        names = {
            preset: load_config(overrides=[f"experiment={preset}"]).experiment.name
            for preset in PRESETS
        }
        assert len(set(names.values())) > 1, f"every preset resolves to the same name: {names}"


class TestLoaderMechanics:
    def test_register_schema_is_idempotent(self):
        register_schema()
        register_schema()

    def test_missing_config_dir_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config directory not found"):
            load_config(config_dir=tmp_path / "absent")

    def test_missing_config_dir_message_suggests_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run from the repository root"):
            load_config(config_dir=tmp_path / "absent")

    def test_composing_from_a_fixture_tree(self, fixture_config_dir):
        cfg = load_config(config_dir=fixture_config_dir)
        assert cfg.experiment.name == "alpha"

    def test_base_values_are_inherited_by_the_preset(self, fixture_config_dir):
        cfg = load_config(config_dir=fixture_config_dir)
        assert cfg.experiment.description == "Shared base every preset inherits"

    def test_preset_overrides_beat_the_base(self, fixture_config_dir):
        cfg = load_config(config_dir=fixture_config_dir)
        assert cfg.experiment.task_id == "E01"

    def test_switching_preset_switches_the_values(self, fixture_config_dir):
        cfg = load_config(overrides=["experiment=beta"], config_dir=fixture_config_dir)
        assert cfg.experiment.name == "beta"
        assert cfg.experiment.task_id == "E02"

    def test_two_composes_in_one_process_do_not_collide(self, fixture_config_dir):
        first = load_config(config_dir=fixture_config_dir)
        second = load_config(overrides=["experiment=beta"], config_dir=fixture_config_dir)
        assert first.experiment.name == "alpha"
        assert second.experiment.name == "beta"

    def test_many_composes_in_one_process(self, fixture_config_dir):
        for _ in range(5):
            load_config(config_dir=fixture_config_dir)

    def test_available_presets_reads_the_fixture_tree(self, fixture_config_dir):
        assert available_presets(config_dir=fixture_config_dir) == ["alpha", "beta"]


class TestSaveResolvedConfig:
    def test_writes_a_file(self, fixture_config_dir, tmp_path):
        cfg = load_config(config_dir=fixture_config_dir)
        assert save_resolved_config(cfg, tmp_path / "config.yaml").is_file()

    def test_creates_parent_directories(self, fixture_config_dir, tmp_path):
        cfg = load_config(config_dir=fixture_config_dir)
        assert save_resolved_config(cfg, tmp_path / "run" / "config.yaml").is_file()

    def test_output_reloads_as_yaml(self, fixture_config_dir, tmp_path):
        cfg = load_config(config_dir=fixture_config_dir)
        path = save_resolved_config(cfg, tmp_path / "config.yaml")
        assert OmegaConf.load(path).experiment.name == "alpha"

    def test_output_is_fully_resolved(self, fixture_config_dir, tmp_path):
        cfg = load_config(config_dir=fixture_config_dir)
        path = save_resolved_config(cfg, tmp_path / "config.yaml")
        assert "${" not in path.read_text(encoding="utf-8")

    def test_the_saved_copy_matches_what_ran(self, fixture_config_dir, tmp_path):
        cfg = load_config(overrides=["run.seed=17"], config_dir=fixture_config_dir)
        path = save_resolved_config(cfg, tmp_path / "config.yaml")
        assert OmegaConf.load(path).run.seed == 17


class TestConfigToDict:
    def test_returns_a_plain_dict(self, fixture_config_dir):
        payload = config_to_dict(load_config(config_dir=fixture_config_dir))
        assert type(payload) is dict

    def test_nested_sections_are_plain_dicts(self, fixture_config_dir):
        payload = config_to_dict(load_config(config_dir=fixture_config_dir))
        assert type(payload["experiment"]) is dict

    def test_values_survive(self, fixture_config_dir):
        payload = config_to_dict(load_config(config_dir=fixture_config_dir))
        assert payload["experiment"]["name"] == "alpha"
