"""The typed config schema and its validation rules."""

import dataclasses

import pytest

from evalaware.configs.schema import (
    AblationConfig,
    CacheConfig,
    Config,
    DataConfig,
    DeviceKind,
    DType,
    EvalConfig,
    ExperimentConfig,
    FigureConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    Precision,
    Profile,
    ReportingConfig,
    RunConfig,
    SweepConfig,
)

SECTIONS = [
    ("paths", PathsConfig),
    ("model", ModelConfig),
    ("data", DataConfig),
    ("cache", CacheConfig),
    ("eval", EvalConfig),
    ("sweep", SweepConfig),
    ("ablation", AblationConfig),
    ("reporting", ReportingConfig),
    ("logging", LoggingConfig),
    ("experiment", ExperimentConfig),
    ("run", RunConfig),
]

ENUMS = [Profile, DeviceKind, DType, Precision]


class TestRootConfig:
    def test_constructs_with_no_arguments(self):
        assert isinstance(Config(), Config)

    @pytest.mark.parametrize(("name", "kind"), SECTIONS)
    def test_every_section_is_present_and_typed(self, name, kind):
        assert isinstance(getattr(Config(), name), kind)

    def test_the_standard_asks_for_at_least_ten_nested_dataclasses(self):
        assert len(SECTIONS) >= 10

    def test_sections_are_not_shared_between_instances(self):
        first, second = Config(), Config()
        first.data.n_items = 999
        assert second.data.n_items != 999

    def test_every_section_is_a_dataclass(self):
        for name, _ in SECTIONS:
            assert dataclasses.is_dataclass(getattr(Config(), name))

    def test_asdict_round_trips_the_whole_tree(self):
        payload = dataclasses.asdict(Config())
        assert {name for name, _ in SECTIONS} <= set(payload)


class TestEnums:
    @pytest.mark.parametrize("enum", ENUMS)
    def test_enums_are_string_valued(self, enum):
        for member in enum:
            assert isinstance(member.value, str)
            assert member == member.value

    @pytest.mark.parametrize("enum", ENUMS)
    def test_enums_are_non_empty(self, enum):
        assert len(list(enum)) >= 2

    def test_profile_members(self):
        assert {p.value for p in Profile} == {"pilot", "full"}

    def test_device_members(self):
        assert {d.value for d in DeviceKind} == {"auto", "cpu", "mps", "cuda"}

    def test_dtype_members(self):
        assert {d.value for d in DType} == {"float32", "float16", "bfloat16"}

    def test_precision_members(self):
        assert {p.value for p in Precision} == {"exact", "mixed", "fast"}

    @pytest.mark.parametrize("value", ["pilot", "full"])
    def test_profile_is_constructible_from_its_string(self, value):
        assert Profile(value).value == value

    def test_unknown_enum_value_is_rejected(self):
        with pytest.raises(ValueError):
            DType("float8")


class TestModelConfig:
    def test_defaults_are_conservative(self):
        model = ModelConfig()
        assert model.trust_remote_code is False
        assert model.dtype is DType.FLOAT32
        assert model.device is DeviceKind.AUTO

    @pytest.mark.parametrize("batch_size", [0, -1, -8])
    def test_batch_size_must_be_positive(self, batch_size):
        with pytest.raises(ValueError, match=r"model\.batch_size must be >= 1"):
            ModelConfig(batch_size=batch_size)

    def test_batch_size_error_names_the_value(self):
        with pytest.raises(ValueError, match="got -3"):
            ModelConfig(batch_size=-3)

    @pytest.mark.parametrize("max_new_tokens", [0, -1])
    def test_max_new_tokens_must_be_positive(self, max_new_tokens):
        with pytest.raises(ValueError, match=r"model\.max_new_tokens must be >= 1"):
            ModelConfig(max_new_tokens=max_new_tokens)

    @pytest.mark.parametrize("temperature", [-0.1, 2.1, 100.0])
    def test_temperature_is_bounded(self, temperature):
        with pytest.raises(ValueError, match=r"model\.temperature must be in \[0, 2\]"):
            ModelConfig(temperature=temperature)

    @pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0, 2.0])
    def test_temperature_accepts_the_valid_range(self, temperature):
        assert ModelConfig(temperature=temperature).temperature == temperature

    @pytest.mark.parametrize("batch_size", [1, 4, 64])
    def test_valid_batch_sizes_are_accepted(self, batch_size):
        assert ModelConfig(batch_size=batch_size).batch_size == batch_size


class TestDataConfig:
    def test_default_split_sums_to_one(self):
        data = DataConfig()
        assert data.train_frac + data.val_frac + data.test_frac == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("train", "val", "test"),
        [(0.5, 0.2, 0.2), (0.8, 0.2, 0.2), (0.0, 0.0, 0.0), (0.34, 0.34, 0.34)],
    )
    def test_splits_that_do_not_sum_to_one_are_rejected(self, train, val, test):
        with pytest.raises(ValueError, match=r"split fractions must sum to 1\.0"):
            DataConfig(train_frac=train, val_frac=val, test_frac=test)

    def test_split_error_names_every_fraction(self):
        with pytest.raises(ValueError, match=r"train=0\.5.*val=0\.2.*test=0\.2"):
            DataConfig(train_frac=0.5, val_frac=0.2, test_frac=0.2)

    def test_split_error_reports_the_bad_total(self):
        with pytest.raises(ValueError, match="got 0.900000"):
            DataConfig(train_frac=0.5, val_frac=0.2, test_frac=0.2)

    @pytest.mark.parametrize("n_items", [0, -1, -100])
    def test_n_items_must_be_positive(self, n_items):
        with pytest.raises(ValueError, match=r"data\.n_items must be >= 1"):
            DataConfig(n_items=n_items)

    @pytest.mark.parametrize(
        ("train", "val", "test"), [(0.6, 0.2, 0.2), (0.8, 0.1, 0.1), (1.0, 0.0, 0.0)]
    )
    def test_valid_splits_are_accepted(self, train, val, test):
        assert DataConfig(train_frac=train, val_frac=val, test_frac=test).train_frac == train

    def test_float_slop_within_tolerance_is_accepted(self):
        DataConfig(train_frac=1 / 3, val_frac=1 / 3, test_frac=1 / 3)


class TestEvalConfig:
    def test_defaults_are_usable(self):
        cfg = EvalConfig()
        assert cfg.threshold == 0.5
        assert cfg.bootstrap_samples >= 100

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_must_be_a_strict_probability(self, alpha):
        with pytest.raises(ValueError, match=r"eval\.bootstrap_alpha must be in \(0, 1\)"):
            EvalConfig(bootstrap_alpha=alpha)

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.5])
    def test_valid_alphas_are_accepted(self, alpha):
        assert EvalConfig(bootstrap_alpha=alpha).bootstrap_alpha == alpha

    @pytest.mark.parametrize("samples", [0, 1, 99])
    def test_too_few_bootstrap_samples_are_rejected(self, samples):
        with pytest.raises(ValueError, match=r"eval\.bootstrap_samples must be >= 100"):
            EvalConfig(bootstrap_samples=samples)

    def test_bootstrap_error_explains_why(self):
        with pytest.raises(ValueError, match="for a usable interval"):
            EvalConfig(bootstrap_samples=10)

    def test_layers_default_is_not_shared(self):
        first, second = EvalConfig(), EvalConfig()
        first.layers.append(99)
        assert 99 not in second.layers

    def test_stratify_by_defaults_to_empty(self):
        assert EvalConfig().stratify_by == []


class TestSweepAndAblation:
    @pytest.mark.parametrize("max_cells", [0, -1])
    def test_max_cells_must_be_positive(self, max_cells):
        with pytest.raises(ValueError, match=r"sweep\.max_cells must be >= 1"):
            SweepConfig(max_cells=max_cells)

    def test_sweep_is_off_by_default(self):
        assert SweepConfig().enabled is False

    def test_sweep_seeds_default_to_three(self):
        assert SweepConfig().seeds == [0, 1, 2]

    def test_sweep_grid_defaults_to_empty(self):
        assert SweepConfig().grid == {}

    def test_ablation_arms_default_to_empty(self):
        cfg = AblationConfig()
        assert cfg.enabled == []
        assert cfg.control_arms == []

    def test_ablation_defaults_to_multiple_seeds(self):
        assert AblationConfig().n_seeds >= 2


class TestReportingAndLogging:
    def test_figure_formats_cover_pdf_svg_and_png(self):
        assert set(FigureConfig().formats) >= {"pdf", "svg", "png"}

    def test_figure_dpi_is_publication_grade(self):
        assert FigureConfig().dpi >= 300

    def test_figure_palette_is_colorblind_safe_by_default(self):
        assert FigureConfig().palette == "colorblind"

    def test_reporting_emits_both_formats_by_default(self):
        reporting = ReportingConfig()
        assert reporting.latex is True
        assert reporting.markdown is True

    def test_reporting_nests_a_figure_config(self):
        assert isinstance(ReportingConfig().figures, FigureConfig)

    def test_logging_defaults_to_info(self):
        assert LoggingConfig().level == "INFO"

    def test_logging_format_carries_the_logger_name(self):
        assert "%(name)s" in LoggingConfig().format


class TestRunAndExperiment:
    def test_run_defaults_to_the_pilot_profile(self):
        assert RunConfig().profile is Profile.PILOT

    def test_run_defaults_to_deterministic_and_resumable(self):
        run = RunConfig()
        assert run.deterministic is True
        assert run.resume is True

    def test_provenance_is_saved_by_default(self):
        run = RunConfig()
        assert run.save_git_sha is True
        assert run.save_resolved_config is True

    def test_unsupported_hardware_is_refused_by_default(self):
        assert RunConfig().allow_unsupported_hardware is False

    def test_experiment_carries_a_task_id(self):
        assert ExperimentConfig().task_id

    def test_experiment_lists_default_to_empty(self):
        experiment = ExperimentConfig()
        assert experiment.stages == []
        assert experiment.tags == []


class TestPathsConfig:
    @pytest.mark.parametrize(
        "field", ["root", "data", "cache", "runs", "results", "figures", "tables"]
    )
    def test_every_output_directory_is_named(self, field):
        assert getattr(PathsConfig(), field)

    def test_figures_and_tables_live_under_results(self):
        paths = PathsConfig()
        assert paths.figures.startswith(paths.results)
        assert paths.tables.startswith(paths.results)
