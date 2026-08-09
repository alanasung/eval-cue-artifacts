"""Typed configuration: the schema, the Hydra loader, and the preset catalogue.

The schema is the contract, the loader composes against it so a misspelled key
fails at compose time, and the preset registry declares which experiment configs
are supposed to exist so one cannot vanish unnoticed.
"""

from __future__ import annotations

from .loader import (
    CONFIG_DIR,
    available_presets,
    config_to_dict,
    load_config,
    register_schema,
    save_resolved_config,
)
from .presets import (
    PRESET_REGISTRY,
    PresetSpec,
    PresetTier,
    expected_task_id,
    get_preset,
    list_presets,
    validate_preset_name,
)
from .schema import (
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

__all__ = [
    "CONFIG_DIR",
    "PRESET_REGISTRY",
    "AblationConfig",
    "CacheConfig",
    "Config",
    "DType",
    "DataConfig",
    "DeviceKind",
    "EvalConfig",
    "ExperimentConfig",
    "FigureConfig",
    "LoggingConfig",
    "ModelConfig",
    "PathsConfig",
    "Precision",
    "PresetSpec",
    "PresetTier",
    "Profile",
    "ReportingConfig",
    "RunConfig",
    "SweepConfig",
    "available_presets",
    "config_to_dict",
    "expected_task_id",
    "get_preset",
    "list_presets",
    "load_config",
    "register_schema",
    "save_resolved_config",
    "validate_preset_name",
]
