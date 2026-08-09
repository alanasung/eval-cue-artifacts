"""Typed configuration schema.

OmegaConf gives dotted access and composition; it does not give you a typed
surface or an error when a key is misspelled. These dataclasses are the typed
surface. They are registered with Hydra's ``ConfigStore`` in :mod:`.loader`, so
composition is validated against them and ``cfg.model.dtpye`` fails at compose
time instead of producing ``None`` three stages later.

Closed choice sets are ``str, Enum`` rather than bare strings so that an invalid
device or dtype is rejected at construction, and so the valid options are
discoverable from the type instead of from a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "AblationConfig",
    "CacheConfig",
    "Config",
    "DataConfig",
    "DeviceKind",
    "DType",
    "EvalConfig",
    "ExperimentConfig",
    "FigureConfig",
    "LoggingConfig",
    "ModelConfig",
    "PathsConfig",
    "Precision",
    "Profile",
    "ReportingConfig",
    "RunConfig",
    "SweepConfig",
]


class Profile(str, Enum):
    """Which scale a run is operating at.

    ``PILOT`` must complete on the local machine. ``FULL`` documents the
    scaled-up run and is expected to require a real GPU.
    """

    PILOT = "pilot"
    FULL = "full"


class DeviceKind(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class DType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class Precision(str, Enum):
    """How aggressively to trade numerics for speed."""

    EXACT = "exact"
    MIXED = "mixed"
    FAST = "fast"


@dataclass
class PathsConfig:
    """Every directory the code writes to. Nothing writes outside these."""

    root: str = "."
    data: str = "data"
    cache: str = ".cache/artifacts"
    runs: str = "runs"
    results: str = "results"
    figures: str = "results/figures"
    tables: str = "results/tables"


@dataclass
class ModelConfig:
    """The model under study."""

    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    dtype: DType = DType.FLOAT32
    device: DeviceKind = DeviceKind.AUTO
    batch_size: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    trust_remote_code: bool = False
    attn_implementation: str = "eager"
    role: str = "default"
    use_chat_template: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"model.batch_size must be >= 1, got {self.batch_size}")
        if self.max_new_tokens < 1:
            raise ValueError(f"model.max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"model.temperature must be in [0, 2], got {self.temperature}")


@dataclass
class DataConfig:
    """Dataset construction and splitting."""

    name: str = "synthetic"
    n_items: int = 512
    seed: int = 0
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    shuffle: bool = True
    max_prompt_tokens: int = 512

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"data split fractions must sum to 1.0, got {total:.6f} "
                f"(train={self.train_frac}, val={self.val_frac}, test={self.test_frac})"
            )
        if self.n_items < 1:
            raise ValueError(f"data.n_items must be >= 1, got {self.n_items}")


@dataclass
class CacheConfig:
    """Artifact cache behaviour."""

    enabled: bool = True
    version: str = "v1"
    strict_version: bool = True
    namespace: str = "default"
    overwrite: bool = False


@dataclass
class EvalConfig:
    """Evaluation and the loud-versus-silent failure analysis."""

    threshold: float = 0.5
    tolerance: float = 0.1
    confidence_margin: float = 0.2
    bootstrap_samples: int = 10_000
    bootstrap_alpha: float = 0.05
    layers: list[int] = field(default_factory=lambda: [2, 4, 6])
    stratify_by: list[str] = field(default_factory=list)
    min_stratum_size: int = 20

    def __post_init__(self) -> None:
        if not 0.0 < self.bootstrap_alpha < 1.0:
            raise ValueError(
                f"eval.bootstrap_alpha must be in (0, 1), got {self.bootstrap_alpha}"
            )
        if self.bootstrap_samples < 100:
            raise ValueError(
                f"eval.bootstrap_samples must be >= 100 for a usable interval, "
                f"got {self.bootstrap_samples}"
            )


@dataclass
class SweepConfig:
    """A cross-product sweep, expressed as config rather than a shell loop."""

    enabled: bool = False
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    grid: dict[str, list[Any]] = field(default_factory=dict)
    max_cells: int = 64

    def __post_init__(self) -> None:
        if self.max_cells < 1:
            raise ValueError(f"sweep.max_cells must be >= 1, got {self.max_cells}")


@dataclass
class AblationConfig:
    """Which ablation arms to run. Each maps to a registered ablation module."""

    enabled: list[str] = field(default_factory=list)
    control_arms: list[str] = field(default_factory=list)
    n_seeds: int = 3


@dataclass
class FigureConfig:
    """Publication-grade figure settings."""

    formats: list[str] = field(default_factory=lambda: ["pdf", "svg", "png"])
    dpi: int = 300
    width_in: float = 6.6
    height_in: float = 2.6
    palette: str = "colorblind"
    font_family: str = "serif"


@dataclass
class ReportingConfig:
    """Table and figure generation."""

    figures: FigureConfig = field(default_factory=FigureConfig)
    latex: bool = True
    markdown: bool = True
    caption_from_numbers: bool = True


@dataclass
class LoggingConfig:
    """Structured logging. Library code logs; scripts print."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    log_to_file: bool = True
    log_llm_calls: bool = True


@dataclass
class ExperimentConfig:
    """Identity of the experiment, matching an id in docs/TASK.md."""

    name: str = "base"
    task_id: str = "E00"
    description: str = "Base experiment"
    stages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    """Execution-level settings."""

    profile: Profile = Profile.PILOT
    seed: int = 0
    deterministic: bool = True
    resume: bool = True
    max_memory_gb: float = 12.0
    allow_unsupported_hardware: bool = False
    save_git_sha: bool = True
    save_resolved_config: bool = True


@dataclass
class Config:
    """Root config, composed from the config groups in ``configs/``."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    roles: dict[str, ModelConfig] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    run: RunConfig = field(default_factory=RunConfig)
