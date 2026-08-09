"""Programmatic Hydra config loading.

Loading through the compose API rather than only through an ``@hydra.main``
decorator is the choice that makes configs testable: a test can compose any
preset with any overrides and assert on the result, and library code can load a
config without pretending to be a CLI entry point.

Every experiment preset in ``configs/experiment/`` inherits from a
``base_experiment`` config that genuinely exists, and ``test_config_composition``
composes every preset to prove it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .schema import Config

__all__ = [
    "CONFIG_DIR",
    "available_presets",
    "config_to_dict",
    "load_config",
    "register_schema",
    "save_resolved_config",
]

# configs/ sits at the repo root, four parents up from this module.
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

_STORE_NAME = "base_schema"


def register_schema() -> None:
    """Register the typed schema so Hydra validates composition against it."""
    store = ConfigStore.instance()
    store.store(name=_STORE_NAME, node=Config)


def load_config(
    config_name: str = "config",
    overrides: list[str] | None = None,
    *,
    config_dir: Path | None = None,
) -> DictConfig:
    """Compose a config and return the resolved ``DictConfig``.

    Args:
        config_name: Base config name, without the ``.yaml`` suffix.
        overrides: CLI-style overrides, e.g.
            ``["experiment=ablation_controls", "model.batch_size=8", "run.seed=7"]``.
        config_dir: Override the config directory. Tests use this to compose
            from a fixture tree.

    Returns:
        The resolved configuration.

    Raises:
        FileNotFoundError: If the config directory does not exist, which almost
            always means the caller is running from the wrong working directory.
    """
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    if not directory.is_dir():
        raise FileNotFoundError(
            f"config directory not found: {directory}. Run from the repository root, "
            f"or pass config_dir explicitly."
        )
    register_schema()
    # Hydra refuses to initialize twice in one process, which happens constantly
    # inside a test session composing many presets.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides or []))


def config_to_dict(cfg: DictConfig, *, resolve: bool = True) -> dict[str, Any]:
    """Convert a config to a plain dict for logging and serialization."""
    return OmegaConf.to_container(cfg, resolve=resolve)  # type: ignore[return-value]


def save_resolved_config(cfg: DictConfig, path: str | Path) -> Path:
    """Write the fully resolved config next to a run's metadata.

    The resolved copy is what makes a run reconstructible: interpolations are
    expanded, so the file records what actually ran rather than what the
    template said.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return target


def available_presets(*, config_dir: Path | None = None) -> list[str]:
    """Names of every experiment preset, excluding the shared base."""
    directory = (Path(config_dir) if config_dir else CONFIG_DIR) / "experiment"
    if not directory.is_dir():
        return []
    return sorted(
        path.stem for path in directory.glob("*.yaml") if path.stem != "base_experiment"
    )
