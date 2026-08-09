"""Logging setup for library code and per-run log files.

The rule this module enforces is that library code logs and entry-point scripts
print. A library that prints cannot be silenced, cannot be redirected, and
cannot tell the reader which module spoke. Every module here therefore holds a
``logging.getLogger(__name__)`` and nothing else.

The second job is the per-run log file. A run that produced a surprising number
is usually investigated days later, when the terminal scrollback is gone. If
the log lives next to ``run_metadata.json`` and ``results.json`` in the run
directory, the evidence stays attached to the numbers it explains.

``configure_logging`` is idempotent. Reconfiguring in a notebook or a test
session otherwise stacks handlers and every line appears three times, which
looks like a loop bug in the code being debugged.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

__all__ = ["LOG_FILE_NAME", "configure_logging", "get_logger", "set_level"]

LOG_FILE_NAME = "run.log"

_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_MANAGED = "_managed_by_configure_logging"

logger = logging.getLogger(__name__)


def _setting(cfg: Any, key: str, default: Any) -> Any:
    """Read one field from a ``LoggingConfig``, a DictConfig, or a plain mapping.

    Config objects arrive in three shapes depending on whether the caller came
    through Hydra, constructed a dataclass, or passed a literal dict in a test.
    Accepting all three keeps callers from having to convert first.
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_level(level: Any) -> int:
    """Turn a level name or number into a logging level integer.

    Raises:
        ValueError: If the name is not a standard level. Silently defaulting an
            unrecognised level to INFO hides a typo that suppresses debug output
            for the rest of the project.
    """
    if isinstance(level, int):
        return level
    name = str(level).upper()
    if name not in _LEVELS:
        raise ValueError(
            f"logging.level must be one of {list(_LEVELS)}, got {level!r}; "
            "fix the value in the logging config group"
        )
    return getattr(logging, name)


def _clear_managed_handlers(target: logging.Logger) -> None:
    for handler in list(target.handlers):
        if getattr(handler, _MANAGED, False):
            target.removeHandler(handler)
            handler.close()


def configure_logging(
    cfg: Any = None,
    *,
    run_dir: str | Path | None = None,
    force: bool = True,
) -> logging.Logger:
    """Install a stderr handler and, when a run directory is given, a file handler.

    Args:
        cfg: A ``LoggingConfig`` (or anything exposing ``level``, ``format``,
            and ``log_to_file``). Omitted in tests, where defaults are fine.
        run_dir: Directory of the current run. When set and ``log_to_file`` is
            enabled, a ``run.log`` is written there at DEBUG level regardless of
            the console level, because the expensive detail is exactly what you
            want on disk and not on screen.
        force: Replace handlers this function installed previously. Turning it
            off lets an embedding application keep its own configuration.

    Returns:
        The root logger, already configured.

    Raises:
        ValueError: If the configured level name is not a standard level.
    """
    level = _normalize_level(_setting(cfg, "level", "INFO"))
    fmt = str(_setting(cfg, "format", _DEFAULT_FORMAT))
    log_to_file = bool(_setting(cfg, "log_to_file", True))

    root = logging.getLogger()
    if force:
        _clear_managed_handlers(root)
    # The root logger must sit at the more permissive of the two levels or the
    # file handler never sees the DEBUG records it exists to capture.
    root.setLevel(min(level, logging.DEBUG) if (log_to_file and run_dir) else level)

    formatter = logging.Formatter(fmt)
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(level)
    setattr(console, _MANAGED, True)
    root.addHandler(console)

    if run_dir is not None and log_to_file:
        path = Path(run_dir) / LOG_FILE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        setattr(file_handler, _MANAGED, True)
        root.addHandler(file_handler)
        logger.debug("run log opened at %s", path)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return the module logger. Present so callers import one name, not two."""
    return logging.getLogger(name)


def set_level(level: str | int, *, name: str | None = None) -> None:
    """Change the level of one logger, or of everything when ``name`` is None.

    Useful for turning a single noisy subsystem up to DEBUG without drowning in
    tokenizer and HTTP client chatter from libraries.
    """
    logging.getLogger(name).setLevel(_normalize_level(level))
