"""Shared helpers for the eval-awareness domain (v2 spine)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.git import git_sha
from ..utils.io import ensure_dir


def result_dict(
    *,
    task: str,
    seed: int,
    n: int,
    git_sha_value: str | None = None,
    **metrics: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task": task,
        "seed": int(seed),
        "git_sha": git_sha_value if git_sha_value is not None else git_sha(),
        "n": int(n),
    }
    out.update(metrics)
    return out


def dump_json_text(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> Path:
    return dump_json_text(path, payload)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cfg_seed(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "run", cfg), "seed", 0))


def cfg_n_items(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "data", cfg), "n_items", 240))


def cfg_param(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return default


def ensure_artifacts(run_dir: Path) -> Path:
    return ensure_dir(run_dir / "artifacts")
