"""Collect self-describing result JSONs into one aggregate payload."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..utils.git import git_sha
from ..utils.io import save_json

__all__ = [
    "AggregatePayload",
    "REQUIRED_FIELDS",
    "aggregate_results",
    "flatten_metrics",
    "write_aggregate",
]

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("task", "seed", "git_sha")


@dataclass
class AggregatePayload:
    """Measured rows kept separate from synthetic harness-validation rows."""

    measured: list[dict[str, Any]] = field(default_factory=list)
    synthetic: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    git_sha: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "n_measured": len(self.measured),
            "n_synthetic": len(self.synthetic),
            "warnings": list(self.warnings),
            "measured": list(self.measured),
            "synthetic": list(self.synthetic),
        }


def flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dicts into dotted numeric keys."""
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: expected a JSON object"
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return None, f"{path}: missing required fields {missing}"
    return payload, None


def aggregate_results(
    search_dirs: Iterable[Path],
    *,
    pattern: str = "*.json",
    include_synthetic: bool = False,
) -> AggregatePayload:
    """Walk search dirs, keep only provenance-complete result payloads."""
    out = AggregatePayload(git_sha=git_sha())
    for root in search_dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob(pattern)):
            if path.name in {"run_metadata.json", "config.yaml"}:
                continue
            if path.name == "results.json" and path.parent.name == "results":
                # skip prior aggregates
                if "measured" in json.loads(path.read_text(encoding="utf-8") or "{}"):
                    continue
            payload, err = _load(path)
            if err:
                out.warnings.append(err)
                logger.warning("skipping candidate: %s", err)
                continue
            assert payload is not None
            row = {
                "path": str(path),
                "task": payload["task"],
                "seed": payload["seed"],
                "git_sha": payload["git_sha"],
                "is_synthetic": bool(payload.get("is_synthetic", False)),
                "n": payload.get("n"),
                "metrics": flatten_metrics(
                    {k: v for k, v in payload.items() if k not in {
                        "task", "seed", "git_sha", "git_dirty", "profile", "model",
                        "created_at", "n", "n_dropped", "dropped_reason", "is_synthetic",
                        "notes", "inputs", "path",
                    }}
                ),
                "raw": payload,
            }
            if row["is_synthetic"]:
                out.synthetic.append(row)
                if include_synthetic:
                    out.measured.append(row)
            else:
                out.measured.append(row)
    return out


def write_aggregate(payload: AggregatePayload, dest: Path) -> Path:
    """Atomic write of the aggregate payload."""
    save_json(dest, payload.to_dict())
    return dest
