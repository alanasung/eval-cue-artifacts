"""Run directories, run metadata, and self-describing result payloads.

The contract this module enforces: **a result file must explain itself**. Given
only ``results.json``, a reader should be able to say which task produced it,
under which seed and commit, over how many examples, from which inputs, and how
many rows were dropped on the way. Numbers that cannot answer those questions
are not reportable.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .git import git_state

__all__ = [
    "ResultPayload",
    "RunMetadata",
    "create_run_dir",
    "save_results",
    "save_run_metadata",
    "utc_now_iso",
    "write_json_atomic",
]

_RUN_DIR_FORMAT = "%Y-%m-%d_%H-%M-%S"
_METADATA_NAME = "run_metadata.json"


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp. Naive timestamps are unorderable across machines."""
    return datetime.now(timezone.utc).isoformat()


def create_run_dir(base: str | Path, *, suffix: str | None = None) -> Path:
    """Create and return a fresh timestamped run directory.

    Collisions are possible when two runs start inside the same second, so a
    numeric disambiguator is appended rather than letting the second run write
    into the first one's directory.

    Args:
        base: Parent directory; created if absent.
        suffix: Optional human-readable tag appended to the timestamp.

    Returns:
        The created directory path.
    """
    base_path = Path(base)
    stamp = datetime.now().strftime(_RUN_DIR_FORMAT)
    name = f"{stamp}_{suffix}" if suffix else stamp
    candidate = base_path / name
    counter = 1
    while candidate.exists():
        candidate = base_path / f"{name}-{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON via temp-file-plus-rename so readers never see a partial file.

    A run killed mid-write otherwise leaves truncated JSON that fails to parse
    days later, usually during aggregation, far from the cause.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_name).replace(target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


@dataclass
class RunMetadata:
    """Everything needed to reconstruct the conditions of a run."""

    script: str
    seed: int
    profile: str
    git_sha: str
    git_branch: str
    git_dirty: bool
    created_at: str = field(default_factory=utc_now_iso)
    hardware: dict[str, Any] = field(default_factory=dict)
    packages: dict[str, str] = field(default_factory=dict)
    determinism: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_run_metadata(
    run_dir: str | Path,
    metadata: Mapping[str, Any],
    *,
    seed: int = 0,
    profile: str = "pilot",
    script: str = "unknown",
) -> Path:
    """Write ``run_metadata.json`` with caller metadata plus git provenance."""
    state = git_state()
    record = RunMetadata(
        script=script,
        seed=seed,
        profile=profile,
        git_sha=state.sha,
        git_branch=state.branch,
        git_dirty=state.dirty,
        extra=dict(metadata),
    )
    return write_json_atomic(Path(run_dir) / _METADATA_NAME, record.to_dict())


@dataclass
class ResultPayload:
    """A self-describing result.

    ``task`` identifies which experiment produced this, matching the id
    convention in ``docs/TASK.md`` so a number can be traced back to the plan
    entry that asked for it.
    """

    task: str
    seed: int
    n: dict[str, int]
    metrics: dict[str, Any]
    profile: str = "pilot"
    model: str | None = None
    inputs: list[str] = field(default_factory=list)
    n_dropped: int = 0
    dropped_reason: str | None = None
    git_sha: str = field(default_factory=lambda: git_state().sha)
    git_dirty: bool = field(default_factory=lambda: git_state().dirty)
    created_at: str = field(default_factory=utc_now_iso)
    is_synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("ResultPayload.task must be a non-empty task id, e.g. 'E01_baseline'")
        if self.n_dropped < 0:
            raise ValueError(f"n_dropped must be non-negative, got {self.n_dropped}")
        if self.n_dropped and not self.dropped_reason:
            raise ValueError(
                f"{self.n_dropped} rows were dropped but dropped_reason is unset; "
                "silent data loss is not permitted in a reported result"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.is_synthetic:
            # Loud, machine-checkable marker. Synthetic harness output must never
            # be mistaken for a measured result during aggregation.
            payload["WARNING"] = "SYNTHETIC DATA - harness validation only, not a measured result"
        return payload


def save_results(path: str | Path, payload: ResultPayload | Mapping[str, Any]) -> Path:
    """Persist a result payload atomically.

    Raises:
        TypeError: If handed a bare mapping missing the provenance fields. Use
            ``ResultPayload`` so provenance cannot be forgotten.
    """
    if isinstance(payload, ResultPayload):
        return write_json_atomic(path, payload.to_dict())
    required = {"task", "seed", "git_sha"}
    missing = required - set(payload)
    if missing:
        raise TypeError(
            f"result payload is missing provenance fields {sorted(missing)}; "
            "construct a ResultPayload instead of a plain dict"
        )
    return write_json_atomic(path, payload)
