"""Tiny checked-in dataset manifests.

A manifest records how a dataset was built (seed, n_items, version, split
fractions) without shipping the rows. Downstream stages refuse to mix artifacts
keyed to different manifest versions, which is what stops a cache from silently
pairing activations from one draw with labels from another.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.io import save_json
from ..utils.validation import require_positive

__all__ = ["DatasetManifest", "ItemRecord", "load_manifest", "write_manifest"]


@dataclass(frozen=True)
class ItemRecord:
    """One row identity in a manifest (no payload)."""

    id: str
    split: str = "train"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    """Self-describing dataset provenance, small enough to commit."""

    name: str
    version: str
    n_items: int
    seed: int
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    is_synthetic: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: list[str] = field(default_factory=list)
    items: list[ItemRecord] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_positive(self.n_items, "n_items")
        if not self.name:
            raise ValueError("manifest.name must be non-empty")
        if not self.version:
            raise ValueError("manifest.version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def save(self, path: str | Path) -> Path:
        """Atomic write; alias kept for test / script compatibility."""
        return write_manifest(path, self)

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        return load_manifest(path)


def write_manifest(path: str | Path, manifest: DatasetManifest) -> Path:
    """Atomic write of a manifest JSON."""
    return save_json(path, manifest.to_dict())


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load and validate a manifest from disk.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required fields are missing or invalid.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target} is not valid JSON: {exc.msg}") from exc
    required = ("name", "version", "n_items", "seed")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{target} missing required fields {missing}")
    items = [
        ItemRecord(
            id=str(row["id"]),
            split=str(row.get("split", "train")),
            meta=dict(row.get("meta", {})),
        )
        for row in payload.get("items", [])
        if isinstance(row, dict) and "id" in row
    ]
    return DatasetManifest(
        name=str(payload["name"]),
        version=str(payload["version"]),
        n_items=int(payload["n_items"]),
        seed=int(payload["seed"]),
        train_frac=float(payload.get("train_frac", 0.6)),
        val_frac=float(payload.get("val_frac", 0.2)),
        test_frac=float(payload.get("test_frac", 0.2)),
        is_synthetic=bool(payload.get("is_synthetic", True)),
        created_at=str(payload.get("created_at", "")),
        notes=list(payload.get("notes", [])),
        items=items,
        extra=dict(payload.get("extra", {})),
    )
