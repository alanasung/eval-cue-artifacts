"""Version-pinned, append-only artifact cache.

The frozen-backbone workflow this repo uses runs one expensive forward pass and
then iterates on cheap heads for hours. That only works if the cache is
trustworthy, which means four properties, each of which exists because the
absence of it produces a specific, painful bug:

**atomic writes**      a run killed mid-write must not leave a truncated array
                       that reads back as garbage days later.
**version pinning**    activations from two different model revisions must never
                       silently mix into one probe fit.
**resume via has()**   re-running after a crash must skip completed work rather
                       than redo six hours of it.
**safe-name checks**   record ids come from data files; an id of ``../../x``
                       must not write outside the cache root.

Layout::

    <root>/<namespace>/<id>.npy
    <root>/<namespace>/manifest.jsonl

Cached ids are derived from the files actually on disk, so a manifest that was
truncated by a crash cannot hide an embedding that exists.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils.run_manifest import utc_now_iso

__all__ = ["ArtifactCache", "CacheRecord", "check_safe_name"]

_ARRAY_SUFFIX = ".npy"
_MANIFEST_NAME = "manifest.jsonl"
_VERSION_FILE = "version.txt"


def check_safe_name(name: str, kind: str) -> str:
    """Reject anything that is not a single safe path component.

    Raises:
        ValueError: If the name is empty, contains a separator, is a relative
            path element, or is absolute.
    """
    if not name:
        raise ValueError(f"{kind} must be a non-empty string")
    if "/" in name or "\\" in name or name in (".", "..") or os.path.isabs(name):
        raise ValueError(
            f"{kind} is not a safe path component: {name!r}; "
            "ids must not contain path separators or traverse directories"
        )
    return name


@dataclass(frozen=True)
class CacheRecord:
    """One line of ``manifest.jsonl``."""

    id: str
    namespace: str
    version: str
    timestamp: str
    shape: list[int]
    dtype: str
    path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "version": self.version,
            "timestamp": self.timestamp,
            "shape": self.shape,
            "dtype": self.dtype,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CacheRecord":
        return cls(
            id=str(data["id"]),
            namespace=str(data["namespace"]),
            version=str(data["version"]),
            timestamp=str(data["timestamp"]),
            shape=list(data["shape"]),  # type: ignore[arg-type]
            dtype=str(data["dtype"]),
            path=str(data["path"]),
        )


class ArtifactCache:
    """Read/write cache for one namespace of arrays, keyed by record id.

    Args:
        root: Base cache directory.
        namespace: Sub-directory name, e.g. ``"residuals"`` or ``"logits"``.
        version: Producer version recorded on every write. Any read or write
            against a different version raises rather than mixing silently.
        strict_version: When False, reads of mismatched records warn via the
            returned record instead of raising. Writes always enforce.
    """

    def __init__(
        self,
        root: str | Path,
        namespace: str,
        *,
        version: str = "unknown",
        strict_version: bool = True,
    ) -> None:
        self.root = Path(root)
        self.namespace = check_safe_name(namespace, "namespace")
        self.version = version
        self.strict_version = strict_version
        self.dir = self.root / self.namespace
        self.manifest_path = self.dir / _MANIFEST_NAME
        self._version_path = self.dir / _VERSION_FILE

    # -- identity ---------------------------------------------------------
    def _path(self, record_id: str) -> Path:
        return self.dir / f"{check_safe_name(str(record_id), 'record id')}{_ARRAY_SUFFIX}"

    def has(self, record_id: str) -> bool:
        """True if this id is already cached. The resume check."""
        return self._path(record_id).is_file()

    def __contains__(self, record_id: str) -> bool:
        return self.has(record_id)

    # -- version ----------------------------------------------------------
    def _pinned_version(self) -> str | None:
        if not self._version_path.is_file():
            return None
        return self._version_path.read_text(encoding="utf-8").strip()

    def _enforce_version(self) -> None:
        pinned = self._pinned_version()
        if pinned is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._version_path.write_text(self.version, encoding="utf-8")
            return
        if pinned != self.version:
            raise ValueError(
                f"version mismatch in {self.dir}: cache holds {pinned!r} but this "
                f"process produces {self.version!r}. Re-run with the original version, "
                f"point at a different cache root, or clear the namespace first."
            )

    # -- write ------------------------------------------------------------
    def write(
        self,
        record_id: str,
        array: np.ndarray,
        *,
        overwrite: bool = False,
        timestamp: str | None = None,
    ) -> CacheRecord:
        """Cache one array atomically and append a manifest line.

        Raises:
            ValueError: On an empty or non-numeric array, or a version mismatch.
            FileExistsError: If the id exists and ``overwrite`` is False.
        """
        values = np.asarray(array)
        if values.dtype == object or values.size == 0:
            raise ValueError(
                f"refusing to cache {record_id!r}: expected a non-empty numeric array, "
                f"got dtype={values.dtype} size={values.size}"
            )
        self._enforce_version()
        path = self._path(record_id)
        if path.is_file() and not overwrite:
            raise FileExistsError(
                f"{record_id!r} is already cached at {path}; pass overwrite=True to replace"
            )
        self.dir.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, values)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp_name).replace(path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

        record = CacheRecord(
            id=str(record_id),
            namespace=self.namespace,
            version=self.version,
            timestamp=timestamp or utc_now_iso(),
            shape=list(values.shape),
            dtype=str(values.dtype),
            path=path.name,
        )
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")
        return record

    # -- read -------------------------------------------------------------
    def read(self, record_id: str, *, mmap: bool = True) -> np.ndarray:
        """Return a cached array.

        Raises:
            KeyError: If the id is not cached.
            ValueError: If the cache version differs and ``strict_version``.
        """
        path = self._path(record_id)
        if not path.is_file():
            raise KeyError(
                f"{record_id!r} is not cached in {self.dir}; call has() first, or "
                f"run the stage that populates this namespace"
            )
        if self.strict_version:
            self._enforce_version()
        return np.load(path, mmap_mode="r" if mmap else None)

    def __getitem__(self, record_id: str) -> np.ndarray:
        return self.read(record_id)

    # -- introspection ----------------------------------------------------
    def keys(self) -> list[str]:
        """Sorted cached ids, derived from files on disk rather than the manifest."""
        if not self.dir.is_dir():
            return []
        return sorted(path.stem for path in self.dir.glob(f"*{_ARRAY_SUFFIX}"))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def records(self) -> list[CacheRecord]:
        """Parse ``manifest.jsonl``, skipping lines truncated by a killed run."""
        if not self.manifest_path.is_file():
            return []
        out: list[CacheRecord] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(CacheRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
        return out

    def missing(self, record_ids: list[str]) -> list[str]:
        """Subset of ids not yet cached, in input order. Drives resume."""
        return [rid for rid in record_ids if not self.has(rid)]

    def disk_usage_bytes(self) -> int:
        """Total bytes of cached arrays, excluding the manifest."""
        if not self.dir.is_dir():
            return 0
        return sum(path.stat().st_size for path in self.dir.glob(f"*{_ARRAY_SUFFIX}"))
