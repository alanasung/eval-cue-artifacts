"""Cross-namespace view of the artifact cache.

:class:`~.artifact_cache.ArtifactCache` deliberately knows about exactly one
namespace, because a writer should not be able to touch a namespace it did not
name. Auditing the cache needs the opposite view: every namespace at once.

The question this module exists to answer is version drift. The frozen-backbone
workflow writes activations once and then fits heads against them for days. If
half the residuals were produced by ``v1`` and half by ``v2`` after a model
revision bump, a probe fitted across both is fitting noise from two different
models, and nothing about the resulting accuracy looks wrong. The manifests are
the only record of which producer wrote what, so :meth:`CacheManifest.drift`
reads them all and reports namespaces holding more than one version.

The secondary job is shape and dtype summarisation, which catches the other
silent corruption: a namespace whose rows are half ``(1, 768)`` and half
``(768,)`` because a squeeze was added partway through a run.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import read_jsonl
from .artifact_cache import CacheRecord

__all__ = ["CacheManifest", "NamespaceSummary", "VersionDrift"]

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.jsonl"
_ARRAY_SUFFIX = ".npy"


@dataclass(frozen=True)
class VersionDrift:
    """One namespace found holding artifacts from more than one producer version."""

    namespace: str
    versions: list[str]
    counts: dict[str, int]

    @property
    def dominant_version(self) -> str:
        """The version that wrote the most rows, i.e. the one worth keeping."""
        return max(self.counts, key=lambda key: self.counts[key])

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "versions": list(self.versions),
            "counts": dict(self.counts),
            "dominant_version": self.dominant_version,
        }


@dataclass(frozen=True)
class NamespaceSummary:
    """Aggregate description of one cache namespace."""

    namespace: str
    n_records: int
    n_files: int
    versions: list[str]
    shapes: dict[str, int]
    dtypes: dict[str, int]
    bytes_on_disk: int

    @property
    def is_homogeneous(self) -> bool:
        """True when one version, one shape, and one dtype cover the namespace.

        A heterogeneous namespace is not automatically wrong, but it is never
        safe to concatenate without looking.
        """
        return len(self.versions) <= 1 and len(self.shapes) <= 1 and len(self.dtypes) <= 1

    @property
    def orphan_files(self) -> int:
        """Arrays on disk with no manifest line, i.e. writes lost to a crash."""
        return max(self.n_files - self.n_records, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "n_records": self.n_records,
            "n_files": self.n_files,
            "orphan_files": self.orphan_files,
            "versions": list(self.versions),
            "shapes": dict(self.shapes),
            "dtypes": dict(self.dtypes),
            "bytes_on_disk": self.bytes_on_disk,
            "is_homogeneous": self.is_homogeneous,
        }


class CacheManifest:
    """Read-only aggregator over every ``manifest.jsonl`` under a cache root.

    Args:
        root: The cache base directory, matching ``paths.cache``. A root that
            does not exist yet is treated as an empty cache rather than an
            error, since auditing before the first run is legitimate.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def namespaces(self) -> list[str]:
        """Sorted namespace names, discovered from directories on disk."""
        if not self.root.is_dir():
            return []
        return sorted(path.name for path in self.root.iterdir() if path.is_dir())

    def records(self, namespace: str | None = None) -> list[CacheRecord]:
        """Manifest lines for one namespace, or for all of them.

        Damaged lines are skipped by the JSONL reader, and lines missing a
        required field are skipped here, so a crash-truncated manifest still
        yields everything that was written cleanly.

        Raises:
            ValueError: If a named namespace has no directory, which almost
                always means a typo rather than an empty cache.
        """
        if namespace is not None and namespace not in self.namespaces():
            raise ValueError(
                f"namespace {namespace!r} is not present under {self.root}; "
                f"known namespaces are {self.namespaces()}"
            )
        targets = [namespace] if namespace is not None else self.namespaces()
        out: list[CacheRecord] = []
        for name in targets:
            for row in read_jsonl(self.root / name / _MANIFEST_NAME):
                try:
                    out.append(CacheRecord.from_dict(row))
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "skipping malformed manifest line in namespace %r: keys=%s",
                        name,
                        sorted(row),
                    )
        return out

    def summarize(self, namespace: str | None = None) -> list[NamespaceSummary]:
        """One :class:`NamespaceSummary` per namespace, in name order."""
        targets = [namespace] if namespace is not None else self.namespaces()
        summaries: list[NamespaceSummary] = []
        for name in targets:
            records = self.records(name)
            directory = self.root / name
            files = list(directory.glob(f"*{_ARRAY_SUFFIX}")) if directory.is_dir() else []
            summaries.append(
                NamespaceSummary(
                    namespace=name,
                    n_records=len(records),
                    n_files=len(files),
                    versions=sorted({record.version for record in records}),
                    shapes=dict(Counter("x".join(str(d) for d in r.shape) for r in records)),
                    dtypes=dict(Counter(record.dtype for record in records)),
                    bytes_on_disk=sum(path.stat().st_size for path in files),
                )
            )
        return summaries

    def drift(self) -> list[VersionDrift]:
        """Namespaces holding artifacts from more than one producer version.

        An empty list is the healthy answer. Anything else must be resolved
        before the namespace is read as a single dataset.
        """
        drifted: list[VersionDrift] = []
        for summary in self.summarize():
            if len(summary.versions) <= 1:
                continue
            counts = Counter(record.version for record in self.records(summary.namespace))
            logger.warning(
                "namespace %r holds %d producer versions %s; "
                "reading it as one dataset mixes artifacts from different producers",
                summary.namespace,
                len(summary.versions),
                summary.versions,
            )
            drifted.append(
                VersionDrift(
                    namespace=summary.namespace,
                    versions=summary.versions,
                    counts=dict(counts),
                )
            )
        return drifted

    def require_no_drift(self) -> None:
        """Fail the run rather than fit a head across mixed producer versions.

        Raises:
            ValueError: If any namespace holds more than one version, naming the
                namespaces and the remedy.
        """
        drifted = self.drift()
        if drifted:
            names = [item.namespace for item in drifted]
            raise ValueError(
                f"cache version drift in namespaces {names} under {self.root}; "
                "clear the affected namespaces or pin cache.version to the "
                "version that produced the majority of the rows"
            )

    def to_dict(self) -> dict[str, object]:
        """Serializable audit of the whole cache, suitable for a run manifest."""
        summaries = self.summarize()
        return {
            "root": str(self.root),
            "n_namespaces": len(summaries),
            "n_records": sum(item.n_records for item in summaries),
            "bytes_on_disk": sum(item.bytes_on_disk for item in summaries),
            "namespaces": [item.to_dict() for item in summaries],
            "drift": [item.to_dict() for item in self.drift()],
        }
