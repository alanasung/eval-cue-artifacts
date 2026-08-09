"""Version-pinned artifact caching and cache auditing.

:class:`ArtifactCache` is the write and read path for one namespace;
:class:`CacheManifest` is the audit view across all of them. The split is
deliberate -- a producer should only be able to touch the namespace it named,
while an auditor needs to see everything at once to detect version drift.
"""

from __future__ import annotations

from .artifact_cache import ArtifactCache, CacheRecord, check_safe_name
from .manifest import CacheManifest, NamespaceSummary, VersionDrift

__all__ = [
    "ArtifactCache",
    "CacheManifest",
    "CacheRecord",
    "NamespaceSummary",
    "VersionDrift",
    "check_safe_name",
]
