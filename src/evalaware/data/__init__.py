"""Dataset construction: synthetic items, splits, and checked-in manifests.

Real corpora are never committed. What lives here is the machinery that builds
pilot-sized synthetic sets, splits them reproducibly, and writes tiny manifests
that name how a dataset was produced so a later run can refuse to mix versions.
"""

from __future__ import annotations

from .manifest import DatasetManifest, ItemRecord, load_manifest, write_manifest
from .splits import SplitBundle, split_items
from .synthetic import SyntheticItem, build_synthetic_items

__all__ = [
    "DatasetManifest",
    "ItemRecord",
    "SplitBundle",
    "SyntheticItem",
    "build_synthetic_items",
    "load_manifest",
    "split_items",
    "write_manifest",
]
