"""Seeding and determinism control.

Every number this repo reports has to be traceable to a seed, a commit, and a
resolved config. This module owns the first of those three.

The determinism story is deliberately honest rather than aspirational: on Apple
Silicon the MPS backend has no deterministic-algorithms mode, so
``torch.use_deterministic_algorithms`` cannot be enabled and reduction order may
vary between runs. Rather than silently pretending otherwise, ``set_seed``
records what it could and could not guarantee, and ``determinism_report``
returns that for inclusion in the run manifest.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

try:  # torch is a hard dependency in practice but the seeding path stays usable without it
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _HAS_TORCH = False

__all__ = [
    "DeterminismReport",
    "determinism_report",
    "seed_everything",
    "set_seed",
    "worker_seed",
]

Backend = Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True)
class DeterminismReport:
    """What seeding actually guaranteed, for the run manifest."""

    seed: int
    deterministic_requested: bool
    torch_available: bool
    backend: str
    guarantees: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "deterministic_requested": self.deterministic_requested,
            "torch_available": self.torch_available,
            "backend": self.backend,
            "guarantees": list(self.guarantees),
            "caveats": list(self.caveats),
        }


def _active_backend() -> str:
    if not _HAS_TORCH:
        return "none"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int, *, deterministic: bool = True) -> DeterminismReport:
    """Seed every RNG this codebase touches.

    Covers ``random``, ``numpy``, ``torch``, ``torch.cuda``, ``PYTHONHASHSEED``,
    and the cuDNN determinism flags. Returns a report describing which
    guarantees actually hold on this machine, which belongs in the manifest.

    Args:
        seed: The seed. Must be non-negative; negative seeds are a common sign
            of an unset config value being passed through.
        deterministic: Request deterministic kernels where the backend supports
            it. Costs throughput, so sweeps may turn it off deliberately.

    Returns:
        A ``DeterminismReport`` naming guarantees and caveats.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}; check the config value")

    guarantees = ["python random", "numpy default_rng and legacy global"]
    caveats: list[str] = []

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    backend = _active_backend()
    if not _HAS_TORCH:
        caveats.append("torch is not installed; only python and numpy RNGs were seeded")
        return DeterminismReport(seed, deterministic, False, backend, guarantees, caveats)

    torch.manual_seed(seed)
    guarantees.append("torch cpu generator")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        guarantees.append("torch cuda generators")

    if deterministic:
        if backend == "mps":
            caveats.append(
                "MPS has no deterministic-algorithms mode, so reduction order may vary "
                "between runs; expect small numeric drift and do not read it as an effect"
            )
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)
            guarantees.append("torch deterministic algorithms (warn_only)")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        guarantees.append("cudnn deterministic, benchmark off")

    caveats.append(
        "sampling-based generation is seeded per call; changing batch composition "
        "changes results even at a fixed seed"
    )
    return DeterminismReport(seed, deterministic, True, backend, guarantees, caveats)


# Kept as an alias because `seed_everything` is the name most people reach for.
seed_everything = set_seed


def determinism_report(seed: int) -> DeterminismReport:
    """Describe determinism guarantees without mutating global RNG state."""
    backend = _active_backend()
    caveats: list[str] = []
    if backend == "mps":
        caveats.append("MPS has no deterministic-algorithms mode")
    if not _HAS_TORCH:
        caveats.append("torch is not installed")
    return DeterminismReport(seed, True, _HAS_TORCH, backend, [], caveats)


def worker_seed(base_seed: int, worker_id: int) -> int:
    """Derive a per-worker seed that will not collide across small worker counts.

    Raises:
        ValueError: If ``worker_id`` is negative.
    """
    if worker_id < 0:
        raise ValueError(f"worker_id must be non-negative, got {worker_id}")
    return base_seed * 1_000 + worker_id
