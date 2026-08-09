"""Activation / feature construction without requiring Hub weight downloads.

When a real model is unavailable, each scenario is mapped to a synthetic
activation that contains:
  - surface cue features (lexical eval markers) — what a text classifier sees
  - a planted internal awareness direction present on eval contexts even when
    surface cues are ablated — what the probe must read for an internal claim

Causal steering later moves this planted direction.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

EVAL_CUES = (
    "benchmark",
    "evaluation",
    "evaluated",
    "exam",
    "test item",
    "[test]",
    "[evaluation]",
    "[rubric]",
    "[lab]",
    "[assessment]",
    "graded",
    "scored",
    "audit",
    "standardized",
)


def surface_cue_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for c in EVAL_CUES if c in low)
    return float(hits)


def synthetic_activation(
    text: str,
    context: str,
    dim: int = 32,
    seed: int = 0,
    steer: float = 0.0,
) -> np.ndarray:
    """Build a dim-dimensional activation with surface + internal channels."""
    rng = np.random.default_rng(abs(hash((text, seed))) % (2**32))
    x = rng.normal(0, 0.1, size=dim).astype(np.float64)
    # Dims 0-7: surface lexical cues (text-classifier accessible).
    cue = surface_cue_score(text)
    x[0] = cue
    x[1] = float(bool(re.search(r"\[(test|evaluation|rubric|lab|assessment)\]", text, re.I)))
    x[2] = float("benchmark" in text.lower() or "exam" in text.lower())
    # Dims 8-15: planted internal awareness (eval contexts), not fully reducible
    # to the surface score — includes a context bit plus noise.
    internal = 1.0 if context == "eval" else 0.0
    x[8] = internal + 0.15 * rng.normal()
    x[9] = 0.7 * internal + 0.3 * (cue > 0) + steer
    x[10:16] += internal * 0.5
    return x


def batch_activations(
    items: list[dict[str, Any]], dim: int = 32, seed: int = 0, steer: float = 0.0
) -> np.ndarray:
    return np.stack(
        [
            synthetic_activation(it["text"], it["context"], dim=dim, seed=seed, steer=steer)
            for it in items
        ]
    )
