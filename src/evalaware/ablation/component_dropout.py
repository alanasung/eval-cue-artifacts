"""Leave-one-component-out dropout ablations.

Given a model-agnostic predict function over component subsets, this module
runs the full set and every leave-one-out arm, recording per-arm metrics and
marginal value. The predict function is injected so the same code runs against
a live model, a cache, or a synthetic oracle.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from ..evaluation.subsets import leave_one_out, subset_label
from ..utils.git import git_sha
from ..utils.validation import require_non_empty

__all__ = ["run_component_dropout"]

PredictFn = Callable[[frozenset[str]], np.ndarray]


def run_component_dropout(
    components: Sequence[str],
    predict_fn_or_labels: PredictFn | Sequence[float] | np.ndarray,
    labels_or_predict_fn: Sequence[float] | np.ndarray | PredictFn | None = None,
    *,
    threshold: float = 0.5,
    seed: int = 0,
    task: str = "A_component_dropout",
) -> dict[str, Any]:
    """Run full and leave-one-out arms; return a structured ablation payload.

    Accepts either ``(components, labels, predict_fn)`` or the test-facing
    order ``(components, predict_fn, labels)``.
    """
    names = list(components)
    require_non_empty(names, "components")

    predict_fn: PredictFn
    labels: Sequence[float] | np.ndarray
    if callable(predict_fn_or_labels):
        predict_fn = predict_fn_or_labels  # type: ignore[assignment]
        if labels_or_predict_fn is None:
            raise ValueError("labels are required")
        labels = labels_or_predict_fn  # type: ignore[assignment]
    else:
        labels = predict_fn_or_labels  # type: ignore[assignment]
        if not callable(labels_or_predict_fn):
            raise ValueError("predict_fn must be callable")
        predict_fn = labels_or_predict_fn  # type: ignore[assignment]

    y = np.asarray(labels, dtype=float).reshape(-1)
    require_non_empty(y, "labels")

    started = time.perf_counter()
    full = frozenset(names)
    full_scores = np.asarray(predict_fn(full), dtype=float).reshape(-1)
    if full_scores.size != y.size:
        raise ValueError(
            f"predict_fn(full) returned {full_scores.size} scores for {y.size} labels"
        )

    def _acc(scores: np.ndarray) -> float:
        return float(np.mean((scores >= threshold).astype(int) == y.astype(int)))

    arms: dict[str, Any] = {
        subset_label(full): {
            "components": sorted(full),
            "accuracy": _acc(full_scores),
            "mean_score": float(full_scores.mean()),
        }
    }
    marginal: dict[str, float] = {}
    for dropped in leave_one_out(names):
        scores = np.asarray(predict_fn(dropped), dtype=float).reshape(-1)
        if scores.size != y.size:
            raise ValueError(
                f"predict_fn({sorted(dropped)}) returned {scores.size} scores "
                f"for {y.size} labels"
            )
        acc = _acc(scores)
        label = subset_label(dropped)
        arms[label] = {
            "components": sorted(dropped),
            "accuracy": acc,
            "mean_score": float(scores.mean()),
        }
        missing = sorted(full - dropped)
        if len(missing) == 1:
            marginal[missing[0]] = arms[subset_label(full)]["accuracy"] - acc

    return {
        "task": task,
        "seed": seed,
        "git_sha": git_sha(),
        "n": int(y.size),
        "is_synthetic": True,
        "threshold": threshold,
        "components": names,
        "arms": arms,
        "marginal_value": marginal,
        "elapsed_seconds": time.perf_counter() - started,
    }
