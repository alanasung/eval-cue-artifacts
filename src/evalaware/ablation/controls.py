"""Control arms that bound what a positive ablation result can mean.

A treatment effect without a shuffle control and a random-direction control is
not interpretable: the same magnitude can come from removing any information at
all. These arms return structured dicts so they compose with the rest of the
ablation surface and with aggregation.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from ..utils.git import git_sha
from ..utils.validation import require_non_empty, require_positive

__all__ = ["run_control_arms"]

FeaturePredictFn = Callable[[np.ndarray], np.ndarray]


def _accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    preds = (scores >= threshold).astype(int)
    return float(np.mean(preds == labels.astype(int)))


def run_control_arms(
    features_or_labels: Sequence[float] | np.ndarray,
    labels_or_scores: Sequence[float] | np.ndarray,
    predict_fn: FeaturePredictFn | None = None,
    *,
    threshold: float = 0.5,
    n_shuffles: int = 20,
    seed: int = 0,
    task: str = "A_controls",
) -> dict[str, Any]:
    """Run identity / shuffle controls.

    Two call shapes are accepted for compatibility:

    1. ``run_control_arms(labels, scores)`` — score-table controls
    2. ``run_control_arms(features, labels, predict_fn)`` — feature-matrix
       controls where ``predict_fn(features) -> scores``
    """
    require_positive(n_shuffles, "n_shuffles")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)

    if predict_fn is not None:
        features = np.asarray(features_or_labels, dtype=float)
        y = np.asarray(labels_or_scores, dtype=float).reshape(-1)
        require_non_empty(y, "labels")
        if features.shape[0] != y.size:
            raise ValueError(
                f"features rows {features.shape[0]} != labels length {y.size}"
            )
        scores = np.asarray(predict_fn(features), dtype=float).reshape(-1)
        if scores.size != y.size:
            raise ValueError(f"predict_fn returned {scores.size} scores for {y.size} labels")
        baseline = _accuracy(y, scores, threshold)
        shuffle_label_scores = []
        for _ in range(n_shuffles):
            shuffled = y.copy()
            rng.shuffle(shuffled)
            shuffle_label_scores.append(_accuracy(shuffled, scores, threshold))
        perm = rng.permutation(features.shape[0])
        shuffled_scores = np.asarray(predict_fn(features[perm]), dtype=float).reshape(-1)
        feature_shuffle_acc = _accuracy(y, shuffled_scores, threshold)
        return {
            "task": task,
            "seed": seed,
            "git_sha": git_sha(),
            "n": int(y.size),
            "is_synthetic": True,
            "real_accuracy": baseline,
            "label_shuffle_accuracy": float(np.mean(shuffle_label_scores)),
            "feature_shuffle_accuracy": feature_shuffle_acc,
            "arms": {
                "identity": {"accuracy": baseline},
                "label_shuffle": {"accuracy_mean": float(np.mean(shuffle_label_scores))},
                "feature_shuffle": {"accuracy": feature_shuffle_acc},
            },
            "elapsed_seconds": time.perf_counter() - started,
        }

    y = np.asarray(features_or_labels, dtype=float).reshape(-1)
    s = np.asarray(labels_or_scores, dtype=float).reshape(-1)
    require_non_empty(y, "labels")
    if y.shape != s.shape:
        raise ValueError(f"labels and scores length disagree: {y.size} vs {s.size}")

    baseline = _accuracy(y, s, threshold)
    shuffle_label_scores = []
    for _ in range(n_shuffles):
        shuffled = y.copy()
        rng.shuffle(shuffled)
        shuffle_label_scores.append(_accuracy(shuffled, s, threshold))
    shuffle_score_scores = []
    for _ in range(n_shuffles):
        shuffled = s.copy()
        rng.shuffle(shuffled)
        shuffle_score_scores.append(_accuracy(y, shuffled, threshold))
    random_acc = _accuracy(y, rng.uniform(0.0, 1.0, size=y.size), threshold)

    return {
        "task": task,
        "seed": seed,
        "git_sha": git_sha(),
        "n": int(y.size),
        "is_synthetic": True,
        "real_accuracy": baseline,
        "threshold": threshold,
        "arms": {
            "identity": {"accuracy": baseline},
            "label_shuffle": {
                "accuracy_mean": float(np.mean(shuffle_label_scores)),
                "accuracy_std": float(np.std(shuffle_label_scores)),
                "n_shuffles": n_shuffles,
            },
            "score_shuffle": {
                "accuracy_mean": float(np.mean(shuffle_score_scores)),
                "accuracy_std": float(np.std(shuffle_score_scores)),
                "n_shuffles": n_shuffles,
            },
            "random_scores": {"accuracy": random_acc},
        },
        "delta_vs_label_shuffle": baseline - float(np.mean(shuffle_label_scores)),
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Controls bound what a positive treatment effect can mean; "
            "they are not themselves evidence for a mechanism."
        ],
    }
