"""Calibration: whether a stated confidence means what it says.

Accuracy answers "how often is it right". Calibration answers "when it says 0.9,
is it right nine times in ten". These come apart constantly, and the gap is
directly safety-relevant: a monitor that is 80% accurate and well calibrated can
be given a threshold, while a monitor that is 80% accurate and reports 0.99 on
everything cannot be thresholded at all.

Two summary numbers are provided because they fail differently. Expected
calibration error is interpretable -- "confidences are off by 0.07 on average" --
but it is a binned statistic, so it can be driven to near zero by choosing few
enough bins. The Brier score is a proper scoring rule with no binning parameter,
so it cannot be gamed that way, but it mixes calibration and discrimination into
one number. Report both, plus :func:`reliability_bins` so the shape of the
miscalibration is visible rather than summarised away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils.validation import require_in_range, require_positive, require_same_length

__all__ = ["ReliabilityBin", "brier_score", "expected_calibration_error", "reliability_bins"]


def _as_probs(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Coerce to a 1-D probability array, rejecting values outside [0, 1].

    Raises:
        ValueError: On an empty array or on a value outside the unit interval,
            which usually means logits were passed where probabilities were
            expected.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty; there is nothing to calibrate")
    low, high = float(array.min()), float(array.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], got range [{low:.4f}, {high:.4f}]; "
            "apply a sigmoid or softmax before scoring calibration"
        )
    return array


def _as_labels(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """Coerce to a 1-D binary array of 0.0/1.0.

    Raises:
        ValueError: If any value is neither 0 nor 1.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    unique = np.unique(array)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(
            f"{name} must be binary 0/1, got distinct values {unique.tolist()}; "
            "binarize against the decision threshold before scoring calibration"
        )
    return array


@dataclass(frozen=True)
class ReliabilityBin:
    """One confidence bucket of a reliability diagram."""

    lower: float
    upper: float
    n: int
    mean_confidence: float
    empirical_accuracy: float

    @property
    def gap(self) -> float:
        """Signed miscalibration. Positive means over-confident."""
        return self.mean_confidence - self.empirical_accuracy

    @property
    def is_sparse(self) -> bool:
        """True when too few points land here for the bin to mean anything.

        Sparse bins dominate the visual impression of a reliability diagram
        while carrying almost no evidence, so they are flagged for the plot.
        """
        return self.n < 10

    def to_dict(self) -> dict[str, object]:
        return {
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "n": self.n,
            "mean_confidence": round(self.mean_confidence, 6),
            "empirical_accuracy": round(self.empirical_accuracy, 6),
            "gap": round(self.gap, 6),
            "is_sparse": self.is_sparse,
        }


def reliability_bins(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bucket predictions by confidence and compare stated to observed accuracy.

    Bins are equal-width on ``[0, 1]``. Empty bins are dropped rather than
    reported with a zero accuracy, which would draw a plunging line in the
    reliability diagram where in fact there is no data at all.

    Args:
        probs: Predicted probability of the positive class.
        labels: Binary ground truth.
        n_bins: Number of equal-width buckets.

    Raises:
        ValueError: On length mismatch, out-of-range probabilities, non-binary
            labels, or a non-positive bin count.
    """
    prob_arr = _as_probs(probs, "probs")
    label_arr = _as_labels(labels, "labels")
    require_same_length(probs=prob_arr, labels=label_arr)
    require_positive(n_bins, "n_bins")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # The rightmost bin is closed so that a confidence of exactly 1.0 lands
    # inside the diagram rather than in a phantom bin past the end.
    index = np.clip(np.digitize(prob_arr, edges[1:-1], right=False), 0, n_bins - 1)

    out: list[ReliabilityBin] = []
    for bin_id in range(n_bins):
        mask = index == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=float(edges[bin_id]),
                upper=float(edges[bin_id + 1]),
                n=count,
                mean_confidence=float(prob_arr[mask].mean()),
                empirical_accuracy=float(label_arr[mask].mean()),
            )
        )
    return out


def expected_calibration_error(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Sample-weighted mean absolute gap between confidence and accuracy.

    Always report ``n_bins`` alongside the number. ECE with three bins and ECE
    with fifty are different statistics, and the smaller one is not the better
    model.

    Raises:
        ValueError: Propagated from :func:`reliability_bins`.
    """
    bins = reliability_bins(probs, labels, n_bins=n_bins)
    total = sum(item.n for item in bins)
    if total == 0:
        return 0.0
    return sum(item.n * abs(item.gap) for item in bins) / total


def brier_score(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    """Mean squared error between predicted probability and outcome.

    A proper scoring rule, so it is minimised only by reporting one's true
    beliefs, and it needs no binning parameter. Lower is better; the trivial
    baseline of always predicting the base rate scores ``p * (1 - p)``, and any
    model above that is worse than a constant.

    Raises:
        ValueError: On length mismatch, out-of-range probabilities, or
            non-binary labels.
    """
    prob_arr = _as_probs(probs, "probs")
    label_arr = _as_labels(labels, "labels")
    require_same_length(probs=prob_arr, labels=label_arr)
    require_in_range(float(prob_arr.mean()), "mean probability", low=0.0, high=1.0)
    return float(np.mean((prob_arr - label_arr) ** 2))
