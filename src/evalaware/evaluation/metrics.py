"""Point estimates with intervals, and the handful of metrics worth hand-checking.

Two decisions shape this module.

The first is that a metric returns an :class:`Estimate`, not a float. A bare
float invites a results table of point estimates with no spread, and a table
like that cannot distinguish a real effect from run-to-run noise. Bundling the
interval with the value means the interval has to be actively discarded to be
omitted, rather than merely forgotten.

The second is that AUROC is implemented here rather than imported from
scikit-learn. It is fifteen lines of rank arithmetic, it removes a heavyweight
dependency from the critical path, and -- most usefully -- the tie handling is
visible. Rank-based AUROC with average ranks for ties is exactly the
Mann-Whitney U statistic, which means the implementation can be checked by hand
against a four-row example, and the tests do exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..utils.validation import require_in_range, require_positive, require_same_length

__all__ = [
    "Estimate",
    "accuracy",
    "auroc",
    "bootstrap_diff",
    "bootstrap_mean",
    "f1",
    "paired_bootstrap",
    "minimum_detectable_effect",
    "tost_equivalence",
]


@dataclass(frozen=True)
class Estimate:
    """A value with a confidence interval and the sample size behind it."""

    value: float
    lo: float
    hi: float
    n: int
    alpha: float = 0.05

    @property
    def excludes_zero(self) -> bool:
        """True when the interval lies wholly above or wholly below zero.

        The one-line answer to "is this difference distinguishable from
        nothing". Reported for every difference so a reader never has to
        eyeball two overlapping intervals to find out.
        """
        return self.lo > 0.0 or self.hi < 0.0

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def format(self, *, digits: int = 3) -> str:
        """Render as ``value [lo, hi]`` for a table cell."""
        return f"{self.value:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "value": round(self.value, 6),
            "lo": round(self.lo, 6),
            "hi": round(self.hi, 6),
            "n": self.n,
            "alpha": self.alpha,
            "excludes_zero": self.excludes_zero,
        }
        # Spanning zero is inconclusive, not evidence of a null.
        if not self.excludes_zero and self.n >= 2:
            payload["minimum_detectable_effect"] = round(
                minimum_detectable_effect(self.n, alpha=self.alpha), 6
            )
            payload["null_claim"] = (
                "inconclusive: interval spans zero; use tost_equivalence before "
                "claiming a null, and compare effects to minimum_detectable_effect"
            )
        return payload


def _as_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Coerce to a 1-D float array, refusing an empty one.

    Raises:
        ValueError: If the array is empty or not one-dimensional.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        raise ValueError(
            f"{name} must be one-dimensional with one entry per example, "
            f"got shape {array.shape}; flatten or select a column before passing it"
        )
    array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(
            f"{name} is empty, so there is nothing to estimate; "
            "check whether an upstream filter removed every example"
        )
    return array


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lower = float(np.percentile(samples, 100.0 * alpha / 2.0))
    upper = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> Estimate:
    """Percentile bootstrap interval for the mean, or any other statistic.

    Args:
        values: One observation per example.
        n_boot: Resamples. Below a few thousand the interval endpoints
            themselves become noisy enough to change conclusions.
        alpha: Two-sided level; ``0.05`` gives a 95% interval.
        seed: Fixed so the interval is reproducible. An interval that moves
            between runs of the same analysis is not evidence of anything.
        statistic: Alternative to the mean, e.g. ``np.median``.

    Raises:
        ValueError: On an empty input or an out-of-range ``alpha``.
    """
    array = _as_1d(values, "values")
    require_in_range(alpha, "alpha", low=0.0, high=1.0, inclusive=False)
    require_positive(n_boot, "n_boot")
    func = statistic or (lambda sample: float(np.mean(sample)))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    samples = np.array([func(array[row]) for row in indices])
    lo, hi = _percentile_ci(samples, alpha)
    return Estimate(value=func(array), lo=lo, hi=hi, n=int(array.size), alpha=alpha)


def bootstrap_diff(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Interval for ``mean(treatment) - mean(control)`` with independent groups.

    Use this only when the two groups really are separate examples. When the
    same examples were measured twice, :func:`paired_bootstrap` is both correct
    and much tighter.

    Raises:
        ValueError: On an empty input or an out-of-range ``alpha``.
    """
    left = _as_1d(treatment, "treatment")
    right = _as_1d(control, "control")
    require_in_range(alpha, "alpha", low=0.0, high=1.0, inclusive=False)
    require_positive(n_boot, "n_boot")

    rng = np.random.default_rng(seed)
    left_idx = rng.integers(0, left.size, size=(n_boot, left.size))
    right_idx = rng.integers(0, right.size, size=(n_boot, right.size))
    samples = left[left_idx].mean(axis=1) - right[right_idx].mean(axis=1)
    lo, hi = _percentile_ci(samples, alpha)
    return Estimate(
        value=float(left.mean() - right.mean()),
        lo=lo,
        hi=hi,
        n=int(min(left.size, right.size)),
        alpha=alpha,
    )


def paired_bootstrap(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Interval for the mean paired difference, resampling examples jointly.

    Ablation arms are almost always paired: the same examples are scored with
    and without a component. Resampling the two arms independently throws away
    that pairing and inflates the interval, which is how a real effect gets
    reported as "not distinguishable from zero".

    Raises:
        ValueError: If the two arms have different lengths, or on bad ``alpha``.
    """
    left = _as_1d(treatment, "treatment")
    right = _as_1d(control, "control")
    require_same_length(treatment=left, control=right)
    require_in_range(alpha, "alpha", low=0.0, high=1.0, inclusive=False)
    require_positive(n_boot, "n_boot")

    deltas = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    samples = deltas[indices].mean(axis=1)
    lo, hi = _percentile_ci(samples, alpha)
    return Estimate(value=float(deltas.mean()), lo=lo, hi=hi, n=int(deltas.size), alpha=alpha)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing the average of their positions.

    Written out rather than imported so the tie handling is inspectable: tied
    scores must share a rank, otherwise a model that outputs a handful of
    distinct values gets an AUROC that depends on sort order.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def auroc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve, from the Mann-Whitney U identity.

    ``AUROC = (sum of positive ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)``,
    which is the probability that a randomly chosen positive outscores a
    randomly chosen negative, with ties counted as half.

    Args:
        labels: Binary ground truth, anything truthy counting as positive.
        scores: Continuous score, higher meaning more positive.

    Returns:
        A value in ``[0, 1]``.

    Raises:
        ValueError: If lengths disagree, or if one class is absent -- AUROC is
            undefined then, and returning 0.5 would look like a real result.
    """
    label_arr = _as_1d(labels, "labels")
    score_arr = _as_1d(scores, "scores")
    require_same_length(labels=label_arr, scores=score_arr)

    positive = label_arr > 0
    n_pos = int(positive.sum())
    n_neg = int(label_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"auroc needs both classes present, got {n_pos} positive and {n_neg} negative; "
            "report the class balance instead of an undefined AUROC"
        )
    ranks = _average_ranks(score_arr)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> float:
    """Fraction of exact matches.

    Raises:
        ValueError: If the two arrays have different lengths.
    """
    truth = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    require_same_length(y_true=truth, y_pred=pred)
    return float(np.mean(truth == pred))


def f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    positive: float = 1.0,
) -> float:
    """Harmonic mean of precision and recall for one class.

    Returns 0.0 when precision and recall are both zero, which is the
    conventional reading of "predicted nothing and found nothing" and avoids a
    nan propagating silently into a results table.

    Raises:
        ValueError: If the two arrays have different lengths.
    """
    truth = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    require_same_length(y_true=truth, y_pred=pred)

    true_pos = float(np.sum((pred == positive) & (truth == positive)))
    false_pos = float(np.sum((pred == positive) & (truth != positive)))
    false_neg = float(np.sum((pred != positive) & (truth == positive)))
    denominator = 2.0 * true_pos + false_pos + false_neg
    return 0.0 if denominator == 0.0 else 2.0 * true_pos / denominator



def minimum_detectable_effect(
    n: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    sigma: float = 1.0,
) -> float:
    """Approximate two-sided MDE for a mean under known sigma.

    An interval that spans zero is not evidence of a null; report this MDE
    alongside every such interval so underpowered tests are visible.
    """
    if n < 2:
        raise ValueError(f"MDE needs n>=2, got {n}")
    require_in_range(alpha, "alpha", low=0.0, high=1.0, inclusive=False)
    require_in_range(power, "power", low=0.0, high=1.0, inclusive=False)
    # Normal approximations: z_{1-a/2} + z_{power}
    from math import ceil
    # Rough constants: 1.96 for 0.05 two-sided, 0.84 for 80% power
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143 if abs(power - 0.8) < 1e-9 else 1.2815515655446004
    return float(sigma * (z_alpha + z_power) / (n ** 0.5))


def tost_equivalence(
    values: Sequence[float] | np.ndarray,
    *,
    low: float,
    high: float,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Two one-sided tests for equivalence to a null band [low, high].

    Returns whether both one-sided bounds support equivalence at level alpha.
    Never claim a null from a CI that merely spans zero without this check.
    """
    if low >= high:
        raise ValueError(f"equivalence band requires low < high, got [{low}, {high}]")
    est = bootstrap_mean(values, n_boot=n_boot, alpha=alpha * 2, seed=seed)
    # TOST: reject H0_lower (mean <= low) and H0_upper (mean >= high)
    equivalent = est.lo > low and est.hi < high
    mde = minimum_detectable_effect(est.n, alpha=alpha, sigma=float(np.std(_as_1d(values, "values"), ddof=1) or 1.0))
    return {
        "equivalent": equivalent,
        "estimate": est.to_dict(),
        "band": [low, high],
        "minimum_detectable_effect": mde,
        "note": (
            "CI spanning zero is inconclusive; equivalence requires the interval "
            "to lie inside the pre-registered band, and effects smaller than MDE "
            "are underpowered."
        ),
    }
