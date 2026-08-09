"""Per-stratum evaluation with explicit small-sample flagging.

An aggregate number hides where a method works. Splitting by stratum -- prompt
template, difficulty band, source dataset, sequence-length bucket -- shows it.
The hazard is that splitting also shrinks every cell, and a stratum with nine
examples produces a swing that looks like a finding and is arithmetic.

So this module never reports a stratum without its ``n``, and it marks every
stratum below ``min_stratum_size`` with a footnote marker that survives into the
markdown and LaTeX tables. The marker is attached at computation time rather
than at rendering time, because a caveat added by the table writer is a caveat
that gets lost the first time someone copies a number out of the JSON.

**Strata are task properties, not people.** Stratifying by a demographic
attribute turns an engineering evaluation into a claim about groups of humans --
a claim this codebase has neither the sampling design, the consent basis, nor
the statistical power to make. Demographic keys are therefore rejected outright
rather than discouraged in a comment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..utils.validation import require_positive, require_same_length

__all__ = [
    "FOOTNOTE_MARKERS",
    "StratifiedReport",
    "StratumResult",
    "stratify",
    "stratify_many",
]

logger = logging.getLogger(__name__)

FOOTNOTE_MARKERS: tuple[str, ...] = ("*", "†", "‡", "§", "¶", "#")

# Rejected outright: this codebase evaluates task properties, not people.
_FORBIDDEN_KEYS = frozenset(
    {
        "sex",
        "gender",
        "age",
        "age_group",
        "race",
        "ethnicity",
        "nationality",
        "religion",
        "disability",
        "sexual_orientation",
        "income",
        "patient_id",
    }
)


def _check_stratum_key(key: str) -> str:
    """Reject demographic stratum keys.

    Raises:
        ValueError: If the key names a protected or personal attribute.
    """
    if key.strip().lower() in _FORBIDDEN_KEYS:
        raise ValueError(
            f"stratum key {key!r} names a demographic attribute, which this evaluation "
            "is not designed or powered to make claims about. Stratify by a task "
            "property instead, such as prompt template, difficulty band, source "
            "dataset, or sequence-length bucket."
        )
    return key


def _marker(index: int) -> str:
    """Footnote marker for the nth flagged stratum, doubling past the alphabet."""
    base = FOOTNOTE_MARKERS[index % len(FOOTNOTE_MARKERS)]
    return base * (index // len(FOOTNOTE_MARKERS) + 1)


@dataclass(frozen=True)
class StratumResult:
    """One stratum's value, size, and small-sample status."""

    key: str
    level: str
    n: int
    value: float
    is_small: bool
    marker: str = ""

    @property
    def label(self) -> str:
        """Level name with its footnote marker attached, ready for a table cell."""
        return f"{self.level}{self.marker}"

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "level": self.level,
            "n": self.n,
            "value": round(self.value, 6),
            "is_small": self.is_small,
            "marker": self.marker,
        }


@dataclass
class StratifiedReport:
    """Overall value plus per-stratum breakdown, with footnotes already written."""

    key: str
    metric: str
    overall: float
    n: int
    min_stratum_size: int
    strata: list[StratumResult] = field(default_factory=list)

    @property
    def small_strata(self) -> list[StratumResult]:
        return [item for item in self.strata if item.is_small]

    @property
    def spread(self) -> float:
        """Range across strata with adequate n.

        Small strata are excluded because they would otherwise define the range
        and make every breakdown look dramatic.
        """
        usable = [item.value for item in self.strata if not item.is_small]
        return float(max(usable) - min(usable)) if len(usable) >= 2 else 0.0

    def footnotes(self) -> list[str]:
        """Footnote lines for the flagged strata, in marker order."""
        return [
            f"{item.marker} n = {item.n} for {item.key}={item.level}, below the "
            f"minimum of {self.min_stratum_size}; this cell is indicative only."
            for item in self.small_strata
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "metric": self.metric,
            "overall": round(self.overall, 6),
            "n": self.n,
            "min_stratum_size": self.min_stratum_size,
            "spread_excluding_small": round(self.spread, 6),
            "strata": [item.to_dict() for item in self.strata],
            "footnotes": self.footnotes(),
        }


def stratify(
    values: Sequence[float] | np.ndarray,
    strata: Sequence[str] | np.ndarray,
    *,
    key: str,
    metric_fn: Callable[[np.ndarray], float] | None = None,
    metric: str = "mean",
    min_stratum_size: int = 20,
) -> StratifiedReport:
    """Compute a metric overall and within each level of one stratum key.

    Args:
        values: One observation per example, e.g. per-example correctness or
            absolute error.
        strata: The level each example belongs to, one per example.
        key: Name of the stratifying variable, used in the footnotes and
            checked against the demographic-attribute block list.
        metric_fn: Aggregator over a stratum's values. Defaults to the mean.
        metric: Name of the metric, recorded in the report.
        min_stratum_size: Levels below this are flagged, not dropped. Dropping
            them would hide that the evaluation has no coverage there.

    Returns:
        A :class:`StratifiedReport` with markers and footnotes already attached.

    Raises:
        ValueError: If the key is demographic, if lengths disagree, or if
            ``min_stratum_size`` is not positive.
    """
    _check_stratum_key(key)
    require_positive(min_stratum_size, "min_stratum_size")
    value_arr = np.asarray(values, dtype=float).reshape(-1)
    level_arr = np.asarray(strata).reshape(-1).astype(str)
    require_same_length(values=value_arr, strata=level_arr)

    aggregate = metric_fn or (lambda sample: float(np.mean(sample)))
    results: list[StratumResult] = []
    flagged = 0
    for level in sorted(set(level_arr.tolist())):
        mask = level_arr == level
        count = int(mask.sum())
        is_small = count < min_stratum_size
        marker = ""
        if is_small:
            marker = _marker(flagged)
            flagged += 1
            logger.warning(
                "stratum %s=%s has n=%d, below min_stratum_size=%d; "
                "its value is indicative only and is marked %r in tables",
                key,
                level,
                count,
                min_stratum_size,
                marker,
            )
        results.append(
            StratumResult(
                key=key,
                level=level,
                n=count,
                value=aggregate(value_arr[mask]),
                is_small=is_small,
                marker=marker,
            )
        )

    return StratifiedReport(
        key=key,
        metric=metric,
        overall=aggregate(value_arr),
        n=int(value_arr.size),
        min_stratum_size=min_stratum_size,
        strata=results,
    )


def stratify_many(
    values: Sequence[float] | np.ndarray,
    strata: Mapping[str, Sequence[str] | np.ndarray],
    *,
    metric_fn: Callable[[np.ndarray], float] | None = None,
    metric: str = "mean",
    min_stratum_size: int = 20,
) -> dict[str, StratifiedReport]:
    """Run :func:`stratify` once per key. Every key is checked before any runs.

    Validating all keys up front means a forbidden key fails before any compute
    is spent, rather than halfway through a set of breakdowns.

    Raises:
        ValueError: Propagated from :func:`stratify`.
    """
    for key in strata:
        _check_stratum_key(key)
    return {
        key: stratify(
            values,
            levels,
            key=key,
            metric_fn=metric_fn,
            metric=metric,
            min_stratum_size=min_stratum_size,
        )
        for key, levels in strata.items()
    }
