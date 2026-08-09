"""Loud-versus-silent failure analysis over arbitrary component sets.

When a component is removed -- a feature ablated, a layer patched, a monitor
disabled, an input modality dropped -- the model gets something wrong. The
safety-relevant question is not only *how often*, but *how visibly*:

**loud**   the observable output moves toward the decision boundary, so the
           failure is monitorable at deployment time.
**silent** the output stays confident and stable while being wrong, so nothing
           downstream can tell.

A model that fails loudly is a model you can build a monitor for. A model that
fails silently is one you cannot. That distinction is the point of this module,
and it is why ``silent_rate`` is the headline number rather than accuracy.

The harness is model-agnostic: callers pass ``predict_fn(present) -> np.ndarray``
and the harness enumerates the component subsets it needs. It works for any
number of components and any backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np

from .protocols import ComponentSet, PredictFn, as_component_set

__all__ = [
    "CATEGORIES",
    "Category",
    "FailureReport",
    "analyze_component_failure",
    "classify_taxonomy",
]

Category = Literal["correct", "imprecise", "critical"]
CATEGORIES: tuple[Category, ...] = ("correct", "imprecise", "critical")


def _abs_err(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=float) - np.asarray(target, dtype=float))


def _gate(pred: np.ndarray, threshold: float) -> np.ndarray:
    """Binary decision derived from the continuous prediction."""
    return np.asarray(pred, dtype=float) >= threshold


def classify_taxonomy(
    target: np.ndarray,
    pred: np.ndarray,
    gate_true: np.ndarray,
    *,
    threshold: float,
    tolerance: float,
) -> np.ndarray:
    """Per-example error category.

    ``critical``  the binary decision derived from the prediction is wrong.
    ``imprecise`` decision correct, but the continuous error exceeds tolerance.
    ``correct``   decision correct and within tolerance.

    The split exists because a decision flip and a magnitude error have
    different consequences, and averaging them into one error rate hides which
    one is happening.
    """
    target_arr = np.asarray(target, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    gate_arr = np.asarray(gate_true, dtype=bool)
    if not (target_arr.shape == pred_arr.shape == gate_arr.shape):
        raise ValueError(
            f"shape mismatch: target={target_arr.shape}, pred={pred_arr.shape}, "
            f"gate_true={gate_arr.shape}; all three must be one row per example"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    err = _abs_err(pred_arr, target_arr)
    gate_wrong = _gate(pred_arr, threshold) != gate_arr
    return np.where(gate_wrong, "critical", np.where(err <= tolerance, "correct", "imprecise"))


def _counts(categories: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(categories == name)) for name in CATEGORIES}


@dataclass
class FailureReport:
    """Serializable failure report. Raw predictions stay out of ``to_dict``."""

    components: list[str]
    n: int
    threshold: float
    tolerance: float
    confidence_margin: float
    conditions: dict[str, dict[str, object]]
    complementarity: dict[str, object]
    dropout: dict[str, dict[str, object]]
    predictions: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def most_silent_component(self) -> str | None:
        """Component whose removal produces the highest silent-failure rate.

        This is the component a deployment monitor would be blindest to.
        """
        rates = {
            name: stats["silent_rate"]
            for name, stats in self.dropout.items()
            if stats.get("silent_rate") is not None
        }
        if not rates:
            return None
        return max(rates, key=lambda key: float(rates[key]))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "task": "component_failure_analysis",
            "components": self.components,
            "n": self.n,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
            "confidence_margin": self.confidence_margin,
            "conditions": self.conditions,
            "complementarity": self.complementarity,
            "dropout": self.dropout,
            "most_silent_component": self.most_silent_component,
        }


def _required_subsets(components: list[str]) -> set[ComponentSet]:
    """Subsets the analysis needs: full, singletons, leave-one-out, and pairs."""
    everything = frozenset(components)
    needed: set[ComponentSet] = {everything}
    for name in components:
        needed.add(frozenset({name}))
        needed.add(everything - {name})
    for left, right in combinations(components, 2):
        needed.add(frozenset({left, right}))
    return needed


def analyze_component_failure(
    components: list[str],
    target: np.ndarray,
    gate_true: np.ndarray,
    predict_fn: PredictFn,
    *,
    threshold: float,
    tolerance: float,
    confidence_margin: float,
    groups: dict[str, np.ndarray] | None = None,
) -> FailureReport:
    """Run the full component-failure analysis.

    Args:
        components: Names of the components that can be present or absent.
        target: Continuous ground truth, one value per example.
        gate_true: Ground-truth binary decision, one per example.
        predict_fn: ``predict_fn(present) -> np.ndarray`` giving a continuous
            prediction per example when only ``present`` is enabled. The harness
            never learns how masking is implemented.
        threshold: Decision boundary on the continuous prediction.
        tolerance: Error band separating ``correct`` from ``imprecise``.
        confidence_margin: Distance from the boundary beyond which a wrong
            prediction counts as *silent* rather than *loud*.
        groups: Optional ``{attribute: array}`` for stratified win attribution.
            The mechanism is generic; supply whatever strata the experiment has.

    Returns:
        A :class:`FailureReport`.

    Raises:
        ValueError: If fewer than two components are given, if array shapes
            disagree, or if ``predict_fn`` returns the wrong length.
    """
    if len(components) < 2:
        raise ValueError(
            f"need at least 2 components for leave-one-out attribution, got {components}"
        )
    if len(set(components)) != len(components):
        raise ValueError(f"component names must be unique, got {components}")
    if confidence_margin < 0:
        raise ValueError(f"confidence_margin must be non-negative, got {confidence_margin}")

    target_arr = np.asarray(target, dtype=float)
    gate_arr = np.asarray(gate_true, dtype=bool)
    if target_arr.shape != gate_arr.shape:
        raise ValueError(
            f"target and gate_true disagree: {target_arr.shape} vs {gate_arr.shape}"
        )
    n = int(target_arr.size)
    everything = frozenset(components)

    preds: dict[ComponentSet, np.ndarray] = {}
    for subset in _required_subsets(components):
        values = np.asarray(predict_fn(as_component_set(subset)), dtype=float)
        if values.shape != target_arr.shape:
            raise ValueError(
                f"predict_fn({sorted(subset)}) returned shape {values.shape}, "
                f"expected {target_arr.shape} (one prediction per example)"
            )
        preds[subset] = values

    full = preds[everything]
    err_full = _abs_err(full, target_arr)

    def condition_summary(pred: np.ndarray) -> dict[str, object]:
        cats = classify_taxonomy(
            target_arr, pred, gate_arr, threshold=threshold, tolerance=tolerance
        )
        return {
            "mae": round(float(_abs_err(pred, target_arr).mean()), 6),
            "taxonomy": _counts(cats),
        }

    conditions = {"full": condition_summary(full)}
    for name in components:
        conditions[f"drop_{name}"] = condition_summary(preds[everything - {name}])

    solo_err = {name: _abs_err(preds[frozenset({name})], target_arr) for name in components}
    drop_err = {name: _abs_err(preds[everything - {name}], target_arr) for name in components}
    marginal_value = {
        name: round(float(drop_err[name].mean() - err_full.mean()), 6) for name in components
    }
    solo_mae = {name: round(float(solo_err[name].mean()), 6) for name in components}

    # Per-example winner: the component whose removal hurts this example most.
    drop_minus_full = np.stack([drop_err[name] - err_full for name in components], axis=1)
    winner_idx = np.argmax(drop_minus_full, axis=1)
    winners = {name: int(np.sum(winner_idx == i)) for i, name in enumerate(components)}

    matrix = [
        [
            round(
                float(
                    _abs_err(
                        preds[frozenset({a}) if a == b else frozenset({a, b})], target_arr
                    ).mean()
                ),
                6,
            )
            for b in components
        ]
        for a in components
    ]

    best_solo = min(solo_mae.values())
    complementarity: dict[str, object] = {
        "full_mae": round(float(err_full.mean()), 6),
        "solo_mae": solo_mae,
        "marginal_value": marginal_value,
        "fusion_gain_vs_best_solo": round(best_solo - float(err_full.mean()), 6),
        "per_example_winners": winners,
        "matrix": {"components": components, "values": matrix},
    }
    if groups:
        complementarity["winners_by_group"] = {
            attribute: {
                str(level): {
                    components[i]: int(np.sum(winner_idx[np.asarray(values) == level] == i))
                    for i in range(len(components))
                }
                for level in np.unique(np.asarray(values))
            }
            for attribute, values in groups.items()
        }

    dropout: dict[str, dict[str, object]] = {}
    for name in components:
        pred_drop = preds[everything - {name}]
        was_right = _gate(full, threshold) == gate_arr
        now_wrong = _gate(pred_drop, threshold) != gate_arr
        induced = was_right & now_wrong
        # Confidence is deployment-observable: distance from the boundary needs
        # no counterfactual. Output shift does need one, so it stays diagnostic.
        confidence = np.abs(pred_drop - threshold)
        shift = np.abs(pred_drop - full)
        silent = induced & (confidence >= confidence_margin)
        loud = induced & (confidence < confidence_margin)
        induced_n = int(induced.sum())
        dropout[f"drop_{name}"] = {
            "induced_critical": induced_n,
            "silent": int(silent.sum()),
            "loud": int(loud.sum()),
            "silent_rate": round(float(silent.sum() / induced_n), 6) if induced_n else None,
            "mean_output_shift": round(float(shift.mean()), 6),
            "mae_increase": round(float(drop_err[name].mean() - err_full.mean()), 6),
        }

    return FailureReport(
        components=list(components),
        n=n,
        threshold=threshold,
        tolerance=tolerance,
        confidence_margin=confidence_margin,
        conditions=conditions,
        complementarity=complementarity,
        dropout=dropout,
        predictions={
            "__target__": target_arr,
            "__gate__": gate_arr,
            **{",".join(sorted(subset)): values for subset, values in preds.items()},
        },
    )
