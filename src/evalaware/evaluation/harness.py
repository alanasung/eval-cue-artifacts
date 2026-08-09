"""One harness that turns a prediction function into a reportable result.

The pieces of an evaluation -- accuracy, discrimination, intervals, loud-versus-
silent failure attribution -- are individually simple and are individually
assembled slightly differently every time somebody writes a new script. The
result is a results directory where each file has a different shape and
aggregation becomes bespoke parsing.

This harness fixes the shape. Given a ``predict_fn``, ground truth, and the
thresholds from the config, it produces a :class:`~..utils.run_manifest.ResultPayload`
carrying the metrics, the failure report, and the provenance, every time.

It takes a callable, never a model. Nothing in this module may import a concrete
model class, which is what lets the same harness run against a HuggingFace
model, a cached prediction table, or a synthetic oracle used to prove the
instrument works before any real model is loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..utils.run_manifest import ResultPayload
from ..utils.validation import require_in_range, require_same_length
from .failure import FailureReport, analyze_component_failure
from .metrics import Estimate, accuracy, auroc, bootstrap_mean, f1
from .protocols import PredictFn
from .subsets import subset_label

__all__ = ["EvaluationHarness", "EvaluationResult"]

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Metrics, failure attribution, and the notes a reader needs to trust them."""

    task: str
    n: int
    metrics: dict[str, Any]
    failure: FailureReport | None = None
    notes: list[str] = field(default_factory=list)
    is_synthetic: bool = False

    @property
    def headline(self) -> str:
        """The one number this evaluation exists to produce.

        Silent failure rate outranks accuracy: a wrong answer that a monitor can
        see is a different kind of problem from one it cannot.
        """
        if self.failure is not None:
            worst = self.failure.most_silent_component
            if worst is not None:
                rate = self.failure.dropout[f"drop_{worst}"]["silent_rate"]
                return f"removing {worst} produces a silent-failure rate of {rate}"
        return f"accuracy {self.metrics.get('accuracy')}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "n": self.n,
            "metrics": self.metrics,
            "notes": list(self.notes),
        }
        if self.failure is not None:
            payload["failure"] = self.failure.to_dict()
        return payload

    def to_payload(self, *, seed: int, profile: str = "pilot", model: str | None = None):
        """Wrap as a provenance-carrying :class:`ResultPayload` ready to save."""
        return ResultPayload(
            task=self.task,
            seed=seed,
            n={"examples": self.n},
            metrics=self.to_dict()["metrics"],
            profile=profile,
            model=model,
            is_synthetic=self.is_synthetic,
            notes=list(self.notes),
        )


class EvaluationHarness:
    """Compose predictions, metrics, and failure analysis into one result.

    Args:
        components: Names of the components that can be enabled or disabled.
            Two or more are required for the failure attribution; with fewer,
            the harness still reports metrics but skips attribution.
        target: Continuous ground truth, one value per example.
        gate_true: Ground-truth binary decision, one per example.
        threshold: Decision boundary on the continuous prediction.
        tolerance: Error band separating a correct answer from an imprecise one.
        confidence_margin: Distance from the boundary past which a wrong answer
            counts as silent rather than loud.
        bootstrap_samples: Resamples behind every reported interval.
        bootstrap_alpha: Two-sided interval level.
        seed: Fixes the bootstrap so intervals are reproducible.
        task: Task id stamped onto the result, matching ``docs/TASK.md``.

    Raises:
        ValueError: If shapes disagree or a level is out of range.
    """

    def __init__(
        self,
        components: Sequence[str],
        target: Sequence[float] | np.ndarray,
        gate_true: Sequence[int] | np.ndarray,
        *,
        threshold: float = 0.5,
        tolerance: float = 0.1,
        confidence_margin: float = 0.2,
        bootstrap_samples: int = 10_000,
        bootstrap_alpha: float = 0.05,
        seed: int = 0,
        task: str = "E00_evaluation",
    ) -> None:
        self.components = list(components)
        self.target = np.asarray(target, dtype=float).reshape(-1)
        self.gate_true = np.asarray(gate_true).reshape(-1).astype(bool)
        require_same_length(target=self.target, gate_true=self.gate_true)
        require_in_range(bootstrap_alpha, "bootstrap_alpha", low=0.0, high=1.0, inclusive=False)
        self.threshold = threshold
        self.tolerance = tolerance
        self.confidence_margin = confidence_margin
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_alpha = bootstrap_alpha
        self.seed = seed
        self.task = task

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        components: Sequence[str],
        target: Sequence[float] | np.ndarray,
        gate_true: Sequence[int] | np.ndarray,
    ) -> "EvaluationHarness":
        """Build a harness from ``cfg.eval``, ``cfg.run``, and ``cfg.experiment``."""
        eval_cfg = getattr(cfg, "eval", cfg)
        return cls(
            components,
            target,
            gate_true,
            threshold=float(getattr(eval_cfg, "threshold", 0.5)),
            tolerance=float(getattr(eval_cfg, "tolerance", 0.1)),
            confidence_margin=float(getattr(eval_cfg, "confidence_margin", 0.2)),
            bootstrap_samples=int(getattr(eval_cfg, "bootstrap_samples", 10_000)),
            bootstrap_alpha=float(getattr(eval_cfg, "bootstrap_alpha", 0.05)),
            seed=int(getattr(getattr(cfg, "run", None), "seed", 0)),
            task=str(getattr(getattr(cfg, "experiment", None), "task_id", "E00")),
        )

    def _interval(self, values: np.ndarray) -> Estimate:
        return bootstrap_mean(
            values,
            n_boot=self.bootstrap_samples,
            alpha=self.bootstrap_alpha,
            seed=self.seed,
        )

    def _core_metrics(self, predictions: np.ndarray) -> dict[str, Any]:
        """Accuracy, F1, AUROC, and the interval on absolute error.

        AUROC is skipped rather than faked when one class is missing: a
        degenerate split makes it undefined, and a placeholder 0.5 would read as
        a measured chance-level result.
        """
        decisions = predictions >= self.threshold
        errors = np.abs(predictions - self.target)
        metrics: dict[str, Any] = {
            "accuracy": round(accuracy(self.gate_true.astype(float), decisions.astype(float)), 6),
            "f1": round(f1(self.gate_true.astype(float), decisions.astype(float)), 6),
            "mae": round(float(errors.mean()), 6),
            "mae_ci": self._interval(errors).to_dict(),
            "accuracy_ci": self._interval((decisions == self.gate_true).astype(float)).to_dict(),
        }
        classes = set(self.gate_true.tolist())
        if len(classes) == 2:
            metrics["auroc"] = round(auroc(self.gate_true.astype(int), predictions), 6)
        else:
            metrics["auroc"] = None
            metrics["auroc_skipped_reason"] = (
                f"gate_true holds a single class {sorted(classes)}, so AUROC is undefined"
            )
        return metrics

    def run(self, predict_fn: PredictFn, *, is_synthetic: bool = False) -> EvaluationResult:
        """Evaluate the full component set and attribute failures to components.

        Args:
            predict_fn: ``predict_fn(present) -> np.ndarray``. Called once for
                the full set and once per subset the failure analysis needs.
            is_synthetic: Mark the result as harness validation rather than a
                measurement. Propagates a loud warning into the saved payload.

        Returns:
            An :class:`EvaluationResult`.

        Raises:
            ValueError: If ``predict_fn`` returns the wrong number of rows.
        """
        everything = frozenset(self.components)
        predictions = np.asarray(predict_fn(everything), dtype=float).reshape(-1)
        if predictions.shape != self.target.shape:
            raise ValueError(
                f"predict_fn({subset_label(everything)}) returned {predictions.shape} rows "
                f"but the ground truth has {self.target.shape}; return one prediction "
                "per example, in the same order"
            )

        metrics = self._core_metrics(predictions)
        notes: list[str] = []
        failure: FailureReport | None = None

        if len(self.components) >= 2:
            failure = analyze_component_failure(
                self.components,
                self.target,
                self.gate_true,
                predict_fn,
                threshold=self.threshold,
                tolerance=self.tolerance,
                confidence_margin=self.confidence_margin,
            )
            metrics["most_silent_component"] = failure.most_silent_component
        else:
            notes.append(
                f"component-failure attribution skipped: it needs at least 2 components, "
                f"got {self.components}"
            )
            logger.info(notes[-1])

        result = EvaluationResult(
            task=self.task,
            n=int(self.target.size),
            metrics=metrics,
            failure=failure,
            notes=notes,
            is_synthetic=is_synthetic,
        )
        logger.info("evaluation %s over n=%d: %s", self.task, result.n, result.headline)
        return result
