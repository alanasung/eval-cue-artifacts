"""Model-agnostic evaluation: protocols, metrics, failure attribution, harness.

Nothing in this subpackage imports a concrete model class. Everything talks to
the callables in :mod:`.protocols`, so the same evaluation code runs against a
live model, a cached prediction table, or a synthetic oracle used to validate
the instrument before any real model exists.
"""

from __future__ import annotations

from .calibration import (
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from .failure import (
    CATEGORIES,
    Category,
    FailureReport,
    analyze_component_failure,
    classify_taxonomy,
)
from .harness import EvaluationHarness, EvaluationResult
from .metrics import (
    Estimate,
    accuracy,
    auroc,
    bootstrap_diff,
    bootstrap_mean,
    f1,
    minimum_detectable_effect,
    paired_bootstrap,
    tost_equivalence,
)
from .protocols import (
    ComponentSet,
    GenerateFn,
    JudgeFn,
    JudgeLabel,
    LLMBackend,
    PredictFn,
    ProbeFn,
    ScoreFn,
    SupportsToDict,
    as_component_set,
)
from .stratified import (
    FOOTNOTE_MARKERS,
    StratifiedReport,
    StratumResult,
    stratify,
    stratify_many,
)
from .subsets import (
    DESIGNS,
    Design,
    enumerate_subsets,
    full_set,
    leave_one_out,
    pairs,
    parse_label,
    powerset,
    singletons,
    subset_label,
)

__all__ = [
    "CATEGORIES",
    "DESIGNS",
    "FOOTNOTE_MARKERS",
    "Category",
    "ComponentSet",
    "Design",
    "Estimate",
    "EvaluationHarness",
    "EvaluationResult",
    "FailureReport",
    "GenerateFn",
    "JudgeFn",
    "JudgeLabel",
    "LLMBackend",
    "PredictFn",
    "ProbeFn",
    "ReliabilityBin",
    "ScoreFn",
    "StratifiedReport",
    "StratumResult",
    "SupportsToDict",
    "accuracy",
    "analyze_component_failure",
    "as_component_set",
    "auroc",
    "bootstrap_diff",
    "bootstrap_mean",
    "brier_score",
    "classify_taxonomy",
    "enumerate_subsets",
    "expected_calibration_error",
    "f1",
    "full_set",
    "leave_one_out",
    "minimum_detectable_effect",
    "paired_bootstrap",
    "pairs",
    "parse_label",
    "powerset",
    "reliability_bins",
    "singletons",
    "stratify",
    "stratify_many",
    "subset_label",
    "tost_equivalence",
]
