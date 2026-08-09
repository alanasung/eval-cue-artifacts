"""Callable protocols that keep evaluation model-agnostic.

Evaluation outlives models. Nothing in ``evaluation/`` may import a concrete
model class; harnesses ask for behaviour through the callables named here, so
the same harness runs against a HuggingFace model, a cached prediction table, a
synthetic oracle used to validate the instrument, or a model that does not exist
yet.

The named aliases matter as much as the types. ``PredictFn`` in a signature says
"this harness needs predictions under a chosen subset of available inputs" far
more clearly than an inline ``Callable[[frozenset[str]], np.ndarray]``.
"""

from __future__ import annotations

from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ComponentSet",
    "GenerateFn",
    "JudgeFn",
    "JudgeLabel",
    "LLMBackend",
    "PredictFn",
    "ProbeFn",
    "ScoreFn",
    "SupportsToDict",
    "as_component_set",
]

# A set of present/enabled components: input modalities, features, layers,
# attention heads, or probe subsets depending on the experiment. Frozen so it
# can key a cache.
ComponentSet = frozenset[str]

# Predictions for every example, given the components left enabled. This is the
# single most important abstraction in the evaluation layer.
PredictFn = Callable[[ComponentSet], np.ndarray]

# Continuous score per example, e.g. a probe readout or a refusal-direction
# projection.
ScoreFn = Callable[[ComponentSet], np.ndarray]

# Raw text generation. Kept separate from PredictFn because generation is
# expensive, stochastic, and usually cached.
GenerateFn = Callable[[list[str]], list[str]]

# Any LLM used as a component of the measurement instrument rather than as the
# subject of study, e.g. a monitor or a simulator.
LLMBackend = Callable[[str], str]

JudgeLabel = Literal["pass", "fail", "refuse", "unclear"]

# Adjudication of a single completion against the prompt that produced it.
JudgeFn = Callable[[str, str], JudgeLabel]

# A fitted probe applied to activations, returning one score per row.
ProbeFn = Callable[[np.ndarray], np.ndarray]


@runtime_checkable
class SupportsToDict(Protocol):
    """Anything that can serialize itself into a result payload."""

    def to_dict(self) -> dict[str, object]:  # pragma: no cover - structural only
        ...


def as_component_set(components: object) -> ComponentSet:
    """Coerce user input into a ``ComponentSet``, failing loudly on bad shapes.

    Accepting a bare string here would silently become a set of characters,
    which produces a confusing empty-intersection bug much later.

    Raises:
        TypeError: If given a string or a non-iterable.
        ValueError: If any component is not a non-empty string.
    """
    if isinstance(components, str):
        raise TypeError(
            f"expected an iterable of component names, got the string {components!r}; "
            f"wrap it as {{{components!r}}} if you meant a single component"
        )
    try:
        items = list(components)  # type: ignore[call-overload]
    except TypeError as exc:
        raise TypeError(f"components must be iterable, got {type(components).__name__}") from exc
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError(f"component names must be non-empty strings, got {item!r}")
    return frozenset(items)
