"""Argument validators with a single, consistent error vocabulary.

Every module in this repo needs the same four or five checks: a count is
positive, a probability is in range, two arrays line up, a set of requested
names is actually available. Written inline, those checks drift: one raises
``AssertionError``, another says ``"bad value"``, a third silently coerces. The
result is that a config typo surfaces as a shape error six stages later.

Centralising them buys two things. Error text is uniform, so a message can be
grepped back to the check that produced it. And every message names both the
offending value and the fix, because an error that only says what went wrong
leaves the reader to guess what to type instead.

Each validator returns the value it checked, so it can be used inline in an
assignment rather than as a separate statement.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Sized, TypeVar

__all__ = [
    "require_in_range",
    "require_non_empty",
    "require_positive",
    "require_same_length",
    "require_subset",
    "validate_layers",
]

T = TypeVar("T")
NumberT = TypeVar("NumberT", int, float)


def require_positive(value: NumberT, name: str, *, allow_zero: bool = False) -> NumberT:
    """Check that a count, size, or scale is positive.

    Args:
        value: The number to check.
        name: Dotted name of the setting, used verbatim in the message so the
            reader knows which key to edit.
        allow_zero: Accept zero as well. Used for counts that may legitimately
            be empty, such as a dropped-row tally.

    Returns:
        ``value`` unchanged.

    Raises:
        ValueError: If the value fails the bound.
    """
    floor = 0 if allow_zero else 1
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}; set {name} to 0 or more")
    if not allow_zero and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}; set {name} to at least {floor}")
    return value


def require_in_range(
    value: float,
    name: str,
    *,
    low: float,
    high: float,
    inclusive: bool = True,
) -> float:
    """Check that a value falls inside a closed or open interval.

    Used for probabilities, alphas, fractions, and temperatures, where an
    out-of-range value usually means a percentage was passed where a fraction
    was expected.

    Raises:
        ValueError: If ``value`` is outside the interval, or if the interval
            itself is inverted (a caller bug worth catching immediately).
    """
    if low > high:
        raise ValueError(
            f"range for {name} is inverted: low={low} is above high={high}; "
            "swap the bounds at the call site"
        )
    inside = low <= value <= high if inclusive else low < value < high
    if not inside:
        bounds = f"[{low}, {high}]" if inclusive else f"({low}, {high})"
        hint = (
            " (values above 1 usually mean a percentage was passed where a fraction is expected)"
            if high <= 1.0 and value > 1.0
            else ""
        )
        raise ValueError(f"{name} must be in {bounds}, got {value}{hint}; clamp or rescale it")
    return value


def require_same_length(**named: Sized) -> int:
    """Check that every named sequence has the same length, and return it.

    Keyword-only by construction: the names are the error message. Comparing
    ``len(a) != len(b)`` inline produces "shape mismatch" errors that do not say
    which array was short, which is the only fact the reader needs.

    Raises:
        ValueError: If fewer than two sequences are given, or if the lengths
            disagree.
    """
    if len(named) < 2:
        raise ValueError(
            f"require_same_length needs at least two named sequences, got {sorted(named)}; "
            "pass them as keywords, e.g. require_same_length(labels=y, scores=s)"
        )
    lengths = {name: len(value) for name, value in named.items()}
    distinct = set(lengths.values())
    if len(distinct) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in sorted(lengths.items()))
        raise ValueError(
            f"sequence lengths disagree: {detail}; every argument must hold one entry "
            "per example, so re-align them before calling"
        )
    return distinct.pop()


def require_non_empty(value: T, name: str) -> T:
    """Check that a container holds at least one item.

    An empty container is the most common symptom of a filter that matched
    nothing, and it produces a ``nan`` mean rather than an error if left alone.

    Raises:
        ValueError: If the container is empty or has no length at all.
    """
    if not isinstance(value, Sized):
        raise ValueError(
            f"{name} must be a sized container, got {type(value).__name__}; "
            "wrap it in a list or tuple before validating"
        )
    if len(value) == 0:
        raise ValueError(
            f"{name} is empty; this usually means an upstream filter matched nothing, "
            "so check the selection criteria rather than the consumer"
        )
    return value


def require_subset(values: Iterable[str], allowed: Iterable[str], name: str) -> list[str]:
    """Check that requested names are all known, and return them in input order.

    Raises:
        ValueError: If any requested name is unknown. The message lists the
            valid options, because "unknown key" without the alternatives forces
            the reader into the source.
    """
    requested = list(values)
    valid: Sequence[str] = sorted(set(allowed))
    unknown = [item for item in requested if item not in set(valid)]
    if unknown:
        raise ValueError(
            f"{name} contains unknown entries {unknown}; valid options are {list(valid)}"
        )
    return requested


def validate_layers(layers: Iterable[int], n_layers: int, *, name: str = "eval.layers") -> list[int]:
    """Validate configured layer indices against a loaded model's depth.

    GPT-2 has 12 layers indexed ``0..11``; a default that includes ``12`` is a
    silent bug until the first capture. Raise here with the valid range named.

    Raises:
        ValueError: If ``n_layers`` is not positive or any index is out of range.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    resolved: list[int] = []
    for index in layers:
        actual = index + n_layers if index < 0 else int(index)
        if not 0 <= actual < n_layers:
            raise ValueError(
                f"{name} index {index} is out of range for a {n_layers}-layer model; "
                f"valid indices are 0..{n_layers - 1} (or -1..-{n_layers} from the end)"
            )
        resolved.append(actual)
    return resolved
