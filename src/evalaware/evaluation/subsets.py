"""Canonical enumeration and naming of component subsets.

Ablation results are keyed by which components were enabled. If two places
disagree about how to spell that key -- one writes ``"vision+text"``, another
``"text,vision"`` -- the aggregation step produces two rows where there should
be one, and the duplicate looks like a genuine second condition. That failure is
silent and survives all the way into a table.

:func:`subset_label` is the single answer to "what is this subset called":
sorted, ``+``-joined, with reserved names for the empty and full sets. Every
cache key, result key, and table row goes through it.

The enumerators exist because the standard ablation designs are small and fixed:
the full set (what the system does), singletons (what each component achieves
alone), leave-one-out (what each component contributes on the margin), and pairs
(where components are redundant). The powerset is available but capped, because
2^n grows past any compute budget somewhere around n=15 and the failure mode is
a job that appears to hang.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Literal

from .protocols import ComponentSet, as_component_set

__all__ = [
    "DESIGNS",
    "Design",
    "enumerate_subsets",
    "full_set",
    "leave_one_out",
    "pairs",
    "parse_label",
    "powerset",
    "singletons",
    "subset_label",
]

Design = Literal["full", "singletons", "leave_one_out", "pairs", "powerset"]
DESIGNS: tuple[Design, ...] = ("full", "singletons", "leave_one_out", "pairs", "powerset")

EMPTY_LABEL = "none"
SEPARATOR = "+"


def subset_label(subset: Iterable[str]) -> str:
    """Canonical name for a component subset: sorted and ``+``-joined.

    Sorting is what makes the name canonical. ``{"b", "a"}`` and ``{"a", "b"}``
    are the same condition and must produce the same key, or the same condition
    appears twice in the results.

    Raises:
        TypeError: If handed a bare string, which would decompose into
            characters.
        ValueError: If any component name is empty or contains the separator,
            since either makes the label ambiguous to parse back.
    """
    members = sorted(as_component_set(subset))
    for name in members:
        if SEPARATOR in name:
            raise ValueError(
                f"component name {name!r} contains the label separator {SEPARATOR!r}, "
                "which makes subset labels ambiguous; rename the component"
            )
    return SEPARATOR.join(members) if members else EMPTY_LABEL


def parse_label(label: str) -> ComponentSet:
    """Invert :func:`subset_label`. Used when reading result keys back."""
    if label == EMPTY_LABEL or not label:
        return frozenset()
    return frozenset(label.split(SEPARATOR))


def full_set(components: Iterable[str]) -> ComponentSet:
    """Everything enabled: the reference condition."""
    return as_component_set(components)


def singletons(components: Iterable[str]) -> list[ComponentSet]:
    """Each component alone, in sorted order.

    Answers "how far does this component get on its own", which is the
    denominator for any claim that components are complementary.
    """
    return [frozenset({name}) for name in sorted(as_component_set(components))]


def leave_one_out(components: Iterable[str]) -> list[ComponentSet]:
    """Everything except one, in sorted order of the removed component.

    The marginal-contribution design. Preferred over singletons for attribution
    because it measures what is lost when a component is removed from a working
    system, which is the deployment-relevant question.
    """
    everything = as_component_set(components)
    return [everything - {name} for name in sorted(everything)]


def pairs(components: Iterable[str]) -> list[ComponentSet]:
    """Every unordered pair, in sorted order. Exposes redundancy between two."""
    names = sorted(as_component_set(components))
    return [frozenset(pair) for pair in combinations(names, 2)]


def powerset(
    components: Iterable[str],
    *,
    max_subsets: int = 256,
    include_empty: bool = False,
) -> list[ComponentSet]:
    """Every subset, smallest first, capped.

    Args:
        components: Component names.
        max_subsets: Refuse to enumerate more than this. The default admits
            eight components; beyond that the caller should choose a design.
        include_empty: Include the empty set, which is the "no components"
            control arm and is often exactly what is wanted.

    Raises:
        ValueError: If the powerset exceeds ``max_subsets``, naming the count
            and the designs to use instead.
    """
    names = sorted(as_component_set(components))
    total = 2 ** len(names) - (0 if include_empty else 1)
    if total > max_subsets:
        raise ValueError(
            f"the powerset of {len(names)} components has {total} subsets, above the "
            f"cap of {max_subsets}; use leave_one_out or pairs instead, or raise "
            "max_subsets deliberately after checking the compute cost"
        )
    out: list[ComponentSet] = []
    start = 0 if include_empty else 1
    for size in range(start, len(names) + 1):
        out.extend(frozenset(combo) for combo in combinations(names, size))
    return out


def enumerate_subsets(
    components: Iterable[str],
    designs: Iterable[Design] = ("full", "singletons", "leave_one_out"),
    *,
    max_subsets: int = 256,
) -> dict[str, ComponentSet]:
    """Union of several designs, deduplicated and keyed by canonical label.

    Returning a label-keyed mapping rather than a list means duplicates across
    designs collapse automatically. With two components, leave-one-out and
    singletons coincide, and running both conditions twice would be wasted
    compute reported as two independent measurements.

    Raises:
        ValueError: On an unknown design name, or if the powerset is over cap.
    """
    everything = as_component_set(components)
    unknown = [name for name in designs if name not in DESIGNS]
    if unknown:
        raise ValueError(f"unknown subset designs {unknown}; valid designs are {list(DESIGNS)}")

    collected: list[ComponentSet] = []
    for design in designs:
        if design == "full":
            collected.append(full_set(everything))
        elif design == "singletons":
            collected.extend(singletons(everything))
        elif design == "leave_one_out":
            collected.extend(leave_one_out(everything))
        elif design == "pairs":
            collected.extend(pairs(everything))
        else:
            collected.extend(powerset(everything, max_subsets=max_subsets))
    return {subset_label(subset): subset for subset in collected}
