"""Stage registration and dependency-ordered execution planning.

A pipeline written as a sequence of function calls in ``main()`` encodes its
dependency order implicitly, in the order the lines happen to appear. That order
is invisible, unverifiable, and wrong the first time someone reorders two calls
that looked independent. The symptom is a stage silently consuming a stale
artifact from the previous run.

Declaring dependencies instead makes the order derivable and checkable. A cycle
becomes an error at planning time rather than a hang; a typo in a dependency
name becomes an error before any compute is spent; and asking for one stage
automatically pulls in what it needs.

The order is deterministic. Ties in the topological sort are broken
alphabetically rather than by dictionary insertion, so the plan does not change
because a module was imported in a different order.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CyclicDependencyError",
    "DuplicateStageError",
    "StageError",
    "StageSpec",
    "UnknownStageError",
    "clear_registry",
    "get_stage",
    "list_stages",
    "resolve_order",
    "stage",
]

logger = logging.getLogger(__name__)

StageFn = Callable[..., Any]


class StageError(ValueError):
    """Base class for pipeline-definition errors, all raised before execution."""


class DuplicateStageError(StageError):
    """Two stages registered under one name."""


class UnknownStageError(StageError):
    """A stage or dependency name that was never registered."""


class CyclicDependencyError(StageError):
    """The dependency graph contains a cycle, so no valid order exists."""


@dataclass(frozen=True)
class StageSpec:
    """One registered pipeline stage."""

    name: str
    fn: StageFn
    deps: tuple[str, ...] = ()
    description: str = ""
    produces: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "deps": list(self.deps),
            "description": self.description,
            "produces": list(self.produces),
        }


_REGISTRY: dict[str, StageSpec] = {}


def _first_docstring_line(fn: StageFn) -> str:
    """Summary line of a stage function, used when no description is given."""
    lines = (fn.__doc__ or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def stage(
    name: str,
    *,
    deps: Iterable[str] = (),
    description: str = "",
    produces: Iterable[str] = (),
) -> Callable[[StageFn], StageFn]:
    """Register a function as a pipeline stage.

    The decorated function is returned unchanged, so it stays directly callable
    and directly testable without going through the runner.

    Args:
        name: Unique stage name, used on the command line and in run records.
        deps: Stages that must complete first. Validated at planning time, not
            here, so declaration order across modules does not matter.
        description: One line shown when listing stages.
        produces: Artifact names this stage writes, for documentation and for
            the resume check.

    Raises:
        DuplicateStageError: If the name is already taken. Re-registering
            silently would let an import-order accident decide which
            implementation runs.
        ValueError: If the name is empty or a stage declares itself as its own
            dependency.
    """
    if not name:
        raise ValueError("stage name must be a non-empty string, e.g. 'build_dataset'")
    if name in deps:
        raise ValueError(f"stage {name!r} lists itself as a dependency; remove it from deps")

    def decorate(fn: StageFn) -> StageFn:
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            raise DuplicateStageError(
                f"stage {name!r} is already registered by "
                f"{existing.fn.__module__}.{existing.fn.__qualname__}; "
                "rename one of them or remove the duplicate import"
            )
        _REGISTRY[name] = StageSpec(
            name=name,
            fn=fn,
            deps=tuple(deps),
            description=description or _first_docstring_line(fn),
            produces=tuple(produces),
        )
        logger.debug("registered stage %r with deps %s", name, list(deps))
        return fn

    return decorate


def list_stages() -> list[str]:
    """Registered stage names, alphabetically."""
    return sorted(_REGISTRY)


def get_stage(name: str) -> StageSpec:
    """Look up one stage.

    Raises:
        UnknownStageError: If the name is not registered, listing what is.
    """
    if name not in _REGISTRY:
        raise UnknownStageError(
            f"unknown stage {name!r}; registered stages are {list_stages()}. "
            "Check the spelling, or import the module that defines it."
        )
    return _REGISTRY[name]


def clear_registry() -> None:
    """Empty the registry. For tests that register throwaway stages."""
    _REGISTRY.clear()


def _validate_deps() -> None:
    """Check every declared dependency exists.

    Raises:
        UnknownStageError: Naming the stage, the missing dependency, and the
            registered alternatives.
    """
    for spec in _REGISTRY.values():
        for dep in spec.deps:
            if dep not in _REGISTRY:
                raise UnknownStageError(
                    f"stage {spec.name!r} depends on {dep!r}, which is not registered; "
                    f"registered stages are {list_stages()}"
                )


def resolve_order(names: Sequence[str] | None = None) -> list[str]:
    """Topologically order the requested stages and everything they need.

    Args:
        names: Stages to run. When omitted, every registered stage is planned.
            Dependencies are pulled in automatically, so asking for a late stage
            on a clean checkout produces a complete plan.

    Returns:
        Stage names in an order where every dependency precedes its dependent.
        Independent stages are ordered alphabetically for determinism.

    Raises:
        UnknownStageError: If a requested stage or a declared dependency is
            unregistered.
        CyclicDependencyError: If the requested subgraph contains a cycle. The
            message names the stages still unresolved, which is the cycle.
    """
    _validate_deps()
    requested = list(names) if names is not None else list_stages()
    for name in requested:
        get_stage(name)

    # Close over dependencies so a partial request still yields a runnable plan.
    needed: set[str] = set()
    frontier = list(requested)
    while frontier:
        current = frontier.pop()
        if current in needed:
            continue
        needed.add(current)
        frontier.extend(get_stage(current).deps)

    remaining = dict.fromkeys(sorted(needed))
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if all(dep in ordered for dep in get_stage(name).deps if dep in needed)
        )
        if not ready:
            raise CyclicDependencyError(
                f"cannot order stages {sorted(remaining)}: their dependencies form a cycle. "
                "Break it by removing one edge, or by splitting the shared work into a "
                "third stage that both depend on."
            )
        for name in ready:
            ordered.append(name)
            remaining.pop(name)
    return ordered
