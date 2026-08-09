"""The catalogue of experiment presets and the task ids they must produce.

:func:`~.loader.available_presets` reports which YAML files exist on disk. That
is not the same question as which presets are *supposed* to exist. A preset
deleted during a refactor disappears silently from a disk listing, and the paper
section that depended on it simply stops being generated.

This registry is the declared side of that pair. It names each preset, says in
one sentence what it is for, and pins the ``task_id`` the preset is expected to
stamp onto its results. A consistency test can then compare declared against
on-disk in both directions, and result aggregation can check that the task id in
a results file matches the preset that claims to have produced it.

Preset names are deliberately generic. A shared spine cannot know whether a
given repo studies probes, monitors, or refusal directions, so the axes named
here are the ones every empirical project has: a smoke run, a baseline, an
ablation arm, a sweep, a scaling curve, and a robustness check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "PRESET_REGISTRY",
    "PresetSpec",
    "PresetTier",
    "expected_task_id",
    "get_preset",
    "list_presets",
    "validate_preset_name",
]

PresetTier = Literal["smoke", "core", "extension"]


@dataclass(frozen=True)
class PresetSpec:
    """Declared identity of one experiment preset.

    ``task_id`` matches an entry in ``docs/TASK.md``. Pinning it here is what
    lets aggregation refuse a results file whose task id does not match the
    preset that wrote it -- the usual sign of a copy-pasted config.
    """

    name: str
    task_id: str
    description: str
    tier: PresetTier = "core"
    requires_gpu: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "task_id": self.task_id,
            "description": self.description,
            "tier": self.tier,
            "requires_gpu": self.requires_gpu,
        }


_SPECS: tuple[PresetSpec, ...] = (
    PresetSpec(
        name="smoke",
        task_id="E00",
        description=(
            "Smallest configuration that exercises every stage end to end. "
            "Runs in under a minute and is what CI executes; its numbers are "
            "not reportable."
        ),
        tier="smoke",
    ),
    PresetSpec(
        name="baseline",
        task_id="E01",
        description=(
            "Reference condition with every component enabled. Every other "
            "preset is read as a difference against this one."
        ),
    ),
    PresetSpec(
        name="ablation_controls",
        task_id="E02",
        description=(
            "Leave-one-out and single-component arms plus the control arms that "
            "distinguish a real effect from the cost of intervening at all."
        ),
    ),
    PresetSpec(
        name="layer_sweep",
        task_id="E03",
        description=(
            "Sweeps the depth at which the measurement is taken, so a result is "
            "not an artifact of one arbitrarily chosen layer."
        ),
    ),
    PresetSpec(
        name="seed_variance",
        task_id="E04",
        description=(
            "Repeats the baseline across seeds to size run-to-run noise. Any "
            "effect smaller than this spread is not an effect."
        ),
    ),
    PresetSpec(
        name="scaling",
        task_id="E05",
        description=(
            "Repeats the core measurement across model sizes to show whether the "
            "effect strengthens, flattens, or reverses with scale."
        ),
        tier="extension",
        requires_gpu=True,
    ),
    PresetSpec(
        name="robustness",
        task_id="E06",
        description=(
            "Perturbs prompt phrasing, ordering, and formatting to separate a "
            "finding about the model from a finding about one template."
        ),
        tier="extension",
    ),
)

PRESET_REGISTRY: dict[str, PresetSpec] = {spec.name: spec for spec in _SPECS}


def list_presets(*, tier: PresetTier | None = None) -> list[str]:
    """Declared preset names, optionally restricted to one tier."""
    return sorted(
        name for name, spec in PRESET_REGISTRY.items() if tier is None or spec.tier == tier
    )


def validate_preset_name(name: str) -> str:
    """Check that a preset name is declared, and return it.

    Raises:
        ValueError: If the name is unknown. The message lists every valid name,
            because the failure is almost always a near-miss spelling and the
            reader needs the alternatives, not the fact of the error.
    """
    if name not in PRESET_REGISTRY:
        raise ValueError(
            f"unknown preset {name!r}; declared presets are {list_presets()}. "
            "Add a PresetSpec to configs/presets.py if this preset is new."
        )
    return name


def get_preset(name: str) -> PresetSpec:
    """Look up one preset spec.

    Raises:
        ValueError: If the name is not declared.
    """
    return PRESET_REGISTRY[validate_preset_name(name)]


def expected_task_id(name: str) -> str:
    """The task id a preset's results are required to carry."""
    return get_preset(name).task_id
