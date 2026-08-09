"""Catalogue of checkpoints known to run, with honest local-hardware limits.

Model names in configs are opaque strings. Nothing stops someone setting a 70B
repo id in a pilot config, and the failure arrives half an hour later as an
out-of-memory error or as swap thrashing that never errors at all.

This registry is the pre-flight answer. Every entry records parameter count,
family, and one boolean that matters more than the rest: whether the checkpoint
actually fits on the 16 GiB unified-memory Apple M4 that pilot runs target.
``fits_on_m4`` is about float32 inference plus activation capture, which is the
heaviest thing this codebase does locally, not about the weights alone.

Entries are ordered smallest first, because the right way to debug a pipeline is
on the smallest model that exercises it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["MODEL_REGISTRY", "Family", "ModelSpec", "get_model_spec", "list_models", "smallest"]

Family = Literal["qwen", "llama", "gpt2", "pythia", "olmo", "gemma"]


@dataclass(frozen=True)
class ModelSpec:
    """One tested checkpoint.

    Args:
        name: HuggingFace repo id, used verbatim as ``model.name``.
        params_b: Parameter count in billions, for sorting and rough budgeting.
        family: Architecture family, which decides how config attributes and
            layer lists are named.
        fits_on_m4: Whether this runs on a 16 GiB Apple M4 with activation
            capture enabled. False means "needs a real GPU", not "will be slow".
        notes: What this checkpoint is good for, and what it is bad at.
    """

    name: str
    params_b: float
    family: Family
    fits_on_m4: bool
    revision: str
    notes: str

    @property
    def approx_float32_gb(self) -> float:
        """Weight footprint in float32. Activations and KV cache are on top."""
        return round(self.params_b * 4.0, 2)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "params_b": self.params_b,
            "family": self.family,
            "fits_on_m4": self.fits_on_m4,
            "revision": self.revision,
            "approx_float32_gb": self.approx_float32_gb,
            "notes": self.notes,
        }


_SPECS: tuple[tuple[str, ModelSpec], ...] = (
    (
        "gpt2",
        ModelSpec(
            name="gpt2",
            params_b=0.124,
            family="gpt2",
            fits_on_m4=True,
            revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
            notes=(
                "Fastest smoke-test model and the one to debug hooks against, since "
                "its block list lives at transformer.h rather than model.layers. "
                "Too weak for instruction-following tasks."
            ),
        ),
    ),
    (
        "pythia-160m",
        ModelSpec(
            name="EleutherAI/pythia-160m",
            params_b=0.16,
            family="pythia",
            fits_on_m4=True,
            revision="main",
            notes=(
                "Public intermediate training checkpoints make this the model for "
                "any claim about how a property emerges during training."
            ),
        ),
    ),
    (
        "pythia-410m",
        ModelSpec(
            name="EleutherAI/pythia-410m",
            params_b=0.41,
            family="pythia",
            fits_on_m4=True,
            revision="main",
            notes="Second point on a Pythia scaling curve, same tokenizer as pythia-160m.",
        ),
    ),
    (
        "qwen2.5-0.5b",
        ModelSpec(
            name="Qwen/Qwen2.5-0.5B-Instruct",
            params_b=0.5,
            family="qwen",
            fits_on_m4=True,
            revision="main",
            notes=(
                "Default pilot model: instruction-tuned, small enough for full-layer "
                "activation capture, and standard llama-style attribute names."
            ),
        ),
    ),
    (
        "llama-3.2-1b",
        ModelSpec(
            name="meta-llama/Llama-3.2-1B-Instruct",
            params_b=1.24,
            family="llama",
            fits_on_m4=True,
            revision="main",
            notes=(
                "Gated on the Hub, so it needs HF_TOKEN; call require_api_key before "
                "loading. Strongest instruction-following in the fits-locally tier."
            ),
        ),
    ),
    (
        "olmo-2-1b",
        ModelSpec(
            name="allenai/OLMo-2-0425-1B-Instruct",
            params_b=1.48,
            family="olmo",
            fits_on_m4=True,
            revision="main",
            notes=(
                "Fully open training data, which is what makes contamination "
                "arguments checkable rather than assumed."
            ),
        ),
    ),
    (
        "qwen2.5-1.5b",
        ModelSpec(
            name="Qwen/Qwen2.5-1.5B-Instruct",
            params_b=1.54,
            family="qwen",
            fits_on_m4=True,
            revision="main",
            notes=(
                "Third point on a Qwen scaling curve. Capture every layer in float16 "
                "here; float32 full-depth capture is tight on 16 GiB."
            ),
        ),
    ),
    (
        "gemma-3-4b",
        ModelSpec(
            name="google/gemma-3-4b-it",
            params_b=4.3,
            family="gemma",
            fits_on_m4=False,
            revision="main",
            notes=(
                "Multimodal wrapper: the text stack is under config.text_config and "
                "model.language_model.layers. Full-profile only, needs a real GPU."
            ),
        ),
    ),
)

MODEL_REGISTRY: dict[str, ModelSpec] = dict(_SPECS)


def list_models(*, fits_on_m4: bool | None = None, family: Family | None = None) -> list[str]:
    """Registry keys, smallest first, optionally filtered.

    Sorted by size rather than alphabetically because the useful question is
    almost always "what is the smallest thing that will do".
    """
    selected = [
        key
        for key, spec in MODEL_REGISTRY.items()
        if (fits_on_m4 is None or spec.fits_on_m4 == fits_on_m4)
        and (family is None or spec.family == family)
    ]
    return sorted(selected, key=lambda key: MODEL_REGISTRY[key].params_b)


def get_model_spec(key: str) -> ModelSpec:
    """Look up a checkpoint by registry key or by full repo id.

    Raises:
        ValueError: If the key is unknown, listing every valid key. Adding a
            checkpoint to the registry is one edit; guessing a repo id from an
            error message is not.
    """
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    for spec in MODEL_REGISTRY.values():
        if spec.name == key:
            return spec
    raise ValueError(
        f"unknown model {key!r}; registered keys are {list_models()}. "
        "Add a ModelSpec to models/registry.py before using a new checkpoint, "
        "so its local-hardware limits are recorded alongside it."
    )


def smallest(*, fits_on_m4: bool = True) -> ModelSpec:
    """The smallest registered checkpoint, i.e. the right one to debug with.

    Raises:
        ValueError: If the filter matches nothing.
    """
    keys = list_models(fits_on_m4=fits_on_m4 or None)
    if not keys:
        raise ValueError(
            f"no registered model matches fits_on_m4={fits_on_m4}; "
            f"registered keys are {list_models()}"
        )
    return MODEL_REGISTRY[keys[0]]
