"""Activation / feature construction: measured residual path with synthetic smoke-only.

Pilot default collects last-token residual activations from a loaded small model.
``force_synthetic=True`` is reserved for smoke/plumbing and never silently
substitutes for a failed measured load.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .model_runtime import residual_blocks, try_load_causal_lm

EVAL_CUES = (
    "benchmark",
    "evaluation",
    "evaluated",
    "exam",
    "test item",
    "[test]",
    "[evaluation]",
    "[rubric]",
    "[lab]",
    "[assessment]",
    "graded",
    "scored",
    "audit",
    "standardized",
)


def surface_cue_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for c in EVAL_CUES if c in low)
    return float(hits)


def synthetic_activation(
    text: str,
    context: str,
    dim: int = 32,
    seed: int = 0,
    steer: float = 0.0,
) -> np.ndarray:
    """Build a dim-dimensional activation with surface + internal channels."""
    rng = np.random.default_rng(abs(hash((text, seed))) % (2**32))
    x = rng.normal(0, 0.1, size=dim).astype(np.float64)
    # Dims 0-7: surface lexical cues (text-classifier accessible).
    cue = surface_cue_score(text)
    x[0] = cue
    x[1] = float(bool(re.search(r"\[(test|evaluation|rubric|lab|assessment)\]", text, re.I)))
    x[2] = float("benchmark" in text.lower() or "exam" in text.lower())
    # Dims 8-15: planted internal awareness (eval contexts), not fully reducible
    # to the surface score — includes a context bit plus noise.
    internal = 1.0 if context == "eval" else 0.0
    x[8] = internal + 0.15 * rng.normal()
    x[9] = 0.7 * internal + 0.3 * (cue > 0) + steer
    x[10:16] += internal * 0.5
    return x


def batch_activations(
    items: list[dict[str, Any]], dim: int = 32, seed: int = 0, steer: float = 0.0
) -> np.ndarray:
    return np.stack(
        [
            synthetic_activation(it["text"], it["context"], dim=dim, seed=seed, steer=steer)
            for it in items
        ]
    )


def collect_activations(
    items: list[dict[str, Any]],
    *,
    model_name: str,
    revision: str | None = None,
    layer: int = -2,
    seed: int = 0,
    force_synthetic: bool = False,
    dim: int = 32,
    steer: float = 0.0,
) -> dict[str, Any]:
    """Collect last-token residual features; synthetic only when forced."""
    if force_synthetic:
        x = batch_activations(items, dim=dim, seed=seed, steer=steer)
        return {
            "activations": x,
            "mode": "synthetic",
            "is_synthetic": True,
            "dim": int(x.shape[1]),
            "layer": None,
            "model_name": model_name,
            "revision": revision,
            "fallback_reason": "force_synthetic=True",
            "notes": ["synthetic activations for smoke/plumbing only"],
        }

    runtime = try_load_causal_lm(
        model_name, revision=revision, force_synthetic=False
    )
    if runtime is None:
        raise RuntimeError(
            f"Could not load weights for {model_name!r} revision={revision!r}. "
            "Measured activation collect refused synthetic substitution. "
            "Download the model, or set force_synthetic=true for smoke only."
        )

    import torch

    model = runtime.model
    tok = runtime.tokenizer
    blocks = residual_blocks(model)
    n_layers = len(blocks)
    layer_idx = layer if layer >= 0 else n_layers + layer
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer {layer} out of range for {n_layers} blocks")

    captured: list[np.ndarray] = []

    def _hook(_module: Any, _inp: Any, out: Any) -> None:
        tensor = out[0] if isinstance(out, tuple) else out
        vec = tensor[0, -1].detach().float().cpu().numpy()
        captured.append(vec.astype(np.float64))

    handle = blocks[layer_idx].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for it in items:
                enc = tok(it["text"], return_tensors="pt", truncation=True, max_length=256)
                enc = {k: v.to(runtime.device) for k, v in enc.items()}
                model(**enc)
                if steer != 0.0 and captured:
                    # Intervention for causal checks: add steer * unit direction.
                    direction = np.ones_like(captured[-1])
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    captured[-1] = captured[-1] + float(steer) * direction
    finally:
        handle.remove()

    x = np.stack(captured, axis=0)
    return {
        "activations": x,
        "mode": "model",
        "is_synthetic": False,
        "dim": int(x.shape[1]),
        "layer": layer_idx,
        "model_name": model_name,
        "revision": runtime.revision,
        "notes": list(runtime.notes),
    }
