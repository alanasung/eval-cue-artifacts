"""Local causal-LM loading with explicit synthetic fallback (no silent Hub in tests)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeModel:
    model: Any
    tokenizer: Any
    name: str
    revision: str | None
    device: str
    notes: list[str]


def try_load_causal_lm(
    model_name: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    force_synthetic: bool = False,
) -> RuntimeModel | None:
    """Load a small open-weight LM, or return None when unavailable / forced synthetic."""
    if force_synthetic or not model_name or model_name in {"x", "none", "synthetic", "missing"}:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None

    if revision is None:
        try:
            from evalaware.models.registry import get_model_spec

            revision = get_model_spec(model_name).revision
        except Exception:
            revision = None

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    try:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        model.to(device)
        model.eval()
        return RuntimeModel(
            model=model,
            tokenizer=tok,
            name=model_name,
            revision=revision,
            device=device,
            notes=[f"loaded {model_name} revision={revision or 'default'} on {device}"],
        )
    except Exception:
        return None


def residual_blocks(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("unsupported model layout for residual hooks")
