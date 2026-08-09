"""Model loading, hardware resolution, hooks, generation, and instrument logging.

This is the only subpackage that imports torch and transformers. Keeping the
boundary sharp is what lets ``evaluation/`` stay model-agnostic: evaluation talks
to the callables in :mod:`..evaluation.protocols`, and anything concrete that
satisfies those callables is built here.
"""

from __future__ import annotations

from .device import (
    DeviceInfo,
    HardwareError,
    describe,
    enable_mps_fallback,
    get_device,
    guard_memory,
    resolve_device,
    resolve_dtype,
)
from .generation import TokenLogprobs, apply_chat_template, generate, sequence_logprob, token_logprobs
from .hooks import (
    ActivationCache,
    ablate,
    capture,
    intervene_attention_heads,
    patch_activations,
    resolve_layers,
    steer,
)
from .loader import LoadedModel, MissingCredentialError, load_model, require_api_key, unload
from .logged_llm import LLM_CONFIG_NAME, LLM_LOG_NAME, LoggedLLM
from .registry import MODEL_REGISTRY, Family, ModelSpec, get_model_spec, list_models, smallest

__all__ = [
    "LLM_CONFIG_NAME",
    "LLM_LOG_NAME",
    "MODEL_REGISTRY",
    "ActivationCache",
    "DeviceInfo",
    "Family",
    "HardwareError",
    "LoadedModel",
    "LoggedLLM",
    "MissingCredentialError",
    "ModelSpec",
    "TokenLogprobs",
    "ablate",
    "apply_chat_template",
    "capture",
    "describe",
    "enable_mps_fallback",
    "generate",
    "get_device",
    "get_model_spec",
    "guard_memory",
    "intervene_attention_heads",
    "list_models",
    "load_model",
    "patch_activations",
    "require_api_key",
    "resolve_device",
    "resolve_dtype",
    "resolve_layers",
    "sequence_logprob",
    "smallest",
    "steer",
    "token_logprobs",
    "unload",
]
