"""Model and tokenizer loading, with architecture details normalised.

Two things make loading worth wrapping.

The first is that every model family names its config fields differently.
Llama and Qwen say ``num_hidden_layers``, GPT-2 says ``n_layer``, Pythia nests
under ``gpt_neox``, and Gemma 3 hides the text stack under ``text_config``.
Code that reaches for ``config.num_hidden_layers`` works until the day someone
runs the same script against GPT-2, then fails with ``AttributeError`` several
minutes into a load. :class:`LoadedModel` normalises the three numbers that
downstream code actually needs.

The second is credentials. A gated checkpoint fails deep inside the HTTP stack
with a 401 that says nothing about which environment variable to export.
:func:`require_api_key` turns that into a message naming the variable and the
reason it is needed, checked before anything expensive happens.

On transformers 5.x the ``from_pretrained`` dtype argument is ``dtype``;
``torch_dtype`` is still accepted for backwards compatibility but is deprecated,
so this module passes ``dtype``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .device import get_device, resolve_device, resolve_dtype

__all__ = [
    "LoadedModel",
    "MissingCredentialError",
    "load_model",
    "require_api_key",
    "unload",
]

logger = logging.getLogger(__name__)

# Ordered by preference: the first attribute a config actually has wins.
_LAYER_ATTRS = ("num_hidden_layers", "n_layer", "n_layers", "num_layers")
_HEAD_ATTRS = ("num_attention_heads", "n_head", "n_heads", "num_heads")
_HIDDEN_ATTRS = ("hidden_size", "n_embd", "d_model", "model_dim")
# Multimodal checkpoints (Gemma 3, several Qwen VL variants) nest the text stack.
_NESTED_CONFIGS = ("text_config", "llm_config", "language_model_config")


class MissingCredentialError(RuntimeError):
    """Raised when an API key needed by this run is not in the environment."""


def require_api_key(var: str, purpose: str) -> str:
    """Read a required credential from the environment, failing early and clearly.

    Called before any expensive setup, so a missing key costs a second rather
    than surfacing as a 401 after a model has already been downloaded.

    Args:
        var: Environment variable name, e.g. ``"HF_TOKEN"``.
        purpose: What the key is for, quoted back in the message so the reader
            knows whether they need it at all.

    Returns:
        The key.

    Raises:
        MissingCredentialError: If the variable is unset or blank.
    """
    value = os.environ.get(var, "").strip()
    if not value:
        raise MissingCredentialError(
            f"{var} is not set, and it is required for {purpose}. "
            f"Export it in your shell or add it to .env, then re-run: export {var}=..."
        )
    return value


def _config_attr(config: Any, names: tuple[str, ...]) -> int | None:
    """First matching attribute on a config, descending into nested text configs."""
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    for nested in _NESTED_CONFIGS:
        sub = getattr(config, nested, None)
        if sub is not None:
            found = _config_attr(sub, names)
            if found is not None:
                return found
    return None


@dataclass
class LoadedModel:
    """A model, its tokenizer, and the architecture facts callers keep asking for.

    Holding the tokenizer alongside the model is not convenience: a tokenizer
    from a different revision than the weights produces plausible-looking
    garbage, and pairing them in one object makes that mismatch hard to create.
    """

    model: Any
    tokenizer: Any
    name: str
    device: str
    dtype: str = "float32"
    revision: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def n_layers(self) -> int:
        """Transformer block count, across llama/qwen/gpt2/pythia/gemma naming.

        Raises:
            AttributeError: If no known attribute is present, naming the ones
                that were tried so a new family can be added in one edit.
        """
        value = _config_attr(self.config, _LAYER_ATTRS)
        if value is None:
            raise AttributeError(
                f"cannot determine layer count for {self.name}: none of {list(_LAYER_ATTRS)} "
                "is present on its config; add the family's attribute name to _LAYER_ATTRS"
            )
        return value

    @property
    def n_heads(self) -> int:
        """Attention head count.

        Raises:
            AttributeError: If no known attribute is present.
        """
        value = _config_attr(self.config, _HEAD_ATTRS)
        if value is None:
            raise AttributeError(
                f"cannot determine head count for {self.name}: none of {list(_HEAD_ATTRS)} "
                "is present on its config; add the family's attribute name to _HEAD_ATTRS"
            )
        return value

    @property
    def hidden_size(self) -> int:
        """Residual stream width, i.e. the second dimension of cached activations.

        Raises:
            AttributeError: If no known attribute is present.
        """
        value = _config_attr(self.config, _HIDDEN_ATTRS)
        if value is None:
            raise AttributeError(
                f"cannot determine hidden size for {self.name}: none of {list(_HIDDEN_ATTRS)} "
                "is present on its config; add the family's attribute name to _HIDDEN_ATTRS"
            )
        return value

    @property
    def n_parameters(self) -> int:
        return sum(param.numel() for param in self.model.parameters())

    def to_dict(self) -> dict[str, object]:
        """Architecture summary for the run manifest."""
        return {
            "name": self.name,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "hidden_size": self.hidden_size,
            "n_parameters": self.n_parameters,
            "notes": list(self.notes),
        }


def _prepare_tokenizer(tokenizer: Any, name: str) -> list[str]:
    """Give the tokenizer a pad token and left padding, reporting what changed.

    Decoder-only checkpoints frequently ship without a pad token. Batched
    generation then either crashes or, worse, pads with token 0 and quietly
    conditions on whatever that id means.
    """
    notes: list[str] = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        notes.append(f"{name} has no pad token; using eos ({tokenizer.eos_token!r}) for padding")
        logger.warning(notes[-1])
    # Right padding puts pad tokens between the prompt and the first generated
    # token, so a decoder-only model continues from padding.
    tokenizer.padding_side = "left"
    return notes


def load_model(cfg: Any, device: str | None = None) -> LoadedModel:
    """Load a causal LM and its tokenizer under the resolved device and dtype.

    Args:
        cfg: A ``ModelConfig`` or a full ``Config``. Reads ``name``,
            ``revision``, ``dtype``, ``device``, ``trust_remote_code``, and
            ``attn_implementation``.
        device: Override the resolved device, e.g. to pin a stage to CPU.

    Returns:
        A :class:`LoadedModel` in eval mode with gradients disabled.

    Raises:
        ValueError: If no model name is configured.
        HardwareError: If the requested device is unavailable.
    """
    # Imported here rather than at module scope: transformers costs several
    # seconds to import, and result aggregation should not pay for it.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = getattr(cfg, "model", cfg)
    name = str(getattr(model_cfg, "name", "") or "")
    if not name:
        raise ValueError(
            "model.name is empty; set it to a HuggingFace repo id such as "
            "'Qwen/Qwen2.5-0.5B-Instruct' (see models/registry.py for tested options)"
        )

    target = device or resolve_device(getattr(model_cfg, "device", "auto"))
    dtype = resolve_dtype(getattr(model_cfg, "dtype", "float32"), target)
    revision = getattr(model_cfg, "revision", None)
    # Prefer a pinned registry revision when the config left revision unset.
    if not revision:
        try:
            from .registry import get_model_spec

            revision = get_model_spec(name).revision
        except ValueError:
            revision = None
    trust_remote_code = bool(getattr(model_cfg, "trust_remote_code", False))

    logger.info("loading %s (revision=%s) on %s as %s", name, revision or "default", target, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    notes = _prepare_tokenizer(tokenizer, name)

    # transformers 5.x renamed torch_dtype to dtype; torch_dtype still works but
    # is deprecated and warns on every load.
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=str(getattr(model_cfg, "attn_implementation", "eager")),
    )
    model.to(target)
    model.eval()
    model.requires_grad_(False)

    # Record the commit the Hub actually resolved, not only the requested tag.
    config_obj = getattr(model, "config", None)
    resolved_commit = revision
    if config_obj is not None:
        resolved_commit = (
            getattr(config_obj, "_commit_hash", None)
            or getattr(config_obj, "commit_hash", None)
            or revision
        )
        if resolved_commit and revision and resolved_commit != revision:
            notes.append(f"resolved commit {resolved_commit} from requested revision {revision!r}")

    info = get_device(cfg)
    notes.extend(info.notes)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        device=target,
        dtype=str(dtype).replace("torch.", ""),
        revision=str(resolved_commit) if resolved_commit else revision,
        notes=notes,
    )


def unload(loaded: LoadedModel) -> None:
    """Drop a model and release accelerator memory.

    Sweeps that load several checkpoints in one process otherwise accumulate
    them: Python's reference to the old model is gone but the allocator caches
    the blocks, and the third checkpoint fails to fit.
    """
    loaded.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
