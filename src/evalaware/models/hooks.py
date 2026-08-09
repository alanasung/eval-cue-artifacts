"""Activation capture and intervention using plain torch forward hooks.

TransformerLens is the obvious alternative and is deliberately not used. It
supports a fixed list of architectures, rewrites weights into its own format on
load, and lags new model releases by weeks. This repo needs to run against
whatever checkpoint is current, including ones that did not exist when the code
was written, so it hooks the modules HuggingFace already gives us.

The cost of that choice is that layer lists live in a different place in every
family, which :func:`resolve_layers` absorbs.

Every context manager here removes its hooks in a ``finally`` block. This is the
single most important property of the module: a hook that survives its context
silently contaminates every subsequent forward pass, and the resulting bug looks
like a model that changed behaviour for no reason -- often several stages later,
in a different experiment.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch import nn

__all__ = [
    "ActivationCache",
    "ablate",
    "capture",
    "intervene_attention_heads",
    "patch_activations",
    "resolve_layers",
    "steer",
]

logger = logging.getLogger(__name__)

# Dotted paths to the block list, in the order they are tried. Covers
# llama/qwen/mistral, gpt2, pythia/gpt-neox, opt, and multimodal wrappers.
_LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "language_model.model.layers",
    "model.language_model.layers",
    "layers",
)


def _lookup(root: Any, dotted: str) -> Any:
    current = root
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    """Return the model's transformer blocks in depth order.

    Raises:
        AttributeError: If none of the known paths matches, listing what was
            tried so the new family can be added in one place.
    """
    for path in _LAYER_PATHS:
        found = _lookup(model, path)
        if isinstance(found, (nn.ModuleList, nn.Sequential)) and len(found) > 0:
            return list(found)
    raise AttributeError(
        f"cannot locate transformer blocks on {type(model).__name__}: none of "
        f"{list(_LAYER_PATHS)} resolved to a non-empty module list. "
        "Add this family's path to _LAYER_PATHS in models/hooks.py."
    )


def _normalize_indices(model: nn.Module, layers: Sequence[int] | None) -> list[int]:
    """Validate layer indices against the model's actual depth.

    Raises:
        ValueError: If an index is out of range. Negative indices are resolved
            from the end, so ``-1`` means the final block.
    """
    depth = len(resolve_layers(model))
    if layers is None:
        return list(range(depth))
    resolved: list[int] = []
    for index in layers:
        actual = index + depth if index < 0 else index
        if not 0 <= actual < depth:
            raise ValueError(
                f"layer index {index} is out of range for a {depth}-layer model; "
                f"valid indices are 0..{depth - 1} (or -1..-{depth} from the end)"
            )
        resolved.append(actual)
    return resolved


def _hidden_states(output: Any) -> torch.Tensor:
    """Pull the residual-stream tensor out of a block's output.

    Blocks return either a tensor or a tuple whose first element is the hidden
    state; which one depends on the family and on whether caching is enabled.
    """
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output: Any, replacement: torch.Tensor) -> Any:
    """Put a modified hidden state back into the shape the block returned."""
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


def _unit(direction: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Unit-norm the direction and match it to the hidden state's device/dtype.

    Raises:
        ValueError: If the direction is the zero vector, which would make the
            projection undefined, or if its width does not match the model.
    """
    vector = direction.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
    if vector.shape[0] != hidden.shape[-1]:
        raise ValueError(
            f"direction has width {vector.shape[0]} but the residual stream is "
            f"{hidden.shape[-1]}; the direction was probably fitted on a different model"
        )
    norm = torch.linalg.vector_norm(vector)
    if float(norm) == 0.0:
        raise ValueError(
            "direction is the zero vector, so its projection is undefined; "
            "check that the direction-fitting stage produced a non-degenerate result"
        )
    return vector / norm


class ActivationCache:
    """Captured activations, keyed by layer index.

    Stores CPU tensors by default. Keeping them on an accelerator is how a
    capture over a few hundred examples exhausts unified memory halfway through,
    long after the code that chose the batch size has returned.
    """

    def __init__(self) -> None:
        self._store: dict[int, list[torch.Tensor]] = {}

    def add(self, layer: int, activation: torch.Tensor) -> None:
        self._store.setdefault(layer, []).append(activation)

    def layers(self) -> list[int]:
        return sorted(self._store)

    def __contains__(self, layer: int) -> bool:
        return layer in self._store

    def __len__(self) -> int:
        return len(self._store)

    def get(self, layer: int) -> torch.Tensor:
        """Concatenate every captured batch for one layer along the batch axis.

        Raises:
            KeyError: If nothing was captured at that layer, which usually means
                the capture context did not include it.
        """
        if layer not in self._store:
            raise KeyError(
                f"no activations captured at layer {layer}; captured layers are "
                f"{self.layers()} -- widen the `layers` argument to capture()"
            )
        return torch.cat(self._store[layer], dim=0)

    def numpy(self, layer: int) -> np.ndarray:
        """Layer activations as a float32 numpy array, ready for the cache."""
        return self.get(layer).to(torch.float32).numpy()

    def clear(self) -> None:
        self._store.clear()

    def to_dict(self) -> dict[str, object]:
        """Shape summary. The arrays themselves never go into a result payload."""
        return {
            "layers": self.layers(),
            "shapes": {str(layer): list(self.get(layer).shape) for layer in self.layers()},
        }


@contextmanager
def capture(
    model: nn.Module,
    layers: Sequence[int] | None = None,
    *,
    cache: ActivationCache | None = None,
    last_token_only: bool = False,
    to_cpu: bool = True,
) -> Iterator[ActivationCache]:
    """Record residual-stream activations at the given blocks.

    Args:
        model: The model to hook.
        layers: Block indices; all of them when omitted. Negative indices count
            from the end.
        cache: Append into an existing cache, e.g. across batches.
        last_token_only: Keep only the final position, which is what a
            next-token probe reads and is far cheaper to store.
        to_cpu: Move captured tensors off the accelerator immediately.

    Yields:
        The :class:`ActivationCache` being filled.

    Raises:
        ValueError: If a layer index is out of range.
    """
    blocks = resolve_layers(model)
    indices = _normalize_indices(model, layers)
    store = cache if cache is not None else ActivationCache()
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(index: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden = _hidden_states(output).detach()
            if last_token_only:
                hidden = hidden[:, -1, :]
            store.add(index, hidden.to("cpu") if to_cpu else hidden)

        return hook

    try:
        for index in indices:
            handles.append(blocks[index].register_forward_hook(make_hook(index)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def steer(
    model: nn.Module,
    layers: Sequence[int],
    direction: torch.Tensor,
    *,
    coefficient: float = 1.0,
    positions: slice | None = None,
) -> Iterator[None]:
    """Add ``coefficient * unit(direction)`` to the residual stream.

    The direction is unit-normalised first so that ``coefficient`` means the
    same thing across layers whose activation norms differ by an order of
    magnitude. Without that, a coefficient tuned at layer 4 is a no-op at
    layer 20 and the sweep looks like the effect vanished with depth.

    Args:
        model: The model to hook.
        layers: Blocks to intervene on.
        direction: Vector of length ``hidden_size``.
        coefficient: Signed strength; negative steers the other way.
        positions: Restrict the edit to a slice of sequence positions. Defaults
            to every position.

    Raises:
        ValueError: On a zero or mis-shaped direction, or a bad layer index.
    """
    blocks = resolve_layers(model)
    indices = _normalize_indices(model, layers)
    span = positions if positions is not None else slice(None)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = _hidden_states(output)
        vector = _unit(direction, hidden)
        edited = hidden.clone()
        edited[:, span, :] = edited[:, span, :] + coefficient * vector
        return _rewrap(output, edited)

    try:
        for index in indices:
            handles.append(blocks[index].register_forward_hook(hook))
        logger.debug("steering layers %s with coefficient %s", indices, coefficient)
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def ablate(
    model: nn.Module,
    layers: Sequence[int],
    direction: torch.Tensor,
    *,
    positions: slice | None = None,
) -> Iterator[None]:
    """Project a direction out of the residual stream at the given blocks.

    This is directional ablation: ``h <- h - (h . d) d`` for unit ``d``. It
    removes the component of the representation along ``d`` while leaving the
    orthogonal complement untouched, which is the intervention that supports the
    claim "this direction carries the behaviour". Zeroing the whole activation
    would not, since it also destroys everything else the layer encoded.

    Raises:
        ValueError: On a zero or mis-shaped direction, or a bad layer index.
    """
    blocks = resolve_layers(model)
    indices = _normalize_indices(model, layers)
    span = positions if positions is not None else slice(None)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = _hidden_states(output)
        vector = _unit(direction, hidden)
        edited = hidden.clone()
        target = edited[:, span, :]
        projection = (target @ vector).unsqueeze(-1) * vector
        edited[:, span, :] = target - projection
        return _rewrap(output, edited)

    try:
        for index in indices:
            handles.append(blocks[index].register_forward_hook(hook))
        logger.debug("ablating direction at layers %s", indices)
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_activations(
    model: nn.Module,
    layer: int,
    source: torch.Tensor,
    *,
    positions: slice | None = None,
) -> Iterator[None]:
    """Paired activation patching: replace residual activations with ``source``.

    ``source`` must be shaped ``[batch, seq, hidden]`` or ``[batch, hidden]``
    when ``positions`` selects a single token. This is the intervention that
    carries causal claims about whether a report tracks an internal feature.
    """
    blocks = resolve_layers(model)
    index = _normalize_indices(model, [layer])[0]
    span = positions if positions is not None else slice(None)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = _hidden_states(output)
        edited = hidden.clone()
        patch = source.to(device=hidden.device, dtype=hidden.dtype)
        if patch.ndim == 2:
            edited[:, span, :] = patch.unsqueeze(1)
        elif patch.ndim == 3:
            edited[:, span, :] = patch[:, span, :] if patch.shape[1] == hidden.shape[1] else patch
        else:
            raise ValueError(
                f"source activations must be 2-D or 3-D, got shape {tuple(patch.shape)}"
            )
        return _rewrap(output, edited)

    try:
        handles.append(blocks[index].register_forward_hook(hook))
        logger.debug("patching activations at layer %s positions %s", index, span)
        yield
    finally:
        for handle in handles:
            handle.remove()


def _resolve_attn(block: nn.Module) -> nn.Module:
    for name in ("attn", "self_attn", "attention", "mha"):
        mod = getattr(block, name, None)
        if isinstance(mod, nn.Module):
            return mod
    raise AttributeError(
        f"cannot locate attention module on {type(block).__name__}; "
        "expected one of attn/self_attn/attention/mha"
    )


@contextmanager
def intervene_attention_heads(
    model: nn.Module,
    layer: int,
    head_indices: Sequence[int],
    pattern: torch.Tensor,
    *,
    n_heads: int | None = None,
) -> Iterator[None]:
    """Replace selected attention-head patterns with a provided pattern tensor.

    ``pattern`` is broadcast onto the chosen heads. This operates on attention
    probabilities when the module exposes them via a forward hook on the
    attention submodule; if the family returns weights in the output tuple,
    those weights are edited in place. Masking is an intervention on the
    pattern, not parameter deletion — callers that need genuine Q/K surgery
    must use a domain-specific surgery module.
    """
    blocks = resolve_layers(model)
    index = _normalize_indices(model, [layer])[0]
    attn = _resolve_attn(blocks[index])
    heads = [int(h) for h in head_indices]
    if not heads:
        raise ValueError("head_indices must be non-empty")
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        # Common HF pattern: (attn_output, attn_weights, ...) when output_attentions
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError(
                "intervene_attention_heads requires attention weights in the module "
                "output; call the model with output_attentions=True"
            )
        weights = output[1]
        if weights.ndim != 4:
            raise ValueError(
                f"expected attention weights shaped [batch, heads, q, k], got {tuple(weights.shape)}"
            )
        total_heads = n_heads if n_heads is not None else weights.shape[1]
        for head in heads:
            if not 0 <= head < total_heads:
                raise ValueError(
                    f"head index {head} out of range for {total_heads} heads at layer {index}"
                )
        edited = weights.clone()
        replacement = pattern.to(device=weights.device, dtype=weights.dtype)
        if replacement.ndim == 2:
            for head in heads:
                edited[:, head, :, :] = replacement
        elif replacement.ndim == 3:
            for i, head in enumerate(heads):
                edited[:, head, :, :] = replacement[i] if replacement.shape[0] == len(heads) else replacement[0]
        else:
            raise ValueError(
                f"pattern must be 2-D [q,k] or 3-D [heads,q,k], got {tuple(replacement.shape)}"
            )
        return (output[0], edited, *output[2:])

    try:
        handles.append(attn.register_forward_hook(hook))
        logger.debug("intervening on heads %s at layer %s", heads, index)
        yield
    finally:
        for handle in handles:
            handle.remove()
