"""Batched generation and token-level log probabilities.

Three details here are the difference between usable output and quiet nonsense.

**Left padding.** Decoder-only models continue from the last position. Pad on
the right and the model continues from padding, producing fluent text that has
nothing to do with the prompt. Nothing errors; the outputs just get worse.

**Slicing off the prompt.** ``generate`` returns prompt plus continuation.
Decoding the whole thing and string-stripping the prompt back off fails whenever
the tokenizer round-trip is not exact, which is often. Slicing by input length
in token space is exact, so this module returns only new tokens.

**Temperature zero means greedy.** Passing ``temperature=0.0`` to ``generate``
is a runtime error in transformers, and passing a tiny value instead is not the
same distribution. Callers write ``temperature=0`` and mean "deterministic", so
this module translates that into ``do_sample=False`` and drops the sampling
parameters, which would otherwise warn on every call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ..utils.validation import require_positive

__all__ = [
    "TokenLogprobs",
    "apply_chat_template",
    "generate",
    "sequence_logprob",
    "token_logprobs",
]

logger = logging.getLogger(__name__)


def _batches(items: Sequence[str], size: int) -> list[Sequence[str]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _generation_kwargs(temperature: float, top_p: float) -> dict[str, Any]:
    """Sampling settings, or the greedy settings when temperature is zero."""
    if temperature <= 0.0:
        return {"do_sample": False}
    return {"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)}


def apply_chat_template(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    use_chat_template: bool = True,
    system: str | None = None,
) -> tuple[list[str], str]:
    """Format prompts with the tokenizer chat template when one exists.

    Raw prompts fed to instruction-tuned models without their chat template do
    not exercise the interface the model was trained for, so refusal and
    instruction-following measurements become meaningless.

    Returns:
        ``(formatted_prompts, path)`` where ``path`` is ``"chat_template"``,
        ``"chat_template_unavailable"``, or ``"raw"``.
    """
    if not use_chat_template:
        return list(prompts), "raw"
    formatter = getattr(tokenizer, "apply_chat_template", None)
    if formatter is None or not getattr(tokenizer, "chat_template", None):
        return list(prompts), "chat_template_unavailable"
    formatted: list[str] = []
    for prompt in prompts:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        formatted.append(
            formatter(messages, tokenize=False, add_generation_prompt=True)
        )
    return formatted, "chat_template"


def generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.95,
    batch_size: int = 4,
    device: str | None = None,
    skip_special_tokens: bool = True,
    use_chat_template: bool = True,
    system: str | None = None,
    return_meta: bool = False,
) -> list[str] | tuple[list[str], dict[str, object]]:
    """Generate a continuation for each prompt, returning only the new text.

    Args:
        model: A causal LM in eval mode.
        tokenizer: Its tokenizer. Padding side is forced to ``"left"`` here even
            if the caller changed it, because right padding silently corrupts
            batched decoder-only generation.
        prompts: One prompt per desired continuation. Order is preserved.
        max_new_tokens: Hard cap on continuation length.
        temperature: ``0`` for greedy and reproducible output. Anything above
            zero samples and is only reproducible under a fixed seed *and* a
            fixed batch composition.
        top_p: Nucleus cutoff, ignored when sampling is off.
        batch_size: Prompts per forward pass. Lower it before lowering anything
            else when memory is tight.
        device: Where to place inputs; taken from the model when omitted.
        skip_special_tokens: Strip specials from the decoded continuation.
        use_chat_template: When True, apply the tokenizer chat template if any.
        system: Optional system message used with the chat template.
        return_meta: When True, also return a dict naming the formatting path.

    Returns:
        One continuation per prompt, in input order. Optionally a metadata dict.

    Raises:
        ValueError: If ``prompts`` is empty, or if a size argument is not
            positive.
    """
    if not prompts:
        raise ValueError(
            "generate() was given no prompts; an empty batch usually means an "
            "upstream filter removed every example, so check the selection step"
        )
    require_positive(max_new_tokens, "max_new_tokens")
    require_positive(batch_size, "batch_size")

    rendered, template_path = apply_chat_template(
        tokenizer, prompts, use_chat_template=use_chat_template, system=system
    )

    target = device or str(next(model.parameters()).device)
    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = _generation_kwargs(temperature, top_p)
    outputs: list[str] = []
    try:
        for batch in _batches(list(rendered), batch_size):
            encoded = tokenizer(list(batch), return_tensors="pt", padding=True).to(target)
            with torch.no_grad():
                produced = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    **kwargs,
                )
            # Left padding makes every row's prompt end at the same column, so
            # one slice point is correct for the whole batch.
            prompt_len = encoded["input_ids"].shape[1]
            new_tokens = produced[:, prompt_len:]
            outputs.extend(
                tokenizer.batch_decode(new_tokens, skip_special_tokens=skip_special_tokens)
            )
    finally:
        tokenizer.padding_side = previous_side

    logger.debug(
        "generated %d continuations (greedy=%s, template=%s)",
        len(outputs),
        temperature <= 0.0,
        template_path,
    )
    if return_meta:
        return outputs, {"chat_template_path": template_path, "n": len(outputs)}
    return outputs


@dataclass(frozen=True)
class TokenLogprobs:
    """Per-token log probabilities for one scored sequence."""

    tokens: list[str]
    logprobs: list[float]

    @property
    def total(self) -> float:
        """Sum over tokens, i.e. the sequence log probability."""
        return float(sum(self.logprobs))

    @property
    def mean(self) -> float:
        """Length-normalised score. Use this to compare unequal-length options.

        Raw totals favour short sequences, so ranking completions by ``total``
        is a length bias dressed up as a preference.
        """
        return self.total / len(self.logprobs) if self.logprobs else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "logprobs": [round(value, 6) for value in self.logprobs],
            "total": round(self.total, 6),
            "mean": round(self.mean, 6),
        }


def token_logprobs(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    prefix: str = "",
    device: str | None = None,
) -> TokenLogprobs:
    """Log probability the model assigns to each token of ``text``.

    Args:
        model: A causal LM in eval mode.
        tokenizer: Its tokenizer.
        text: The scored continuation.
        prefix: Context conditioned on but not scored, e.g. the prompt. Scoring
            a continuation without excluding its prompt inflates every number
            with tokens the model was told.
        device: Where to run; taken from the model when omitted.

    Returns:
        A :class:`TokenLogprobs` covering only the tokens of ``text``.

    Raises:
        ValueError: If ``text`` tokenizes to nothing, which leaves no score to
            report and is usually an empty string that should have been filtered.
    """
    target = device or str(next(model.parameters()).device)
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"] if prefix else []
    text_ids = tokenizer(text, add_special_tokens=not prefix)["input_ids"]
    if not text_ids:
        raise ValueError(
            f"text tokenized to zero tokens: {text!r}; drop empty completions before scoring"
        )

    all_ids = list(prefix_ids) + list(text_ids)
    input_ids = torch.tensor([all_ids], device=target)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.to(torch.float32)

    # Position i predicts token i+1, so the score for the first scored token
    # comes from the logits one position earlier.
    log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
    start = max(len(prefix_ids) - 1, 0)
    scored_ids = all_ids[start + 1 :]
    picked = log_probs[start:, :].gather(1, torch.tensor([scored_ids], device=target).T)
    return TokenLogprobs(
        tokens=tokenizer.convert_ids_to_tokens(scored_ids),
        logprobs=[float(value) for value in picked.squeeze(-1).cpu()],
    )


def sequence_logprob(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    prefix: str = "",
    normalize: bool = True,
    device: str | None = None,
) -> float:
    """Scalar score for one continuation, length-normalised by default."""
    scored = token_logprobs(model, tokenizer, text, prefix=prefix, device=device)
    return scored.mean if normalize else scored.total
