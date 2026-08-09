"""Synthetic item builders for pilot runs and harness validation.

Synthetic data is labelled as such in every payload. Numbers produced from it
are harness-validation results, never measured claims about a model or a
corpus. The builders here plant structure the evaluation instrument is supposed
to recover, so a failing recovery rate is a bug in the instrument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from ..utils.validation import require_positive

__all__ = ["SyntheticItem", "build_synthetic_items", "planted_score_table"]

ItemKind = Literal["neutral", "positive", "negative", "adversarial"]


@dataclass(frozen=True)
class SyntheticItem:
    """One synthetic evaluation example with a planted label and score."""

    id: str
    prompt: str
    label: int
    score: float
    kind: ItemKind
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["is_synthetic"] = True
        return payload

    def __getitem__(self, key: str) -> Any:
        """Mapping access so tests/scripts can treat items like dict rows."""
        if key == "is_synthetic":
            return True
        if key in self.meta:
            return self.meta[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


def build_synthetic_items(
    n_items: int = 512,
    *,
    seed: int = 0,
    positive_rate: float = 0.5,
    score_noise: float = 0.15,
) -> list[SyntheticItem]:
    """Build a balanced synthetic set with noisy but recoverable scores.

    Args:
        n_items: Target size. Pilot configs use >= 512 so the test split after
            a 60/20/20 cut still supports bootstrap intervals.
        seed: Deterministic RNG seed.
        positive_rate: Fraction of items with label 1.
        score_noise: Gaussian noise on the latent score; higher makes the
            planted structure harder to recover.

    Raises:
        ValueError: If ``n_items`` is not positive or ``positive_rate`` is out
            of range.
    """
    require_positive(n_items, "n_items")
    if not 0.0 < positive_rate < 1.0:
        raise ValueError(f"positive_rate must be in (0, 1), got {positive_rate}")
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_items * positive_rate))
    n_pos = min(max(n_pos, 1), n_items - 1)
    labels = np.array([1] * n_pos + [0] * (n_items - n_pos), dtype=int)
    rng.shuffle(labels)

    items: list[SyntheticItem] = []
    for index, label in enumerate(labels):
        latent = 0.8 if label == 1 else 0.2
        score = float(np.clip(latent + rng.normal(0.0, score_noise), 0.0, 1.0))
        kind: ItemKind
        if abs(score - 0.5) < 0.05:
            kind = "adversarial"
        elif label == 1:
            kind = "positive"
        elif label == 0 and score > 0.55:
            kind = "adversarial"
        else:
            kind = "negative" if label == 0 else "neutral"
        items.append(
            SyntheticItem(
                id=f"syn-{seed:04d}-{index:05d}",
                prompt=f"Synthetic prompt {index} (seed={seed}, label={label}).",
                label=int(label),
                score=score,
                kind=kind,
                meta={"seed": seed, "is_synthetic": True, "index": index},
            )
        )
    return items


def planted_score_table(
    n_items: int = 512,
    *,
    seed: int = 0,
    components: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Planted per-component scores for failure-harness recovery tests.

    Returns arrays keyed by component name plus ``label`` and ``full``. The
    ``echo``-analogue component is constructed to be more valuable than the
    others so the harness has a known ranking to recover.
    """
    require_positive(n_items, "n_items")
    names = list(components) if components is not None else ["primary", "secondary", "control"]
    if len(names) < 2:
        raise ValueError(f"need at least two components, got {names}")
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=n_items)
    primary = np.clip(label.astype(float) * 0.7 + rng.normal(0, 0.1, n_items), 0, 1)
    secondary = np.clip(0.4 + rng.normal(0, 0.15, n_items), 0, 1)
    control = np.clip(rng.uniform(0.3, 0.7, n_items), 0, 1)
    table: dict[str, np.ndarray] = {
        "label": label.astype(float),
        names[0]: primary,
        names[1]: secondary,
    }
    if len(names) > 2:
        table[names[2]] = control
    for extra in names[3:]:
        table[extra] = np.clip(rng.uniform(0.2, 0.8, n_items), 0, 1)
    table["full"] = np.clip(
        0.6 * table[names[0]] + 0.3 * table[names[1]] + 0.1 * rng.uniform(0, 1, n_items),
        0,
        1,
    )
    return table
