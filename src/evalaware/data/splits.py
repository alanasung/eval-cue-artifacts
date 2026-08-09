"""Train / validation / test splits with a fixed RNG and no leakage.

A split that looks random but is actually sorted by label is how a probe
overfits the train set and looks perfect on a "held-out" set that is just the
same strata in a different order. Everything here is driven by an explicit seed
and returns the indices so the same partition can be rebuilt from a manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np

from ..utils.validation import require_in_range, require_positive

__all__ = ["SplitBundle", "split_items", "split_indices"]

T = TypeVar("T")


@dataclass(frozen=True)
class SplitBundle:
    """One partition of a dataset, with the indices that produced it."""

    train: list[object]
    val: list[object]
    test: list[object]
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]
    seed: int

    @property
    def n(self) -> dict[str, int]:
        return self.sizes()

    def sizes(self) -> dict[str, int]:
        """Split sizes keyed by partition name."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.sizes(),
            "seed": self.seed,
            "train_idx": list(self.train_idx),
            "val_idx": list(self.val_idx),
            "test_idx": list(self.test_idx),
        }


def split_indices(
    n_items: int,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Return train/val/test index lists for ``n_items`` examples.

    Raises:
        ValueError: If fractions do not sum to 1, or if any split would be empty
            when ``n_items`` is large enough that emptiness is avoidable.
    """
    require_positive(n_items, "n_items")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"split fractions must sum to 1.0, got {total:.6f} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )
    for name, value in (("train_frac", train_frac), ("val_frac", val_frac), ("test_frac", test_frac)):
        require_in_range(value, name, low=0.0, high=1.0)

    order = np.arange(n_items)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)

    n_train = int(round(n_items * train_frac))
    n_val = int(round(n_items * val_frac))
    # Put the remainder in test so rounding never drops or doubles an item.
    n_train = min(n_train, n_items)
    n_val = min(n_val, n_items - n_train)
    n_test = n_items - n_train - n_val

    train_idx = order[:n_train].tolist()
    val_idx = order[n_train : n_train + n_val].tolist()
    test_idx = order[n_train + n_val :].tolist()
    if n_items >= 10 and min(n_train, n_val, n_test) == 0:
        raise ValueError(
            f"split produced an empty partition for n_items={n_items} "
            f"(train={n_train}, val={n_val}, test={n_test}); raise n_items or "
            "adjust the fractions"
        )
    return train_idx, val_idx, test_idx


def split_items(
    items: Sequence[T],
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
) -> SplitBundle:
    """Partition a sequence into train/val/test with recorded indices."""
    train_idx, val_idx, test_idx = split_indices(
        len(items),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        shuffle=shuffle,
    )
    materialised = list(items)
    return SplitBundle(
        train=[materialised[i] for i in train_idx],
        val=[materialised[i] for i in val_idx],
        test=[materialised[i] for i in test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=seed,
    )
