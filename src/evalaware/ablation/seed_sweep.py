"""Cross-seed sweeps expressed as a library function returning a structured dict.

Shell loops over seeds hide failures and lose per-cell provenance. This module
runs a callable over a seed grid, records elapsed time and metrics per cell,
and refuses to silently skip a cell that raised.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..utils.git import git_sha
from ..utils.validation import require_non_empty, require_positive

__all__ = ["run_seed_sweep"]

CellFn = Callable[[int], dict[str, Any]]


def run_seed_sweep(
    cell_fn: CellFn,
    seeds: Sequence[int] | None = None,
    *,
    task: str = "A_seed_sweep",
    metric_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run ``cell_fn(seed)`` for every seed and aggregate per-cell metrics.

    Args:
        cell_fn: Callable that runs one seeded cell and returns a metrics dict.
            May include nested numeric fields; top-level numerics are summarised.
        seeds: Seed grid. Defaults to ``[0, 1, 2]``.
        task: Task id stamped into the payload.
        metric_keys: Optional allow-list of metric names to summarise.

    Returns:
        A payload with ``cells``, per-metric mean/std, and ``elapsed_seconds``.

    Raises:
        ValueError: If ``seeds`` is empty, or a cell returns a non-dict.
    """
    grid = list(seeds) if seeds is not None else [0, 1, 2]
    require_non_empty(grid, "seeds")
    require_positive(len(grid), "len(seeds)")

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    for seed in grid:
        cell_started = time.perf_counter()
        result = cell_fn(int(seed))
        if not isinstance(result, dict):
            raise ValueError(
                f"cell_fn(seed={seed}) must return a dict, got {type(result).__name__}"
            )
        row = dict(result)
        row.setdefault("seed", int(seed))
        row["elapsed_seconds"] = time.perf_counter() - cell_started
        cells[f"seed{seed}"] = row

    # Summarise numeric top-level metrics across cells.
    keys: list[str] = []
    if metric_keys is not None:
        keys = list(metric_keys)
    else:
        for row in cells.values():
            for key, value in row.items():
                if key in {"seed", "elapsed_seconds", "git_sha", "task", "notes"}:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in keys:
                        keys.append(key)

    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in cells.values()
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        summary[key] = {"mean": mean, "std": var**0.5, "n": float(len(values))}

    return {
        "task": task,
        "seed": int(grid[0]),
        "git_sha": git_sha(),
        "n": len(grid),
        "n_cells": len(grid),
        "seeds": [int(s) for s in grid],
        "is_synthetic": True,
        "cells": cells,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
