"""First-class ablation arms returning structured dicts.

Ablations are library functions, not shell loops. Each arm records the metrics
that matter, the seed, elapsed time, and whether the run was synthetic, so a
later aggregation step can rebuild every table from the raw JSONs alone.
"""

from __future__ import annotations

from .component_dropout import run_component_dropout
from .controls import run_control_arms
from .seed_sweep import run_seed_sweep

__all__ = [
    "run_component_dropout",
    "run_control_arms",
    "run_seed_sweep",
]
