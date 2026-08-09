"""Run the cross-product sweep described by the ``sweep`` config group.

A sweep is configuration, not a shell loop. The grid lives in
``configs/sweep/*.yaml``, this script expands it into concrete cells, and each
cell is executed by ``scripts/run_experiment.py`` so that a swept run and a
single run go through identical seeding and manifest code.

Two things are recorded per cell that are easy to omit and painful to lack
later: ``elapsed_seconds``, because a metric that improves in lockstep with the
compute a cell received is a budget effect rather than a real one, and the exact
override list, because "which cell was that?" is otherwise unanswerable once the
run directories are timestamped.

Usage:
    python scripts/run_sweep.py --preset sweep_layers
    python scripts/run_sweep.py --preset sweep_scale --dry-run
    python scripts/run_sweep.py --preset sweep_layers --max-cells 6 --keep-going
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_experiment.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand and execute a configured sweep.")
    parser.add_argument("--preset", required=True, help="Experiment preset defining the sweep.")
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Extra Hydra override applied to every cell; repeatable.",
    )
    parser.add_argument(
        "--max-cells", type=int, default=None,
        help="Cap the number of cells, overriding sweep.max_cells.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument(
        "--keep-going", action="store_true",
        help="Continue after a failing cell instead of stopping at the first one.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Where to write the sweep manifest. Defaults to <results>/sweeps/<preset>.json.",
    )
    return parser


def format_override(key: str, value: object) -> str:
    """Render one grid value as a Hydra override string.

    Lists need Hydra's bracket syntax with no spaces; anything with a space in it
    has to be quoted or the shell-free ``subprocess`` call still confuses Hydra's
    parser.
    """
    if isinstance(value, (list, tuple)):
        inner = ",".join(str(item) for item in value)
        return f"{key}=[{inner}]"
    text = str(value)
    return f"{key}='{text}'" if " " in text else f"{key}={text}"


def expand_grid(grid: dict, seeds: list[int]) -> list[dict]:
    """Cartesian product of the grid, crossed with the seed list.

    Each returned cell carries both a stable ``id`` (used for the result
    filename) and the override list that produced it.
    """
    keys = sorted(grid)
    value_lists = [list(grid[key]) for key in keys] or [[None]]
    cells: list[dict] = []
    for combo, seed in itertools.product(itertools.product(*value_lists), seeds):
        assignments = {} if keys == [] else dict(zip(keys, combo))
        overrides = [format_override(k, v) for k, v in assignments.items()]
        overrides.append(f"run.seed={seed}")
        parts = [f"{k.split('.')[-1]}-{_slug(v)}" for k, v in assignments.items()]
        cells.append(
            {
                "id": "_".join([*parts, f"seed{seed}"]),
                "assignments": {k: v for k, v in assignments.items()},
                "seed": seed,
                "overrides": overrides,
            }
        )
    return cells


def _slug(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value)
    return str(value).replace("/", "-")


def run_cell(preset: str, cell: dict, extra: list[str]) -> dict:
    """Execute one cell and return its outcome record."""
    command = [sys.executable, str(RUNNER), f"experiment={preset}", *cell["overrides"], *extra]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        **cell,
        "command": command[1:],
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
        "stderr_tail": completed.stderr.strip().splitlines()[-5:],
    }


def main(argv: list[str] | None = None) -> int:
    """Expand the configured grid and run every cell."""
    args = build_parser().parse_args(argv)

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.configs.loader import config_to_dict, load_config
    from evalaware.utils.run_manifest import utc_now_iso, write_json_atomic

    cfg = load_config(overrides=[f"experiment={args.preset}", *args.override])
    if not cfg.sweep.enabled:
        print(
            f"preset {args.preset!r} has sweep.enabled=false; "
            "choose a sweep preset or override sweep=layers",
            file=sys.stderr,
        )
        return 2

    grid = config_to_dict(cfg.sweep.grid)
    seeds = [int(seed) for seed in cfg.sweep.seeds]
    cells = expand_grid(grid, seeds)
    limit = args.max_cells or int(cfg.sweep.max_cells)
    if len(cells) > limit:
        print(f"grid expands to {len(cells)} cells; truncating to max_cells={limit}")
        cells = cells[:limit]

    print(f"sweep {args.preset}: {len(cells)} cells over grid keys {sorted(grid)} and seeds {seeds}")
    for cell in cells:
        print(f"  {cell['id']:40s} {' '.join(cell['overrides'])}")
    if args.dry_run:
        return 0

    records: list[dict] = []
    started = time.perf_counter()
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell['id']}", flush=True)
        record = run_cell(args.preset, cell, args.override)
        records.append(record)
        if record["returncode"] != 0:
            print(f"  cell failed (exit {record['returncode']})", file=sys.stderr)
            for line in record["stderr_tail"]:
                print(f"    {line}", file=sys.stderr)
            if not args.keep_going:
                break

    failed = [record for record in records if record["returncode"] != 0]
    out = Path(args.out) if args.out else Path(cfg.paths.results) / "sweeps" / f"{args.preset}.json"
    write_json_atomic(
        out,
        {
            "task": cfg.experiment.task_id,
            "preset": args.preset,
            "created_at": utc_now_iso(),
            "grid": grid,
            "seeds": seeds,
            "n_cells": len(records),
            "n_failed": len(failed),
            "total_elapsed_seconds": round(time.perf_counter() - started, 3),
            "cells": records,
        },
    )
    print(f"\n{len(records) - len(failed)}/{len(records)} cells succeeded; manifest at {out}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
