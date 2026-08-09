"""Run the ablation arms named in the ``ablation`` config group.

An ablation table with only treatment arms cannot distinguish "removing this
component hurt" from "intervening at all hurt". This script therefore keeps the
declared control arms separate in its output rather than mixing them into one
list of rows: control arms are the ones expected *not* to move the metric, and a
control arm that moves is a reason to stop and fix the pipeline before reading
anything else in the table.

Each (arm, seed) pair is one invocation of ``scripts/run_experiment.py``, so an
ablation run and a baseline run are produced by the same code path.

Usage:
    python scripts/run_ablation.py --preset ablation_controls
    python scripts/run_ablation.py --preset ablation_controls --arms shuffled_label --seeds 3
    python scripts/run_ablation.py --preset ablation_seeds --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_experiment.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configured ablation arms across seeds.")
    parser.add_argument("--preset", required=True, help="Experiment preset defining the ablation.")
    parser.add_argument(
        "--arms", nargs="*", default=None,
        help="Subset of ablation.enabled to run. Defaults to all of them.",
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Number of seeds per arm, overriding ablation.n_seeds.",
    )
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Extra Hydra override applied to every arm; repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue past a failing arm.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Manifest path. Defaults to <results>/ablations/<preset>.json.",
    )
    return parser


def plan_arms(enabled: list[str], controls: list[str], n_seeds: int) -> list[dict]:
    """One record per (arm, seed), tagged with whether the arm is a control."""
    return [
        {
            "arm": arm,
            "seed": seed,
            "is_control": arm in controls,
            "overrides": [f"ablation.enabled=[{arm}]", f"run.seed={seed}"],
        }
        for arm in enabled
        for seed in range(n_seeds)
    ]


def run_arm(preset: str, item: dict, extra: list[str]) -> dict:
    command = [sys.executable, str(RUNNER), f"experiment={preset}", *item["overrides"], *extra]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        **item,
        "command": command[1:],
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stderr_tail": completed.stderr.strip().splitlines()[-5:],
    }


def main(argv: list[str] | None = None) -> int:
    """Run every requested ablation arm and write the ablation manifest."""
    args = build_parser().parse_args(argv)

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.configs.loader import load_config
    from evalaware.utils.run_manifest import utc_now_iso, write_json_atomic

    cfg = load_config(overrides=[f"experiment={args.preset}", *args.override])
    enabled = list(args.arms if args.arms is not None else cfg.ablation.enabled)
    controls = list(cfg.ablation.control_arms)
    n_seeds = int(args.seeds if args.seeds is not None else cfg.ablation.n_seeds)

    if not enabled:
        print(
            f"preset {args.preset!r} enables no ablation arms; "
            "override ablation=controls or pass --arms",
            file=sys.stderr,
        )
        return 2
    unknown = [arm for arm in controls if arm not in enabled]
    if unknown:
        print(f"warning: control arms not in the enabled list and will not run: {unknown}")

    items = plan_arms(enabled, controls, n_seeds)
    print(f"ablation {args.preset}: {len(enabled)} arms x {n_seeds} seeds = {len(items)} runs")
    for item in items:
        tag = "control" if item["is_control"] else "treatment"
        print(f"  {item['arm']:24s} seed={item['seed']} ({tag})")
    if args.dry_run:
        return 0

    records: list[dict] = []
    started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['arm']} seed={item['seed']}", flush=True)
        record = run_arm(args.preset, item, args.override)
        records.append(record)
        if record["returncode"] != 0:
            print(f"  arm failed (exit {record['returncode']})", file=sys.stderr)
            for line in record["stderr_tail"]:
                print(f"    {line}", file=sys.stderr)
            if not args.keep_going:
                break

    failed = [record for record in records if record["returncode"] != 0]
    out = (
        Path(args.out)
        if args.out
        else Path(cfg.paths.results) / "ablations" / f"{args.preset}.json"
    )
    write_json_atomic(
        out,
        {
            "task": cfg.experiment.task_id,
            "preset": args.preset,
            "created_at": utc_now_iso(),
            "arms": enabled,
            "control_arms": controls,
            "n_seeds": n_seeds,
            "n_failed": len(failed),
            "total_elapsed_seconds": round(time.perf_counter() - started, 3),
            "runs": records,
        },
    )
    print(f"\n{len(records) - len(failed)}/{len(records)} arm runs succeeded; manifest at {out}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
