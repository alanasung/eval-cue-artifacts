"""Prove the configuration plumbing works, with no data and no model weights.

This is the first thing to run on a fresh clone and the first thing to run after
a config refactor. It composes the config, seeds the RNGs, creates a run
directory, and writes the resolved config plus the git SHA -- the four things
every real run depends on -- while touching no dataset, downloading no
checkpoint, and importing no model.

A failure here is a configuration failure, which is worth separating from a
pipeline failure: the two have completely different fixes and the config one is
the one that wastes an afternoon of GPU time if it surfaces late.

Usage:
    python scripts/run_config_smoke_test.py
    python scripts/run_config_smoke_test.py --preset pilot -o run.seed=7 -o model=gpt2
    python scripts/run_config_smoke_test.py --all-presets --output-dir .scratch/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose the config, seed, and write a run directory with no data present.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Experiment preset to compose, e.g. pilot. Omit for the base config.",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Compose every preset in configs/experiment/ instead of just one.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra-style override; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to create the run directory. Defaults to paths.runs from the config.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Compose and validate only; do not create a run directory.",
    )
    return parser


def _compose(preset: str | None, overrides: list[str]):
    from evalaware.configs.loader import load_config

    full = list(overrides)
    if preset:
        full.insert(0, f"experiment={preset}")
    return load_config(overrides=full)


def _smoke_one(preset: str | None, args: argparse.Namespace) -> int:
    from evalaware.configs.loader import save_resolved_config
    from evalaware.utils.git import git_state
    from evalaware.utils.reproducibility import set_seed
    from evalaware.utils.run_manifest import create_run_dir, save_run_metadata

    label = preset or "config"
    cfg = _compose(preset, args.override)
    report = set_seed(int(cfg.run.seed), deterministic=bool(cfg.run.deterministic))
    state = git_state()

    print(f"[{label}] composed: experiment={cfg.experiment.name} task={cfg.experiment.task_id}")
    print(f"[{label}] model={cfg.model.name} device={cfg.model.device} dtype={cfg.model.dtype}")
    print(f"[{label}] seed={cfg.run.seed} git={state.sha[:12]} dirty={state.dirty}")
    for caveat in report.caveats:
        print(f"[{label}] caveat: {caveat}")

    if args.no_write:
        return 0

    base = args.output_dir or cfg.paths.runs
    run_dir = create_run_dir(base, suffix=f"smoke-{label}")
    save_resolved_config(cfg, run_dir / "config.yaml")
    save_run_metadata(
        run_dir,
        {
            "purpose": "config smoke test",
            "preset": label,
            "overrides": list(args.override),
            "determinism": report.to_dict(),
            "git": state.to_dict(),
        },
        seed=int(cfg.run.seed),
        profile=str(cfg.run.profile),
        script="scripts/run_config_smoke_test.py",
    )
    print(f"[{label}] wrote {run_dir / 'config.yaml'} and {run_dir / 'run_metadata.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Compose one preset or all of them, and write the run scaffolding."""
    args = build_parser().parse_args(argv)

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.configs.loader import available_presets

    targets: list[str | None]
    if args.all_presets:
        targets = list(available_presets())
        if not targets:
            print("no presets found in configs/experiment/", file=sys.stderr)
            return 1
    else:
        targets = [args.preset]

    failures = 0
    for target in targets:
        try:
            failures += _smoke_one(target, args)
        except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not crash
            print(f"[{target or 'config'}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1

    print(f"\n{len(targets) - failures}/{len(targets)} configurations composed cleanly")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
