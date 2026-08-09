"""List registered experiment stages and their dependencies.

Usage:
    python scripts/list_stages.py
"""

from __future__ import annotations

import argparse
import importlib


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List registered pipeline stages.")
    parser.add_argument("--package", default="evalaware")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stages_mod = importlib.import_module(f"{args.package}.stages")
    except ModuleNotFoundError:
        print(f"no stages module for {args.package}")
        return 1
    registry = dict(getattr(stages_mod, "STAGES", {}))
    if not registry:
        print("STAGES is empty")
        return 0
    for name in sorted(registry):
        fn = registry[name]
        doc = (fn.__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else ""
        print(f"{name:24s} {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
