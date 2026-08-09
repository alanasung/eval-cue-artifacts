"""Collect every result JSON in the tree into one aggregate payload.

One command regenerates the aggregate, and the tables and figures are generated
from the aggregate rather than from whatever happened to be in someone's shell
history. That is the only way a results table stays consistent with the runs
behind it.

Two rules are enforced while collecting, both of which exist because the
alternative has burned real projects:

1. A file without ``task``, ``seed``, and ``git_sha`` is not a result. It is
   reported as a warning and excluded, rather than being folded in as an
   anonymous row nobody can trace.
2. Synthetic output is separated, never merged. A harness-validation payload and
   a measured payload look identical once they are rows in a table, so they are
   kept in different lists and the synthetic count is printed every time.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --search runs results --out results/results.json
    python scripts/aggregate_results.py --include-synthetic --strict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = ("task", "seed", "git_sha")
PROVENANCE_FIELDS = (
    "task", "seed", "git_sha", "git_dirty", "profile", "model", "created_at", "n",
    "n_dropped", "dropped_reason", "is_synthetic", "notes", "inputs",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate result JSONs into one payload.")
    parser.add_argument(
        "--search", nargs="*", default=["results", "runs"],
        help="Directories to walk for result JSONs.",
    )
    parser.add_argument(
        "--pattern", default="*.json", help="Glob applied within each search directory.",
    )
    parser.add_argument("--out", default="results/results.json", help="Aggregate output path.")
    parser.add_argument(
        "--include-synthetic", action="store_true",
        help="Also emit synthetic rows in the measured list. Off by default, and it "
             "should stay off for anything that becomes a reported table.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any candidate file failed the provenance check.",
    )
    return parser


def flatten_metrics(payload: dict, prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dicts into dotted keys, keeping numeric leaves only.

    Non-numeric leaves (per-seed records, note strings, nested lists) are dropped
    here on purpose: they belong in the raw payload, not in a table row.
    """
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def load_candidate(path: Path) -> tuple[dict | None, str | None]:
    """Read one JSON file, returning ``(payload, error)``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path}: unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(data, dict):
        return None, f"{path}: top level is {type(data).__name__}, expected an object"
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return None, f"{path}: missing provenance fields {missing}"
    return data, None


def to_row(path: Path, payload: dict) -> dict:
    """One flat row: provenance columns plus flattened numeric metrics."""
    row = {"source": str(path)}
    for field in PROVENANCE_FIELDS:
        if field in payload:
            row[field] = payload[field]
    row["metrics"] = flatten_metrics(payload.get("metrics", {}))
    return row


def collect(search: list[Path], pattern: str) -> tuple[list[dict], list[dict], list[str]]:
    """Walk the search paths and split results into measured, synthetic, and warnings."""
    measured: list[dict] = []
    synthetic: list[dict] = []
    warnings: list[str] = []
    seen: set[Path] = set()

    for directory in search:
        if not directory.is_dir():
            warnings.append(f"{directory}: not a directory, skipped")
            continue
        for path in sorted(directory.rglob(pattern)):
            resolved = path.resolve()
            if resolved in seen or path.name in ("run_metadata.json", "results.json"):
                continue
            seen.add(resolved)
            payload, error = load_candidate(path)
            if error:
                warnings.append(error)
                continue
            row = to_row(path, payload or {})
            (synthetic if row.get("is_synthetic") else measured).append(row)
    return measured, synthetic, warnings


def main(argv: list[str] | None = None) -> int:
    """Aggregate every discoverable result JSON into one file."""
    args = build_parser().parse_args(argv)

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.utils.git import git_state
    from evalaware.utils.run_manifest import utc_now_iso, write_json_atomic

    search = [Path(item) if Path(item).is_absolute() else REPO_ROOT / item for item in args.search]
    # Prefer the library aggregator when available; keep the script collector as
    # a fallback for aggregates that need the `rows` shape used by older tables.
    try:
        from evalaware.reporting.aggregate import aggregate_results, write_aggregate

        payload = aggregate_results(search, pattern=args.pattern, include_synthetic=args.include_synthetic)
        out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
        write_aggregate(payload, out)
        print(f"measured results : {payload.n_measured if hasattr(payload,'n_measured') else len(payload.measured)}")
        print(f"synthetic results: {len(payload.synthetic)} (kept separate; not reportable)")
        print(f"unusable files   : {len(payload.warnings)}")
        print(f"wrote {out} via evalaware.reporting.aggregate")
        return 1 if (args.strict and payload.warnings) else 0
    except Exception as exc:  # noqa: BLE001 - fall back
        print(f"note: library aggregator unavailable ({exc}); using script fallback")

    measured, synthetic, warnings = collect(search, args.pattern)

    for warning in warnings:
        print(f"warning: {warning}")

    rows = measured + synthetic if args.include_synthetic else measured
    state = git_state()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    write_json_atomic(
        out,
        {
            "task": "aggregate",
            "created_at": utc_now_iso(),
            "git_sha": state.sha,
            "git_dirty": state.dirty,
            "searched": [str(path) for path in search],
            "n_measured": len(measured),
            "n_synthetic": len(synthetic),
            "n_warnings": len(warnings),
            "includes_synthetic_rows": bool(args.include_synthetic),
            "rows": rows,
            "synthetic_rows": synthetic,
            "warnings": warnings,
        },
    )

    print(f"measured results : {len(measured)}")
    print(f"synthetic results: {len(synthetic)} (kept separate; not reportable)")
    print(f"unusable files   : {len(warnings)}")
    print(f"wrote {out}")
    if not measured:
        print(
            "\nNo measured results were found. The tables generated from this file "
            "will be empty, which is the correct state before any run has happened."
        )
    return 1 if (args.strict and warnings) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
