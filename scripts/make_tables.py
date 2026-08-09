"""Render the aggregate payload as Markdown and paper-ready LaTeX.

Tables are generated, never hand-edited. A hand-edited table drifts from the
runs behind it within a week, and the drift is invisible because both artefacts
look equally authoritative.

Three details are worth naming because they are the ones usually skipped.
Numeric columns are right-aligned with ``|---:|`` so decimal points line up.
Every metric carries a direction arrow so a reader knows whether lower is
better without consulting the methods section. And a row whose sample count is
below ``--min-n`` gets a footnote marker rather than being silently reported
next to rows twenty times its size.

Usage:
    python scripts/make_tables.py
    python scripts/make_tables.py --in results/results.json --out-dir results/tables
    python scripts/make_tables.py --metrics accuracy silent_rate --lower-is-better silent_rate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Substrings that mark a metric as one where smaller is better. Used only when
# the caller does not name the directions explicitly.
LOWER_IS_BETTER_HINTS = ("error", "mae", "rmse", "loss", "ece", "gap", "cost", "silent", "kl")
SMALL_N_MARKER = "*"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Markdown and LaTeX results tables.")
    parser.add_argument("--in", dest="source", default="results/results.json")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument(
        "--metrics", nargs="*", default=None,
        help="Metric columns to include, in order. Defaults to every metric present.",
    )
    parser.add_argument(
        "--group-by", default="task",
        help="Provenance column used as the row label.",
    )
    parser.add_argument("--lower-is-better", nargs="*", default=None)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument(
        "--min-n", type=int, default=20,
        help="Rows with fewer examples than this are footnoted as small strata.",
    )
    parser.add_argument("--caption", default="Results. No measured numbers yet.")
    parser.add_argument("--label", default="tab:results")
    return parser


def direction(metric: str, explicit: list[str] | None) -> str:
    """Return the arrow shown next to a metric name."""
    if explicit is not None:
        return "down" if metric in explicit else "up"
    lowered = metric.lower()
    return "down" if any(hint in lowered for hint in LOWER_IS_BETTER_HINTS) else "up"


def row_n(row: dict) -> int:
    """Total example count for a row, however the payload spelled it."""
    value = row.get("n")
    if isinstance(value, dict):
        return int(sum(int(item) for item in value.values() if isinstance(item, (int, float))))
    return int(value) if isinstance(value, (int, float)) else 0


def collect_metrics(rows: list[dict], requested: list[str] | None) -> list[str]:
    if requested:
        return list(requested)
    names: set[str] = set()
    for row in rows:
        names.update(row.get("metrics", {}))
    return sorted(names)


def fmt(value: object, precision: int) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def markdown_table(rows: list[dict], metrics: list[str], args: argparse.Namespace) -> str:
    arrows = {name: ("(lower better)" if direction(name, args.lower_is_better) == "down"
                     else "(higher better)") for name in metrics}
    header = [args.group_by, "seed", "n", *[f"{name} {arrows[name]}" for name in metrics]]
    align = ["---", "---:", "---:", *["---:"] * len(metrics)]

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    footnoted = False
    for row in rows:
        count = row_n(row)
        marker = SMALL_N_MARKER if 0 < count < args.min_n else ""
        footnoted = footnoted or bool(marker)
        cells = [
            str(row.get(args.group_by, "?")),
            str(row.get("seed", "?")),
            f"{count}{marker}",
            *[fmt(row.get("metrics", {}).get(name), args.precision) for name in metrics],
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if not rows:
        lines.append("| _no measured results yet_ | -- | -- |" + " -- |" * len(metrics))
    if footnoted:
        lines.append("")
        lines.append(f"{SMALL_N_MARKER} fewer than {args.min_n} examples; treat as indicative.")
    return "\n".join(lines)


def latex_table(rows: list[dict], metrics: list[str], args: argparse.Namespace) -> str:
    arrows = {
        name: (r"$\downarrow$" if direction(name, args.lower_is_better) == "down" else r"$\uparrow$")
        for name in metrics
    }
    columns = "l" + "r" * (2 + len(metrics))
    header = " & ".join(
        [args.group_by.replace("_", " "), "seed", "n"]
        + [f"{name.replace('_', ' ')} {arrows[name]}" for name in metrics]
    )
    body = []
    for row in rows:
        cells = [
            str(row.get(args.group_by, "?")).replace("_", r"\_"),
            str(row.get("seed", "?")),
            str(row_n(row)),
            *[fmt(row.get("metrics", {}).get(name), args.precision) for name in metrics],
        ]
        body.append(" & ".join(cells) + r" \\")
    if not rows:
        body.append(r"\multicolumn{" + str(3 + len(metrics)) + r"}{c}{no measured results yet} \\")

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            rf"  \caption{{{args.caption}}}",
            rf"  \label{{{args.label}}}",
            rf"  \begin{{tabular}}{{{columns}}}",
            r"    \toprule",
            f"    {header} \\\\",
            r"    \midrule",
            *[f"    {line}" for line in body],
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Write results.md and results.tex from the aggregate payload."""
    args = build_parser().parse_args(argv)

    source = Path(args.source) if Path(args.source).is_absolute() else REPO_ROOT / args.source
    if not source.is_file():
        print(
            f"{source} does not exist. Run `python scripts/aggregate_results.py` first; "
            "it creates the aggregate even when no runs have happened yet."
        )
        return 2

    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    # Prefer the library table writer when the aggregate carries measured rows
    # in the reporting package shape.
    try:
        from evalaware.reporting.tables import write_tables

        measured = payload.get("measured")
        if isinstance(measured, list):
            out_dir = (
                Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
            )
            write_tables(
                measured,
                out_dir,
                metrics=args.metrics,
                group_by=args.group_by,
                lower_is_better=args.lower_is_better,
                precision=args.precision,
                min_n=args.min_n,
            )
            print(f"wrote tables via evalaware.reporting.tables -> {out_dir}")
            return 0
    except Exception as exc:  # noqa: BLE001 - fall back to local renderer
        print(f"note: library table writer unavailable ({exc}); using script fallback")

    metrics = collect_metrics(rows, args.metrics)

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = (
        f"<!-- generated by scripts/make_tables.py from {source.name} "
        f"at git {payload.get('git_sha', 'unknown')[:12]}; do not edit by hand -->"
    )
    note = (
        "**Provenance.** Every row above came from a run recorded under `runs/`. "
        f"This table was generated from `{source.name}` "
        f"({payload.get('n_measured', 0)} measured results, "
        f"{payload.get('n_synthetic', 0)} synthetic results excluded). "
        "Synthetic harness-validation output is never included here."
    )
    md_path = out_dir / "results.md"
    md_path.write_text(
        "\n\n".join([provenance, markdown_table(rows, metrics, args), note]) + "\n",
        encoding="utf-8",
    )

    tex_path = out_dir / "results.tex"
    tex_path.write_text(
        "% generated by scripts/make_tables.py; do not edit by hand\n"
        + latex_table(rows, metrics, args)
        + "\n",
        encoding="utf-8",
    )

    print(f"rows      : {len(rows)}")
    print(f"metrics   : {metrics or '(none present)'}")
    print(f"wrote     : {md_path}")
    print(f"wrote     : {tex_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
