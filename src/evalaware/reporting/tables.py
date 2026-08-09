"""Markdown and booktabs LaTeX tables from an aggregate payload."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["write_tables", "direction_arrow"]

LOWER_HINTS = ("error", "mae", "rmse", "loss", "ece", "gap", "cost", "silent", "kl", "obfusc")


def direction_arrow(metric: str, lower_is_better: Sequence[str] | None = None) -> str:
    if lower_is_better is not None:
        return r"$\downarrow$" if metric in lower_is_better else r"$\uparrow$"
    return r"$\downarrow$" if any(h in metric.lower() for h in LOWER_HINTS) else r"$\uparrow$"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cell(values: list[float], precision: int) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{precision}f}"
    return f"{statistics.mean(values):.{precision}f}±{statistics.stdev(values):.{precision}f}"


def write_tables(
    measured: list[dict[str, Any]],
    out_dir: Path,
    *,
    metrics: Sequence[str] | None = None,
    group_by: str = "task",
    lower_is_better: Sequence[str] | None = None,
    precision: int = 4,
    min_n: int = 20,
    caption: str = "Results.",
    label: str = "tab:results",
) -> dict[str, Path]:
    """Emit ``results.md`` and ``results.tex`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not measured:
        md = out_dir / "results.md"
        tex = out_dir / "results.tex"
        _write_text(md, "_No measured results yet._\n")
        _write_text(
            tex,
            "\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\begin{tabular}{l}\n\\toprule\nNo measured results yet.\\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        )
        return {"markdown": md, "latex": tex}

    all_metrics: list[str] = []
    for row in measured:
        for key in row.get("metrics", {}):
            if key not in all_metrics:
                all_metrics.append(key)
    cols = list(metrics) if metrics else all_metrics
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in measured:
        key = str(row.get("raw", {}).get(group_by, row.get(group_by, "unknown")))
        groups.setdefault(key, []).append(row)

    header = ["group", *cols, "n"]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in cols] + ["---:"]) + " |",
    ]
    tex_cols = "l" + "r" * (len(cols) + 1)
    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{tex_cols}}}",
        "\\toprule",
        " & ".join(
            [group_by]
            + [f"{c} {direction_arrow(c, lower_is_better)}" for c in cols]
            + ["n"]
        )
        + " \\\\",
        "\\midrule",
    ]
    footnotes = False
    for group, rows in sorted(groups.items()):
        n_vals = [int(r["n"]) for r in rows if r.get("n") is not None]
        n = sum(n_vals) if n_vals else 0
        mark = "*" if 0 < n < min_n else ""
        if mark:
            footnotes = True
        cells_md = [f"{group}{mark}"]
        cells_tex = [f"{group}{mark}"]
        for col in cols:
            vals = [float(r["metrics"][col]) for r in rows if col in r.get("metrics", {})]
            cell = _cell(vals, precision)
            cells_md.append(cell)
            cells_tex.append(cell)
        cells_md.append(str(n))
        cells_tex.append(str(n))
        md_lines.append("| " + " | ".join(cells_md) + " |")
        tex_lines.append(" & ".join(cells_tex) + " \\\\")
    tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    if footnotes:
        md_lines.append("")
        md_lines.append(f"\\*{min_n}: stratum smaller than min-n={min_n}.")
    md_path = out_dir / "results.md"
    tex_path = out_dir / "results.tex"
    _write_text(md_path, "\n".join(md_lines) + "\n")
    _write_text(tex_path, "\n".join(tex_lines))
    return {"markdown": md_path, "latex": tex_path}
