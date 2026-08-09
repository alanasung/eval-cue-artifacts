"""Publication-grade figures with captions derived from plotted numbers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["PALETTE", "RC_PARAMS", "write_caption", "write_figures"]

logger = logging.getLogger(__name__)

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

RC_PARAMS = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def write_caption(metric: str, groups: dict[str, float], *, digits: int = 3) -> str:
    """Generate a paper caption from the plotted numbers."""
    if not groups:
        return (
            f"Figure placeholder for {metric}: no measured values were available, "
            "so this panel must not be reported as an empirical result."
        )
    ordered = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_val = ordered[0]
    worst_name, worst_val = ordered[-1]
    return (
        f"{metric} across experimental groups. "
        f"Highest: {best_name} at {best_val:.{digits}f}; "
        f"lowest: {worst_name} at {worst_val:.{digits}f}. "
        "Error bars omitted when per-group replication is below two seeds."
    )


def write_figures(
    measured: list[dict[str, Any]],
    out_dir: Path,
    *,
    metric: str = "accuracy",
    group_by: str = "task",
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
    width_in: float = 6.6,
    height_in: float = 2.6,
) -> dict[str, Path]:
    """Write a bar chart for ``metric`` grouped by ``group_by``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[float]] = {}
    for row in measured:
        key = str(row.get("raw", {}).get(group_by, row.get(group_by, "unknown")))
        val = row.get("metrics", {}).get(metric)
        if val is None:
            continue
        groups.setdefault(key, []).append(float(val))
    means = {k: sum(v) / len(v) for k, v in groups.items()}

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in))
        ax0, ax1 = axes
        if means:
            names = list(means)
            vals = [means[n] for n in names]
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
            ax0.bar(names, vals, color=colors)
            ax0.set_ylabel(metric)
            ax0.set_title("(a)")
            ax0.tick_params(axis="x", rotation=30)
            ax1.plot(range(len(vals)), vals, marker="o", color=PALETTE[0])
            ax1.set_xticks(range(len(names)), names, rotation=30)
            ax1.set_title("(b)")
            ax1.set_ylabel(metric)
        else:
            for ax, label in ((ax0, "(a)"), (ax1, "(b)")):
                ax.text(0.5, 0.5, f"no data for {metric}", ha="center", va="center")
                ax.set_axis_off()
                ax.set_title(label)
        fig.tight_layout()
        written: dict[str, Path] = {}
        stem = out_dir / f"{metric}_by_{group_by}"
        for fmt in formats:
            dest = Path(f"{stem}.{fmt}")
            fig.savefig(dest, dpi=dpi if fmt == "png" else None)
            written[fmt] = dest
        plt.close(fig)

    caption = write_caption(metric, means)
    cap_path = out_dir / f"{metric}_by_{group_by}.caption.txt"
    cap_path.write_text(caption + "\n", encoding="utf-8")
    written["caption"] = cap_path
    return written
