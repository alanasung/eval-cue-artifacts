"""Aggregate results into JSON, Markdown, LaTeX, and publication figures."""

from __future__ import annotations

from .aggregate import AggregatePayload, aggregate_results, flatten_metrics, write_aggregate
from .figures import PALETTE, RC_PARAMS, write_caption, write_figures
from .tables import write_tables

__all__ = [
    "AggregatePayload",
    "PALETTE",
    "RC_PARAMS",
    "aggregate_results",
    "flatten_metrics",
    "write_aggregate",
    "write_caption",
    "write_figures",
    "write_tables",
]
