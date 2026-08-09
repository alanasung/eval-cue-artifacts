"""Experiment stages for unverbalized evaluation awareness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig

from evalaware.evalaware.common import cfg_n_items, cfg_param, cfg_seed, ensure_artifacts
from evalaware.evalaware.construct import run_construct
from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.inflation import run_inflation
from evalaware.evalaware.monitors import run_monitors
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.rewrite import run_rewrite
from evalaware.evalaware.verbalized import run_verbalized


def corpora(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    holdout = cfg_param(cfg, "holdout_families", ["privacy", "tool_use"])
    return run_corpora(
        n_items=max(48, cfg_n_items(cfg)),
        seed=cfg_seed(cfg),
        artifacts=ensure_artifacts(run_dir),
        holdout_families=list(holdout) if holdout else None,
    )


def probe(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_probe(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        corpora_metrics={"artifact": str(artifacts / "corpora.json")},
        dim=int(cfg_param(cfg, "activation_dim", 32) or 32),
    )


def monitors(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_monitors(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        corpora_metrics={"artifact": str(artifacts / "corpora.json")},
        probe_metrics={
            "artifact": str(artifacts / "probe.json"),
            "auroc_heldout": 0.0,
            "weights_path": str(artifacts / "probe_weights.npy"),
        },
        supervising=str(cfg_param(cfg, "supervising_monitor", "probe") or "probe"),  # type: ignore[arg-type]
    )


def construct(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    from evalaware.evalaware.common import read_json

    probe_payload = read_json(artifacts / "probe.json")
    return run_construct(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        corpora_metrics={"artifact": str(artifacts / "corpora.json")},
        probe_metrics={
            "artifact": str(artifacts / "probe.json"),
            "auroc_heldout": float(probe_payload.get("auroc_heldout", 0.0)),
            "weights_path": str(artifacts / "probe_weights.npy"),
        },
        margin_min=float(cfg_param(cfg, "margin_min", 0.02) or 0.02),
    )


def verbalized(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_verbalized(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        probe_metrics={"artifact": str(artifacts / "probe.json")},
    )


def inflation(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_inflation(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        verbalized_metrics={"artifact": str(artifacts / "verbalized.json")},
    )


def rewrite(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    from evalaware.evalaware.common import read_json

    mon = read_json(artifacts / "monitors.json")
    return run_rewrite(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        inflation_metrics={"artifact": str(artifacts / "inflation.json")},
        monitor_metrics=mon,
        rounds=int(cfg_param(cfg, "rewrite_rounds", 2) or 2),
    )


STAGES: dict[str, Callable[[DictConfig, Path], dict[str, Any]]] = {
    "corpora": corpora,
    "probe": probe,
    "monitors": monitors,
    "construct": construct,
    "verbalized": verbalized,
    "inflation": inflation,
    "rewrite": rewrite,
}

# make pilot / smoke contract
_std = list(STAGES.values())
_names = list(STAGES.keys())
if len(_std) >= 5:
    STAGES.update({
        "build_dataset": _std[0],
        "collect": _std[1],
        "fit": _std[2],
        "evaluate": _std[3],
        "report": _std[4],
    })

