"""Experiment stages for unverbalized evaluation awareness."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from evalaware.evalaware.common import cfg_n_items, cfg_param, cfg_seed, ensure_artifacts
from evalaware.evalaware.construct import run_construct
from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.inflation import run_inflation
from evalaware.evalaware.monitors import run_monitors
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.rewrite import run_rewrite
from evalaware.evalaware.verbalized import run_verbalized


def _force_synthetic(cfg: DictConfig) -> bool:
    if bool(getattr(cfg, "force_synthetic", False)):
        return True
    exp = getattr(cfg, "experiment", None)
    if exp is not None and str(getattr(exp, "name", "")) == "smoke":
        return True
    run = getattr(cfg, "run", None)
    if run is not None and str(getattr(run, "profile", "")) in {"smoke", "debug"}:
        return True
    data = getattr(cfg, "data", None)
    if data is not None and str(getattr(data, "name", "")) == "synthetic":
        return True
    return False


def _model_name(cfg: DictConfig) -> str:
    model = getattr(cfg, "model", None)
    return str(getattr(model, "name", "synthetic") if model is not None else "synthetic")


def _revision(cfg: DictConfig) -> str | None:
    model = getattr(cfg, "model", None)
    rev = getattr(model, "revision", None) if model is not None else None
    return str(rev) if rev not in (None, "", "null") else None


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
        model_name=_model_name(cfg),
        revision=_revision(cfg),
        force_synthetic=_force_synthetic(cfg),
        layer=int(cfg_param(cfg, "probe_layer", -2) or -2),
    )


def monitors(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    from evalaware.evalaware.common import read_json

    probe_payload = read_json(artifacts / "probe.json")
    return run_monitors(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        corpora_metrics={"artifact": str(artifacts / "corpora.json")},
        probe_metrics={
            "artifact": str(artifacts / "probe.json"),
            "auroc_heldout": float(probe_payload.get("auroc_heldout", 0.0)),
            "weights_path": str(artifacts / "probe_weights.npy"),
            "activations_path": str(artifacts / "activations.npy"),
            "is_synthetic": bool(probe_payload.get("is_synthetic", True)),
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
            "activations_path": str(artifacts / "activations.npy"),
            "is_synthetic": bool(probe_payload.get("is_synthetic", True)),
        },
        margin_min=float(cfg_param(cfg, "margin_min", 0.02) or 0.02),
        model_name=_model_name(cfg),
        revision=_revision(cfg),
        force_synthetic=_force_synthetic(cfg),
        dim=int(cfg_param(cfg, "activation_dim", 32) or 32),
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
    construct_payload = read_json(artifacts / "construct.json")
    return run_rewrite(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        inflation_metrics={"artifact": str(artifacts / "inflation.json")},
        monitor_metrics=mon,
        construct_metrics={
            "artifact": str(artifacts / "construct.json"),
            "internal_awareness_licensed": construct_payload.get("claim")
            == "internal_awareness_licensed",
            "construct_status": construct_payload.get("construct_status"),
            "claim": construct_payload.get("claim"),
        },
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
