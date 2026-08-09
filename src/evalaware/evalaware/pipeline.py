
"""End-to-end stage implementations with synthetic fallback when weights are absent."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from ..utils.git import git_sha
from ..utils.io import ensure_dir, save_json
from ..data.synthetic import build_synthetic_items
from ..data.splits import split_items
from ..evaluation.metrics import accuracy, bootstrap_mean, minimum_detectable_effect
from ..reporting.aggregate import aggregate_results
from ..reporting.tables import write_tables
from ..reporting.figures import write_figures

logger = logging.getLogger(__name__)


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def _n_items(cfg: DictConfig) -> int:
    return int(getattr(cfg.data, "n_items", 512))


def _base(task: str, cfg: DictConfig, n: int, **metrics: float) -> dict[str, Any]:
    payload = {
        "task": task,
        "seed": _seed(cfg),
        "git_sha": git_sha(),
        "n": n,
        "profile": str(getattr(cfg.run, "profile", "pilot")),
        "model": str(getattr(cfg.model, "name", "")),
        "revision": str(getattr(cfg.model, "revision", "") or ""),
        "is_synthetic": True,
        "notes": "synthetic/pilot path; label measured rows only after real weights run",
    }
    payload.update(metrics)
    return payload


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    n = _n_items(cfg)
    items = build_synthetic_items(n, seed=_seed(cfg))
    # Project-specific enrichment hook
    try:
        from .enrich import enrich_items  # type: ignore
        items = enrich_items(items, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.info("enrich_items unavailable or failed (%s); using synthetic items", exc)
    bundle = split_items(
        items,
        train_frac=float(cfg.data.train_frac),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
        seed=_seed(cfg),
        shuffle=bool(cfg.data.shuffle),
    )
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    save_json(out / "train.json", bundle.train)
    save_json(out / "val.json", bundle.val)
    save_json(out / "test.json", bundle.test)
    payload = _base("build_dataset", cfg, n, **{f"n_{k}": v for k, v in bundle.sizes().items()})
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    save_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    data_dir = run_dir / "artifacts" / "dataset"
    train = json.loads((data_dir / "train.json").read_text()) if (data_dir / "train.json").is_file() else build_synthetic_items(64, seed=_seed(cfg))
    features = []
    labels = []
    for row in train:
        comps = row.get("components") or {"c0": float(row.get("label", 0))}
        features.append([float(comps.get(k, 0.0)) for k in sorted(comps)])
        labels.append(int(row.get("label", 0)))
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    # Optional model path
    model_note = "numpy_features"
    try:
        from ..models.loader import load_model
        from ..models.hooks import capture
        import torch
        loaded = load_model(cfg)
        # tiny forward on a few prompts if tokenizer works
        prompts = [str(r.get("prompt", "hello")) for r in train[: min(8, len(train))]]
        tok = loaded.tokenizer
        encoded = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        encoded = {k: v.to(loaded.device) for k, v in encoded.items()}
        with capture(loaded.model, layers=[0], last_token_only=True) as cache:
            loaded.model(**encoded)
        acts = cache.numpy(0)
        # pad/truncate feature matrix with activation summary
        summary = acts.mean(axis=1, keepdims=True)
        # align lengths
        m = min(len(x), len(summary))
        x = np.concatenate([x[:m], summary[:m]], axis=1)
        model_note = f"model:{loaded.name}@{loaded.revision}"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "collect: model weights unavailable or forward failed (%s). "
            "Continuing with synthetic component features. Install/download the "
            "configured checkpoint to produce measured (non-synthetic) activations.",
            exc,
        )
        model_note = f"fallback:{type(exc).__name__}"
    out = ensure_dir(run_dir / "artifacts" / "collect")
    np.save(out / "features.npy", x)
    np.save(out / "labels.npy", y)
    payload = _base("collect", cfg, int(len(y)), n_features=int(x.shape[1] if x.ndim == 2 else 0))
    payload["model_note"] = model_note
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    # If we truly used a model, mark not synthetic for activations — keep synthetic flag if fallback
    if model_note.startswith("model:"):
        payload["is_synthetic"] = False
    save_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    col = run_dir / "artifacts" / "collect"
    x = np.load(col / "features.npy") if (col / "features.npy").is_file() else np.random.default_rng(_seed(cfg)).normal(size=(64, 4))
    y = np.load(col / "labels.npy") if (col / "labels.npy").is_file() else np.zeros(len(x), dtype=int)
    # logistic-style linear probe via least squares on [0,1]
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    rng = np.random.default_rng(_seed(cfg))
    # simple ridge
    xt = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    lam = 1e-2
    w = np.linalg.solve(xt.T @ xt + lam * np.eye(xt.shape[1]), xt.T @ y.astype(float))
    pred = xt @ w
    prob = 1.0 / (1.0 + np.exp(-np.clip(pred, -20, 20)))
    out = ensure_dir(run_dir / "artifacts" / "fit")
    np.save(out / "weights.npy", w)
    np.save(out / "pred.npy", prob)
    try:
        from .fit_extra import fit_extra  # type: ignore
        extra = fit_extra(cfg, run_dir, x, y, prob)
    except Exception:
        extra = {}
    payload = _base("fit", cfg, int(len(y)), train_accuracy=float(accuracy(y, (prob > 0.5).astype(int))))
    payload.update(extra)
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    save_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    fit_dir = run_dir / "artifacts" / "fit"
    col = run_dir / "artifacts" / "collect"
    y = np.load(col / "labels.npy") if (col / "labels.npy").is_file() else np.zeros(32, dtype=int)
    prob = np.load(fit_dir / "pred.npy") if (fit_dir / "pred.npy").is_file() else np.full(len(y), 0.5)
    m = min(len(y), len(prob))
    y, prob = y[:m], prob[:m]
    pred = (prob > 0.5).astype(int)
    acc = accuracy(y, pred)
    est = bootstrap_mean((pred == y).astype(float), n_boot=500, seed=_seed(cfg))
    mde = minimum_detectable_effect(max(m, 2), sigma=0.5)
    try:
        from .evaluate_extra import evaluate_extra  # type: ignore
        extra = evaluate_extra(cfg, run_dir, y, prob)
    except Exception:
        extra = {}
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    payload = _base(
        "evaluate",
        cfg,
        int(m),
        accuracy=float(acc),
        accuracy_lo=float(est.lo),
        accuracy_hi=float(est.hi),
        minimum_detectable_effect=float(mde),
    )
    payload.update(extra)
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    if (run_dir / "artifacts" / "collect" / "results.json").is_file():
        collect_meta = json.loads((run_dir / "artifacts" / "collect" / "results.json").read_text())
        payload["is_synthetic"] = bool(collect_meta.get("is_synthetic", True))
    save_json(out / "results.json", payload)
    return payload


def stage_report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    results_root = ensure_dir(Path(str(cfg.paths.results)))
    # copy stage results into results/
    for name in ("build_dataset", "collect", "fit", "evaluate"):
        src = run_dir / "artifacts" / name / "results.json"
        if src.is_file():
            dest = results_root / f"{name}_results.json"
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    agg = aggregate_results([results_root, run_dir / "artifacts"])
    save_json(results_root / "results.json", agg.to_dict())
    write_tables(agg.measured, Path(str(cfg.paths.tables)))
    write_figures(agg.measured, Path(str(cfg.paths.figures)), metric="accuracy", formats=("png", "pdf", "svg"))
    payload = _base("report", cfg, len(agg.measured), n_measured=len(agg.measured), n_synthetic=len(agg.synthetic))
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    payload["is_synthetic"] = any(r.get("is_synthetic") for r in agg.measured) if agg.measured else True
    save_json(run_dir / "artifacts" / "report" / "results.json" if False else results_root / "report_results.json", payload)
    ensure_dir(run_dir / "artifacts" / "report")
    save_json(run_dir / "artifacts" / "report" / "results.json", payload)
    return payload
