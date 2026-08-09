"""Awareness probe fitting and held-out family validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import auroc
from .common import dump_json_text, read_json, result_dict, write_json
from .features import batch_activations


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Closed-form ridge classifier weights (with bias in last coordinate)."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    eye = np.eye(xb.shape[1])
    eye[-1, -1] = 0.0
    w = np.linalg.solve(xb.T @ xb + l2 * eye, xb.T @ y)
    return w


def _score(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xb @ w


def run_probe(
    *,
    seed: int,
    artifacts: Path,
    corpora_metrics: dict[str, Any],
    dim: int = 32,
) -> dict[str, Any]:
    items = read_json(Path(corpora_metrics["artifact"]))
    train = [it for it in items if it["split"] == "train"]
    test = [it for it in items if it["split"] == "test"]
    if len(train) < 8 or len(test) < 8:
        # Fall back to random split if family holdout left a cell thin.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(items))
        cut = int(0.7 * len(items))
        train = [items[i] for i in idx[:cut]]
        test = [items[i] for i in idx[cut:]]

    x_train = batch_activations(train, dim=dim, seed=seed)
    y_train = np.array([1.0 if it["context"] == "eval" else 0.0 for it in train])
    x_test = batch_activations(test, dim=dim, seed=seed)
    y_test = np.array([it["context"] == "eval" for it in test], dtype=bool)

    w = _fit_ridge(x_train, y_train, l2=1.0)
    scores = _score(w, x_test)
    auc = auroc(y_test.tolist(), scores.tolist())

    # Per-family held-out AUROC where both classes exist.
    family_auc: dict[str, float] = {}
    for fam in sorted({it["family"] for it in test}):
        mask = [it["family"] == fam for it in test]
        s = [float(scores[i]) for i, m in enumerate(mask) if m]
        y = [bool(y_test[i]) for i, m in enumerate(mask) if m]
        if len(set(y)) < 2:
            continue
        family_auc[fam] = auroc(y, s)

    np.save(artifacts / "probe_weights.npy", w)
    scored = []
    all_x = batch_activations(items, dim=dim, seed=seed)
    all_scores = _score(w, all_x)
    for it, sc in zip(items, all_scores):
        row = dict(it)
        row["probe_score"] = float(sc)
        row["probe_positive"] = bool(sc > 0.5)
        scored.append(row)
    path = dump_json_text(
        artifacts / "probe.json",
        {
            "items": scored,
            "auroc_heldout": auc,
            "family_auroc": family_auc,
            "dim": dim,
            "claim": (
                "Probe AUROC alone is insufficient for an internal-awareness claim; "
                "see construct stage for text-classifier margin + causal steering."
            ),
        },
    )
    return result_dict(
        task="probe",
        seed=seed,
        n=len(items),
        artifact=str(path),
        weights_path=str(artifacts / "probe_weights.npy"),
        auroc_heldout=float(auc),
        family_auroc=family_auc,
        n_train=len(train),
        n_test=len(test),
    )
