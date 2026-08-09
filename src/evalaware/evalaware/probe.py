"""Awareness probe fitting and held-out family validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import auroc
from .common import dump_json_text, read_json, result_dict
from .features import collect_activations


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
    model_name: str = "synthetic",
    revision: str | None = None,
    force_synthetic: bool = False,
    layer: int = -2,
) -> dict[str, Any]:
    items = read_json(Path(corpora_metrics["artifact"]))
    train_idx = [i for i, it in enumerate(items) if it["split"] == "train"]
    test_idx = [i for i, it in enumerate(items) if it["split"] == "test"]
    if len(train_idx) < 8 or len(test_idx) < 8:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(items))
        cut = int(0.7 * len(items))
        train_idx = perm[:cut].tolist()
        test_idx = perm[cut:].tolist()

    # Library callers that omit a model (unit tests) stay on the synthetic path.
    if model_name in {"", "synthetic", "x", "none", "missing"}:
        force_synthetic = True

    collected = collect_activations(
        items,
        model_name=model_name,
        revision=revision,
        layer=layer,
        seed=seed,
        force_synthetic=force_synthetic,
        dim=dim,
    )
    x_all = np.asarray(collected["activations"], dtype=np.float64)
    train = [items[i] for i in train_idx]
    test = [items[i] for i in test_idx]
    x_train = x_all[np.asarray(train_idx, dtype=int)]
    y_train = np.array([1.0 if it["context"] == "eval" else 0.0 for it in train])
    x_test = x_all[np.asarray(test_idx, dtype=int)]
    y_test = np.array([it["context"] == "eval" for it in test], dtype=bool)

    w = _fit_ridge(x_train, y_train, l2=1.0)
    scores = _score(w, x_test)
    auc = auroc(y_test.tolist(), scores.tolist())

    family_auc: dict[str, float] = {}
    for fam in sorted({it["family"] for it in test}):
        mask = [it["family"] == fam for it in test]
        s = [float(scores[i]) for i, m in enumerate(mask) if m]
        y = [bool(y_test[i]) for i, m in enumerate(mask) if m]
        if len(set(y)) < 2:
            continue
        family_auc[fam] = auroc(y, s)

    np.save(artifacts / "probe_weights.npy", w)
    np.save(artifacts / "activations.npy", x_all)
    dump_json_text(
        artifacts / "activations_meta.json",
        {
            "mode": collected["mode"],
            "is_synthetic": collected["is_synthetic"],
            "dim": collected["dim"],
            "layer": collected.get("layer"),
            "model_name": collected.get("model_name"),
            "revision": collected.get("revision"),
            "notes": collected.get("notes", []),
            "fallback_reason": collected.get("fallback_reason"),
        },
    )
    all_scores = _score(w, x_all)
    scored = []
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
            "dim": int(collected["dim"]),
            "activation_mode": collected["mode"],
            "is_synthetic": collected["is_synthetic"],
            "claim": (
                "Probe AUROC alone is insufficient for an internal-awareness claim; "
                "see construct stage for text-classifier margin + causal checks."
            ),
        },
    )
    return result_dict(
        task="probe",
        seed=seed,
        n=len(items),
        artifact=str(path),
        weights_path=str(artifacts / "probe_weights.npy"),
        activations_path=str(artifacts / "activations.npy"),
        auroc_heldout=float(auc),
        family_auroc=family_auc,
        n_train=len(train),
        n_test=len(test),
        is_synthetic=bool(collected["is_synthetic"]),
        activation_mode=collected["mode"],
        force_synthetic=bool(force_synthetic),
        model_name=model_name,
        revision=revision,
    )
