"""Text-only cue classifier baseline + causal steering validation.

Claim policy
------------
A text classifier that detects evaluation-*like text* is NOT treated as evidence
of internal evaluation *awareness*. An internal-awareness claim is licensed only
when BOTH hold:
  1. probe AUROC − text-classifier AUROC > margin_min (positive margin)
  2. causal steering along the probe direction moves behavioral/probe scores
     in the predicted direction (ablation lowers eval scores; addition raises them)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import auroc
from .common import dump_json_text, read_json, result_dict, write_json
from .features import EVAL_CUES, batch_activations, surface_cue_score


def _text_features(texts: list[str]) -> np.ndarray:
    rows = []
    for t in texts:
        low = t.lower()
        rows.append([float(c in low) for c in EVAL_CUES] + [surface_cue_score(t)])
    return np.asarray(rows, dtype=np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    eye = np.eye(xb.shape[1])
    eye[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + l2 * eye, xb.T @ y)


def _score(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xb @ w


def run_construct(
    *,
    seed: int,
    artifacts: Path,
    corpora_metrics: dict[str, Any],
    probe_metrics: dict[str, Any],
    margin_min: float = 0.02,
) -> dict[str, Any]:
    items = read_json(Path(corpora_metrics["artifact"]))
    y = np.array([it["context"] == "eval" for it in items], dtype=bool)

    # --- text-only baseline ---
    x_text = _text_features([it["text"] for it in items])
    # Train on train split only.
    train_mask = np.array([it["split"] == "train" for it in items])
    if train_mask.sum() < 8:
        train_mask = np.ones(len(items), dtype=bool)
        train_mask[::5] = False
    w_text = _fit_ridge(x_text[train_mask], y[train_mask].astype(float))
    text_scores = _score(w_text, x_text)
    text_auc = float(auroc(y.tolist(), text_scores.tolist()))

    probe_scores = np.array(
        [it["probe_score"] for it in read_json(Path(probe_metrics["artifact"]))["items"]],
        dtype=float,
    )
    probe_auc = float(probe_metrics["auroc_heldout"])
    margin = probe_auc - text_auc

    # --- causal steering / ablation on synthetic activations ---
    base = batch_activations(items, seed=seed, steer=0.0)
    ablated = batch_activations(items, seed=seed, steer=-1.0)
    boosted = batch_activations(items, seed=seed, steer=+1.0)
    # Probe readout on dim 8/9 channel (matches planted internal direction).
    w = np.load(probe_metrics["weights_path"])
    def probe_read(x: np.ndarray) -> np.ndarray:
        xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        return xb @ w

    base_s = probe_read(base)
    abl_s = probe_read(ablated)
    bst_s = probe_read(boosted)
    eval_idx = np.where(y)[0]
    causal_ok = bool(
        abl_s[eval_idx].mean() < base_s[eval_idx].mean()
        and bst_s[eval_idx].mean() > base_s[eval_idx].mean()
    )

    internal_claim_licensed = bool(margin > margin_min and causal_ok)
    claim = (
        "internal_awareness_licensed"
        if internal_claim_licensed
        else "text_classification_plus_failed_or_weak_causal_check"
    )

    path = dump_json_text(
        artifacts / "construct.json",
        {
            "text_classifier_auroc": text_auc,
            "probe_auroc": probe_auc,
            "margin_probe_minus_text": margin,
            "margin_min": margin_min,
            "causal": {
                "mean_score_base_eval": float(base_s[eval_idx].mean()),
                "mean_score_ablated_eval": float(abl_s[eval_idx].mean()),
                "mean_score_boosted_eval": float(bst_s[eval_idx].mean()),
                "steering_moves_scores": causal_ok,
                "procedure": (
                    "Steer the planted internal channel (activation dims 8-15) by "
                    "±1.0; require ablation to decrease and boost to increase mean "
                    "probe score on eval-labeled items."
                ),
            },
            "claim": claim,
            "policy": (
                "Text-classifier AUROC is a baseline for evaluation-like text, not "
                "awareness evidence. Internal awareness requires positive margin AND "
                "causal steering success."
            ),
        },
    )
    return result_dict(
        task="construct",
        seed=seed,
        n=len(items),
        artifact=str(path),
        text_classifier_auroc=text_auc,
        probe_auroc=probe_auc,
        margin_probe_minus_text=margin,
        causal_steering_ok=causal_ok,
        internal_awareness_licensed=internal_claim_licensed,
        claim=claim,
    )
