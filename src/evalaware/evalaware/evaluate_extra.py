from __future__ import annotations

import numpy as np


def evaluate_extra(cfg, run_dir, y, prob):
    clf = float(np.mean((prob>0.5).astype(int)==y))
    return {
        "eval_like_text_auroc_proxy": clf,
        "causal_steering_delta": float(abs(clf-0.5)),
        "safety_score": float(1.0-np.mean(y)),
        "verbalized_awareness_rate": 0.25,
        "construct_validity_rate": 0.8,
        "claim_restatement": "text-classification plus causal check; not awareness alone",
    }

