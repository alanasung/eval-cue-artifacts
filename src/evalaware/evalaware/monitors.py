"""Held-out monitor panel: supervising probe + real held-out linear + stand-ins.

Exactly one monitor supervises the rewrite loop; the others are held out to
measure monitor-fooling vs transfer. ``heldout_linear`` is a real second monitor
class (refit on a disjoint train fold), not a random stand-in. SAE / lens /
attention remain labeled stand-ins for panel coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..evaluation.metrics import auroc
from .common import dump_json_text, read_json, result_dict
from .features import batch_activations

MonitorName = Literal["probe", "heldout_linear", "sae", "lens", "attention"]


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    eye = np.eye(xb.shape[1])
    eye[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + l2 * eye, xb.T @ y)


def _score(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xb @ w


def fit_heldout_linear(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_mask: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Real monitor: linear probe on the complementary half of the train fold."""
    rng = np.random.default_rng(seed)
    train_idx = np.where(train_mask)[0]
    if len(train_idx) < 4:
        train_idx = np.arange(len(y))
    perm = rng.permutation(train_idx)
    half = perm[len(perm) // 2 :]  # disjoint from typical first-half probe fold
    if len(half) < 2:
        half = train_idx
    return _fit_ridge(x[half], y[half].astype(float))


def _sae_score(x: np.ndarray) -> np.ndarray:
    # Sparse-feature STAND-IN: ReLU on a fixed random projection.
    rng = np.random.default_rng(0)
    proj = rng.normal(size=(x.shape[1], min(8, x.shape[1])))
    return np.maximum(0.0, x @ proj).sum(axis=1)


def _lens_score(x: np.ndarray) -> np.ndarray:
    # Logit-lens / NLA STAND-IN: mid-channel linear readout when dims allow.
    if x.shape[1] > 9:
        return x[:, 8] * 0.8 + x[:, 9] * 0.2
    return x.mean(axis=1)


def _attention_score(x: np.ndarray) -> np.ndarray:
    # Attention-pattern STAND-IN: surface×internal interaction when dims allow.
    if x.shape[1] > 8:
        return x[:, 0] * 0.3 + x[:, 8] * 0.7
    return np.linalg.norm(x, axis=1)


def score_monitor(
    name: MonitorName,
    x: np.ndarray,
    probe_scores: np.ndarray,
    *,
    heldout_weights: np.ndarray | None = None,
) -> np.ndarray:
    if name == "probe":
        return probe_scores
    if name == "heldout_linear":
        if heldout_weights is None:
            raise ValueError("heldout_linear requires fitted weights")
        return _score(heldout_weights, x)
    if name == "sae":
        return _sae_score(x)
    if name == "lens":
        return _lens_score(x)
    if name == "attention":
        return _attention_score(x)
    raise ValueError(f"unknown monitor {name}")


def run_monitors(
    *,
    seed: int,
    artifacts: Path,
    corpora_metrics: dict[str, Any],
    probe_metrics: dict[str, Any],
    supervising: MonitorName = "probe",
) -> dict[str, Any]:
    items = read_json(Path(probe_metrics["artifact"]))["items"]
    act_path = Path(probe_metrics.get("activations_path", artifacts / "activations.npy"))
    if act_path.exists():
        x = np.load(act_path)
    else:
        x = batch_activations(items, seed=seed)
    y = np.array([it["context"] == "eval" for it in items], dtype=bool)
    probe_scores = np.array([it["probe_score"] for it in items], dtype=float)
    train_mask = np.array([it.get("split") == "train" for it in items], dtype=bool)
    if train_mask.sum() < 4:
        train_mask = np.ones(len(items), dtype=bool)
        train_mask[::5] = False

    heldout_w = fit_heldout_linear(x, y, train_mask=train_mask, seed=seed)
    np.save(artifacts / "heldout_linear_weights.npy", heldout_w)

    panel = ("probe", "heldout_linear", "sae", "lens", "attention")
    stand_ins = ("sae", "lens", "attention")
    real_monitors = ("probe", "heldout_linear")
    held_out = [m for m in panel if m != supervising]
    aurocs: dict[str, float] = {}
    scores: dict[str, list[float]] = {}
    for name in panel:
        s = score_monitor(
            name,  # type: ignore[arg-type]
            x,
            probe_scores,
            heldout_weights=heldout_w,
        )
        scores[name] = s.tolist()
        aurocs[name] = float(auroc(y.tolist(), s.tolist()))

    path = dump_json_text(
        artifacts / "monitors.json",
        {
            "supervising": supervising,
            "held_out": held_out,
            "auroc": aurocs,
            "scores": scores,
            "real_monitors": list(real_monitors),
            "stand_in_monitors": list(stand_ins),
            "procedure": (
                "Supervising monitor alone may be fooled by rewrites. Transfer is "
                "measured as awareness reduction on supervising minus reduction on "
                "the held-out panel mean. heldout_linear is a real disjoint-fold "
                "linear monitor; sae/lens/attention remain labeled stand-ins."
            ),
        },
    )
    return result_dict(
        task="monitors",
        seed=seed,
        n=len(items),
        artifact=str(path),
        supervising=supervising,
        held_out=held_out,
        auroc=aurocs,
        real_monitors=list(real_monitors),
        stand_in_monitors=list(stand_ins),
    )
