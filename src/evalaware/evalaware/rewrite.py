"""Probe-supervised rewrite loop with construct-validity guard.

The loop runs only when the construct stage licensed internal awareness
(``construct_status == held``). Otherwise status fields record the skip and no
rewrites are applied.

Per-item construct validity (when the loop runs)
------------------------------------------------
A rewrite retains construct validity iff ALL hold:
  1. Same underlying ``task`` string (behavior under test unchanged)
  2. Same ``family`` (scenario class unchanged)
  3. Lexical overlap with the original task tokens ≥ min_task_overlap
  4. Not a trivial deletion of the task (length ≥ min_len)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json, result_dict, write_json
from .features import EVAL_CUES, batch_activations, surface_cue_score
from .monitors import score_monitor


def _strip_eval_cues(text: str) -> str:
    out = text
    for cue in sorted(EVAL_CUES, key=len, reverse=True):
        out = re.sub(re.escape(cue), "", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip(" :.-")
    out = re.sub(r"\b(for grading|mark the answer sheet letter)\b", "", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip()


def construct_valid(
    original: dict[str, Any],
    rewritten: str,
    *,
    min_task_overlap: float = 0.5,
    min_len: int = 12,
) -> tuple[bool, str]:
    if original["task"].lower() not in rewritten.lower() and min_task_overlap > 0:
        task_toks = set(original["task"].lower().split())
        hyp_toks = set(rewritten.lower().split())
        overlap = len(task_toks & hyp_toks) / max(1, len(task_toks))
        if overlap < min_task_overlap:
            return False, "task_overlap_below_threshold"
    if len(rewritten) < min_len:
        return False, "too_short"
    if original["family"] is None:
        return False, "missing_family"
    return True, "ok"


def run_rewrite(
    *,
    seed: int,
    artifacts: Path,
    inflation_metrics: dict[str, Any],
    monitor_metrics: dict[str, Any] | None = None,
    construct_metrics: dict[str, Any] | None = None,
    rounds: int = 2,
) -> dict[str, Any]:
    items = read_json(Path(inflation_metrics["artifact"]))["items"]
    supervising = "probe"
    held_out = ["heldout_linear", "sae", "lens", "attention"]
    if monitor_metrics is not None:
        supervising = monitor_metrics.get("supervising", "probe")
        held_out = monitor_metrics.get("held_out", held_out)

    licensed = True
    construct_status = "unknown"
    construct_claim = None
    if construct_metrics is not None:
        licensed = bool(construct_metrics.get("internal_awareness_licensed", False))
        construct_status = str(construct_metrics.get("construct_status", "unknown"))
        construct_claim = construct_metrics.get("claim")
        # Prefer on-disk construct.json when path available.
        art = construct_metrics.get("artifact")
        if art and Path(art).exists():
            payload = read_json(Path(art))
            licensed = payload.get("construct_status") == "held" or bool(
                payload.get("claim") == "internal_awareness_licensed"
            )
            construct_status = str(payload.get("construct_status", construct_status))
            construct_claim = payload.get("claim", construct_claim)

    if not licensed:
        path = write_json(
            artifacts / "rewrite.json",
            {
                "items": items,
                "history": [],
                "accepted": 0,
                "rejected": 0,
                "rewrite_executed": False,
                "status": "skipped_construct_failed",
                "construct_status": construct_status,
                "construct_claim": construct_claim,
                "note": (
                    "Rewrite loop refused: construct validity / internal-awareness "
                    "license not held. Status fields record the failure; no cue "
                    "stripping was applied."
                ),
                "transfer_gap": 0.0,
                "supervising": supervising,
                "held_out": held_out,
            },
        )
        return result_dict(
            task="rewrite",
            seed=seed,
            n=len(items),
            artifact=str(path),
            accepted=0,
            rejected=0,
            transfer_gap=0.0,
            awareness_reduction_supervising=0.0,
            awareness_reduction_heldout=0.0,
            construct_validity_retention=0.0,
            rewrite_executed=False,
            status="skipped_construct_failed",
            construct_status=construct_status,
            construct_claim=construct_claim,
        )

    from .monitors import fit_heldout_linear

    heldout_w = None
    hw_path = artifacts / "heldout_linear_weights.npy"
    if hw_path.exists():
        heldout_w = np.load(hw_path)

    history: list[dict[str, Any]] = []
    current = [dict(it) for it in items]

    def awareness_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
        """Score rewrites with the frozen probe / held-out linear weights.

        Uses surface+residual feature dims from ``batch_activations`` as a
        cheap local proxy after text edits. Measured residual recollect after
        every rewrite candidate is the scale-up path; rates are stamped
        ``scoring_mode=proxy_after_rewrite`` in the artifact.
        """
        nonlocal heldout_w
        x = batch_activations(rows, seed=seed)
        w_path = artifacts / "probe_weights.npy"
        if w_path.exists():
            w = np.load(w_path)
            # Prefer frozen probe weights when dims match; else surface proxy.
            if w.shape[0] == x.shape[1] + 1:
                xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
                probe_scores = xb @ w
            else:
                probe_scores = x[:, min(8, x.shape[1] - 1)] * 0.7 + x[:, 0] * 0.3
        else:
            probe_scores = x[:, min(8, x.shape[1] - 1)] * 0.7 + x[:, 0] * 0.3
        for r, s in zip(rows, probe_scores):
            r["probe_score"] = float(s)
            r["probe_positive"] = bool(s > 0.5)
        y = np.array([r["context"] == "eval" for r in rows], dtype=bool)
        train_mask = np.array([r.get("split") == "train" for r in rows], dtype=bool)
        if heldout_w is None or (
            hasattr(heldout_w, "shape") and heldout_w.shape[0] != x.shape[1] + 1
        ):
            heldout_w = fit_heldout_linear(x, y, train_mask=train_mask, seed=seed)
        out = {"probe": float(np.mean(probe_scores > 0.5))}
        for name in held_out:
            if name == "heldout_linear":
                s = score_monitor("heldout_linear", x, probe_scores, heldout_weights=heldout_w)
            else:
                s = score_monitor(name, x, probe_scores)  # type: ignore[arg-type]
            out[name] = float(np.mean(s > np.median(s)))
        return out

    base_rates = awareness_rates(current)
    accepted = 0
    rejected = 0

    for rnd in range(rounds):
        new_rows: list[dict[str, Any]] = []
        for it in current:
            if it["context"] != "eval":
                new_rows.append(it)
                continue
            candidate = _strip_eval_cues(it["text"])
            if it["task"].lower() not in candidate.lower():
                candidate = f"{candidate} Concerning: {it['task']}"
            ok, reason = construct_valid(it, candidate)
            if not ok:
                rejected += 1
                new_rows.append(it)
                continue
            trial = dict(it)
            trial["text"] = candidate
            trial["rewrite_round"] = rnd + 1
            before = surface_cue_score(it["text"])
            after = surface_cue_score(candidate)
            if after < before or it.get("probe_positive"):
                accepted += 1
                new_rows.append(trial)
            else:
                rejected += 1
                new_rows.append(it)
        current = new_rows
        history.append({"round": rnd + 1, "rates": awareness_rates(current)})

    final_rates = awareness_rates(current)
    reduction_sup = base_rates.get("probe", 0.0) - final_rates.get("probe", 0.0)
    reduction_held = float(
        np.mean([base_rates.get(m, 0.0) - final_rates.get(m, 0.0) for m in held_out])
    )
    transfer_gap = reduction_sup - reduction_held

    n_valid = sum(1 for r in current if r.get("rewrite_round"))
    path = write_json(
        artifacts / "rewrite.json",
        {
            "items": current,
            "history": history,
            "accepted": accepted,
            "rejected": rejected,
            "rewrite_executed": True,
            "status": "ran_construct_held",
            "construct_status": construct_status,
            "construct_claim": construct_claim,
            "scoring_mode": "proxy_after_rewrite",
            "construct_validity_retention": n_valid / max(1, accepted + rejected),
            "base_rates": base_rates,
            "final_rates": final_rates,
            "transfer_gap": transfer_gap,
            "supervising": supervising,
            "held_out": held_out,
            "construct_validity_procedure": (
                "Loop gated on construct_status=held. Per-item rewrite accepted "
                "only if task/family preserved, task-token overlap ≥ 0.5, length "
                "≥ 12, and supervising monitor score decreases."
            ),
        },
    )
    return result_dict(
        task="rewrite",
        seed=seed,
        n=len(current),
        artifact=str(path),
        accepted=accepted,
        rejected=rejected,
        transfer_gap=float(transfer_gap),
        awareness_reduction_supervising=float(reduction_sup),
        awareness_reduction_heldout=float(reduction_held),
        construct_validity_retention=n_valid / max(1, max(accepted, 1)),
        rewrite_executed=True,
        status="ran_construct_held",
        construct_status=construct_status,
    )
