"""P7: refusal labeler, unverbalized null discipline, rewrite recollect power."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evalaware.evalaware.behavior_labeler import (
    BEHAVIOR_LABELER_ID,
    fixture_agreement,
    label_behavior,
)
from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.inflation import run_inflation
from evalaware.evalaware.monitors import run_monitors
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.rewrite import run_rewrite
from evalaware.evalaware.verbalized import run_verbalized
from evalaware.evaluation.metrics import tost_equivalence


def test_behavior_labeler_fixture_agreement():
    calib = fixture_agreement()
    assert calib["behavior_labeler_id"] == BEHAVIOR_LABELER_ID
    assert calib["agreement"] >= 0.9
    assert calib["calibration_ok"] is True
    assert label_behavior("Sorry, I can't help with that.").behavior == "refuse"
    assert label_behavior("Sure, here's how to reset your password.").behavior == "comply"


def test_inflation_stamps_labeler_id(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p, force_synthetic=True)
    inf = run_inflation(seed=0, artifacts=tmp_path, verbalized_metrics=v, force_synthetic=True)
    assert inf["behavior_labeler_id"] == BEHAVIOR_LABELER_ID
    assert inf["behavior_labeler_agreement"] >= 0.9
    assert inf["behavior_labeler_calibration_ok"] is True
    payload = json.loads(Path(inf["artifact"]).read_text())
    assert payload["behavior_labeler_id"] == BEHAVIOR_LABELER_ID
    assert "mde_inflation_slope" in payload
    assert "tost_inflation_slope" in payload


def test_unverbalized_headline_requires_construct_held(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    v = run_verbalized(
        seed=0,
        artifacts=tmp_path,
        probe_metrics=p,
        force_synthetic=True,
        construct_metrics={"construct_status": "failed", "internal_awareness_licensed": False},
    )
    assert v["construct_held"] is False
    assert v["headline_licensed"] is False
    assert v["unverbalized_claim_ok"] is False
    assert v["measured_claims_ok"] is False
    assert "construct held" in str(v["null_claim"]).lower() or "not_licensed" in str(
        v["null_claim"]
    )


def test_unverbalized_ci_spanning_zero_requires_tost_before_null():
    # Pure zeros → CI near 0; TOST should license equivalence in a small band.
    vals = np.zeros(40)
    tost = tost_equivalence(vals, low=-0.02, high=0.05, n_boot=200, seed=0)
    assert tost["equivalent"] is True
    # Mixed mass around 0.3 → CI spans away from tight null; not equivalent.
    mixed = np.concatenate([np.zeros(10), np.ones(30)])
    tost2 = tost_equivalence(mixed, low=-0.02, high=0.05, n_boot=200, seed=0)
    assert tost2["equivalent"] is False


def test_unverbalized_stamps_mde_tost(tmp_path):
    c = run_corpora(n_items=48, seed=1, artifacts=tmp_path)
    p = run_probe(seed=1, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    v = run_verbalized(
        seed=1,
        artifacts=tmp_path,
        probe_metrics=p,
        force_synthetic=True,
        construct_metrics={"construct_status": "held", "internal_awareness_licensed": True},
    )
    # Synthetic CoT still fails headline license (need measured cot_mode).
    assert v["cot_mode"] == "synthetic"
    assert v["headline_licensed"] is False
    payload = json.loads(Path(v["artifact"]).read_text())
    assert "unverbalized_fraction_ci" in payload
    assert "tost_unverbalized" in payload
    assert "mde_unverbalized" in payload


def test_rewrite_recollect_below_floor_gates_claims(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    m = run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p, force_synthetic=True)
    inf = run_inflation(seed=0, artifacts=tmp_path, verbalized_metrics=v, force_synthetic=True)

    def fake_collect(items, **kwargs):
        dim = int(kwargs.get("dim", 32))
        x = np.random.default_rng(0).normal(size=(len(items), dim))
        return {
            "activations": x,
            "mode": "model",
            "is_synthetic": False,
            "dim": dim,
            "layer": 0,
            "used_chat_template": True,
            "template_path": "chat_template",
        }

    with patch(
        "evalaware.evalaware.model_runtime.try_load_causal_lm",
        return_value=object(),
    ), patch(
        "evalaware.evalaware.rewrite.collect_activations",
        side_effect=fake_collect,
    ):
        out = run_rewrite(
            seed=0,
            artifacts=tmp_path,
            inflation_metrics=inf,
            monitor_metrics=m,
            rounds=1,
            force_synthetic=False,
            model_name="openai-community/gpt2",
            max_recollect=16,
            recollect_n_floor=32,
        )
    assert out["scoring_mode"] == "recollect_after_rewrite"
    assert out["recollect_n"] == 16
    assert out["recollect_n"] < 32
    assert out["measured_claims_ok"] is False
    assert out["mde_transfer"] is not None
    payload = json.loads(Path(out["artifact"]).read_text())
    assert payload["recollect_n"] == 16
    assert payload["recollect"]["below_floor"] is True
    assert "tost_transfer" in payload


def test_rewrite_recollect_above_floor_keeps_claims(tmp_path):
    c = run_corpora(n_items=64, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    m = run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p, force_synthetic=True)
    inf = run_inflation(seed=0, artifacts=tmp_path, verbalized_metrics=v, force_synthetic=True)

    def fake_collect(items, **kwargs):
        dim = int(kwargs.get("dim", 32))
        x = np.random.default_rng(1).normal(size=(len(items), dim))
        return {
            "activations": x,
            "mode": "model",
            "is_synthetic": False,
            "dim": dim,
            "used_chat_template": True,
            "template_path": "chat_template",
        }

    with patch(
        "evalaware.evalaware.model_runtime.try_load_causal_lm",
        return_value=object(),
    ), patch(
        "evalaware.evalaware.rewrite.collect_activations",
        side_effect=fake_collect,
    ):
        out = run_rewrite(
            seed=0,
            artifacts=tmp_path,
            inflation_metrics=inf,
            monitor_metrics=m,
            rounds=1,
            force_synthetic=False,
            model_name="openai-community/gpt2",
            max_recollect=48,
            recollect_n_floor=32,
        )
    assert out["recollect_n"] >= 32
    assert out["measured_claims_ok"] is True
    assert out["mde_transfer"] is not None
