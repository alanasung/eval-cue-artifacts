"""P4: force_synthetic smoke-only; measured path monkeypatched (no Hub)."""

from __future__ import annotations

from unittest.mock import patch

from omegaconf import OmegaConf

from evalaware.evalaware.construct import run_construct
from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.features import collect_activations
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.rewrite import run_rewrite
from evalaware.evalaware.verbalized import run_verbalized
from evalaware.evalaware.inflation import run_inflation
from evalaware.evalaware.monitors import run_monitors
from evalaware.stages import _force_synthetic, probe as stage_probe


def test_force_synthetic_smoke_only():
    smoke = OmegaConf.create(
        {"force_synthetic": True, "experiment": {"name": "smoke"}, "data": {"name": "synthetic"}}
    )
    pilot = OmegaConf.create(
        {"force_synthetic": False, "experiment": {"name": "pilot"}, "data": {"name": "pilot"}}
    )
    assert _force_synthetic(smoke) is True
    assert _force_synthetic(pilot) is False


def test_collect_respects_force_flag(tmp_path):
    cfg = OmegaConf.create(
        {
            "force_synthetic": True,
            "experiment": {"name": "smoke"},
            "run": {"seed": 0},
            "data": {"n_items": 48, "name": "synthetic"},
            "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "deadbeef"},
            "activation_dim": 32,
        }
    )
    run_corpora(n_items=48, seed=0, artifacts=tmp_path / "artifacts")
    with patch("evalaware.evalaware.features.try_load_causal_lm") as load:
        out = stage_probe(cfg, tmp_path)
        load.assert_not_called()
    assert out["is_synthetic"] is True
    assert out["force_synthetic"] is True


def test_measured_collect_fail_closed_without_weights():
    items = [{"text": "hello eval", "context": "eval"}]
    with patch("evalaware.evalaware.features.try_load_causal_lm", return_value=None):
        try:
            collect_activations(
                items,
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                force_synthetic=False,
            )
            ok = False
        except RuntimeError:
            ok = True
        assert ok


def test_construct_status_fields(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    out = run_construct(
        seed=0,
        artifacts=tmp_path,
        corpora_metrics=c,
        probe_metrics=p,
        force_synthetic=True,
        margin_min=10.0,
    )
    assert out["construct_status"] == "failed"
    assert out["internal_awareness_licensed"] is False
    assert "construct_validity_failed" in out["claim"]


def test_rewrite_skips_when_construct_fails(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    m = run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p)
    inf = run_inflation(seed=0, artifacts=tmp_path, verbalized_metrics=v)
    cons = run_construct(
        seed=0,
        artifacts=tmp_path,
        corpora_metrics=c,
        probe_metrics=p,
        force_synthetic=True,
        margin_min=10.0,
    )
    out = run_rewrite(
        seed=0,
        artifacts=tmp_path,
        inflation_metrics=inf,
        monitor_metrics=m,
        construct_metrics=cons,
        rounds=1,
    )
    assert out["rewrite_executed"] is False
    assert out["status"] == "skipped_construct_failed"


def test_heldout_linear_is_real_monitor(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    m = run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    assert "heldout_linear" in m["real_monitors"]
    assert "heldout_linear" in m["held_out"]
    assert "sae" in m["stand_in_monitors"]


def test_pilot_yaml_honest_n_and_force_flag():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "configs" / "experiment" / "pilot.yaml").read_text()
    assert "n_items: 512" in text
    assert "force_synthetic: false" in text
    smoke = (Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke.yaml").read_text()
    assert "force_synthetic: true" in smoke
