"""P6: rewrite recollect stamps, construct behavioral fields, chat-template collect."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from evalaware.evalaware.construct import run_construct
from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.features import collect_activations
from evalaware.evalaware.inflation import run_inflation
from evalaware.evalaware.monitors import run_monitors
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.rewrite import run_rewrite
from evalaware.evalaware.verbalized import run_verbalized


def test_construct_stamps_algebraic_and_behavioral(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    out = run_construct(
        seed=0,
        artifacts=tmp_path,
        corpora_metrics=c,
        probe_metrics=p,
        force_synthetic=True,
    )
    assert "algebraic_ok" in out
    assert "behavioral_ok" in out
    payload = __import__("json").loads(__import__("pathlib").Path(out["artifact"]).read_text())
    assert "behavioral_ok" in payload["checks"]
    assert "behavioral" in payload
    if out["construct_status"] == "held":
        assert out["algebraic_ok"] is True
        assert out["behavioral_ok"] is True


def test_construct_held_requires_behavioral(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)

    def fail_behavioral(*_a, **_k):
        return False, {"passes": False, "procedure": "forced fail"}

    with patch(
        "evalaware.evalaware.construct._behavioral_heldout_agreement",
        side_effect=fail_behavioral,
    ):
        out = run_construct(
            seed=0,
            artifacts=tmp_path,
            corpora_metrics=c,
            probe_metrics=p,
            force_synthetic=True,
            margin_min=-1.0,
        )
    assert out["behavioral_ok"] is False
    assert out["construct_status"] == "failed"
    assert out["internal_awareness_licensed"] is False


def test_rewrite_proxy_smoke_stamps(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    m = run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p, force_synthetic=True)
    inf = run_inflation(seed=0, artifacts=tmp_path, verbalized_metrics=v, force_synthetic=True)
    # License rewrite without requiring construct held (unit path).
    out = run_rewrite(
        seed=0,
        artifacts=tmp_path,
        inflation_metrics=inf,
        monitor_metrics=m,
        rounds=1,
        force_synthetic=True,
        model_name="synthetic",
    )
    assert out["scoring_mode"] == "proxy_after_rewrite"
    assert out["measured_claims_ok"] is False
    assert out["transfer_uses_stand_ins"] is False
    payload = __import__("json").loads(__import__("pathlib").Path(out["artifact"]).read_text())
    assert payload["scoring_mode"] == "proxy_after_rewrite"
    assert payload["transfer_uses_stand_ins"] is False
    assert "sae" in payload["stand_in_monitors"]


def test_rewrite_recollect_mode_stamps(tmp_path):
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
            recollect_n_floor=8,
        )
    assert out["scoring_mode"] == "recollect_after_rewrite"
    assert out["measured_claims_ok"] is True
    payload = __import__("json").loads(__import__("pathlib").Path(out["artifact"]).read_text())
    assert payload["scoring_mode"] == "recollect_after_rewrite"
    assert payload["recollect"]["n_recollected"] <= 16
    assert payload["recollect"]["used_chat_template"] is True


def test_rewrite_recollect_fail_closed(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    m = run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p, force_synthetic=True)
    inf = run_inflation(seed=0, artifacts=tmp_path, verbalized_metrics=v, force_synthetic=True)

    with patch(
        "evalaware.evalaware.model_runtime.try_load_causal_lm",
        return_value=None,
    ):
        try:
            run_rewrite(
                seed=0,
                artifacts=tmp_path,
                inflation_metrics=inf,
                monitor_metrics=m,
                rounds=1,
                force_synthetic=False,
                model_name="openai-community/gpt2",
            )
            ok = False
        except RuntimeError as exc:
            ok = "recollect" in str(exc).lower() or "weights" in str(exc).lower()
        assert ok


def test_collect_activations_stamps_chat_template():
    items = [{"text": "hello eval", "context": "eval"}]

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return f"<chat>{messages[-1]['content']}</chat>"

        def __call__(self, text, return_tensors="pt", truncation=True, max_length=256):
            import torch

            ids = torch.ones(1, 4, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class Block:
        def register_forward_hook(self, fn):
            self._fn = fn

            class H:
                def remove(self):
                    pass

            return H()

        def __call__(self, *a, **k):
            return None

    class Model:
        def __init__(self):
            self._block = Block()

        def __call__(self, **enc):
            import torch

            # Fire hook with fake residual
            out = torch.randn(1, 4, 8)
            self._block._fn(self._block, None, out)
            return out

    rt = type(
        "RT",
        (),
        {
            "model": Model(),
            "tokenizer": Tok(),
            "device": "cpu",
            "revision": "test",
            "notes": [],
        },
    )()

    with patch(
        "evalaware.evalaware.features.try_load_causal_lm", return_value=rt
    ), patch(
        "evalaware.evalaware.features.residual_blocks", return_value=[rt.model._block]
    ):
        out = collect_activations(
            items,
            model_name="fake-chat",
            force_synthetic=False,
            layer=0,
            dim=8,
        )
    assert out["used_chat_template"] is True
    assert out["template_path"] == "chat_template"
    assert out["is_synthetic"] is False


def test_monitors_exclude_stand_ins_from_headline(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    m = run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    assert m["transfer_uses_stand_ins"] is False
    assert "sae" in m["stand_in_monitors"]
    assert "sae" not in m["headline_held_out"]
