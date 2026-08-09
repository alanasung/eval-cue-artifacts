"""X-fix regression tests for the shared spine."""

from __future__ import annotations

import os

import numpy as np
import pytest


def test_mps_fallback_flag_in_device_info_dict():
    from evalaware.models.device import DeviceInfo

    info = DeviceInfo(
        device="mps",
        dtype="float16",
        total_memory_gb=16.0,
        available_memory_gb=8.0,
        unified_memory=True,
        platform="Darwin arm64",
        torch_version="2.6.0",
        notes=[],
        mps_fallback_enabled=True,
    )
    payload = info.to_dict()
    assert payload["mps_fallback_enabled"] is True


def test_enable_mps_fallback_sets_env(monkeypatch):
    from evalaware.models import device as device_mod

    monkeypatch.setitem(os.environ, "PYTORCH_ENABLE_MPS_FALLBACK", "0")
    # Force the setter path when MPS is unavailable: function returns False but
    # when available it should setdefault to 1.
    if device_mod.torch.backends.mps.is_available():
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        assert device_mod.enable_mps_fallback() is True
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    else:
        assert device_mod.enable_mps_fallback() is False


def test_apply_chat_template_raw_path():
    from evalaware.models.generation import apply_chat_template

    class Tok:
        chat_template = None

    out, path = apply_chat_template(Tok(), ["hello"], use_chat_template=True)
    assert out == ["hello"]
    assert path == "chat_template_unavailable"

    out2, path2 = apply_chat_template(Tok(), ["hello"], use_chat_template=False)
    assert path2 == "raw"


def test_apply_chat_template_when_present():
    from evalaware.models.generation import apply_chat_template

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "FMT:" + messages[-1]["content"]

    out, path = apply_chat_template(Tok(), ["hi"], use_chat_template=True)
    assert path == "chat_template"
    assert out == ["FMT:hi"]


def test_role_keyed_config():
    from evalaware.configs.schema import Config, ModelConfig

    cfg = Config()
    cfg.roles = {
        "reference": ModelConfig(name="gpt2", role="reference"),
        "monitor": ModelConfig(name="gpt2", role="monitor"),
    }
    assert set(cfg.roles) == {"reference", "monitor"}
    assert cfg.model.name


def test_data_default_n_items_pilot_sized():
    from evalaware.configs.schema import DataConfig, EvalConfig

    assert DataConfig().n_items >= 512
    assert 12 not in EvalConfig().layers
    assert EvalConfig().layers == [2, 4, 6]


def test_validate_layers_rejects_gpt2_layer_12():
    from evalaware.utils.validation import validate_layers

    with pytest.raises(ValueError, match="out of range"):
        validate_layers([2, 4, 12], n_layers=12)


def test_validate_layers_accepts_negatives():
    from evalaware.utils.validation import validate_layers

    assert validate_layers([-1], n_layers=12) == [11]


def test_registry_revision_pinned():
    from evalaware.models.registry import MODEL_REGISTRY, get_model_spec

    for key, spec in MODEL_REGISTRY.items():
        assert isinstance(spec.revision, str) and spec.revision
    assert get_model_spec("gpt2").revision != ""


def test_mde_and_tost():
    from evalaware.evaluation.metrics import Estimate, minimum_detectable_effect, tost_equivalence

    mde = minimum_detectable_effect(100, sigma=1.0)
    assert mde > 0
    # Tight around zero inside a wide band -> equivalent
    values = np.zeros(200) + np.random.default_rng(0).normal(0, 0.01, 200)
    result = tost_equivalence(values, low=-0.5, high=0.5, n_boot=500, seed=0)
    assert "equivalent" in result
    assert "minimum_detectable_effect" in result
    est = Estimate(value=0.0, lo=-0.1, hi=0.1, n=50)
    payload = est.to_dict()
    assert payload["excludes_zero"] is False
    assert "minimum_detectable_effect" in payload
    assert "null_claim" in payload


def test_patch_activations_and_attn_exported():
    from evalaware.models.hooks import intervene_attention_heads, patch_activations

    assert callable(patch_activations)
    assert callable(intervene_attention_heads)


def test_data_split_and_manifest(tmp_path):
    from evalaware.data import (
        DatasetManifest,
        build_synthetic_items,
        split_items,
        write_manifest,
        load_manifest,
    )

    items = build_synthetic_items(64, seed=1)
    bundle = split_items(items, seed=1)
    assert sum(bundle.n.values()) == 64
    man = DatasetManifest(name="syn", version="v1", n_items=64, seed=1)
    path = write_manifest(tmp_path / "m.json", man)
    loaded = load_manifest(path)
    assert loaded.n_items == 64


def test_ablation_controls_and_dropout():
    from evalaware.ablation import run_component_dropout, run_control_arms, run_seed_sweep

    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    ctrl = run_control_arms(labels, scores, n_shuffles=5, seed=0)
    assert "arms" in ctrl and ctrl["task"]

    components = ["a", "b"]

    def predict(active):
        if active == frozenset(components):
            return scores
        return scores * 0.5

    drop = run_component_dropout(components, labels, predict, seed=0)
    assert "marginal_value" in drop

    sweep = run_seed_sweep(lambda s: {"accuracy": float(s) / 10.0}, seeds=[0, 1, 2])
    assert sweep["summary"]["accuracy"]["n"] == 3.0


def test_reporting_roundtrip(tmp_path):
    pytest.importorskip("matplotlib")
    from evalaware.reporting import aggregate_results, write_figures, write_tables
    from evalaware.utils.run_manifest import write_json_atomic

    src = tmp_path / "runs" / "r1"
    src.mkdir(parents=True)
    write_json_atomic(
        src / "out.json",
        {"task": "E00", "seed": 0, "git_sha": "deadbeef", "accuracy": 0.75, "n": 10},
    )
    payload = aggregate_results([tmp_path / "runs"])
    assert payload.measured
    tables = write_tables(payload.measured, tmp_path / "tables")
    assert tables["markdown"].is_file()
    figs = write_figures(payload.measured, tmp_path / "figs", metric="accuracy", formats=("png",))
    assert "png" in figs
