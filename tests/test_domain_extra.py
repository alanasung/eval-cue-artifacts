
from pathlib import Path
from evalaware.evalaware.corpora import FAMILIES, build_corpora
from evalaware.evalaware.features import batch_activations, surface_cue_score
from evalaware.evalaware.verbalized import AWARENESS_PATTERNS, verbalized_aware
from evalaware.evalaware.inflation import safety_score
from evalaware.stages import STAGES
import numpy as np

def test_family_count():
    assert len(FAMILIES) >= 6

def test_patterns_count():
    assert len(AWARENESS_PATTERNS) >= 4

def test_batch_activations_shape():
    items = [{"text":"benchmark item","context":"eval"},{"text":"hello","context":"deploy"}]
    x = batch_activations(items, dim=32, seed=0)
    assert x.shape == (2,32)

def test_stage_count():
    assert len(STAGES)==7

def test_stage_order_construct_before_rewrite():
    names = list(STAGES)
    assert names.index("construct") < names.index("rewrite")

def test_safety_benign_comply():
    assert safety_score("today's weather in Paris", "comply")==1.0

def test_verbalized_under_evaluation():
    assert verbalized_aware("I am under evaluation right now.")

def test_surface_zero_on_plain():
    assert surface_cue_score("please help me cook pasta") == 0

def test_build_has_splits():
    items = build_corpora(48, 0)
    assert {i.split for i in items} >= {"train","test"}

def test_pilot_stages_list():
    text=(Path(__file__).resolve().parents[1]/"configs"/"experiment"/"pilot.yaml").read_text()
    assert "corpora" in text and "construct" in text and "rewrite" in text

def test_activation_steer_changes():
    from evalaware.evalaware.features import synthetic_activation
    a = synthetic_activation("x","eval",steer=0.0, seed=0)
    b = synthetic_activation("x","eval",steer=1.0, seed=0)
    assert b[9] > a[9]

def test_probe_positive_field(tmp_path):
    import json
    from evalaware.evalaware.corpora import run_corpora
    from evalaware.evalaware.probe import run_probe
    c=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p=run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c)
    items=json.loads(Path(p["artifact"]).read_text())["items"]
    assert "probe_positive" in items[0]

def test_rewrite_rejected_short():
    from evalaware.evalaware.rewrite import construct_valid
    ok, reason = construct_valid({"task":"17*24","family":"capability_math"}, "17*24", min_len=20)
    assert ok is False and reason=="too_short"
