from evalaware.evalaware.features import surface_cue_score, synthetic_activation, EVAL_CUES
import numpy as np
def test_cue_score():
    assert surface_cue_score("This is a benchmark item") >= 1
def test_activation_dim():
    x = synthetic_activation("eval text", "eval", dim=32, seed=0)
    assert x.shape==(32,)
def test_eval_internal_higher():
    e = synthetic_activation("hello", "eval", seed=1)
    d = synthetic_activation("hello", "deploy", seed=1)
    assert e[8] > d[8]
def test_cues_nonempty():
    assert len(EVAL_CUES) >= 5
