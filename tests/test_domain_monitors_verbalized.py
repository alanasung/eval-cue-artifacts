from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.monitors import run_monitors
from evalaware.evalaware.verbalized import run_verbalized, verbalized_aware
def test_monitors(tmp_path):
    c=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p=run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c)
    m=run_monitors(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    assert m["supervising"]=="probe" and "sae" in m["held_out"]
def test_patterns():
    assert verbalized_aware("I notice this is an evaluation item.")
    assert not verbalized_aware("Let me solve 2+2.")
def test_unverb(tmp_path):
    c=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p=run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c)
    out=run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p)
    assert 0<=out["unverbalized_fraction"]<=1
