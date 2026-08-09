from pathlib import Path
from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.construct import run_construct
def test_probe(tmp_path):
    c=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p=run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c)
    assert 0.5 <= p["auroc_heldout"] <= 1.0
def test_construct_policy(tmp_path):
    import json
    c=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p=run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c)
    out=run_construct(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p)
    payload=json.loads(Path(out["artifact"]).read_text())
    assert "text" in payload["policy"].lower()
    assert "causal_steering_ok" in out
def test_not_licensed_with_huge_margin(tmp_path):
    c=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p=run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c)
    out=run_construct(seed=0, artifacts=tmp_path, corpora_metrics=c, probe_metrics=p, margin_min=10.0)
    assert out["internal_awareness_licensed"] is False
def test_keys(tmp_path):
    c=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p=run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c)
    for k in ("task","seed","git_sha","n"): assert k in p
