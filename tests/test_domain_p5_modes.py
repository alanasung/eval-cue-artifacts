"""P5 honesty stamps for verbalized / inflation modes."""

from evalaware.evalaware.corpora import run_corpora
from evalaware.evalaware.inflation import run_inflation
from evalaware.evalaware.probe import run_probe
from evalaware.evalaware.verbalized import run_verbalized


def test_verbalized_synthetic_mode_stamped(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p, force_synthetic=True)
    assert v["cot_mode"] == "synthetic"
    assert v["is_synthetic"] is True
    assert "measured_claims_ok" in v


def test_inflation_simulated_mode_stamped(tmp_path):
    c = run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    p = run_probe(seed=0, artifacts=tmp_path, corpora_metrics=c, force_synthetic=True)
    v = run_verbalized(seed=0, artifacts=tmp_path, probe_metrics=p, force_synthetic=True)
    inf = run_inflation(seed=0, artifacts=tmp_path, verbalized_metrics=v, force_synthetic=True)
    assert inf["behavior_mode"] == "simulated"
    assert inf["is_synthetic"] is True
