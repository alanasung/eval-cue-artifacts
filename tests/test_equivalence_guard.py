"""CI spanning zero is not a null claim without TOST + MDE."""
import numpy as np
from evalaware.evaluation.metrics import bootstrap_mean, tost_equivalence

def test_span_zero_not_equivalence():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.0, 1.0, size=30)
    est = bootstrap_mean(vals, n_boot=300, seed=0)
    # Likely spans zero; equivalence to tiny band should fail
    out = tost_equivalence(vals, low=-0.01, high=0.01, n_boot=300, seed=0)
    assert "minimum_detectable_effect" in out
    assert out["estimate"]["lo"] <= out["estimate"]["hi"]
