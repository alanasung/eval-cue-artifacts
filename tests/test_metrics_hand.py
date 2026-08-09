import numpy as np
import pytest
from evalaware.evaluation.metrics import accuracy, auroc, f1, minimum_detectable_effect, tost_equivalence

def test_accuracy_perfect():
    assert accuracy([0,1,1,0], [0,1,1,0]) == 1.0

def test_accuracy_none():
    assert accuracy([0,1], [1,0]) == 0.0

def test_f1_basic():
    assert f1([1,1,0,0], [1,0,0,0]) == pytest.approx(2/3, rel=1e-6)

def test_auroc_separated():
    assert auroc([0,0,1,1], [0.1,0.2,0.8,0.9]) == pytest.approx(1.0)

def test_auroc_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        auroc([1,1,1], [0.1,0.2,0.3])

def test_mde_positive():
    assert minimum_detectable_effect(100, sigma=1.0) > 0

def test_mde_rejects_tiny_n():
    with pytest.raises(ValueError):
        minimum_detectable_effect(1)

def test_tost_structure():
    out = tost_equivalence(np.zeros(50), low=-0.2, high=0.2, n_boot=200, seed=0)
    assert "equivalent" in out and "minimum_detectable_effect" in out
