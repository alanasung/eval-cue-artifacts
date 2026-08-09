import numpy as np
from evalaware.ablation.controls import run_control_arms
from evalaware.ablation.component_dropout import run_component_dropout
from evalaware.ablation.seed_sweep import run_seed_sweep

def test_controls_keys():
    x = np.random.default_rng(0).normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int)
    out = run_control_arms(x, y, lambda f: f[:, 0], seed=0)
    assert "real_accuracy" in out

def test_component_dropout():
    labels = np.array([0,1,0,1,1,0])
    def predict(comps):
        return np.array([0.1,0.9,0.1,0.8,0.7,0.2])
    out = run_component_dropout(["a","b"], predict, labels, seed=0)
    assert "marginal_value" in out

def test_seed_sweep():
    out = run_seed_sweep(lambda s: {"seed": s, "acc": 0.5}, [0,1,2])
    assert out["n_cells"] == 3
