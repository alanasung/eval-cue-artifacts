from pathlib import Path
from omegaconf import OmegaConf
from evalaware.stages import STAGES
def test_e2e(tmp_path):
    cfg = OmegaConf.create({
        "run":{"seed":0},
        "data":{"n_items":48,"name":"synthetic"},
        "model":{"name":"x"},
        "force_synthetic":True,
        "experiment":{"name":"smoke"},
        "holdout_families":["privacy","tool_use"],
        "activation_dim":32,
        "supervising_monitor":"probe",
        "margin_min":0.02,
        "rewrite_rounds":1,
    })
    for name in STAGES:
        out = STAGES[name](cfg, tmp_path)
        assert out["task"]==name
    assert (tmp_path/"artifacts"/"corpora.json").exists()
def test_pilot_n():
    text=(Path(__file__).resolve().parents[1]/"configs"/"experiment"/"pilot.yaml").read_text()
    assert "n_items: 512" in text
    assert "force_synthetic: false" in text
