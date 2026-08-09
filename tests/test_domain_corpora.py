import pytest
from pathlib import Path
from evalaware.evalaware.corpora import build_corpora, run_corpora
def test_min():
    with pytest.raises(ValueError):
        build_corpora(32,0)
def test_balance():
    items=build_corpora(48,0)
    assert sum(i.context=="eval" for i in items)==24
def test_families():
    assert len({i.family for i in build_corpora(48,0)})>=4
def test_run(tmp_path):
    out=run_corpora(n_items=48, seed=0, artifacts=tmp_path)
    assert out["n"]==48 and Path(out["artifact"]).exists()
def test_holdout():
    items=build_corpora(60,1,holdout_families=["privacy"])
    assert all(i.split=="test" for i in items if i.family=="privacy")
