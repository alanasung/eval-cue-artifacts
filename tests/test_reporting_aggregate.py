import json
from evalaware.reporting.aggregate import aggregate_results, flatten_metrics
from evalaware.reporting.tables import write_tables
from evalaware.reporting.figures import write_caption, write_figures

def test_flatten_metrics():
    assert flatten_metrics({"a": {"b": 1.5}, "c": "x"}) == {"a.b": 1.5}

def test_aggregate_skips_incomplete(tmp_path):
    good = {"task": "t", "seed": 0, "git_sha": "abc", "accuracy": 0.5, "n": 10}
    (tmp_path/"a.json").write_text(json.dumps(good))
    (tmp_path/"b.json").write_text(json.dumps({"task": "t"}))
    agg = aggregate_results([tmp_path])
    assert len(agg.measured) == 1
    assert agg.warnings

def test_tables_empty(tmp_path):
    paths = write_tables([], tmp_path)
    assert paths["markdown"].exists()

def test_tables_rows(tmp_path):
    rows = [{"task":"t","metrics":{"accuracy":0.8},"n":30,"raw":{"task":"t"}}]
    paths = write_tables(rows, tmp_path)
    text = paths["latex"].read_text()
    assert "toprule" in text

def test_caption_and_figures(tmp_path):
    cap = write_caption("accuracy", {"a": 0.9, "b": 0.1})
    assert "Highest" in cap
    rows = [{"task":"t","metrics":{"accuracy":0.8},"n":30,"raw":{"task":"t"}}]
    written = write_figures(rows, tmp_path, formats=("png",))
    assert written["png"].exists()
