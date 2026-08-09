from evalaware.utils.io import append_jsonl, read_jsonl, save_json, load_json

def test_jsonl_roundtrip(tmp_path):
    path = tmp_path/"a.jsonl"
    append_jsonl(path, {"x": 1})
    append_jsonl(path, {"x": 2})
    rows = read_jsonl(path)
    assert len(rows) == 2

def test_save_load_json(tmp_path):
    path = tmp_path/"a.json"
    save_json(path, {"a": 1})
    assert load_json(path)["a"] == 1
