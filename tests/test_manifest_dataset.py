
from evalaware.data.manifest import DatasetManifest, ItemRecord

def test_manifest_roundtrip(tmp_path):
    m = DatasetManifest(
        name="pilot", version="v1", n_items=2, seed=0,
        items=[ItemRecord(id="a", split="train"), ItemRecord(id="b", split="test")],
    )
    path = m.save(tmp_path / "m.json")
    # Prefer classmethod/load helper if present
    if hasattr(DatasetManifest, "load"):
        loaded = DatasetManifest.load(path)
    else:
        from evalaware.data.manifest import load_manifest
        loaded = load_manifest(path)
    assert loaded.n_items == 2
    assert loaded.items[0].id == "a"
