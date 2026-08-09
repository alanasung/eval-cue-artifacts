import pytest
from evalaware.data.splits import split_items
from evalaware.data.synthetic import build_synthetic_items

def test_split_sizes_sum():
    items = list(range(100))
    bundle = split_items(items, seed=0)
    assert sum(bundle.sizes().values()) == 100

def test_split_reproducible():
    items = list(range(40))
    a = split_items(items, seed=3)
    b = split_items(items, seed=3)
    assert a.train == b.train

def test_split_bad_frac():
    with pytest.raises(ValueError):
        split_items([1,2,3], train_frac=0.5, val_frac=0.5, test_frac=0.5)

def test_synthetic_count():
    items = build_synthetic_items(32, seed=1)
    assert len(items) == 32
    assert items[0]["is_synthetic"] is True

def test_synthetic_rejects_zero():
    with pytest.raises(ValueError):
        build_synthetic_items(0)
