import pytest
from evalaware.utils.validation import require_positive, require_in_range

def test_require_positive():
    assert require_positive(3, "x") == 3
    with pytest.raises(ValueError):
        require_positive(0, "x")

def test_require_in_range():
    assert require_in_range(0.5, "a", low=0, high=1, inclusive=False) == 0.5
