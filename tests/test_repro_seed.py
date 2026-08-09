from evalaware.utils.reproducibility import set_seed
import random
import numpy as np

def test_set_seed_reproducible():
    set_seed(123)
    a = random.random()
    b = np.random.rand()
    set_seed(123)
    assert random.random() == a
    assert np.random.rand() == b
