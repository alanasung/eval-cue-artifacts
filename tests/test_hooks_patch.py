import torch
from transformers import GPT2Config, GPT2LMHeadModel
from evalaware.models.hooks import capture, patch_activations, resolve_layers, steer, ablate

def _tiny():
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=100, n_positions=64)
    return GPT2LMHeadModel(cfg)

def test_resolve_layers():
    m = _tiny()
    assert len(resolve_layers(m)) == 2

def test_capture_last_token():
    m = _tiny()
    ids = torch.randint(0, 100, (2, 8))
    with capture(m, layers=[0], last_token_only=True) as cache:
        m(ids)
    assert cache.numpy(0).shape[0] == 2

def test_patch_and_steer():
    m = _tiny()
    ids = torch.randint(0, 100, (1, 6))
    with capture(m, layers=[0], last_token_only=False) as cache:
        m(ids)
    src = cache.get(0)
    direction = torch.randn(32)
    with patch_activations(m, 0, src):
        m(ids)
    with steer(m, [0], direction, coefficient=0.5, positions=slice(-1, None)):
        m(ids)
    with ablate(m, [0], direction, positions=slice(-1, None)):
        m(ids)

def test_bad_layer():
    m = _tiny()
    import pytest
    with pytest.raises(ValueError, match="out of range"):
        with capture(m, layers=[99]):
            pass
