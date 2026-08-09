from evalaware.configs.schema import Config, ModelConfig

def test_roles_default_empty():
    cfg = Config()
    assert cfg.roles == {}

def test_roles_accept_mapping():
    cfg = Config(roles={"reference": ModelConfig(name="gpt2"), "simulator": ModelConfig(name="gpt2")})
    assert set(cfg.roles) == {"reference", "simulator"}

def test_data_default_n_items():
    assert Config().data.n_items >= 512

def test_eval_layers_valid_for_gpt2():
    assert max(Config().eval.layers) < 12
