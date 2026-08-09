from evalaware.experiments.registry import clear_registry, stage, list_stages, resolve_order


def test_register_and_order():
    clear_registry()

    @stage("a")
    def a(cfg, run_dir):
        return {}

    @stage("b", deps=("a",))
    def b(cfg, run_dir):
        return {}

    assert set(list_stages()) >= {"a", "b"}
    assert resolve_order(["b"]) == ["a", "b"]
    clear_registry()
