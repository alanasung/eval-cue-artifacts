from evalaware.stages import STAGES
def test_names():
    assert set(STAGES) >= {"corpora","probe","construct","rewrite","monitors"}
def test_no_ni():
    import inspect
    import evalaware.stages as s
    assert "NotImplementedError" not in inspect.getsource(s)
def test_callable():
    assert all(callable(v) for v in STAGES.values())
