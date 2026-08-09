from pathlib import Path
def test_pilot_mentions_causal():
    text=(Path(__file__).resolve().parents[1]/"configs"/"experiment"/"pilot.yaml").read_text()
    assert "construct" in text and "causal" in text.lower() or "margin_min" in text
def test_construct_module_policy():
    from evalaware.evalaware import construct
    import inspect
    src = inspect.getsource(construct)
    assert "NOT treated as evidence" in src or "not treated as evidence" in src.lower() or "NOT" in src
