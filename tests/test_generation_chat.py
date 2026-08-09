from evalaware.models.generation import apply_chat_template

class _Tok:
    chat_template = "x"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "WRAPPED:" + messages[-1]["content"]

class _Bare:
    chat_template = None

def test_chat_template_applied():
    out, path = apply_chat_template(_Tok(), ["hi"], use_chat_template=True)
    assert path == "chat_template"
    assert out[0].startswith("WRAPPED:")

def test_chat_template_raw():
    out, path = apply_chat_template(_Tok(), ["hi"], use_chat_template=False)
    assert path == "raw"
    assert out == ["hi"]

def test_chat_template_unavailable():
    out, path = apply_chat_template(_Bare(), ["hi"], use_chat_template=True)
    assert path == "chat_template_unavailable"
