from evalaware.models.logged_llm import LoggedLLM


def test_logged_llm_records(tmp_path):
    def backend(prompt: str) -> str:
        return "resp:" + prompt

    llm = LoggedLLM(backend, tmp_path, model_version="test-1")
    assert llm("hi") == "resp:hi"
    assert (tmp_path / "llm_log.jsonl").exists()
