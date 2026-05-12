from part_finder.config import configure_langsmith_env, normalize_ollama_host


def test_normalize_ollama_host_strips_openai_compatible_path():
    assert normalize_ollama_host("http://localhost:11434/v1") == "http://localhost:11434"


def test_normalize_ollama_host_keeps_base_url():
    assert normalize_ollama_host("http://localhost:11434") == "http://localhost:11434"


def test_configure_langsmith_env_sets_langchain_compat_vars(monkeypatch):
    monkeypatch.setenv("PART_FINDER_TRACE_LANGSMITH", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://example.langsmith.test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    assert configure_langsmith_env() is True
    assert __import__("os").environ["LANGCHAIN_TRACING_V2"] == "true"
    assert __import__("os").environ["LANGCHAIN_ENDPOINT"] == "https://example.langsmith.test"
    assert __import__("os").environ["LANGCHAIN_API_KEY"] == "test-key"
    assert __import__("os").environ["LANGCHAIN_PROJECT"] == "test-project"
