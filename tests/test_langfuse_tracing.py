import sys
from types import SimpleNamespace

from part_finder.config import get_langfuse_config
from part_finder.agent import answer_query_result
from part_finder.llm_router import llm_route
from part_finder.tracing import get_langfuse_client, log_search_failure


def test_langfuse_env_absent_does_not_block_agent(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")
    monkeypatch.delenv("PART_FINDER_TRACE_LANGFUSE", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    get_langfuse_client.cache_clear()

    result = answer_query_result("클램프 파트넘버 찾아줘", top_k=3)

    assert result.rows


def test_langfuse_client_creation_failure_is_non_blocking(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")
    monkeypatch.setenv("PART_FINDER_TRACE_LANGFUSE", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setattr("part_finder.tracing.get_langfuse_config", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    get_langfuse_client.cache_clear()

    result = answer_query_result("쓰로틀밸브 찾아줘", top_k=3)

    assert result.rows


def test_tracing_disabled_pytest_path_passes(monkeypatch):
    monkeypatch.setenv("PART_FINDER_TRACE_LANGFUSE", "0")
    get_langfuse_client.cache_clear()

    assert get_langfuse_client() is None


def test_failure_log_independent_from_langfuse(tmp_path, monkeypatch):
    monkeypatch.setenv("PART_FINDER_TRACE_LANGFUSE", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    path = tmp_path / "search_failures.jsonl"
    monkeypatch.setattr("part_finder.tracing.FAILURE_LOG_PATH", path)

    log_search_failure({"user_query": "missing", "failure_type": "no_result"})

    assert path.exists()
    assert "missing" in path.read_text(encoding="utf-8")


def test_langfuse_base_url_env_is_supported(monkeypatch):
    monkeypatch.setenv("PART_FINDER_TRACE_LANGFUSE", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.local")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    config = get_langfuse_config()

    assert config.trace_langfuse is True
    assert config.langfuse_base_url == "https://langfuse.example.local"


def test_langfuse_client_uses_explicit_secret_and_base_url(monkeypatch):
    captured: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("PART_FINDER_TRACE_LANGFUSE", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.local")
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=FakeLangfuse))
    get_langfuse_client.cache_clear()

    assert get_langfuse_client() is not None
    assert captured["public_key"] == "public"
    assert captured["secret_key"] == "secret"
    assert captured["base_url"] == "https://langfuse.example.local"


def test_llm_route_passes_langfuse_callback_to_langchain(monkeypatch):
    captured: dict[str, object] = {}
    handler = object()

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def invoke(self, prompt, config=None):
            captured["config"] = config
            return SimpleNamespace(
                content='{"intent":"lookup_part","query_type":"english","normalized_query":"O-ring",'
                '"candidate_queries":["O-ring"],"requested_fields":["part_number"],'
                '"preferred_tools":["hybrid_search_tool"],"tool_name":"hybrid_search_tool"}'
            )

    monkeypatch.setenv("PART_FINDER_USE_LLM", "1")
    monkeypatch.setitem(sys.modules, "langchain_ollama", SimpleNamespace(ChatOllama=FakeChatOllama))
    monkeypatch.setattr("part_finder.llm_router.load_part_catalog", lambda: ["O-ring"])
    monkeypatch.setattr("part_finder.llm_router.get_langfuse_callback_handler", lambda trace=None: handler)
    monkeypatch.setattr("part_finder.llm_router.start_trace", lambda *args, **kwargs: SimpleNamespace(trace_id="trace", id="span"))
    monkeypatch.setattr("part_finder.llm_router.trace_span", lambda *args, **kwargs: None)
    monkeypatch.setattr("part_finder.llm_router.end_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr("part_finder.llm_router.flush_traces", lambda *args, **kwargs: None)

    route = llm_route("O-ring")

    assert route.used_llm is True
    assert captured["config"]["callbacks"] == [handler]  # type: ignore[index]
