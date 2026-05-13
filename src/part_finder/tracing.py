from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

from part_finder.config import DATA_DIR, get_langfuse_config


F = TypeVar("F", bound=Callable[..., Any])
FAILURE_LOG_PATH = DATA_DIR / "search_failures.jsonl"


def _safe_payload(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


@lru_cache(maxsize=1)
def get_langfuse_client() -> Any | None:
    """Create a Langfuse client when configured; otherwise return no-op None."""
    try:
        config = get_langfuse_config()
    except Exception:
        return None
    if not config.trace_langfuse:
        return None

    try:
        from langfuse import Langfuse

        if config.langfuse_base_url:
            os.environ["LANGFUSE_BASE_URL"] = config.langfuse_base_url
        if config.langfuse_host:
            os.environ["LANGFUSE_HOST"] = config.langfuse_host
        if config.langfuse_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse_public_key
        if config.langfuse_secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse_secret_key
        return Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            base_url=config.langfuse_base_url,
            host=config.langfuse_host,
        )
    except Exception:
        return None


def _trace_context(trace: Any | None) -> dict[str, str] | None:
    trace_id = getattr(trace, "trace_id", None)
    span_id = getattr(trace, "id", None)
    if not trace_id:
        return None
    context = {"trace_id": str(trace_id)}
    if span_id:
        context["parent_span_id"] = str(span_id)
    return context


def get_langfuse_callback_handler(trace: Any | None = None) -> Any | None:
    """Return a LangChain callback handler attached to the active Langfuse trace."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        config = get_langfuse_config()
        if config.langfuse_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse_public_key
        if config.langfuse_secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse_secret_key
        if config.langfuse_base_url:
            os.environ["LANGFUSE_BASE_URL"] = config.langfuse_base_url
        return CallbackHandler(public_key=get_langfuse_config().langfuse_public_key, trace_context=_trace_context(trace))
    except Exception:
        return None


def start_trace(name: str, user_query: str = "", metadata: dict[str, Any] | None = None) -> Any | None:
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        project = get_langfuse_config().langfuse_project
        trace_metadata = {"project": project, **(metadata or {})}
        if hasattr(client, "start_observation"):
            return client.start_observation(
                name=name,
                as_type="chain",
                input=user_query,
                metadata=_safe_payload(trace_metadata),
            )
        if hasattr(client, "trace"):
            return client.trace(name=name, input=user_query, metadata=_safe_payload(trace_metadata))
    except Exception:
        return None
    return None


def trace_span(
    trace: Any | None,
    name: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if trace is None:
        return
    try:
        payload = {
            "name": name,
            "input": _safe_payload(input_data),
            "output": _safe_payload(output_data),
            "metadata": _safe_payload(metadata or {}),
        }
        if hasattr(trace, "start_observation"):
            span = trace.start_observation(as_type="span", **payload)
            if hasattr(span, "end"):
                span.end()
            return
        if hasattr(trace, "span"):
            span = trace.span(**payload)
            if hasattr(span, "end"):
                span.end()
            return
        if hasattr(trace, "update"):
            trace.update(metadata={name: payload})
    except Exception:
        return


def end_trace(trace: Any | None, output_data: Any = None) -> None:
    if trace is None:
        return
    try:
        if output_data is not None and hasattr(trace, "update"):
            trace.update(output=_safe_payload(output_data))
        if hasattr(trace, "end"):
            trace.end()
    except Exception:
        return


def flush_traces() -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        if hasattr(client, "flush"):
            client.flush()
    except Exception:
        return


def traceable_run(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """Compatibility decorator; Langfuse spans are recorded explicitly."""
    return lambda func: func


def traced_tool(name: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def log_search_failure(payload: dict[str, Any]) -> None:
    """Persist low-confidence/no-result searches independent of tracing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with FAILURE_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
