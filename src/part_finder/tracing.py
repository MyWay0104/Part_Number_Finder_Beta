from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, TypeVar

from part_finder.config import DATA_DIR, configure_langsmith_env


F = TypeVar("F", bound=Callable[..., Any])
FAILURE_LOG_PATH = DATA_DIR / "search_failures.jsonl"


def langsmith_enabled() -> bool:
    """Return True only when LangSmith tracing is configured in .env."""
    return configure_langsmith_env()


def traceable_run(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """Apply LangSmith tracing when available without making it a hard dependency."""
    if not langsmith_enabled():
        return lambda func: func

    try:
        from langsmith import traceable
    except ImportError:
        return lambda func: func

    return traceable(name=name, run_type=run_type)


def log_search_failure(payload: dict[str, Any]) -> None:
    """Persist low-confidence/no-result searches for later alias and catalog tuning."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with FAILURE_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def traced_tool(name: str) -> Callable[[F], F]:
    """Small decorator for tool functions so tests still run without LangSmith."""
    def decorator(func: F) -> F:
        traced = traceable_run(name=name, run_type="tool")(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return traced(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
