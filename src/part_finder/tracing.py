from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from part_finder.config import DATA_DIR

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime.
    pass

# The user's .env may already contain LANGSMITH_TRACING=true for other projects.
# For this closed-network PoC, avoid blocking CLI/test runs unless explicitly
# enabled for this app. Existing LangSmith key/endpoint/project are still used
# when PART_FINDER_TRACE_LANGSMITH=1 is set.
if not str(os.getenv("PART_FINDER_TRACE_LANGSMITH", "0")).strip().lower() in {"1", "true", "yes", "on"}:
    os.environ["LANGSMITH_TRACING"] = "false"


F = TypeVar("F", bound=Callable[..., Any])
FAILURE_LOG_PATH = DATA_DIR / "search_failures.jsonl"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def langsmith_enabled() -> bool:
    """Return True only when LangSmith tracing is configured in .env."""
    return (
        _truthy(os.getenv("PART_FINDER_TRACE_LANGSMITH"))
        and _truthy(os.getenv("LANGSMITH_TRACING"))
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


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
