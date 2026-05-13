from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


# Keep all paths relative to the repository root so the CLI and tests behave the
# same way from a local checkout or a closed network machine.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ALIASES_PATH = DATA_DIR / "aliases.json"
SEMANTIC_HINTS_PATH = DATA_DIR / "semantic_hints.json"
DEFAULT_CSV_PATH = DATA_DIR / "part_numbers.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "Part_Number.db"
ENV_PATH = PROJECT_ROOT / ".env"

PART_NUMBER_PATTERN = r"P\d{7}"


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def load_project_env() -> None:
    """Load repo-local .env values without overriding explicit shell env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is optional at runtime.
        return

    load_dotenv(ENV_PATH, override=False)


def normalize_ollama_host(host: str | None) -> str | None:
    """Return the base Ollama host expected by langchain-ollama."""
    if not host:
        return None
    normalized = host.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized or None


def configure_ollama_env() -> str | None:
    """Load .env and normalize Ollama env vars for ChatOllama."""
    load_project_env()

    # Support the existing .env typo without requiring users to edit secrets.
    host = os.getenv("OLLAMA_HOST") or os.getenv("OLLMA_HOST")
    normalized_host = normalize_ollama_host(host)
    if normalized_host:
        os.environ["OLLAMA_HOST"] = normalized_host
    return normalized_host


def get_ollama_model() -> str:
    load_project_env()
    return os.getenv("OLLAMA_MODEL", "gemma3:4b")


def is_llm_enabled() -> bool:
    load_project_env()
    value = os.getenv("PART_FINDER_USE_LLM", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class LangfuseConfig:
    trace_langfuse: bool
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_base_url: str | None
    langfuse_host: str | None
    langfuse_project: str | None


def get_langfuse_config() -> LangfuseConfig:
    """Return optional Langfuse tracing settings.

    LangSmith flags are still read for backward compatibility, but they no
    longer enable the default tracing backend.
    """
    load_project_env()

    trace_flag = os.getenv("PART_FINDER_TRACE_LANGFUSE")
    trace_enabled = truthy(trace_flag)
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or None
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") or None
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or None
    host = os.getenv("LANGFUSE_HOST") or None
    project = os.getenv("LANGFUSE_PROJECT") or "part-number-finder"
    return LangfuseConfig(
        trace_langfuse=bool(trace_enabled and public_key and secret_key),
        langfuse_public_key=public_key,
        langfuse_secret_key=secret_key,
        langfuse_base_url=base_url,
        langfuse_host=host,
        langfuse_project=project,
    )


def configure_langsmith_env() -> bool:
    """Backward-compatible no-op for older callers.

    The project now uses Langfuse. Explicit LangSmith tracing is disabled here
    so restricted environments do not try to contact LangSmith.
    """
    load_project_env()
    aliases = {
        "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
        "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
    }
    for source, target in aliases.items():
        value = os.getenv(source)
        if value:
            os.environ.setdefault(target, value)
    if truthy(os.getenv("LANGSMITH_TRACING")):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    return bool(
        truthy(os.getenv("LANGSMITH_TRACING"))
        and os.getenv("LANGSMITH_API_KEY")
        and os.getenv("LANGSMITH_PROJECT")
    )


load_project_env()
