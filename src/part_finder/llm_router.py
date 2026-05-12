from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from part_finder.data_loader import load_part_catalog
from part_finder.normalizer import detect_query_type, normalize_query
from part_finder.tracing import traceable_run

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime.
    pass


@dataclass(frozen=True)
class RouteDecision:
    query_type: str
    normalized_query: str
    candidate_queries: tuple[str, ...] = ()
    tool_name: str = "hybrid_search_tool"
    used_llm: bool = False


ALLOWED_QUERY_TYPES = {"abbreviation", "english", "korean", "mixed"}
ALLOWED_TOOLS = {
    "abbreviation_search_tool",
    "english_name_search_tool",
    "korean_name_search_tool",
    "hybrid_search_tool",
}


def rule_based_route(query: str) -> RouteDecision:
    """Deterministic fallback route used when LLM routing is disabled or fails."""
    query_type = detect_query_type(query)
    tool_name = {
        "abbreviation": "abbreviation_search_tool",
        "english": "english_name_search_tool",
        "korean": "korean_name_search_tool",
        "mixed": "hybrid_search_tool",
    }.get(query_type, "hybrid_search_tool")
    return RouteDecision(
        query_type=query_type,
        normalized_query=normalize_query(query),
        candidate_queries=(normalize_query(query), query),
        tool_name=tool_name,
        used_llm=False,
    )


def _parse_json_response(content: str) -> dict[str, Any] | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _candidate_tuple(values: object, normalized_query: str, original_query: str) -> tuple[str, ...]:
    candidates: list[str] = []
    if isinstance(values, list):
        candidates.extend(str(value).strip() for value in values if str(value).strip())
    candidates.extend([normalized_query, original_query])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return tuple(deduped)


@traceable_run(name="llm_route", run_type="chain")
def llm_route(query: str) -> RouteDecision:
    """Use local Ollama only for routing and normalization, never for final PN choice.

    Enabled by default for the agentic PoC. Set PART_FINDER_USE_LLM=0 to force
    deterministic rule-only routing. The default model is qwen2.5:7b because it
    tends to follow structured JSON instructions better than smaller models.
    """
    if str(os.getenv("PART_FINDER_USE_LLM", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return rule_based_route(query)

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return rule_based_route(query)

    # Support the existing .env typo without requiring the user to edit secrets.
    if os.getenv("OLLMA_HOST") and not os.getenv("OLLAMA_HOST"):
        os.environ["OLLAMA_HOST"] = str(os.getenv("OLLMA_HOST"))

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    catalog = load_part_catalog()
    catalog_text = "\n".join(f"- {name}" for name in catalog[:80])
    prompt = f"""
You are the query-understanding router for a semiconductor part-number finder.
Return JSON only. Do not invent part numbers.

Allowed query_type values: abbreviation, english, korean, mixed.
Allowed tool_name values:
- abbreviation_search_tool
- english_name_search_tool
- korean_name_search_tool
- hybrid_search_tool

Task:
1. Find the part-name candidate in the user query.
2. If the user wrote Korean pronunciation, abbreviation, or typo, infer the closest English catalog term.
3. Select one tool_name.
4. Return multiple candidate_queries ordered from most likely to least likely.
5. If uncertain, keep a cleaned user phrase and choose hybrid_search_tool.

Korean pronunciation examples:
- 오링, 오 링, 오-링, owe ling -> O-ring
- 샤워헤드, 샤워 해드 -> Shower Head
- 쿼츠튜브 -> Quartz Tube
- 터보펌프 -> Turbo Pump
- 드라이펌프 -> Dry Pump
- 포커스링 -> Focus Ring
- 게이트밸브 -> Gate Valve
- 엠에프씨 -> MFC

Important:
- normalized_query must be the closest term from the local catalog when possible.
- candidate_queries should include possible English names, abbreviations, and useful original variants.
- Do not output part numbers.

Known catalog examples from the local data:
{catalog_text}

User query: {query}

JSON schema:
{{"query_type":"...", "normalized_query":"...", "candidate_queries":["..."], "tool_name":"..."}}
""".strip()

    try:
        llm = ChatOllama(model=model, temperature=0)
        response = llm.invoke(prompt)
        parsed = _parse_json_response(str(response.content))
    except Exception:
        return rule_based_route(query)

    if not parsed:
        return rule_based_route(query)

    query_type = str(parsed.get("query_type", ""))
    normalized_query = str(parsed.get("normalized_query", ""))
    tool_name = str(parsed.get("tool_name", "hybrid_search_tool"))
    if query_type not in ALLOWED_QUERY_TYPES or not normalized_query or tool_name not in ALLOWED_TOOLS:
        return rule_based_route(query)

    return RouteDecision(
        query_type=query_type,
        normalized_query=normalized_query,
        candidate_queries=_candidate_tuple(parsed.get("candidate_queries"), normalized_query, query),
        tool_name=tool_name,
        used_llm=True,
    )
