from __future__ import annotations

from part_finder.formatter import format_answer
from part_finder.llm_router import llm_route
from part_finder.search import (
    abbreviation_search_tool,
    english_name_search_tool,
    hybrid_search_tool,
    korean_name_search_tool,
)
from part_finder.tracing import log_search_failure, traceable_run


def _call_tool(tool_name: str, query_type: str, query: str, top_k: int) -> list[dict[str, object]]:
    if tool_name == "abbreviation_search_tool" or query_type == "abbreviation":
        return abbreviation_search_tool(query, top_k=top_k)
    if tool_name == "english_name_search_tool" or query_type == "english":
        return english_name_search_tool(query, top_k=top_k)
    if tool_name == "korean_name_search_tool" or query_type == "korean":
        return korean_name_search_tool(query, top_k=top_k)
    return hybrid_search_tool(query, top_k=top_k)


@traceable_run(name="part_number_finder_agent", run_type="chain")
def answer_query(query: str, top_k: int = 3) -> str:
    """Small tool-routing workflow that behaves like a search agent.

    By default the router is rule-based and deterministic. Set
    PART_FINDER_USE_LLM=1 to let a local Ollama model classify/normalize the
    query before calling the same deterministic finder tools.
    """
    route = llm_route(query)
    query_type = route.query_type
    normalized = route.normalized_query
    tool_name = route.tool_name
    candidate_queries = route.candidate_queries or (normalized, query)

    results: list[dict[str, object]] = []
    used_candidate = normalized
    for candidate in candidate_queries:
        results = _call_tool(tool_name, query_type, candidate, top_k)
        if results and float(results[0].get("score", 0.0)) >= 70.0:
            used_candidate = candidate
            break

    if not results or float(results[0].get("score", 0.0)) < 70.0:
        log_search_failure(
            {
                "query": query,
                "query_type": query_type,
                "normalized_query": normalized,
                "candidate_queries": list(candidate_queries),
                "tool_name": tool_name,
                "best_score": float(results[0].get("score", 0.0)) if results else 0.0,
                "results_count": len(results),
            }
        )

    normalized_for_answer = normalized if route.used_llm else used_candidate
    return format_answer(query, normalized_for_answer, results)
