from __future__ import annotations

from dataclasses import dataclass

from part_finder.formatter import format_answer, format_confirmation_prompt
from part_finder.llm_router import llm_route
from part_finder.normalizer import normalize_query, simplify_text
from part_finder.search import (
    abbreviation_search_tool,
    aggregate_part_rows_tool,
    english_name_search_tool,
    filter_part_rows_tool,
    hybrid_search_tool,
    korean_name_search_tool,
    semantic_catalog_match_tool,
    vector_semantic_search_tool,
)
from part_finder.tracing import log_search_failure, traceable_run


LOW_CONFIDENCE_THRESHOLD = 70.0
SEMANTIC_CONFIDENCE_THRESHOLD = 35.0


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    best_score: float
    needs_retry: bool
    rows: list[dict[str, object]] | None = None
    intent: str = "lookup_part"
    requested_fields: tuple[str, ...] = ("part_number",)
    pending_confirmation: dict[str, object] | None = None


def _score_value(result: dict[str, object] | None) -> float:
    if not result:
        return 0.0

    value = result.get("score", 0.0)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _detected_part_alias(query: str) -> str:
    normalized = normalize_query(query)
    if normalized and simplify_text(normalized) != simplify_text(query):
        return normalized
    return ""


def _matches_detected_part(candidate: str, results: list[dict[str, object]], detected_part: str) -> bool:
    if not detected_part:
        return True

    expected = simplify_text(detected_part)
    candidate_key = simplify_text(normalize_query(candidate))
    if candidate_key == expected:
        return True

    if not results:
        return False

    best = results[0]
    result_text = " ".join(str(best.get(field) or "") for field in ["part_name", "description", "matched_query"])
    return expected in simplify_text(result_text)


def _call_tool(
    tool_name: str,
    query_type: str,
    query: str,
    top_k: int,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    if tool_name == "semantic_catalog_match_tool":
        return semantic_catalog_match_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
    if tool_name == "vector_semantic_search_tool":
        return vector_semantic_search_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
    if tool_name == "filter_part_rows_tool":
        return filter_part_rows_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
    if tool_name == "aggregate_part_rows_tool":
        return aggregate_part_rows_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
    if tool_name == "abbreviation_search_tool" or query_type == "abbreviation":
        return abbreviation_search_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
    if tool_name == "english_name_search_tool" or query_type == "english":
        return english_name_search_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
    if tool_name == "korean_name_search_tool" or query_type == "korean":
        return korean_name_search_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
    return hybrid_search_tool(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)


def _is_confirmation(query: str) -> bool:
    stripped = query.strip().lower()
    return stripped in {"확인", "맞아", "맞습니다", "yes", "y", "ok", "okay"} or simplify_text(query) in {
        "yes",
        "y",
        "ok",
        "okay",
    }


def _looks_semantic_request(query: str) -> bool:
    simple = simplify_text(query)
    return any(token in simple for token in ["arm", "pick", "robot", "handler", "transfer"])


def _looks_filter_request(query: str) -> bool:
    simple = simplify_text(query)
    return any(token in simple for token in ["전부", "전체", "모두", "포함", "contains", "include", "all"])


def _filter_keyword(query: str, routed_keyword: str, normalized: str) -> str:
    simple = simplify_text(query)
    if "valve" in simple:
        return "Valve"
    if "밸브" in query:
        return "Valve"
    return routed_keyword or normalized


def _confirmation_payload(
    query: str,
    route_intent: str,
    requested_fields: tuple[str, ...],
    rows: list[dict[str, object]],
    equipment_query: str,
    vendor_query: str,
) -> dict[str, object] | None:
    if not rows:
        return None
    normalized_query = str(rows[0].get("matched_query") or rows[0].get("description") or rows[0].get("part_name") or "")
    return {
        "original_query": query,
        "intent": "lookup_details" if route_intent == "lookup_details" else "lookup_part",
        "normalized_query": normalized_query,
        "requested_fields": requested_fields,
        "equipment_query": equipment_query,
        "vendor_query": vendor_query,
    }


def _last_confirm(
    rows: list[dict[str, object]],
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> tuple[str, list[dict[str, object]]]:
    """Confirm final rows came from tools and satisfy the confidence policy."""
    confirmed: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        part_number = str(row.get("part_number") or "")
        if not part_number or part_number in seen:
            continue
        try:
            score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score < threshold:
            continue
        seen.add(part_number)
        confirmed.append(row)
    return ("confirmed" if confirmed else "needs_clarification", confirmed)


@traceable_run(name="part_number_finder_agent", run_type="chain")
def answer_query_result(
    query: str,
    top_k: int = 3,
    pending_confirmation: dict[str, object] | None = None,
) -> AnswerResult:
    """Small tool-routing workflow that behaves like a search agent.

    By default the router is rule-based and deterministic. Set
    PART_FINDER_USE_LLM=1 to let a local Ollama model classify/normalize the
    query before calling the same deterministic finder tools.
    """
    if pending_confirmation and _is_confirmation(query):
        normalized = str(pending_confirmation.get("normalized_query") or "")
        equipment_query = str(pending_confirmation.get("equipment_query") or "")
        vendor_query = str(pending_confirmation.get("vendor_query") or "")
        requested_fields = tuple(pending_confirmation.get("requested_fields") or ("part_number",))
        intent = str(pending_confirmation.get("intent") or "lookup_part")
        results = hybrid_search_tool(normalized, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
        best_score = _score_value(results[0] if results else None)
        return AnswerResult(
            answer=format_answer(query, normalized, results, requested_fields=requested_fields, intent=intent),
            best_score=best_score,
            needs_retry=best_score < LOW_CONFIDENCE_THRESHOLD,
            rows=results,
            intent=intent,
            requested_fields=requested_fields,
            pending_confirmation=None,
        )

    route = llm_route(query)
    intent = route.intent
    if _looks_filter_request(query):
        intent = "filter_parts"
    query_type = route.query_type
    normalized = route.normalized_query
    tool_name = route.tool_name
    candidate_queries = route.candidate_queries or (normalized, query)
    equipment_query = route.equipment_query
    vendor_query = route.vendor_query
    requested_fields = route.requested_fields or ("part_number",)
    detected_part = _detected_part_alias(query)

    if intent == "filter_parts":
        keyword = _filter_keyword(query, route.contains_keyword, normalized)
        results = filter_part_rows_tool(keyword, top_k=max(top_k, 20), equipment_query=equipment_query, vendor_query=vendor_query)
        best_score = _score_value(results[0] if results else None)
        return AnswerResult(
            answer=format_answer(query, keyword, results, requested_fields=requested_fields, intent=intent),
            best_score=best_score,
            needs_retry=best_score < LOW_CONFIDENCE_THRESHOLD,
            rows=results,
            intent=intent,
            requested_fields=requested_fields,
        )

    if intent == "aggregate_parts":
        results = aggregate_part_rows_tool(
            normalized,
            top_k=top_k,
            equipment_query=equipment_query,
            vendor_query=vendor_query,
            sort_by=route.sort_by or "part_name_length",
        )
        best_score = _score_value(results[0] if results else None)
        return AnswerResult(
            answer=format_answer(query, normalized, results, requested_fields=requested_fields, intent=intent),
            best_score=best_score,
            needs_retry=best_score < LOW_CONFIDENCE_THRESHOLD,
            rows=results,
            intent=intent,
            requested_fields=requested_fields,
        )

    semantic_required = intent == "semantic_candidate" or route.requires_confirmation or _looks_semantic_request(query)
    if semantic_required:
        semantic_query = " ".join([query, normalized, *route.semantic_candidates])
        results = semantic_catalog_match_tool(semantic_query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
        best_score = _score_value(results[0] if results else None)
        payload = _confirmation_payload(query, intent, requested_fields, results, equipment_query, vendor_query)
        if payload and best_score >= SEMANTIC_CONFIDENCE_THRESHOLD:
            return AnswerResult(
                answer=format_confirmation_prompt(query, results, route.confirmation_reason),
                best_score=best_score,
                needs_retry=False,
                rows=results,
                intent="semantic_candidate",
                requested_fields=requested_fields,
                pending_confirmation=payload,
            )

    results: list[dict[str, object]] = []
    used_candidate = normalized
    for candidate in candidate_queries:
        results = _call_tool(tool_name, query_type, candidate, top_k, equipment_query, vendor_query)
        if results and not _matches_detected_part(candidate, results, detected_part):
            results = []
            continue
        if results and _score_value(results[0]) >= LOW_CONFIDENCE_THRESHOLD:
            used_candidate = candidate
            break

    best_score = _score_value(results[0] if results else None)
    if best_score < LOW_CONFIDENCE_THRESHOLD:
        semantic_query = " ".join([query, normalized, *route.semantic_candidates])
        semantic_results = semantic_catalog_match_tool(
            semantic_query,
            top_k=top_k,
            equipment_query=equipment_query,
            vendor_query=vendor_query,
        )
        semantic_score = _score_value(semantic_results[0] if semantic_results else None)
        payload = _confirmation_payload(query, intent, requested_fields, semantic_results, equipment_query, vendor_query)
        if payload and semantic_score >= SEMANTIC_CONFIDENCE_THRESHOLD:
            return AnswerResult(
                answer=format_confirmation_prompt(query, semantic_results, route.confirmation_reason),
                best_score=semantic_score,
                needs_retry=False,
                rows=semantic_results,
                intent="semantic_candidate",
                requested_fields=requested_fields,
                pending_confirmation=payload,
            )

    if best_score < LOW_CONFIDENCE_THRESHOLD:
        log_search_failure(
            {
                "query": query,
                "intent": intent,
                "query_type": query_type,
                "normalized_query": normalized,
                "candidate_queries": list(candidate_queries),
                "semantic_candidates": list(route.semantic_candidates),
                "equipment_query": equipment_query,
                "vendor_query": vendor_query,
                "tool_name": tool_name,
                "best_score": best_score,
                "results_count": len(results),
            }
        )

    confirm_status, confirmed_results = _last_confirm(results)
    final_results = confirmed_results if confirm_status == "confirmed" else results
    return AnswerResult(
        answer=format_answer(query, used_candidate, final_results, requested_fields=requested_fields, intent=intent),
        best_score=best_score,
        needs_retry=confirm_status != "confirmed",
        rows=final_results,
        intent=intent,
        requested_fields=requested_fields,
    )


def answer_query(query: str, top_k: int = 3) -> str:
    return answer_query_result(query, top_k=top_k).answer
