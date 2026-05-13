from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from part_finder.formatter import format_answer, format_confirmation_prompt
from part_finder.llm_router import PartLookupItem, QueryPlan, RouteDecision, llm_decompose_query, llm_route
from part_finder.normalizer import normalize_query, simplify_text
from part_finder.search import (
    agentic_part_search_tool,
    aggregate_part_rows_tool,
    filter_part_rows_tool,
)
from part_finder.tracing import end_trace, flush_traces, log_search_failure, start_trace, trace_span, traceable_run


HIGH_CONFIDENCE_THRESHOLD = 0.85
CONFIRMATION_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.12


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
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _confidence_value(result: dict[str, object] | None) -> float:
    score = _score_value(result)
    return score / 100 if score > 1.0 else score


def _base_part_name(row: dict[str, object]) -> str:
    value = str(row.get("description") or row.get("part_name") or "")
    return value.split(" for ", 1)[0].strip()


def _row_matches_part(row: dict[str, object], expected_part: str) -> bool:
    expected_key = simplify_text(expected_part)
    if not expected_key:
        return True
    fields = [
        str(row.get("part_name") or ""),
        str(row.get("description") or ""),
        _base_part_name(row),
    ]
    return any(expected_key in simplify_text(field) for field in fields)


def _expected_part_name(query: str, route: Any) -> str:
    detected = normalize_query(query)
    route_normalized = str(getattr(route, "normalized_query", "") or "")
    if detected and simplify_text(detected) != simplify_text(query):
        return detected
    if route_normalized and simplify_text(route_normalized) != simplify_text(query):
        return route_normalized
    return ""


def _is_confirmation(query: str) -> bool:
    stripped = query.strip().lower()
    if stripped in {"확인", "네", "맞아", "맞습니다"}:
        return True
    return stripped in {"확인", "?뺤씤", "맞아", "맞습니다", "yes", "y", "ok", "okay"}


def _public_trace_rows(rows: list[dict[str, object]], top_k: int = 5) -> list[dict[str, object]]:
    allowed = [
        "part_number",
        "part_name",
        "vendor",
        "equipment_module",
        "score",
        "match_source",
        "matched_query",
        "search_reason",
    ]
    return [{field: row.get(field, "") for field in allowed} for row in rows[:top_k]]


def _failure_payload(
    user_query: str,
    route: Any,
    candidates: list[dict[str, object]],
    failure_type: str,
) -> dict[str, object]:
    return {
        "user_query": user_query,
        "query": user_query,
        "route": {
            "intent": getattr(route, "intent", ""),
            "query_type": getattr(route, "query_type", ""),
            "normalized_query": getattr(route, "normalized_query", ""),
            "preferred_tools": list(getattr(route, "preferred_tools", ()) or ()),
        },
        "candidate_queries": list(getattr(route, "candidate_queries", ()) or ()),
        "semantic_queries": list(getattr(route, "semantic_queries", ()) or getattr(route, "semantic_candidates", ()) or ()),
        "top_candidates": _public_trace_rows(candidates),
        "failure_type": failure_type,
        "suggested_alias_candidates": list(getattr(route, "phonetic_english_candidates", ()) or ()),
    }


def _confirmation_payload(
    query: str,
    route: Any,
    rows: list[dict[str, object]],
) -> dict[str, object] | None:
    if not rows:
        return None
    return {
        "original_query": query,
        "intent": getattr(route, "intent", "lookup_part"),
        "normalized_query": getattr(route, "normalized_query", ""),
        "requested_fields": tuple(getattr(route, "requested_fields", ("part_number",)) or ("part_number",)),
        "equipment_query": getattr(route, "equipment", "") or getattr(route, "equipment_query", ""),
        "vendor_query": getattr(route, "vendor", "") or getattr(route, "vendor_query", ""),
        "candidate_rows": rows,
    }


def _route_from_plan_item(plan: QueryPlan, item: PartLookupItem) -> RouteDecision:
    return RouteDecision(
        query_type="mixed",
        normalized_query=item.normalized_query,
        intent=plan.intent if plan.intent in {"lookup_part", "lookup_details", "semantic_candidate"} else "lookup_part",
        candidate_queries=item.candidate_queries or (item.normalized_query, item.raw_text),
        semantic_candidates=item.semantic_queries,
        semantic_queries=item.semantic_queries,
        requested_fields=plan.requested_fields,
        requires_confirmation=plan.intent == "semantic_candidate",
        equipment_query=plan.equipment_query,
        vendor_query=plan.vendor_query,
        vendor=plan.vendor_query,
        equipment=plan.equipment_query,
        tool_name="hybrid_search_tool",
        preferred_tools=("hybrid_search_tool", "semantic_catalog_match_tool", "vector_semantic_search_tool"),
        needs_semantic_search=bool(item.semantic_queries),
        needs_confirmation=plan.intent == "semantic_candidate",
        used_llm=plan.used_llm,
    )


def _search_and_validate(
    query: str,
    route: RouteDecision,
    top_k: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    candidates = agentic_part_search_tool(route, top_k=top_k)
    detected_part = _expected_part_name(query, route)
    semantic_intent = route.intent == "semantic_candidate" or route.query_type == "conceptual"
    expected_part = (detected_part or route.normalized_query) if not semantic_intent else detected_part
    if expected_part:
        filtered = [
            row
            for row in candidates
            if _row_matches_part(row, expected_part)
            or simplify_text(expected_part)
            in simplify_text(str(row.get("matched_query") or ""))
        ]
        if filtered:
            candidates = filtered
    validated = validate_candidates(candidates, route, expected_part=expected_part)
    return list(validated["rows"]), validated, candidates


def _format_multi_part_answer(
    original_query: str,
    plan: QueryPlan,
    grouped_rows: list[tuple[str, list[dict[str, object]]]],
    missing: list[tuple[str, list[dict[str, object]]]],
) -> str:
    lines = ["요청하신 파트별 검색 결과입니다."]
    for part_name, rows in grouped_rows:
        lines.append(f"- {part_name}:")
        for row in rows:
            context = " / ".join(
                value
                for value in [
                    str(row.get("vendor") or "").strip(),
                    str(row.get("equipment_module") or "").strip(),
                ]
                if value
            )
            description = str(row.get("description") or row.get("part_name") or "").strip()
            part_number = str(row.get("part_number") or "").strip()
            suffix = f" ({context})" if context else ""
            lines.append(f"  - {description}: {part_number}{suffix}")
    for part_name, alternatives in missing:
        context = " / ".join(value for value in [plan.vendor_query, plan.equipment_query] if value)
        prefix = f"{context} 조건의 " if context else ""
        if alternatives:
            alt_text = ", ".join(
                f"{row.get('description') or row.get('part_name')} {row.get('part_number')} ({row.get('vendor')}/{row.get('equipment_module')})"
                for row in alternatives[:3]
            )
            lines.append(f"- {part_name}: {prefix}결과는 CSV에서 찾지 못했습니다. 다른 조건 후보는 {alt_text} 입니다.")
        else:
            lines.append(f"- {part_name}: {prefix}결과를 CSV에서 찾지 못했습니다.")
    return "\n".join(lines)


def _run_multi_part_workflow(query: str, top_k: int) -> AnswerResult | None:
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return None

    from typing import TypedDict

    class WorkflowState(TypedDict, total=False):
        query: str
        plan: QueryPlan
        grouped_rows: list[tuple[str, list[dict[str, object]]]]
        missing: list[tuple[str, list[dict[str, object]]]]
        all_rows: list[dict[str, object]]

    def decompose(state: WorkflowState) -> WorkflowState:
        return {"plan": llm_decompose_query(state["query"])}

    def search_items(state: WorkflowState) -> WorkflowState:
        plan = state["plan"]
        grouped_rows: list[tuple[str, list[dict[str, object]]]] = []
        missing: list[tuple[str, list[dict[str, object]]]] = []
        all_rows: list[dict[str, object]] = []
        for item in plan.items:
            route = _route_from_plan_item(plan, item)
            rows, _, _ = _search_and_validate(item.normalized_query, route, top_k=top_k)
            if rows:
                grouped_rows.append((item.normalized_query, rows))
                all_rows.extend(rows)
                continue
            if plan.vendor_query or plan.equipment_query:
                relaxed_route = RouteDecision(
                    **{
                        **route.__dict__,
                        "vendor": "",
                        "vendor_query": "",
                        "equipment": "",
                        "equipment_query": "",
                    }
                )
                alternatives, _, _ = _search_and_validate(item.normalized_query, relaxed_route, top_k=top_k)
                missing.append((item.normalized_query, alternatives))
            else:
                missing.append((item.normalized_query, []))
        return {"grouped_rows": grouped_rows, "missing": missing, "all_rows": all_rows}

    graph = StateGraph(WorkflowState)
    graph.add_node("decompose_query", decompose)
    graph.add_node("search_each_part", search_items)
    graph.add_edge(START, "decompose_query")
    graph.add_edge("decompose_query", "search_each_part")
    graph.add_edge("search_each_part", END)
    app = graph.compile()
    state = app.invoke({"query": query})
    plan = state["plan"]
    if len(plan.items) <= 1:
        return None
    rows = list(state.get("all_rows") or [])
    answer = _format_multi_part_answer(
        query,
        plan,
        list(state.get("grouped_rows") or []),
        list(state.get("missing") or []),
    )
    return AnswerResult(
        answer=answer,
        best_score=_confidence_value(rows[0] if rows else None),
        needs_retry=not rows,
        rows=rows,
        intent=plan.intent,
        requested_fields=plan.requested_fields,
    )


def validate_candidates(candidates: list[dict[str, object]], route: Any, expected_part: str = "") -> dict[str, object]:
    """Validate candidates and decide whether answer can be finalized."""
    validated: list[dict[str, object]] = []
    seen: set[str] = set()
    vendor = str(getattr(route, "vendor", "") or getattr(route, "vendor_query", "") or "")
    equipment = str(getattr(route, "equipment", "") or getattr(route, "equipment_query", "") or "")
    for row in candidates:
        part_number = str(row.get("part_number") or "")
        if not part_number or part_number in seen:
            continue
        if vendor and simplify_text(vendor) not in simplify_text(str(row.get("vendor") or "")):
            continue
        if equipment and simplify_text(equipment) not in simplify_text(str(row.get("equipment_module") or "")):
            continue
        if expected_part and not _row_matches_part(row, expected_part):
            continue
        seen.add(part_number)
        validated.append(row)

    top_score = _confidence_value(validated[0]) if validated else 0.0
    second_score = _confidence_value(validated[1]) if len(validated) > 1 else 0.0
    margin = top_score - second_score
    source = str(validated[0].get("match_source") or "") if validated else ""
    semantic_only = source in {"semantic_catalog", "vector_semantic"}
    conceptual = getattr(route, "query_type", "") == "conceptual"
    exact_or_alias = source in {"exact", "alias", "abbreviation", "phonetic"}

    if not validated:
        status = "no_result"
    elif conceptual or semantic_only or getattr(route, "needs_confirmation", False) or getattr(route, "requires_confirmation", False):
        status = "needs_confirmation"
    elif exact_or_alias and top_score >= HIGH_CONFIDENCE_THRESHOLD:
        status = "confirmed"
    elif top_score >= HIGH_CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
        status = "confirmed"
    elif top_score >= CONFIRMATION_THRESHOLD:
        status = "needs_confirmation"
    else:
        status = "low_score"

    return {
        "status": status,
        "rows": validated,
        "top_score": top_score,
        "margin": margin,
        "semantic_only": semantic_only,
        "conceptual": conceptual,
    }


@traceable_run(name="part_number_finder_agent", run_type="chain")
def answer_query_result(
    query: str,
    top_k: int = 3,
    pending_confirmation: dict[str, object] | None = None,
) -> AnswerResult:
    trace = start_trace("part_number_finder_agent", query, {})
    try:
        if pending_confirmation and _is_confirmation(query):
            rows = list(pending_confirmation.get("candidate_rows") or [])
            requested_fields = tuple(pending_confirmation.get("requested_fields") or ("part_number",))
            intent = str(pending_confirmation.get("intent") or "lookup_part")
            normalized = str(pending_confirmation.get("normalized_query") or "")
            answer = format_answer(query, normalized, rows, requested_fields=requested_fields, intent=intent)
            trace_span(trace, "final_response", input_data={"candidate_rows": _public_trace_rows(rows)}, output_data={"answer": answer})
            return AnswerResult(
                answer=answer,
                best_score=_confidence_value(rows[0] if rows else None),
                needs_retry=not rows,
                rows=rows,
                intent=intent,
                requested_fields=requested_fields,
                pending_confirmation=None,
            )

        multi_result = _run_multi_part_workflow(query, top_k=top_k)
        if multi_result is not None:
            trace_span(
                trace,
                "langgraph_multi_part_workflow",
                input_data={"user_query": query},
                output_data={"rows": _public_trace_rows(multi_result.rows or [])},
            )
            return multi_result

        route = llm_route(query)
        trace_span(trace, "route_query", input_data={"user_query": query}, output_data=route.__dict__)

        intent = route.intent
        requested_fields = route.requested_fields or ("part_number",)

        if intent == "filter_parts":
            keyword = route.contains_keyword or route.normalized_query
            rows = filter_part_rows_tool(
                keyword,
                top_k=max(top_k, 20),
                equipment_query=route.equipment or route.equipment_query,
                vendor_query=route.vendor or route.vendor_query,
            )
            answer = format_answer(query, keyword, rows, requested_fields=requested_fields, intent=intent)
            return AnswerResult(answer=answer, best_score=_confidence_value(rows[0] if rows else None), needs_retry=not rows, rows=rows, intent=intent, requested_fields=requested_fields)

        if intent == "aggregate_parts":
            rows = aggregate_part_rows_tool(
                route.normalized_query,
                top_k=top_k,
                equipment_query=route.equipment or route.equipment_query,
                vendor_query=route.vendor or route.vendor_query,
                sort_by=route.sort_by or "part_name_length",
            )
            answer = format_answer(query, route.normalized_query, rows, requested_fields=requested_fields, intent=intent)
            return AnswerResult(answer=answer, best_score=_confidence_value(rows[0] if rows else None), needs_retry=not rows, rows=rows, intent=intent, requested_fields=requested_fields)

        rows, validated, candidates = _search_and_validate(query, route, top_k=top_k)
        detected_part = _expected_part_name(query, route)
        trace_span(trace, "search_candidates", input_data=route.__dict__, output_data=_public_trace_rows(candidates))

        trace_span(trace, "validate_candidates", input_data=_public_trace_rows(candidates), output_data={**validated, "rows": _public_trace_rows(rows)})

        if not rows:
            log_search_failure(_failure_payload(query, route, candidates, "no_result"))
            return AnswerResult(
                answer=format_answer(query, route.normalized_query, []),
                best_score=0.0,
                needs_retry=True,
                rows=[],
                intent=intent,
                requested_fields=requested_fields,
            )

        status = str(validated["status"])
        best_score = float(validated["top_score"])
        if status == "confirmed":
            answer = format_answer(query, route.normalized_query, rows, requested_fields=requested_fields, intent=intent)
            trace_span(trace, "final_response", input_data={"candidate_rows": _public_trace_rows(rows)}, output_data={"answer": answer})
            return AnswerResult(
                answer=answer,
                best_score=best_score,
                needs_retry=False,
                rows=rows,
                intent=intent,
                requested_fields=requested_fields,
            )

        failure_type = "ambiguous" if status == "needs_confirmation" else "low_score"
        log_search_failure(_failure_payload(query, route, rows, failure_type))
        payload = _confirmation_payload(query, route, rows)
        answer = format_confirmation_prompt(query, rows, route.confirmation_reason)
        trace_span(trace, "final_response", input_data={"candidate_rows": _public_trace_rows(rows)}, output_data={"answer": answer, "confirmation": True})
        return AnswerResult(
            answer=answer,
            best_score=best_score,
            needs_retry=status == "low_score",
            rows=rows,
            intent="semantic_candidate" if status == "needs_confirmation" else intent,
            requested_fields=requested_fields,
            pending_confirmation=payload,
        )
    finally:
        end_trace(trace)
        flush_traces()


def answer_query(query: str, top_k: int = 3) -> str:
    return answer_query_result(query, top_k=top_k).answer
