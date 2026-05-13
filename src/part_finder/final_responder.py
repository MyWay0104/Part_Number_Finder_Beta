from __future__ import annotations

import json
import re
from typing import Any

from part_finder.config import configure_ollama_env, get_ollama_model, is_llm_enabled
from part_finder.tracing import start_trace, trace_span


LOW_CONFIDENCE_MESSAGE = (
    "현재 CSV에서 조건에 맞는 Part Number를 찾지 못했습니다."
)


def _score_100(row: dict[str, object]) -> float:
    try:
        score = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return score * 100 if score <= 1.0 else score


def _public_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "part_number": row.get("part_number", ""),
        "part_name": row.get("part_name", ""),
        "description": row.get("description", ""),
        "vendor": row.get("vendor", ""),
        "equipment_module": row.get("equipment_module", ""),
        "vendor_part_number": row.get("vendor_part_number", ""),
        "score": row.get("score", 0.0),
        "match_source": row.get("match_source", ""),
        "matched_query": row.get("matched_query", ""),
        "search_reason": row.get("search_reason", ""),
        "semantic_match": bool(row.get("semantic_match")),
        "vector_match": bool(row.get("vector_match")),
    }


def _field_label(field: str) -> str:
    return {
        "part_number": "Part Number",
        "part_name": "Part Name",
        "description": "English Name",
        "vendor": "Vendor",
        "equipment_module": "Equipment/Module",
        "vendor_part_number": "Vendor Part Number",
    }.get(field, field)


def _fallback_answer(
    original_query: str,
    normalized_query: str,
    results: list[dict[str, object]],
    requested_fields: tuple[str, ...],
    intent: str,
) -> str:
    if not results:
        return LOW_CONFIDENCE_MESSAGE

    best = results[0]
    best_score = _score_100(best)
    if best_score < 70.0:
        return LOW_CONFIDENCE_MESSAGE

    display_name = str(best.get("description") or best.get("part_name") or normalized_query).strip()
    normalized_display = normalized_query.strip()
    if normalized_display and display_name.startswith(normalized_display):
        display_name = normalized_display
    context = " / ".join(
        value
        for value in [
            str(best.get("vendor") or "").strip(),
            str(best.get("equipment_module") or "").strip(),
        ]
        if value
    )
    prefix = f"{context} 기준 " if context else ""
    uncertain = str(best.get("match_source") or "") in {"semantic_catalog", "vector_semantic"} or best_score < 85.0
    lead = "가장 가까운 후보로는 " if uncertain else ""

    if intent in {"lookup_details", "filter_parts", "aggregate_parts"} or requested_fields != ("part_number",):
        lines = []
        for row in results:
            values = []
            for field in requested_fields:
                value = str(row.get(field) or "").strip()
                if value:
                    values.append(f"{_field_label(field)}: {value}")
            if values:
                lines.append("- " + " / ".join(values))
        if lines:
            return f"{lead}{prefix}{display_name} 관련 결과입니다.\n" + "\n".join(lines)

    part_numbers = ", ".join(str(result["part_number"]) for result in results if result.get("part_number"))
    compact_query = re.sub(r"[^a-z0-9]+", "", original_query.lower())
    if normalized_query == "Window Quartz" and "wq" in compact_query:
        return f"질문하신 W/Q는 {normalized_query}로 매칭되었습니다. 파트넘버는 {part_numbers} 입니다."
    return f"{lead}{prefix}{display_name}의 파트넘버는 {part_numbers} 입니다."


def _call_final_llm(payload: dict[str, Any]) -> str:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return ""

    configure_ollama_env()
    model = get_ollama_model()
    prompt = f"""
You are the final response agent for a part-number finder chatbot.
Answer in natural Korean unless the user clearly used only English.

Hard rules:
- Use only the part numbers and fields supplied in CANDIDATE_ROWS.
- Do not invent, modify, or complete a part number.
- If a match is semantic/vector or confidence is low, say it is the closest candidate.
- Do not ask follow-up or confirmation questions. The current CLI is single-turn.
- If LAST_CONFIRM_STATUS is not "confirmed", summarize the available candidates without asking the user to confirm.
- Keep the answer concise.

PAYLOAD:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    try:
        llm = ChatOllama(model=model, temperature=0.2)
        response = llm.invoke(prompt)
    except Exception:
        return ""
    return str(getattr(response, "content", response)).strip()


def final_answer(
    original_query: str,
    normalized_query: str,
    results: list[dict[str, object]],
    requested_fields: tuple[str, ...] = ("part_number",),
    intent: str = "lookup_part",
    last_confirm_status: str = "confirmed",
) -> str:
    fallback = _fallback_answer(original_query, normalized_query, results, requested_fields, intent)
    trace = start_trace("final_responder", original_query, {})

    def record(answer: str, fallback_used: bool, hallucination_guard_triggered: bool) -> str:
        trace_span(
            trace,
            "final_response",
            input_data={"candidate_rows_count": len(results), "last_confirm_status": last_confirm_status},
            output_data={
                "answer": answer,
                "fallback_used": fallback_used,
                "hallucination_guard_triggered": hallucination_guard_triggered,
            },
        )
        return answer

    if not is_llm_enabled() or not results or last_confirm_status != "confirmed":
        return record(fallback, True, False)

    payload = {
        "user_query": original_query,
        "normalized_query": normalized_query,
        "intent": intent,
        "requested_fields": list(requested_fields),
        "last_confirm_status": last_confirm_status,
        "candidate_rows": [_public_row(row) for row in results],
    }
    llm_answer = _call_final_llm(payload)
    if not llm_answer:
        return record(fallback, True, False)

    followup_patterns = ["더 알려주시면", "찾아보겠습니다", "확인하시겠습니까", "확인 필요", "'확인'", "확인이라고"]
    if any(pattern in llm_answer for pattern in followup_patterns) or "?" in llm_answer:
        return record(fallback, True, False)

    allowed_part_numbers = {str(row.get("part_number") or "") for row in results}
    mentioned_part_numbers = set(re.findall(r"P\d{7}", llm_answer))
    hallucination_guard_triggered = bool(mentioned_part_numbers - allowed_part_numbers)
    has_allowed_part_number = any(part_number and part_number in llm_answer for part_number in allowed_part_numbers)
    if hallucination_guard_triggered or not has_allowed_part_number:
        return record(fallback, True, hallucination_guard_triggered)
    return record(llm_answer, False, False)


def confirmation_prompt(
    original_query: str,
    candidates: list[dict[str, object]],
    reason: str = "",
) -> str:
    if not candidates:
        return LOW_CONFIDENCE_MESSAGE

    lines = []
    for index, row in enumerate(candidates[:5], start=1):
        name = str(row.get("description") or row.get("part_name") or row.get("matched_query") or "").strip()
        part_number = str(row.get("part_number") or "").strip()
        vendor = str(row.get("vendor") or "").strip()
        module = str(row.get("equipment_module") or "").strip()
        context = " / ".join(value for value in [vendor, module] if value)
        lines.append(f"{index}. {name} - {part_number}{f' ({context})' if context else ''}")

    reason_text = f" {reason}" if reason else ""
    return f"표현이 정확한 파트명과 완전히 일치하지 않아 가장 가까운 후보를 찾았습니다.{reason_text}\n" + "\n".join(lines)
