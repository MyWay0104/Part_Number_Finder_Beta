from __future__ import annotations

import json
import re
from typing import Any

from part_finder.config import configure_ollama_env, get_ollama_model, is_llm_enabled


LOW_CONFIDENCE_MESSAGE = (
    "지금 데이터에서는 확실한 파트넘버를 찾지 못했습니다. "
    "장비명, 벤더, 파트 영문명, 기능 설명 중 하나를 조금 더 알려주시면 다시 찾아볼게요."
)


def _public_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "part_number": row.get("part_number", ""),
        "part_name": row.get("part_name", ""),
        "description": row.get("description", ""),
        "vendor": row.get("vendor", ""),
        "equipment_module": row.get("equipment_module", ""),
        "vendor_part_number": row.get("vendor_part_number", ""),
        "score": row.get("score", 0.0),
        "matched_query": row.get("matched_query", ""),
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

    best_score = float(results[0].get("score", 0.0))
    if best_score < 70.0:
        return LOW_CONFIDENCE_MESSAGE

    best = results[0]
    display_name = str(best.get("description") or best.get("part_name") or normalized_query).strip()
    if normalized_query and display_name.startswith(normalized_query):
        display_name = normalized_query
    context = " / ".join(
        value
        for value in [
            str(best.get("vendor") or "").strip(),
            str(best.get("equipment_module") or "").strip(),
        ]
        if value
    )
    part_numbers = ", ".join(str(result["part_number"]) for result in results)

    compact_query = re.sub(r"[^a-z0-9]+", "", original_query.lower())
    if normalized_query == "Window Quartz" and "wq" in compact_query:
        return f"질문하신 W/Q는 {normalized_query}로 매칭되었습니다. 파트넘버는 {results[0]['part_number']} 입니다."

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
        prefix = f"{context} 기준 " if context else ""
        return f"{prefix}{display_name} 관련 결과입니다.\n" + "\n".join(lines)

    prefix = f"{context} 기준 " if context else ""
    return f"{prefix}{display_name}의 파트넘버는 {part_numbers} 입니다."


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
- If LAST_CONFIRM_STATUS is not "confirmed", ask a short clarification question.
- Keep the answer concise and conversational, not a rigid template.
- Mention uncertainty when match_source is semantic/vector or confidence is low.

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
    if not is_llm_enabled() or not results or last_confirm_status != "confirmed":
        return fallback

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
        return fallback

    allowed_part_numbers = {str(row.get("part_number") or "") for row in results}
    if not any(part_number and part_number in llm_answer for part_number in allowed_part_numbers):
        return fallback
    return llm_answer


def confirmation_prompt(
    original_query: str,
    candidates: list[dict[str, object]],
    reason: str = "",
) -> str:
    if not candidates:
        return LOW_CONFIDENCE_MESSAGE

    best = candidates[0]
    candidate_name = str(best.get("description") or best.get("part_name") or best.get("matched_query") or "").strip()
    part_number = str(best.get("part_number") or "").strip()
    score = float(best.get("score", 0.0))
    return (
        f"말씀하신 표현은 데이터의 정확한 파트명과 완전히 일치하지는 않습니다. "
        f"의미상으로는 {candidate_name}"
        f"{f' ({part_number})' if part_number else ''} 쪽이 가장 가까워 보입니다"
        f"{f' (유사도 {score:.1f})' if score else ''}. "
        "이 파트를 찾으신 게 맞으면 '확인'이라고 답해주세요. 아니면 장비명이나 기능을 조금 더 알려주세요."
    )
