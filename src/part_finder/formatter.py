from __future__ import annotations

from part_finder.normalizer import normalize_query, simplify_text


LOW_CONFIDENCE_MESSAGE = (
    "정확한 파트넘버가 아닐 수도 있습니다. 찾으시는 파트넘버가 아니라면, "
    "파트의 영문명을 입력해주세요."
)

DISPLAY_NAMES = {
    "PM Kit": "PM Kit",
    "Window Quartz": "Window Quartz",
    "Liner Quartz": "Liner Quartz",
    "Throttle Valve": "Throttle Valve",
    "Slit Valve": "Slit Valve",
    "Gate Valve": "Gate Valve",
    "Vacuum Gauge": "Vacuum Gauge",
    "Ceramic Plate": "Ceramic Plate",
    "Ion Source": "Ion Source",
    "O-ring": "O-ring",
    "Robot Blade": "Robot Blade",
}


def _is_abbreviation_match(original_query: str, normalized_query: str) -> bool:
    simple = simplify_text(original_query)
    return normalized_query == "Window Quartz" and any(token in simple for token in ["wq"])


def _context_text(results: list[dict[str, object]]) -> str:
    if not results:
        return ""

    first = results[0]
    context = []
    vendor = str(first.get("vendor") or "").strip()
    equipment = str(first.get("equipment_module") or "").strip()
    if vendor:
        context.append(vendor)
    if equipment:
        context.append(equipment)
    return " / ".join(context)


def _display_name(normalized_query: str, results: list[dict[str, object]]) -> str:
    if normalized_query:
        return DISPLAY_NAMES.get(normalized_query, normalized_query)
    if results:
        value = str(results[0].get("description") or results[0].get("part_name") or "")
        return value.split(" for ", 1)[0].strip()
    return normalized_query


def _field_label(field: str) -> str:
    return {
        "part_number": "Part Number",
        "part_name": "Part Name",
        "description": "English Name",
        "vendor": "Vendor",
        "equipment_module": "Equipment/Module",
        "vendor_part_number": "Vendor Part Number",
    }.get(field, field)


def _format_detailed_rows(results: list[dict[str, object]], requested_fields: tuple[str, ...]) -> str:
    lines: list[str] = []
    for result in results:
        values = []
        for field in requested_fields:
            value = str(result.get(field) or "").strip()
            if value:
                values.append(f"{_field_label(field)}: {value}")
        if values:
            lines.append("- " + " / ".join(values))
    return "\n".join(lines)


def format_confirmation_prompt(
    original_query: str,
    candidates: list[dict[str, object]],
    reason: str = "",
) -> str:
    if not candidates:
        return LOW_CONFIDENCE_MESSAGE

    candidate_name = _display_name(str(candidates[0].get("matched_query") or ""), candidates)
    return (
        "입력하신 표현은 현재 파트 목록의 정확한 파트명과 일치하지 않습니다.\n"
        f"다만 문맥상 {candidate_name}와 유사한 요청으로 보입니다. "
        f"찾으시는 파트가 {candidate_name}가 맞다면 \"확인\"이라고 입력해 주세요."
    )


def format_answer(
    original_query: str,
    normalized_query: str,
    results: list[dict[str, object]],
    requested_fields: tuple[str, ...] = ("part_number",),
    intent: str = "lookup_part",
) -> str:
    """Format a stable Korean answer without calling an LLM."""
    if not results:
        return LOW_CONFIDENCE_MESSAGE

    best_score = float(results[0].get("score", 0.0))
    if best_score < 70.0:
        return LOW_CONFIDENCE_MESSAGE

    part_numbers = [str(result["part_number"]) for result in results]
    number_text = ", ".join(part_numbers)
    requested_fields = requested_fields or ("part_number",)

    if _is_abbreviation_match(original_query, normalized_query):
        return f"질문하신 W/Q는 {normalized_query}로 매칭되었습니다. 파트넘버는 {part_numbers[0]} 입니다."

    display_name = _display_name(normalized_query, results)
    context = _context_text(results)
    if intent in {"lookup_details", "filter_parts", "aggregate_parts"} or requested_fields != ("part_number",):
        if intent in {"filter_parts", "aggregate_parts"}:
            header = f"{display_name} 결과입니다."
        else:
            header = f"{context} 기준 {display_name} 결과입니다." if context else f"{display_name} 결과입니다."
        return f"{header}\n{_format_detailed_rows(results, requested_fields)}"

    if context:
        return f"{context} 기준 {display_name}의 파트넘버는 {number_text} 입니다."
    return f"{display_name}의 파트넘버는 {number_text} 입니다."


def answer_from_query(query: str, top_k: int = 3) -> str:
    from part_finder.search import search_part_numbers

    normalized = normalize_query(query)
    return format_answer(query, normalized, search_part_numbers(query, top_k=top_k))
