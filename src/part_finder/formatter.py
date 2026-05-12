from __future__ import annotations

from part_finder.normalizer import normalize_query, simplify_text


DISPLAY_NAMES = {
    "PM Kit": "피엠킷(pm kit)",
    "Window Quartz": "Window Quartz",
    "Liner Quartz": "라이너쿼츠",
    "Throttle Valve": "쓰로틀밸브",
    "O-ring": "O-ring",
}


def _is_abbreviation_match(original_query: str, normalized_query: str) -> bool:
    simple = simplify_text(original_query)
    return normalized_query == "Window Quartz" and any(token in simple for token in ["wq"])


def format_answer(original_query: str, normalized_query: str, results: list[dict[str, object]]) -> str:
    """Format a stable Korean answer without calling an LLM."""
    if not results:
        return "입력하신 명칭과 유사한 Part Number를 찾지 못했습니다. 검색어를 더 구체적으로 입력해주세요."

    part_numbers = [str(result["part_number"]) for result in results]
    number_text = ", ".join(part_numbers)
    best_score = float(results[0].get("score", 0.0))

    if _is_abbreviation_match(original_query, normalized_query):
        return f"질문하신 W/Q는 {normalized_query}로 매칭했습니다. 파트넘버는 {part_numbers[0]} 입니다."

    if best_score < 70.0:
        candidates = ", ".join(
            f"{result.get('part_name') or result.get('description')}({result['part_number']})" for result in results
        )
        return f"정확한 매칭은 아니지만, 가장 유사한 후보는 다음과 같습니다: {candidates}"

    display_name = DISPLAY_NAMES.get(normalized_query, normalized_query)
    return f"{display_name}의 파트넘버는 {number_text} 입니다."


def answer_from_query(query: str, top_k: int = 3) -> str:
    from part_finder.search import search_part_numbers

    normalized = normalize_query(query)
    return format_answer(query, normalized, search_part_numbers(query, top_k=top_k))
