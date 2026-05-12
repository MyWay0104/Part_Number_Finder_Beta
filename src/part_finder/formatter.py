from __future__ import annotations

from part_finder.final_responder import LOW_CONFIDENCE_MESSAGE, confirmation_prompt, final_answer
from part_finder.normalizer import normalize_query


def format_confirmation_prompt(
    original_query: str,
    candidates: list[dict[str, object]],
    reason: str = "",
) -> str:
    return confirmation_prompt(original_query, candidates, reason)


def format_answer(
    original_query: str,
    normalized_query: str,
    results: list[dict[str, object]],
    requested_fields: tuple[str, ...] = ("part_number",),
    intent: str = "lookup_part",
) -> str:
    """Create a natural final answer using only tool-returned rows."""
    return final_answer(
        original_query=original_query,
        normalized_query=normalized_query,
        results=results,
        requested_fields=requested_fields or ("part_number",),
        intent=intent,
    )


def answer_from_query(query: str, top_k: int = 3) -> str:
    from part_finder.search import search_part_numbers

    normalized = normalize_query(query)
    return format_answer(query, normalized, search_part_numbers(query, top_k=top_k))
