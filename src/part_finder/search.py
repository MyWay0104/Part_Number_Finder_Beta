from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only when rapidfuzz is absent.
    fuzz = None

from part_finder.data_loader import load_part_data
from part_finder.normalizer import normalize_query, simplify_text
from part_finder.tracing import traced_tool


def _score(query: str, candidate: str) -> float:
    """Return a deterministic 0-100 similarity score."""
    if not query or not candidate:
        return 0.0
    if fuzz:
        return float(max(fuzz.WRatio(query, candidate), fuzz.partial_ratio(query, candidate)))
    return SequenceMatcher(None, simplify_text(query), simplify_text(candidate)).ratio() * 100


def _candidate_texts(row: dict[str, str]) -> Iterable[str]:
    fields = [
        row.get("part_name", ""),
        row.get("description", ""),
        row.get("equipment_module", ""),
        row.get("vendor_part_number", ""),
        row.get("vendor", ""),
    ]
    combined = " ".join(field for field in fields if field)
    return [*fields, combined]


@traced_tool("search_part_numbers")
def search_part_numbers(query: str, top_k: int = 3) -> list[dict[str, object]]:
    """Search part rows with alias normalization, fuzzy matching, and PN dedupe."""
    normalized_query = normalize_query(query)
    rows = load_part_data()
    if not rows:
        return []

    ranked: list[dict[str, object]] = []
    simple_query = simplify_text(normalized_query)
    simple_original_query = simplify_text(query)
    for index, row in enumerate(rows):
        scores = [_score(normalized_query, text) for text in _candidate_texts(row)]

        # Exact simplified containment should outrank fuzzy near misses.
        exact_bonus = 0.0
        priority = 4
        part_name_key = simplify_text(row.get("part_name", ""))
        description_key = simplify_text(row.get("description", ""))
        if part_name_key and part_name_key in simple_original_query:
            priority = 0
            exact_bonus = 100.0
        elif simple_query and part_name_key == simple_query:
            priority = 1
        elif simple_query and description_key == simple_query:
            priority = 2
        for text in _candidate_texts(row):
            simple_text = simplify_text(text)
            if simple_query and simple_query in simple_text:
                exact_bonus = 100.0
                priority = min(priority, 3)
                break

        score = max([exact_bonus, *scores])
        ranked.append(
            {
                **row,
                "score": round(score, 2),
                "matched_query": normalized_query,
                "_priority": priority,
                "_index": index,
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["_priority"]),
            int(item["_index"]),
            str(item["part_number"]),
        )
    )

    results: list[dict[str, object]] = []
    seen_part_numbers: set[str] = set()
    for item in ranked:
        part_number = str(item["part_number"])
        if part_number in seen_part_numbers:
            continue
        if float(item["score"]) < 45.0:
            continue
        seen_part_numbers.add(part_number)
        item.pop("_priority", None)
        item.pop("_index", None)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


@traced_tool("abbreviation_search_tool")
def abbreviation_search_tool(query: str, top_k: int = 3) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k)


@traced_tool("english_name_search_tool")
def english_name_search_tool(query: str, top_k: int = 3) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k)


@traced_tool("korean_name_search_tool")
def korean_name_search_tool(query: str, top_k: int = 3) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k)


@traced_tool("hybrid_search_tool")
def hybrid_search_tool(query: str, top_k: int = 3) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k)
