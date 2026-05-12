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
from part_finder.vector_index import SEMANTIC_HINTS, vector_search


def _score(query: str, candidate: str) -> float:
    """Return a deterministic 0-100 similarity score."""
    if not query or not candidate:
        return 0.0
    if fuzz:
        return float(max(fuzz.WRatio(query, candidate), fuzz.partial_ratio(query, candidate)))
    return SequenceMatcher(None, simplify_text(query), simplify_text(candidate)).ratio() * 100


def _part_texts(row: dict[str, str]) -> Iterable[str]:
    return [
        row.get("part_name", ""),
        row.get("description", ""),
    ]


def _candidate_texts(row: dict[str, str]) -> Iterable[str]:
    fields = [
        *list(_part_texts(row)),
        row.get("equipment_module", ""),
        row.get("vendor_part_number", ""),
        row.get("vendor", ""),
    ]
    combined = " ".join(field for field in fields if field)
    return [*fields, combined]


def _base_part_name(row: dict[str, str]) -> str:
    value = row.get("description", "") or row.get("part_name", "")
    value = value.split(" for ", 1)[0].strip()
    return value


def _semantic_texts(row: dict[str, str]) -> Iterable[str]:
    base_name = _base_part_name(row)
    return [
        row.get("part_name", ""),
        row.get("description", ""),
        row.get("equipment_module", ""),
        base_name,
        *SEMANTIC_HINTS.get(base_name, []),
    ]


def _field_matches(value: str, expected: str) -> bool:
    if not expected:
        return True
    simple_value = simplify_text(value)
    simple_expected = simplify_text(expected)
    return bool(simple_expected and simple_expected in simple_value)


def _dedupe_rows(rows: Iterable[dict[str, object]], limit: int) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    seen_part_numbers: set[str] = set()
    for item in rows:
        part_number = str(item["part_number"])
        if part_number in seen_part_numbers:
            continue
        seen_part_numbers.add(part_number)
        results.append(item)
        if len(results) >= limit:
            break
    return results


@traced_tool("search_part_numbers")
def search_part_numbers(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    """Search part rows with alias normalization, fuzzy matching, and PN dedupe."""
    normalized_query = normalize_query(query)
    rows = load_part_data()
    if not rows:
        return []

    ranked: list[dict[str, object]] = []
    simple_query = simplify_text(normalized_query)
    simple_original_query = simplify_text(query)
    for index, row in enumerate(rows):
        if equipment_query and not _field_matches(row.get("equipment_module", ""), equipment_query):
            continue
        if vendor_query and not _field_matches(row.get("vendor", ""), vendor_query):
            continue

        scores = [_score(normalized_query, text) for text in _part_texts(row)]

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
        for text in _part_texts(row):
            simple_text = simplify_text(text)
            if simple_query and simple_query in simple_text:
                exact_bonus = 100.0
                priority = min(priority, 3)
                break

        context_bonus = 0.0
        if equipment_query:
            context_bonus += 5.0
        if vendor_query:
            context_bonus += 5.0

        score = min(100.0, max([exact_bonus, *scores]) + context_bonus)
        ranked.append(
            {
                **row,
                "score": round(score, 2),
                "matched_query": normalized_query,
                "matched_equipment": equipment_query,
                "matched_vendor": vendor_query,
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

    filtered: list[dict[str, object]] = []
    for item in ranked:
        if float(item["score"]) < 45.0:
            continue
        item.pop("_priority", None)
        item.pop("_index", None)
        filtered.append(item)
    return _dedupe_rows(filtered, top_k)


@traced_tool("semantic_catalog_match_tool")
def semantic_catalog_match_tool(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    """Find likely catalog rows from contextual wording, not only names/aliases."""
    rows = load_part_data()
    if not rows:
        return []

    vector_results = vector_semantic_search_tool(
        query,
        top_k=top_k,
        equipment_query=equipment_query,
        vendor_query=vendor_query,
    )
    if vector_results and float(vector_results[0].get("score", 0.0)) >= 35.0:
        return vector_results

    ranked: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if equipment_query and not _field_matches(row.get("equipment_module", ""), equipment_query):
            continue
        if vendor_query and not _field_matches(row.get("vendor", ""), vendor_query):
            continue

        score = max(_score(query, text) for text in _semantic_texts(row))
        ranked.append(
            {
                **row,
                "score": round(score, 2),
                "matched_query": _base_part_name(row),
                "matched_equipment": equipment_query,
                "matched_vendor": vendor_query,
                "semantic_match": True,
                "_index": index,
            }
        )

    ranked.sort(key=lambda item: (-float(item["score"]), int(item["_index"]), str(item["part_number"])))
    filtered: list[dict[str, object]] = []
    for item in ranked:
        if float(item["score"]) < 55.0:
            continue
        item.pop("_index", None)
        filtered.append(item)
    return _dedupe_rows(filtered, top_k)


@traced_tool("vector_semantic_search_tool")
def vector_semantic_search_tool(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    """Search CSV row chunks through a local TF-IDF vector index."""
    ranked: list[dict[str, object]] = []
    for item in vector_search(query, top_k=max(top_k * 4, 10)):
        if equipment_query and not _field_matches(str(item.get("equipment_module") or ""), equipment_query):
            continue
        if vendor_query and not _field_matches(str(item.get("vendor") or ""), vendor_query):
            continue
        ranked.append(item)
        if len(ranked) >= top_k:
            break
    return ranked


@traced_tool("filter_part_rows_tool")
def filter_part_rows_tool(
    keyword: str,
    top_k: int = 20,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    """Return rows containing a keyword in searchable part fields."""
    simple_keyword = simplify_text(normalize_query(keyword))
    if not simple_keyword:
        return []

    results: list[dict[str, object]] = []
    for index, row in enumerate(load_part_data()):
        if equipment_query and not _field_matches(row.get("equipment_module", ""), equipment_query):
            continue
        if vendor_query and not _field_matches(row.get("vendor", ""), vendor_query):
            continue
        haystack = " ".join(_candidate_texts(row))
        if simple_keyword not in simplify_text(haystack):
            continue
        results.append(
            {
                **row,
                "score": 100.0,
                "matched_query": keyword,
                "_index": index,
            }
        )

    for item in results:
        item.pop("_index", None)
    return _dedupe_rows(results, top_k)


@traced_tool("aggregate_part_rows_tool")
def aggregate_part_rows_tool(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
    sort_by: str = "part_name_length",
) -> list[dict[str, object]]:
    """Handle simple catalog-wide ranking requests such as longest part name."""
    rows: list[dict[str, object]] = []
    for index, row in enumerate(load_part_data()):
        if equipment_query and not _field_matches(row.get("equipment_module", ""), equipment_query):
            continue
        if vendor_query and not _field_matches(row.get("vendor", ""), vendor_query):
            continue
        rows.append({**row, "score": 100.0, "matched_query": query, "_index": index})

    if sort_by == "description_length":
        rows.sort(key=lambda item: (-len(str(item.get("description", ""))), int(item["_index"])))
    else:
        rows.sort(key=lambda item: (-len(str(item.get("part_name", ""))), int(item["_index"])))

    for item in rows:
        item.pop("_index", None)
    return _dedupe_rows(rows, top_k)


@traced_tool("abbreviation_search_tool")
def abbreviation_search_tool(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)


@traced_tool("english_name_search_tool")
def english_name_search_tool(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)


@traced_tool("korean_name_search_tool")
def korean_name_search_tool(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)


@traced_tool("hybrid_search_tool")
def hybrid_search_tool(
    query: str,
    top_k: int = 3,
    equipment_query: str = "",
    vendor_query: str = "",
) -> list[dict[str, object]]:
    return search_part_numbers(query, top_k=top_k, equipment_query=equipment_query, vendor_query=vendor_query)
