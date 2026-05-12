from __future__ import annotations

import re

from part_finder.data_loader import load_aliases


_KOREAN_RE = re.compile(r"[가-힣]")
_ENGLISH_RE = re.compile(r"[A-Za-z]")


def simplify_text(value: str) -> str:
    """Create a comparison key that ignores case, spacing, and most punctuation."""
    value = value.strip().lower()
    value = value.replace("/", "")
    return re.sub(r"[^0-9a-z가-힣]+", "", value)


def normalize_query(query: str) -> str:
    """Normalize user input by applying alias targets before fuzzy search."""
    cleaned = re.sub(r"\s+", " ", query.strip())
    if not cleaned:
        return ""

    simplified_query = simplify_text(cleaned)
    aliases = load_aliases()
    for target, alias_values in aliases.items():
        values = [target, *alias_values]
        for alias in values:
            simplified_alias = simplify_text(alias)
            if simplified_alias and simplified_alias in simplified_query:
                return target

    # Keep English terms readable and title-cased; Korean queries are left as-is.
    if _ENGLISH_RE.search(cleaned) and not _KOREAN_RE.search(cleaned):
        return cleaned.title()
    return cleaned


def detect_query_type(query: str) -> str:
    """Classify the input shape for routing or future LangGraph nodes."""
    stripped = query.strip()
    simplified = simplify_text(stripped)
    aliases = load_aliases()
    abbreviation_keys = {"wq", "tv", "pmkit", "mfc", "rfmatch", "escchuck"}
    for target, alias_values in aliases.items():
        for alias in [target, *alias_values]:
            if simplify_text(alias) in abbreviation_keys and simplify_text(alias) in simplified:
                return "abbreviation"

    has_korean = bool(_KOREAN_RE.search(stripped))
    has_english = bool(_ENGLISH_RE.search(stripped))
    if has_korean and has_english:
        return "mixed"
    if has_korean:
        return "korean"
    if has_english:
        return "english"
    return "mixed"
