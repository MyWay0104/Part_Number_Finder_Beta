from __future__ import annotations

import re

from part_finder.data_loader import load_aliases


_KOREAN_RE = re.compile(r"[가-힣]")
_ENGLISH_RE = re.compile(r"[A-Za-z]")

BUILTIN_ALIASES = {
    "Vacuum Gauge": ["진공게이지", "베큠게이지", "vacuum gauge", "pressure gauge", "gauge"],
    "Gate Valve": ["게이트밸브", "gate valve"],
    "Slit Valve": ["슬릿밸브", "slit valve"],
    "Throttle Valve": ["쓰로틀밸브", "쓰로틀 밸브", "드로틀밸브", "드로틀 밸브", "throttle valve", "t/v", "tv"],
    "Ceramic Plate": ["세라믹플레이트", "ceramic plate"],
    "Ion Source": ["이온소스", "ion source"],
    "Liner Quartz": ["라이너쿼츠", "쿼츠라이너", "liner quartz", "quartz liner"],
    "Window Quartz": ["윈도우쿼츠", "window quartz", "wq", "w/q"],
    "Clamp Ring": ["클램프", "클램프링", "클램프 링", "clamp", "clamp ring"],
    "Robot Blade": ["블레이드", "로봇블레이드", "로봇 블레이드", "robot blade", "end effector"],
    "ESC Chuck": ["척", "chuck", "esc chuck"],
    "O-ring": ["오링", "오 링", "o-ring", "o ring", "oring", "owe ling"],
    "MFC": ["엠에프씨", "mfc", "mass flow controller"],
    "RF Match": ["알에프", "rf", "rf match", "rfmatch"],
}

PROPER_KOREAN_ALIASES = {
    "Robot Blade": ["로봇블레이드", "로봇 블레이드", "로봇 blade"],
    "Turbo Pump": ["터보펌프", "터보 펌프"],
    "Quartz Tube": ["쿼츠튜브", "쿼츠 튜브", "quartz tube", "quart tube"],
    "Liner Quartz": ["라이너쿼츠", "라이너 쿼츠", "리니어쿼츠", "l/q", "lq"],
    "Window Quartz": ["윈도우쿼츠", "윈도우 쿼츠", "w/q", "wq"],
    "PM Kit": ["피엠킷", "pmkit", "pm kit", "p/m kit"],
    "Throttle Valve": ["스로틀밸브", "스로틀 밸브", "t/v", "tv"],
    "O-ring": ["오링", "오 링", "o ring", "oring", "o-ring", "owe ling"],
    "Vacuum Gauge": ["진공게이지", "진공 게이지", "베큠게이지", "베큠 게이지"],
    "Slit Valve": ["슬릿밸브", "슬릿 밸브"],
    "Gate Valve": ["게이트밸브", "게이트 밸브"],
    "Clamp Ring": ["클램프링", "클램프 링", "클램프"],
    "MFC": ["엠에프씨", "mfc", "mass flow controller"],
}


def _aliases_with_builtins() -> dict[str, list[str]]:
    aliases = load_aliases()
    merged = {target: list(values) for target, values in aliases.items()}
    for target, values in BUILTIN_ALIASES.items():
        merged.setdefault(target, [])
        merged[target].extend(value for value in values if value not in merged[target])
    for target, values in PROPER_KOREAN_ALIASES.items():
        merged.setdefault(target, [])
        merged[target].extend(value for value in values if value not in merged[target])
    return merged


def simplify_text(value: str) -> str:
    """Create a comparison key that ignores case, spacing, and punctuation."""
    value = value.strip().lower().replace("/", "")
    return "".join(char for char in value if char.isalnum())


def _has_korean_like_text(value: str) -> bool:
    return bool(_KOREAN_RE.search(value)) or any(ord(char) > 127 and char.isalpha() for char in value)


def _alias_matches(cleaned_query: str, simplified_query: str, alias: str) -> bool:
    simplified_alias = simplify_text(alias)
    if not simplified_alias:
        return False
    if len(simplified_alias) <= 3:
        token_keys = [simplify_text(token) for token in re.split(r"\s+", cleaned_query)]
        return simplified_alias in token_keys or simplified_query.startswith(simplified_alias)
    return simplified_alias in simplified_query


def normalize_query(query: str) -> str:
    """Normalize user input by applying alias targets before fuzzy search."""
    cleaned = re.sub(r"\s+", " ", query.strip())
    if not cleaned:
        return ""

    simplified_query = simplify_text(cleaned)
    aliases = _aliases_with_builtins()
    for target, alias_values in aliases.items():
        values = [target, *alias_values]
        for alias in values:
            if _alias_matches(cleaned, simplified_query, alias):
                return target

    if _ENGLISH_RE.search(cleaned) and not _has_korean_like_text(cleaned):
        return cleaned.title()
    return cleaned


def detect_query_type(query: str) -> str:
    """Classify the input shape for routing or future LangGraph nodes."""
    stripped = query.strip()
    simplified = simplify_text(stripped)
    aliases = _aliases_with_builtins()
    abbreviation_keys = {"wq", "tv", "pmkit", "mfc", "rfmatch", "escchuck"}
    for target, alias_values in aliases.items():
        for alias in [target, *alias_values]:
            alias_key = simplify_text(alias)
            if alias_key in abbreviation_keys and _alias_matches(stripped, simplified, alias):
                return "abbreviation"

    has_korean = _has_korean_like_text(stripped)
    has_english = bool(_ENGLISH_RE.search(stripped))
    if has_korean and has_english:
        return "mixed"
    if has_korean:
        return "korean"
    if has_english:
        return "english"
    return "mixed"
