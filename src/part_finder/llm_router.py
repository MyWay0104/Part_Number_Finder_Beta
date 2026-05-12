from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from part_finder.config import configure_ollama_env, get_ollama_model, is_llm_enabled
from part_finder.data_loader import load_part_catalog
from part_finder.normalizer import detect_query_type, normalize_query, simplify_text
from part_finder.tracing import traceable_run


@dataclass(frozen=True)
class RouteDecision:
    query_type: str
    normalized_query: str
    intent: str = "lookup_part"
    candidate_queries: tuple[str, ...] = ()
    semantic_candidates: tuple[str, ...] = ()
    requested_fields: tuple[str, ...] = ("part_number",)
    contains_keyword: str = ""
    sort_by: str = ""
    requires_confirmation: bool = False
    confirmation_reason: str = ""
    equipment_query: str = ""
    vendor_query: str = ""
    tool_name: str = "hybrid_search_tool"
    used_llm: bool = False


ALLOWED_INTENTS = {
    "lookup_part",
    "lookup_details",
    "filter_parts",
    "aggregate_parts",
    "semantic_candidate",
    "clarify",
}
ALLOWED_QUERY_TYPES = {"abbreviation", "english", "korean", "mixed"}
ALLOWED_TOOLS = {
    "abbreviation_search_tool",
    "english_name_search_tool",
    "korean_name_search_tool",
    "hybrid_search_tool",
    "semantic_catalog_match_tool",
    "filter_part_rows_tool",
    "aggregate_part_rows_tool",
}

EQUIPMENT_ALIASES = {
    "Endura": ["endura", "엔듀라"],
    "Vantage Radox": ["vantage radox", "vantage", "radox", "벤티지라독스", "벤티지 라독스", "라독스"],
    "Etch Module": ["etch module", "etch", "에치모듈", "에치 모듈"],
    "Implant Module": ["implant module", "implant", "임플란트모듈", "임플란트 모듈"],
    "Helios XP": ["helios xp", "helios", "헬리오스"],
}

VENDOR_ALIASES = {
    "Lam Research": ["lam research", "lam", "램리서치", "램 리서치", "램"],
    "ASML": ["asml", "에이에스엠엘"],
    "AMAT": ["amat", "applied materials", "어플라이드", "어플라이드머티어리얼즈"],
    "TEL": ["tel", "tokyo electron", "도쿄일렉트론"],
}


def _canonical_from_alias(value: str, alias_map: dict[str, list[str]]) -> str:
    simplified = simplify_text(value)
    for target, aliases in alias_map.items():
        for alias in [target, *aliases]:
            if simplified == simplify_text(alias):
                return target
    return ""


def _canonical_part_query(value: str, original_query: str, catalog: list[str]) -> str:
    deterministic = normalize_query(original_query)
    if deterministic and simplify_text(deterministic) != simplify_text(original_query):
        return deterministic

    simplified = simplify_text(value)
    for catalog_name in catalog:
        if simplified == simplify_text(catalog_name):
            return catalog_name

    normalized = normalize_query(value)
    if normalized and simplify_text(normalized) != simplified:
        return normalized
    return value


def _detect_named_value(query: str, alias_map: dict[str, list[str]]) -> str:
    simplified = simplify_text(query)
    for target, aliases in alias_map.items():
        for alias in [target, *aliases]:
            if simplify_text(alias) in simplified:
                return target
    return ""


def _requested_fields(query: str) -> tuple[str, ...]:
    simplified = simplify_text(query)
    fields = ["part_number"]
    if any(token in simplified for token in ["name", "partname", "english", "eng", "영문", "이름"]):
        fields.extend(["part_name", "description"])
    if any(token in simplified for token in ["vendor", "maker", "manufacturer", "벤더", "제조사"]):
        fields.append("vendor")
    if any(token in simplified for token in ["equipment", "module", "model", "장비", "모델"]):
        fields.append("equipment_module")
    deduped: list[str] = []
    for field in fields:
        if field not in deduped:
            deduped.append(field)
    return tuple(deduped)


def _rule_based_intent(query: str) -> str:
    simplified = simplify_text(query)
    if any(token in simplified for token in ["전부", "전체", "모두", "포함", "contains", "include", "all"]):
        return "filter_parts"
    if any(token in simplified for token in ["가장긴", "최장", "longest", "max"]):
        return "aggregate_parts"
    if any(token in simplified for token in ["arm", "pick", "robot", "handler", "transfer"]):
        return "semantic_candidate"
    if len(_requested_fields(query)) > 1:
        return "lookup_details"
    return "lookup_part"


def _semantic_candidates(values: object, normalized_query: str) -> tuple[str, ...]:
    candidates: list[str] = []
    if isinstance(values, list):
        candidates.extend(str(value).strip() for value in values if str(value).strip())
    if normalized_query:
        candidates.append(normalized_query)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return tuple(deduped)


def rule_based_route(query: str) -> RouteDecision:
    """Deterministic fallback route used when LLM routing is disabled or fails."""
    query_type = detect_query_type(query)
    tool_name = {
        "abbreviation": "abbreviation_search_tool",
        "english": "english_name_search_tool",
        "korean": "korean_name_search_tool",
        "mixed": "hybrid_search_tool",
    }.get(query_type, "hybrid_search_tool")
    return RouteDecision(
        intent=_rule_based_intent(query),
        query_type=query_type,
        normalized_query=normalize_query(query),
        candidate_queries=(normalize_query(query), query),
        semantic_candidates=(),
        requested_fields=_requested_fields(query),
        contains_keyword=normalize_query(query) if _rule_based_intent(query) == "filter_parts" else "",
        sort_by="part_name_length" if _rule_based_intent(query) == "aggregate_parts" else "",
        requires_confirmation=_rule_based_intent(query) == "semantic_candidate",
        equipment_query=_detect_named_value(query, EQUIPMENT_ALIASES),
        vendor_query=_detect_named_value(query, VENDOR_ALIASES),
        tool_name=tool_name,
        used_llm=False,
    )


def _parse_json_response(content: str) -> dict[str, Any] | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _candidate_tuple(values: object, normalized_query: str, original_query: str) -> tuple[str, ...]:
    candidates: list[str] = []
    deterministic_query = normalize_query(original_query)
    if deterministic_query and simplify_text(deterministic_query) != simplify_text(original_query):
        candidates.append(deterministic_query)
    if isinstance(values, list):
        candidates.extend(str(value).strip() for value in values if str(value).strip())
    candidates.extend([normalized_query, original_query])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return tuple(deduped)


@traceable_run(name="llm_route", run_type="chain")
def llm_route(query: str) -> RouteDecision:
    """Use local Ollama only for routing and normalization, never for final PN choice.

    Enabled by default for the agentic PoC. Set PART_FINDER_USE_LLM=0 to force
    deterministic rule-only routing. The default model is gemma3:4b.
    """
    if not is_llm_enabled():
        return rule_based_route(query)

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return rule_based_route(query)

    configure_ollama_env()
    model = get_ollama_model()
    catalog = load_part_catalog()
    catalog_text = "\n".join(f"- {name}" for name in catalog[:80])
    prompt = f"""
You are the query-understanding router for a semiconductor part-number finder.
Return JSON only. Do not invent part numbers.
The final user answer is generated by deterministic Korean templates, not by you.
Your JSON field values must use the allowed English catalog/vendor/module names.

Allowed query_type values: abbreviation, english, korean, mixed.
Allowed intent values: lookup_part, lookup_details, filter_parts, aggregate_parts,
semantic_candidate, clarify.
Allowed tool_name values:
- abbreviation_search_tool
- english_name_search_tool
- korean_name_search_tool
- hybrid_search_tool
- semantic_catalog_match_tool
- filter_part_rows_tool
- aggregate_part_rows_tool

Task:
1. Find the user's intent and requested output fields.
2. Find the part-name candidate in the user query.
3. If the user wrote Korean pronunciation, abbreviation, or typo, infer the closest English catalog term.
4. If the user describes a function/context not present as an exact part name, infer semantic_candidates from the local catalog only.
5. Select one tool_name.
6. Return multiple candidate_queries ordered from most likely to least likely.
7. If semantic inference is needed, set requires_confirmation true and do not present it as certain.
8. If uncertain, keep a cleaned user phrase and choose hybrid_search_tool.

Korean pronunciation examples:
- 오링, 오 링, 오-링, owe ling -> O-ring
- 샤워헤드, 샤워 해드 -> Shower Head
- 쿼츠튜브 -> Quartz Tube
- 터보펌프 -> Turbo Pump
- 드라이펌프 -> Dry Pump
- 포커스링 -> Focus Ring
- 게이트밸브 -> Gate Valve
- 엠에프씨 -> MFC

Important:
- normalized_query must be the closest term from the local catalog when possible.
- candidate_queries should include possible English names, abbreviations, and useful original variants.
- requested_fields can include part_number, part_name, description, vendor, equipment_module, vendor_part_number.
- Use semantic_candidate when the user says contextual words like robot arm, pick, handler, transfer, pressure sensor, gas flow controller, or isolation door and the exact part name is not in the query.
- Do not output part numbers.

Known catalog examples from the local data:
{catalog_text}

User query: {query}

Return equipment_query only when the user names a model/module such as Endura,
Vantage Radox, Etch Module, Implant Module, or Helios XP.
Return vendor_query only when the user names a vendor such as Lam Research,
ASML, AMAT, or TEL.
If the user says 베큠게이지, 진공게이지, or vacuum gauge, prefer Vacuum Gauge
and do not confuse it with Gate Valve.
If you are unsure, return an empty equipment_query/vendor_query rather than
inventing Korean, Chinese, or mixed-language labels.

JSON schema:
{{"intent":"...", "query_type":"...", "normalized_query":"...", "candidate_queries":["..."], "semantic_candidates":["..."], "requested_fields":["part_number"], "contains_keyword":"...", "sort_by":"...", "requires_confirmation":false, "confirmation_reason":"...", "equipment_query":"...", "vendor_query":"...", "tool_name":"..."}}
""".strip()

    try:
        llm = ChatOllama(model=model, temperature=0)
        response = llm.invoke(
            prompt,
            config={
                "run_name": "ollama_gemma_route",
                "tags": ["part-finder", "ollama", model],
            },
        )
        parsed = _parse_json_response(str(getattr(response, "content", response)))
    except Exception:
        return rule_based_route(query)

    if not parsed:
        return rule_based_route(query)

    intent = str(parsed.get("intent", "")) or _rule_based_intent(query)
    query_type = str(parsed.get("query_type", ""))
    normalized_query = _canonical_part_query(str(parsed.get("normalized_query", "")), query, catalog)
    tool_name = str(parsed.get("tool_name", "hybrid_search_tool"))
    if intent not in ALLOWED_INTENTS:
        intent = _rule_based_intent(query)
    if query_type not in ALLOWED_QUERY_TYPES or not normalized_query or tool_name not in ALLOWED_TOOLS:
        return rule_based_route(query)

    requested = parsed.get("requested_fields")
    requested_values = requested if isinstance(requested, list) else []
    requested_fields = tuple(
        field
        for field in (str(value) for value in requested_values)
        if field
        in {
            "part_number",
            "part_name",
            "description",
            "vendor",
            "equipment_module",
            "vendor_part_number",
        }
    )
    if not requested_fields:
        requested_fields = _requested_fields(query)
    elif "part_number" not in requested_fields:
        requested_fields = ("part_number", *requested_fields)

    equipment_query = _canonical_from_alias(str(parsed.get("equipment_query") or ""), EQUIPMENT_ALIASES)
    vendor_query = _canonical_from_alias(str(parsed.get("vendor_query") or ""), VENDOR_ALIASES)

    return RouteDecision(
        intent=intent,
        query_type=query_type,
        normalized_query=normalized_query,
        candidate_queries=_candidate_tuple(parsed.get("candidate_queries"), normalized_query, query),
        semantic_candidates=_semantic_candidates(parsed.get("semantic_candidates"), normalized_query),
        requested_fields=requested_fields,
        contains_keyword=str(parsed.get("contains_keyword") or ""),
        sort_by=str(parsed.get("sort_by") or ""),
        requires_confirmation=bool(parsed.get("requires_confirmation") or intent == "semantic_candidate"),
        confirmation_reason=str(parsed.get("confirmation_reason") or ""),
        equipment_query=equipment_query or _detect_named_value(query, EQUIPMENT_ALIASES),
        vendor_query=vendor_query or _detect_named_value(query, VENDOR_ALIASES),
        tool_name=tool_name,
        used_llm=True,
    )
