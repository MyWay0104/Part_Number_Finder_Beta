from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from part_finder.config import configure_ollama_env, get_ollama_model, is_llm_enabled
from part_finder.data_loader import load_part_catalog
from part_finder.normalizer import detect_query_type, normalize_query, simplify_text
from part_finder.tracing import end_trace, flush_traces, get_langfuse_callback_handler, start_trace, trace_span, traceable_run


@dataclass(frozen=True)
class RouteDecision:
    query_type: str
    normalized_query: str
    intent: str = "lookup_part"
    candidate_queries: tuple[str, ...] = ()
    semantic_candidates: tuple[str, ...] = ()
    phonetic_english_candidates: tuple[str, ...] = ()
    semantic_queries: tuple[str, ...] = ()
    conceptual_description: str = ""
    requested_fields: tuple[str, ...] = ("part_number",)
    contains_keyword: str = ""
    sort_by: str = ""
    requires_confirmation: bool = False
    confirmation_reason: str = ""
    equipment_query: str = ""
    vendor_query: str = ""
    vendor: str = ""
    equipment: str = ""
    tool_name: str = "hybrid_search_tool"
    preferred_tools: tuple[str, ...] = ("hybrid_search_tool",)
    needs_semantic_search: bool = False
    needs_confirmation: bool = False
    used_llm: bool = False

    @property
    def equipment_query_value(self) -> str:
        return self.equipment or self.equipment_query

    @property
    def vendor_query_value(self) -> str:
        return self.vendor or self.vendor_query


@dataclass(frozen=True)
class PartLookupItem:
    raw_text: str
    normalized_query: str
    candidate_queries: tuple[str, ...] = ()
    semantic_queries: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class QueryPlan:
    items: tuple[PartLookupItem, ...]
    requested_fields: tuple[str, ...] = ("part_number",)
    vendor_query: str = ""
    equipment_query: str = ""
    intent: str = "lookup_part"
    used_llm: bool = False


ALLOWED_INTENTS = {
    "lookup_part",
    "lookup_details",
    "filter_parts",
    "aggregate_parts",
    "semantic_candidate",
    "clarify",
}
ALLOWED_QUERY_TYPES = {"abbreviation", "english", "korean", "mixed", "conceptual"}
ALLOWED_TOOLS = {
    "abbreviation_search_tool",
    "english_name_search_tool",
    "korean_name_search_tool",
    "hybrid_search_tool",
    "semantic_catalog_match_tool",
    "vector_semantic_search_tool",
    "filter_part_rows_tool",
    "aggregate_part_rows_tool",
}

PHONETIC_MAP = {
    "클램프": ("clamp",),
    "쓰로틀밸브": ("throttle valve",),
    "쓰로틀 밸브": ("throttle valve",),
    "드로틀밸브": ("throttle valve",),
    "드로틀 밸브": ("throttle valve",),
    "라이너쿼츠": ("liner quartz", "quartz liner"),
    "쿼츠라이너": ("liner quartz", "quartz liner"),
    "오링": ("o-ring",),
    "오 링": ("o-ring",),
    "엠에프씨": ("mfc", "mass flow controller"),
    "알에프": ("rf",),
    "블레이드": ("blade", "robot blade", "end effector"),
    "척": ("chuck",),
    "게이지": ("gauge",),
    "밸브": ("valve",),
    "센서": ("sensor",),
}

SEMANTIC_EXPANSIONS = {
    "clamp": ("clamp", "wafer clamp", "holding clamp", "clamping part", "wafer holder", "chuck"),
    "throttle valve": ("throttle valve", "pressure control valve", "flow control valve", "gas flow valve"),
    "liner quartz": ("liner quartz", "quartz liner", "chamber liner", "process chamber protection"),
    "o-ring": ("o-ring", "seal", "sealing", "leak prevention"),
    "mfc": ("mfc", "mass flow controller", "gas flow", "flow control"),
    "blade": ("robot blade", "end effector", "wafer transfer", "robot arm pick", "blade"),
    "chuck": ("chuck", "wafer chuck", "wafer holder", "wafer holding", "esc chuck"),
    "gauge": ("vacuum gauge", "pressure gauge", "measurement", "sensor"),
    "valve": ("valve", "gate valve", "slit valve", "throttle valve", "pressure control"),
    "sensor": ("sensor", "measurement", "gauge"),
}

CONCEPT_RULES = (
    (("웨이퍼", "잡"), "part used to hold or clamp wafer", ("wafer clamp", "wafer holder", "clamp", "chuck", "blade")),
    (("wafer", "hold"), "part used to hold or clamp wafer", ("wafer clamp", "wafer holder", "clamp", "chuck")),
    (("압력", "조절", "밸브"), "valve used to control pressure or gas flow", ("throttle valve", "pressure control valve", "flow control valve")),
    (("pressure", "control"), "part used to control pressure or gas flow", ("throttle valve", "pressure control valve", "mfc")),
    (("로봇", "웨이퍼", "집"), "robot end effector used to pick and transfer wafer", ("robot blade", "end effector", "wafer transfer", "robot arm pick", "blade")),
    (("robot", "wafer", "pick"), "robot end effector used to pick and transfer wafer", ("robot blade", "end effector", "wafer transfer", "robot arm pick", "blade")),
)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            deduped.append(item)
    return tuple(deduped)


def _list_field(parsed: dict[str, Any], key: str) -> tuple[str, ...]:
    values = parsed.get(key)
    if not isinstance(values, list):
        return ()
    return _dedupe(str(value).strip() for value in values if str(value).strip())


def _phonetic_candidates(query: str) -> tuple[str, ...]:
    simple = simplify_text(query)
    values: list[str] = []
    for korean, candidates in PHONETIC_MAP.items():
        if simplify_text(korean) in simple:
            values.extend(candidates)
    return _dedupe(values)


def _conceptual_match(query: str) -> tuple[str, tuple[str, ...]]:
    simple = query.lower()
    compact = simplify_text(query)
    for tokens, description, semantic_queries in CONCEPT_RULES:
        if all(token.lower() in simple or simplify_text(token) in compact for token in tokens):
            return description, semantic_queries
    return "", ()


def _semantic_queries(normalized_query: str, phonetic: tuple[str, ...], conceptual: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = [*conceptual]
    for candidate in [normalized_query, *phonetic]:
        key = simplify_text(candidate)
        values.extend(SEMANTIC_EXPANSIONS.get(key, ()))
        values.extend(SEMANTIC_EXPANSIONS.get(candidate.lower(), ()))
    return _dedupe(values)


def _trace_route(user_query: str, route: RouteDecision, fallback_used: bool) -> None:
    trace = start_trace("llm_router", user_query, {"fallback_used": fallback_used})
    trace_span(
        trace,
        "route_query",
        input_data={"user_query": user_query},
        output_data={
            "query_type": route.query_type,
            "normalized_query": route.normalized_query,
            "candidate_queries": list(route.candidate_queries),
            "phonetic_english_candidates": list(route.phonetic_english_candidates),
            "semantic_queries": list(route.semantic_queries),
            "preferred_tools": list(route.preferred_tools),
            "fallback_used": fallback_used,
        },
    )
    end_trace(trace)
    flush_traces()

EQUIPMENT_ALIASES = {
    "Endura": ["endura", "엔듀라"],
    "Vantage Radox": [
        "vantage radox",
        "vantage",
        "radox",
        "벤티지라독스",
        "벤티지 라독스",
        "벤티지",
        "밴티지",
        "라독스",
    ],
    "Etch Module": ["etch module", "etch", "에치모듈", "에치 모듈"],
    "Implant Module": ["implant module", "implant", "임플란트모듈", "임플란트 모듈"],
    "Helios XP": ["helios xp", "helios", "헬리오스"],
}

VENDOR_ALIASES = {
    "Lam Research": ["lam research", "lam", "램리서치", "램 리서치", "램"],
    "ASML": ["asml", "에이에스엠엘"],
    "AMAT": [
        "amat",
        "applied materials",
        "에이멧",
        "에이맷",
        "어플라이드",
        "어플라이드머티어리얼즈",
        "어플라이드머터리얼즈",
    ],
    "TEL": ["tel", "tokyo electron", "도쿄일렉트론"],
}

PROPER_EQUIPMENT_ALIASES = {
    "Endura": ["엔듀라", "앤듀라"],
    "Vantage Radox": ["밴티지", "벤티지", "라도스", "밴티지 라독스", "벤티지 라독스"],
    "Etch Module": ["에치", "에칭", "에치 모듈"],
    "Implant Module": ["임플란트", "임플란트 모듈"],
    "Helios XP": ["헬리오스", "헬리오스 xp"],
}

PROPER_VENDOR_ALIASES = {
    "Lam Research": ["램리서치", "램 리서치", "lam"],
    "ASML": ["에이에스엠엘", "asml"],
    "AMAT": ["어플라이드", "어플라이드 머티어리얼즈", "에이멧", "amat", "applied materials"],
    "TEL": ["티이엘", "도쿄일렉트론", "tel", "tokyo electron"],
}

PART_NAME_ALIASES = {
    "Robot Blade": ["로봇블레이드", "로봇 블레이드", "robot blade", "end effector", "robot arm blade"],
    "Turbo Pump": ["터보펌프", "터보 펌프", "turbo pump"],
    "Quartz Tube": ["쿼츠튜브", "쿼츠 튜브", "quartz tube", "quart tube"],
    "Liner Quartz": ["라이너쿼츠", "라이너 쿼츠", "리니어쿼츠", "liner quartz", "quartz liner", "l/q", "lq"],
    "Window Quartz": ["윈도우쿼츠", "윈도우 쿼츠", "window quartz", "w/q", "wq"],
    "PM Kit": ["피엠킷", "pmkit", "pm kit", "p/m kit"],
    "Throttle Valve": ["스로틀밸브", "스로틀 밸브", "throttle valve", "t/v", "tv"],
    "O-ring": ["오링", "오 링", "o-ring", "o ring", "oring", "owe ling"],
    "Vacuum Gauge": ["진공게이지", "진공 게이지", "베큠게이지", "베큠 게이지", "vacuum gauge"],
    "Slit Valve": ["슬릿밸브", "슬릿 밸브", "slit valve"],
    "Gate Valve": ["게이트밸브", "게이트 밸브", "gate valve"],
    "Clamp Ring": ["클램프링", "클램프 링", "클램프", "clamp ring"],
    "MFC": ["엠에프씨", "mfc", "mass flow controller"],
}


def _canonical_from_alias(value: str, alias_map: dict[str, list[str]]) -> str:
    simplified = simplify_text(value)
    for target, aliases in alias_map.items():
        for alias in [target, *aliases]:
            if simplified == simplify_text(alias):
                return target
    return ""


def _strip_context_only_candidates(
    candidates: tuple[str, ...],
    equipment_query: str,
    vendor_query: str,
) -> tuple[str, ...]:
    """Remove equipment/vendor-only phrases from part-name search candidates."""
    context_values = _dedupe(
        [
            equipment_query,
            vendor_query,
            *EQUIPMENT_ALIASES.keys(),
            *VENDOR_ALIASES.keys(),
            *(alias for values in EQUIPMENT_ALIASES.values() for alias in values),
            *(alias for values in VENDOR_ALIASES.values() for alias in values),
        ]
    )
    context_keys = {simplify_text(value) for value in context_values if value}
    return _dedupe(candidate for candidate in candidates if simplify_text(candidate) not in context_keys)


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


def _merged_aliases(primary: dict[str, list[str]], extra: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {key: list(values) for key, values in primary.items()}
    for key, values in extra.items():
        merged.setdefault(key, [])
        merged[key].extend(value for value in values if value not in merged[key])
    return merged


def _resolve_catalog_part(value: str, catalog: list[str] | None = None) -> str:
    normalized = normalize_query(value)
    catalog_values = catalog or load_part_catalog()
    normalized_key = simplify_text(normalized)
    for catalog_name in catalog_values:
        if normalized_key == simplify_text(catalog_name):
            return catalog_name
    for target, aliases in PART_NAME_ALIASES.items():
        for alias in [target, *aliases]:
            alias_key = simplify_text(alias)
            if alias_key and alias_key in simplify_text(value):
                return target
    if normalized and simplify_text(normalized) != simplify_text(value):
        return normalized
    return value.strip()


def _rule_based_part_items(query: str, catalog: list[str] | None = None) -> tuple[PartLookupItem, ...]:
    catalog_values = catalog or load_part_catalog()
    found: list[PartLookupItem] = []
    seen: set[str] = set()
    query_key = simplify_text(query)

    alias_sources: dict[str, list[str]] = {}
    for target, aliases in PART_NAME_ALIASES.items():
        alias_sources.setdefault(target, [])
        alias_sources[target].extend([target, *aliases])
    for target in catalog_values:
        alias_sources.setdefault(target, [])
        alias_sources[target].append(target)

    for target, aliases in alias_sources.items():
        for alias in aliases:
            alias_key = simplify_text(alias)
            if alias_key and alias_key in query_key and simplify_text(target) not in seen:
                resolved = _resolve_catalog_part(target, catalog_values)
                seen.add(simplify_text(resolved))
                semantic = _semantic_queries(resolved, (), ())
                found.append(
                    PartLookupItem(
                        raw_text=alias,
                        normalized_query=resolved,
                        candidate_queries=_dedupe([resolved, alias]),
                        semantic_queries=semantic,
                        confidence=0.95,
                    )
                )
                break

    if found:
        return tuple(found)

    route = rule_based_route(query)
    return (
        PartLookupItem(
            raw_text=query,
            normalized_query=route.normalized_query,
            candidate_queries=route.candidate_queries,
            semantic_queries=route.semantic_queries,
            confidence=0.6,
        ),
    )


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
    conceptual_description, conceptual_queries = _conceptual_match(query)
    query_type = "conceptual" if conceptual_description else detect_query_type(query)
    tool_name = {
        "abbreviation": "abbreviation_search_tool",
        "english": "english_name_search_tool",
        "korean": "korean_name_search_tool",
        "mixed": "hybrid_search_tool",
        "conceptual": "hybrid_search_tool",
    }.get(query_type, "hybrid_search_tool")
    normalized = normalize_query(query)
    phonetic = _phonetic_candidates(query)
    if simplify_text(normalized) != simplify_text(query):
        generic_terms = {"valve", "sensor", "gauge"}
        phonetic = tuple(value for value in phonetic if value.lower() not in generic_terms)
    semantic_queries = _semantic_queries(normalized, phonetic, conceptual_queries)
    needs_semantic = bool(conceptual_description or phonetic or semantic_queries)
    preferred_tools = (
        "hybrid_search_tool",
        "vector_semantic_search_tool",
        "semantic_catalog_match_tool",
    ) if needs_semantic else (tool_name,)
    equipment_query = _detect_named_value(query, _merged_aliases(EQUIPMENT_ALIASES, PROPER_EQUIPMENT_ALIASES))
    vendor_query = _detect_named_value(query, _merged_aliases(VENDOR_ALIASES, PROPER_VENDOR_ALIASES))
    candidate_queries = _strip_context_only_candidates(
        _dedupe([normalized, *phonetic, query]),
        equipment_query,
        vendor_query,
    )
    return RouteDecision(
        intent=_rule_based_intent(query),
        query_type=query_type,
        normalized_query=normalized,
        candidate_queries=candidate_queries,
        semantic_candidates=semantic_queries,
        phonetic_english_candidates=phonetic,
        semantic_queries=semantic_queries,
        conceptual_description=conceptual_description,
        requested_fields=_requested_fields(query),
        contains_keyword=normalized if _rule_based_intent(query) == "filter_parts" else "",
        sort_by="part_name_length" if _rule_based_intent(query) == "aggregate_parts" else "",
        requires_confirmation=_rule_based_intent(query) == "semantic_candidate" or query_type == "conceptual",
        equipment_query=equipment_query,
        vendor_query=vendor_query,
        tool_name=tool_name,
        preferred_tools=preferred_tools,
        needs_semantic_search=needs_semantic,
        needs_confirmation=query_type == "conceptual",
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
        route = rule_based_route(query)
        _trace_route(query, route, fallback_used=True)
        return route

    trace = start_trace("llm_router", query, {"fallback_used": False})

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        route = rule_based_route(query)
        trace_span(trace, "route_query", input_data={"user_query": query}, output_data={**route.__dict__, "fallback_used": True})
        end_trace(trace)
        flush_traces()
        return route

    configure_ollama_env()
    model = get_ollama_model()
    catalog = load_part_catalog()
    catalog_text = "\n".join(f"- {name}" for name in catalog[:80])
    prompt = f"""
You are the query-understanding router for a semiconductor part-number finder.
Return JSON only. Do not invent part numbers.
The final user answer is generated by deterministic Korean templates, not by you.
Your JSON field values must use the allowed English catalog/vendor/module names.

Allowed query_type values: abbreviation, english, korean, mixed, conceptual.
Allowed intent values: lookup_part, lookup_details, filter_parts, aggregate_parts,
semantic_candidate, clarify.
Allowed tool_name values:
- abbreviation_search_tool
- english_name_search_tool
- korean_name_search_tool
- hybrid_search_tool
- semantic_catalog_match_tool
- vector_semantic_search_tool
- filter_part_rows_tool
- aggregate_part_rows_tool

Hard constraints:
- Do not invent part numbers.
- Return search queries only, never final answers.
- If Korean pronunciation likely maps to English technical term, include English candidates.
- If conceptual question, generate multiple semantic search queries.
- Return strict JSON only.

Task:
1. Find the user's intent and requested output fields.
2. Find the part-name candidate in the user query.
3. If the user wrote Korean pronunciation, abbreviation, or typo, infer the closest English catalog term.
4. If the user describes a function/context not present as an exact part name, infer semantic_candidates from the local catalog only.
5. Select one tool_name.
6. Return multiple candidate_queries ordered from most likely to least likely.
7. If semantic inference is needed, prefer vector_semantic_search_tool or semantic_catalog_match_tool,
   set requires_confirmation true, and do not present it as certain.
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
- Equipment/module names and vendor names are context filters, not part-name candidates.
- Never put an equipment name such as Endura or Vantage Radox in candidate_queries
  unless the phrase also contains a part name.
- The fields "equipment" and "vendor" must use only allowed equipment/vendor names.
  If the value is not a known equipment/vendor, return null or an empty string.
- candidate_queries should include possible English names, abbreviations, and useful original variants.
- requested_fields can include part_number, part_name, description, vendor, equipment_module, vendor_part_number.
- Use semantic_candidate when the user says contextual words like robot arm, pick, handler, transfer, pressure sensor, gas flow controller, or isolation door and the exact part name is not in the query.
- Do not output part numbers.
- The final answer will be written by another LLM pass using only tool-returned rows.

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

Few-shot structure examples:
- 엔듀라 로봇 블레이드 파트넘버 -> normalized_query Robot Blade,
  equipment_query Endura, vendor_query empty, candidate_queries ["Robot Blade"]
- 벤티지 엣지링 파트넘버 -> normalized_query Edge Ring,
  equipment_query Vantage Radox, vendor_query empty, candidate_queries ["Edge Ring"]
- 밴티지 로봇 블레이드 파트넘버 -> normalized_query Robot Blade,
  equipment_query Vantage Radox, vendor_query empty, candidate_queries ["Robot Blade"]
- 어플라이드 포커스링 파트넘버 -> normalized_query Focus Ring,
  equipment_query empty, vendor_query AMAT, candidate_queries ["Focus Ring"]
- 에이멧 세라믹링 파트넘버 -> normalized_query Ceramic Ring,
  equipment_query empty, vendor_query AMAT, candidate_queries ["Ceramic Ring"]

JSON schema:
{{"intent":"...", "query_type":"abbreviation|english|korean|mixed|conceptual", "normalized_query":"...", "candidate_queries":["..."], "phonetic_english_candidates":["..."], "semantic_queries":["..."], "conceptual_description":"...", "requested_fields":["part_number","part_name","vendor","equipment_module"], "vendor":null, "equipment":null, "preferred_tools":["hybrid_search_tool","vector_semantic_search_tool","semantic_catalog_match_tool"], "needs_semantic_search":true, "needs_confirmation":true, "contains_keyword":"...", "sort_by":"...", "requires_confirmation":false, "confirmation_reason":"...", "equipment_query":"...", "vendor_query":"...", "tool_name":"..."}}
""".strip()

    try:
        llm = ChatOllama(model=model, temperature=0)
        langfuse_handler = get_langfuse_callback_handler(trace)
        invoke_config: dict[str, Any] = {
            "run_name": "ollama_gemma_route",
            "tags": ["part-finder", "ollama", model],
        }
        if langfuse_handler is not None:
            invoke_config["callbacks"] = [langfuse_handler]
        response = llm.invoke(
            prompt,
            config=invoke_config,
        )
        parsed = _parse_json_response(str(getattr(response, "content", response)))
    except Exception:
        route = rule_based_route(query)
        trace_span(trace, "route_query", input_data={"user_query": query}, output_data={**route.__dict__, "fallback_used": True})
        end_trace(trace)
        flush_traces()
        return route

    if not parsed:
        route = rule_based_route(query)
        trace_span(trace, "route_query", input_data={"user_query": query}, output_data={**route.__dict__, "fallback_used": True})
        end_trace(trace)
        flush_traces()
        return route

    intent = str(parsed.get("intent", "")) or _rule_based_intent(query)
    query_type = str(parsed.get("query_type", ""))
    normalized_query = _canonical_part_query(str(parsed.get("normalized_query", "")), query, catalog)
    tool_name = str(parsed.get("tool_name", "hybrid_search_tool"))
    if intent not in ALLOWED_INTENTS:
        intent = _rule_based_intent(query)
    if query_type not in ALLOWED_QUERY_TYPES or not normalized_query or tool_name not in ALLOWED_TOOLS:
        route = rule_based_route(query)
        trace_span(trace, "route_query", input_data={"user_query": query}, output_data={**route.__dict__, "fallback_used": True})
        end_trace(trace)
        flush_traces()
        return route

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
    parsed_equipment = _canonical_from_alias(str(parsed.get("equipment") or ""), EQUIPMENT_ALIASES)
    parsed_vendor = _canonical_from_alias(str(parsed.get("vendor") or ""), VENDOR_ALIASES)
    detected_equipment = _detect_named_value(query, _merged_aliases(EQUIPMENT_ALIASES, PROPER_EQUIPMENT_ALIASES))
    detected_vendor = _detect_named_value(query, _merged_aliases(VENDOR_ALIASES, PROPER_VENDOR_ALIASES))
    final_equipment = detected_equipment or parsed_equipment or equipment_query
    final_vendor = detected_vendor or parsed_vendor or vendor_query

    phonetic = _dedupe([*_phonetic_candidates(query), *_list_field(parsed, "phonetic_english_candidates")])
    conceptual_description, conceptual_queries = _conceptual_match(query)
    if parsed.get("conceptual_description"):
        conceptual_description = str(parsed.get("conceptual_description") or conceptual_description)
    semantic_queries = _dedupe([
        *_semantic_queries(normalized_query, phonetic, conceptual_queries),
        *_list_field(parsed, "semantic_queries"),
        *_semantic_candidates(parsed.get("semantic_candidates"), normalized_query),
    ])
    preferred_tools = tuple(
        tool
        for tool in _list_field(parsed, "preferred_tools")
        if tool in ALLOWED_TOOLS
    ) or (tool_name,)
    needs_semantic = bool(parsed.get("needs_semantic_search") or semantic_queries or query_type == "conceptual")
    needs_confirmation = bool(parsed.get("needs_confirmation") or parsed.get("requires_confirmation") or query_type == "conceptual")
    route = RouteDecision(
        intent=intent,
        query_type=query_type,
        normalized_query=normalized_query,
        candidate_queries=_strip_context_only_candidates(
            _candidate_tuple(parsed.get("candidate_queries"), normalized_query, query),
            final_equipment,
            final_vendor,
        ),
        semantic_candidates=semantic_queries,
        phonetic_english_candidates=phonetic,
        semantic_queries=semantic_queries,
        conceptual_description=conceptual_description,
        requested_fields=requested_fields,
        contains_keyword=str(parsed.get("contains_keyword") or ""),
        sort_by=str(parsed.get("sort_by") or ""),
        requires_confirmation=bool(parsed.get("requires_confirmation") or intent == "semantic_candidate"),
        confirmation_reason=str(parsed.get("confirmation_reason") or ""),
        equipment_query=final_equipment,
        vendor_query=final_vendor,
        vendor=final_vendor,
        equipment=final_equipment,
        tool_name=tool_name,
        preferred_tools=preferred_tools,
        needs_semantic_search=needs_semantic,
        needs_confirmation=needs_confirmation,
        used_llm=True,
    )
    trace_span(
        trace,
        "route_query",
        input_data={"user_query": query},
        output_data={
            "query_type": route.query_type,
            "normalized_query": route.normalized_query,
            "candidate_queries": list(route.candidate_queries),
            "phonetic_english_candidates": list(route.phonetic_english_candidates),
            "semantic_queries": list(route.semantic_queries),
            "preferred_tools": list(route.preferred_tools),
            "fallback_used": False,
        },
    )
    end_trace(trace)
    flush_traces()
    return route


def _query_plan_from_parsed(query: str, parsed: dict[str, Any], catalog: list[str]) -> QueryPlan:
    fallback = rule_based_route(query)
    rule_items = list(_rule_based_part_items(query, catalog))
    items_value = parsed.get("items")
    items: list[PartLookupItem] = []
    if isinstance(items_value, list):
        for value in items_value:
            if not isinstance(value, dict):
                continue
            raw = str(value.get("raw_text") or value.get("part_text") or value.get("query") or "").strip()
            normalized = _resolve_catalog_part(str(value.get("normalized_query") or raw), catalog)
            if not normalized:
                continue
            candidates = _dedupe([
                normalized,
                raw,
                *_list_field(value, "candidate_queries"),
            ])
            semantic = _dedupe([
                *_semantic_queries(normalized, (), ()),
                *_list_field(value, "semantic_queries"),
            ])
            items.append(
                PartLookupItem(
                    raw_text=raw or normalized,
                    normalized_query=normalized,
                    candidate_queries=candidates,
                    semantic_queries=semantic,
                    confidence=float(value.get("confidence") or 0.8),
                )
            )

    if not items:
        items = rule_items
    else:
        seen_items = {simplify_text(item.normalized_query) for item in items}
        for item in rule_items:
            if simplify_text(item.normalized_query) not in seen_items:
                items.append(item)
                seen_items.add(simplify_text(item.normalized_query))

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
    ) or fallback.requested_fields
    if "part_number" not in requested_fields:
        requested_fields = ("part_number", *requested_fields)

    detected_equipment = _detect_named_value(query, _merged_aliases(EQUIPMENT_ALIASES, PROPER_EQUIPMENT_ALIASES))
    detected_vendor = _detect_named_value(query, _merged_aliases(VENDOR_ALIASES, PROPER_VENDOR_ALIASES))
    parsed_equipment = _canonical_from_alias(str(parsed.get("equipment_query") or parsed.get("equipment") or ""), _merged_aliases(EQUIPMENT_ALIASES, PROPER_EQUIPMENT_ALIASES))
    parsed_vendor = _canonical_from_alias(str(parsed.get("vendor_query") or parsed.get("vendor") or ""), _merged_aliases(VENDOR_ALIASES, PROPER_VENDOR_ALIASES))
    intent = str(parsed.get("intent") or fallback.intent)
    if intent not in ALLOWED_INTENTS:
        intent = fallback.intent
    return QueryPlan(
        items=tuple(items),
        requested_fields=requested_fields,
        vendor_query=detected_vendor or parsed_vendor or fallback.vendor_query,
        equipment_query=detected_equipment or parsed_equipment or fallback.equipment_query,
        intent=intent,
        used_llm=True,
    )


@traceable_run(name="llm_decompose_query", run_type="chain")
def llm_decompose_query(query: str) -> QueryPlan:
    """Return a possibly multi-item lookup plan for agentic search.

    This is intentionally conservative: if the LLM is unavailable or returns an
    invalid shape, the deterministic route remains the fallback.
    """
    catalog = load_part_catalog()
    fallback_route = rule_based_route(query)
    fallback_plan = QueryPlan(
        items=_rule_based_part_items(query, catalog),
        requested_fields=fallback_route.requested_fields,
        vendor_query=fallback_route.vendor_query,
        equipment_query=fallback_route.equipment_query,
        intent=fallback_route.intent,
        used_llm=False,
    )
    if not is_llm_enabled():
        return fallback_plan

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return fallback_plan

    configure_ollama_env()
    model = get_ollama_model()
    catalog_text = "\n".join(f"- {name}" for name in catalog[:80])
    prompt = f"""
You are a query decomposition agent for a semiconductor part-number finder.
Return JSON only. Do not invent part numbers.

Goal:
- Split the user's message into every requested part-name lookup item.
- Preserve global vendor/equipment filters separately.
- Normalize each item to the closest local catalog part family.
- Sentence order may be unusual, Korean/English may be mixed, and the user may ask for multiple parts.

Known catalog examples:
{catalog_text}

Useful mappings:
- 로봇블레이드, robot arm blade, end effector -> Robot Blade
- 터보펌프 -> Turbo Pump
- Quart Tube, 쿼츠튜브 -> Quartz Tube
- L/Q -> Liner Quartz
- W/Q -> Window Quartz
- pmkit -> PM Kit
- T/V, 스로틀밸브 -> Throttle Valve
- 오링 -> O-ring

Return schema:
{{"intent":"lookup_part|lookup_details|semantic_candidate|clarify", "vendor_query":"AMAT|ASML|TEL|Lam Research|", "equipment_query":"Endura|Vantage Radox|Etch Module|Implant Module|Helios XP|", "requested_fields":["part_number"], "items":[{{"raw_text":"...", "normalized_query":"...", "candidate_queries":["..."], "semantic_queries":["..."], "confidence":0.0}}]}}

User query: {query}
""".strip()
    try:
        llm = ChatOllama(model=model, temperature=0)
        response = llm.invoke(prompt)
        parsed = _parse_json_response(str(getattr(response, "content", response)))
    except Exception:
        return fallback_plan

    if not parsed:
        return fallback_plan
    try:
        plan = _query_plan_from_parsed(query, parsed, catalog)
    except Exception:
        return fallback_plan
    return plan
