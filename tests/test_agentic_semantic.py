from part_finder.agent import answer_query_result
from part_finder.final_responder import final_answer
from part_finder.llm_router import llm_route
from part_finder.search import agentic_part_search_tool


def test_clamp_router_adds_phonetic_and_semantic_search(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    route = llm_route("클램프 파트넘버 찾아줘")
    rows = agentic_part_search_tool(route, top_k=5)

    assert "clamp" in route.phonetic_english_candidates
    assert route.needs_semantic_search
    assert any("vector_semantic_search_tool" == tool for tool in route.preferred_tools)
    assert rows
    assert any("Clamp" in str(row.get("description")) for row in rows)


def test_throttle_valve_phonetic_query_returns_candidate(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    route = llm_route("쓰로틀밸브 찾아줘")
    rows = agentic_part_search_tool(route, top_k=5)

    assert "throttle valve" in route.phonetic_english_candidates or route.normalized_query == "Throttle Valve"
    assert rows
    assert any("Throttle Valve" in str(row.get("description")) for row in rows)


def test_conceptual_wafer_holding_uses_confirmation(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    route = llm_route("웨이퍼 잡아주는 부품")
    result = answer_query_result("웨이퍼 잡아주는 부품", top_k=5)

    assert route.query_type == "conceptual"
    assert any(query in route.semantic_queries for query in ("wafer clamp", "wafer holder", "chuck"))
    assert result.pending_confirmation
    assert "확인" not in result.answer
    assert "더 알려" not in result.answer


def test_robot_wafer_pick_candidate_in_top_k(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    result = answer_query_result("로봇이 웨이퍼 집을 때 쓰는 부품", top_k=5)

    assert result.rows
    assert any("Robot Blade" in str(row.get("description")) for row in result.rows)
    assert result.pending_confirmation


def test_named_part_and_equipment_prioritize_part_family(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    for query in ["엔듀라 로봇 블레이드 파트넘버", "엔듀라 로봇 블레이드 파트넘버 알려줘."]:
        result = answer_query_result(query, top_k=3)

        assert result.rows
        assert result.rows[0]["part_number"] == "P2200043"
        assert all("Robot Blade" in str(row.get("description")) for row in result.rows)
        assert all(row.get("equipment_module") == "Endura" for row in result.rows)
        assert all("Edge Ring" not in str(row.get("description")) for row in result.rows)
        assert "더 알려" not in result.answer
        assert "확인" not in result.answer


def test_vantage_and_amat_aliases_are_context_not_part_queries(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    vantage = answer_query_result("밴티지 세라믹링 파트넘버", top_k=3)
    amat = answer_query_result("에이멧 세라믹링 파트넘버", top_k=3)

    assert vantage.rows
    assert all("Ceramic Ring" in str(row.get("description")) for row in vantage.rows)
    assert all(row.get("equipment_module") == "Vantage Radox" for row in vantage.rows)
    assert amat.rows
    assert all("Ceramic Ring" in str(row.get("description")) for row in amat.rows)
    assert all(row.get("vendor") == "AMAT" for row in amat.rows)


def test_final_responder_blocks_part_number_outside_candidate_rows(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "1")
    monkeypatch.setattr("part_finder.final_responder._call_final_llm", lambda payload: "파트넘버는 P9999999 입니다.")

    answer = final_answer(
        "robot blade part number",
        "Robot Blade",
        [
            {
                "part_number": "P2200013",
                "part_name": "Robot Blade 019",
                "description": "Robot Blade for Implant Module",
                "score": 1.0,
            }
        ],
    )

    assert "P9999999" not in answer
    assert "P2200013" in answer
