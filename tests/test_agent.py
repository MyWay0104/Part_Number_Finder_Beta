from part_finder.agent import answer_query_result
from part_finder.llm_router import RouteDecision


def test_agent_uses_vendor_context_for_vacuum_gauge(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    result = answer_query_result("램리서치 장비의 베큠게이지 파트넘버 알려줘", top_k=3)

    assert not result.needs_retry
    assert "Vacuum Gauge" in result.answer
    assert "Gate Valve" not in result.answer
    assert "Lam Research" in result.answer


def test_agent_uses_equipment_context(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    result = answer_query_result("엔듀라 모델 슬릿밸브 파트넘버", top_k=3)

    assert not result.needs_retry
    assert "Slit Valve" in result.answer
    assert "Endura" in result.answer


def test_agent_rejects_llm_candidate_that_conflicts_with_detected_part(monkeypatch):
    def fake_route(query: str) -> RouteDecision:
        return RouteDecision(
            query_type="mixed",
            normalized_query="Gate Valve",
            candidate_queries=("Gate Valve", "Vacuum Gauge"),
            vendor_query="Lam Research",
            tool_name="hybrid_search_tool",
            used_llm=True,
        )

    monkeypatch.setattr("part_finder.agent.llm_route", fake_route)

    result = answer_query_result("램리서치 장비의 베큠게이지 파트넘버 알려줘", top_k=3)

    assert not result.needs_retry
    assert "Vacuum Gauge" in result.answer
    assert "Gate Valve" not in result.answer


def test_agent_asks_confirmation_for_semantic_robot_arm_pick(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    result = answer_query_result("ASML equipment robot arm pick part number", top_k=3)

    assert not result.needs_retry
    assert result.pending_confirmation
    assert "Robot Blade" in result.answer
    assert "확인" not in result.answer


def test_agent_returns_semantic_candidate_after_confirmation(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")
    first = answer_query_result("ASML equipment robot arm pick part number", top_k=3)

    result = answer_query_result("확인", top_k=3, pending_confirmation=first.pending_confirmation)

    assert not result.needs_retry
    assert "Robot Blade" in result.answer
    assert "P2200013" in result.answer


def test_agent_does_not_drift_part_family_when_vendor_has_no_match(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    result = answer_query_result("로봇블레이드 파트넘버. AMAT 장비 기준으로.", top_k=3)

    assert result.needs_retry
    assert result.rows == []
    assert "Gate Valve" not in result.answer
    assert "Cooling Plate" not in result.answer


def test_agentic_langgraph_multi_part_lookup(monkeypatch):
    monkeypatch.setenv("PART_FINDER_USE_LLM", "0")

    result = answer_query_result("AMAT 기준으로 터보펌프랑 로봇블레이드 파트넘버 알려줘", top_k=3)

    assert "Turbo Pump" in result.answer
    assert "P2200046" in result.answer
    assert "Robot Blade" in result.answer
    assert "AMAT 조건" in result.answer
