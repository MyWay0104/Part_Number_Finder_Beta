from part_finder.search import search_part_numbers, semantic_catalog_match_tool


def test_search_returns_top_k_and_is_deterministic():
    first = search_part_numbers("피엠킷 혹은 피엠키트의 파트넘버 알려줘", top_k=3)
    second = search_part_numbers("피엠킷 혹은 피엠키트의 파트넘버 알려줘", top_k=3)
    assert len(first) == 3
    assert first == second


def test_no_result_for_unrelated_query():
    assert search_part_numbers("없는부품명XYZ123", top_k=3) == []


def test_catalog_wide_korean_pronunciation_searches():
    o_ring_results = search_part_numbers("오링 파트넘버 알려줘", top_k=3)
    assert o_ring_results
    assert all("O-ring" in str(result["part_name"]) or "O-ring" in str(result["description"]) for result in o_ring_results)

    shower_results = search_part_numbers("샤워헤드 part no 알려줘", top_k=3)
    assert shower_results
    assert any("Shower Head" in str(result["part_name"]) for result in shower_results)

    pump_results = search_part_numbers("터보펌프 파트넘버 알려줘", top_k=3)
    assert pump_results
    assert any("Turbo Pump" in str(result["part_name"]) for result in pump_results)


def test_catalog_wide_typo_search():
    results = search_part_numbers("Owe ling part number", top_k=3)
    assert results
    assert any("O-ring" in str(result["part_name"]) for result in results)


def test_search_filters_by_vendor_for_vacuum_gauge():
    results = search_part_numbers("베큠게이지", top_k=3, vendor_query="Lam Research")
    assert results
    assert all(result["vendor"] == "Lam Research" for result in results)
    assert all("Vacuum Gauge" in str(result["part_name"]) for result in results)
    assert all("Gate Valve" not in str(result["part_name"]) for result in results)


def test_search_filters_by_equipment_model():
    results = search_part_numbers("슬릿밸브", top_k=3, equipment_query="Endura")
    assert results
    assert all(result["equipment_module"] == "Endura" for result in results)
    assert all("Slit Valve" in str(result["part_name"]) for result in results)


def test_semantic_catalog_match_maps_robot_arm_pick_to_robot_blade():
    results = semantic_catalog_match_tool(
        "ASML equipment robot arm pick part number",
        top_k=3,
        vendor_query="ASML",
    )

    assert results
    assert all(result["vendor"] == "ASML" for result in results)
    assert all("Robot Blade" in str(result["description"]) for result in results)
