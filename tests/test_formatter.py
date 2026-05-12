from part_finder.formatter import format_answer


def test_format_no_result():
    assert "찾지 못했습니다" in format_answer("unknown", "unknown", [])


def test_format_abbreviation_answer():
    answer = format_answer(
        "W/Q의 파트넘버 알려줘",
        "Window Quartz",
        [{"part_number": "P2100452", "part_name": "Window Quartz", "score": 100}],
    )
    assert answer == "질문하신 W/Q는 Window Quartz로 매칭했습니다. 파트넘버는 P2100452 입니다."


def test_format_catalog_name_answer():
    answer = format_answer(
        "오링 파트넘버 알려줘",
        "O-ring",
        [{"part_number": "P2200014", "part_name": "O-ring 014", "score": 100}],
    )
    assert answer == "O-ring의 파트넘버는 P2200014 입니다."
