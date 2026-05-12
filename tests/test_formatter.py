from part_finder.formatter import LOW_CONFIDENCE_MESSAGE, format_answer


def test_format_no_result():
    assert format_answer("unknown", "unknown", []) == LOW_CONFIDENCE_MESSAGE


def test_format_low_score():
    answer = format_answer(
        "unknown",
        "unknown",
        [{"part_number": "P0000001", "part_name": "Unknown", "score": 69.99}],
    )
    assert answer == LOW_CONFIDENCE_MESSAGE


def test_format_abbreviation_answer():
    answer = format_answer(
        "W/Q 파트넘버 알려줘",
        "Window Quartz",
        [{"part_number": "P2100452", "part_name": "Window Quartz", "score": 100}],
    )
    assert answer == "질문하신 W/Q는 Window Quartz로 매칭되었습니다. 파트넘버는 P2100452 입니다."


def test_format_catalog_name_answer_with_context():
    answer = format_answer(
        "램리서치 장비의 베큠게이지 파트넘버 알려줘",
        "Vacuum Gauge",
        [
            {
                "part_number": "P2200044",
                "part_name": "Vacuum Gauge 050",
                "equipment_module": "Etch Module",
                "vendor": "Lam Research",
                "score": 100,
            }
        ],
    )
    assert answer == "Lam Research / Etch Module 기준 Vacuum Gauge의 파트넘버는 P2200044 입니다."
