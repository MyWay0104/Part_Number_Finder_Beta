from part_finder.normalizer import detect_query_type, normalize_query


def test_pm_kit_korean_aliases_normalize_to_pm_kit():
    assert normalize_query("피엠킷 파트넘버 알려줘") == "PM Kit"
    assert normalize_query("피엠키트 파트넘버 알려줘") == "PM Kit"


def test_abbreviation_and_english_aliases_normalize():
    assert normalize_query("W/Q의 파트넘버 알려줘") == "Window Quartz"
    assert normalize_query("Liner Quartz 의 파트넘버 알려줘") == "Liner Quartz"
    assert normalize_query("오링 파트넘버 알려줘") == "O-ring"
    assert normalize_query("Owe ling part number") == "O-ring"
    assert normalize_query("샤워헤드 파트넘버") == "Shower Head"


def test_detect_query_type():
    assert detect_query_type("W/Q") == "abbreviation"
    assert detect_query_type("MFC") == "abbreviation"
    assert detect_query_type("라이너쿼츠") == "korean"
    assert detect_query_type("Liner Quartz") == "english"
