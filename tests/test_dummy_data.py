import csv
import re
import subprocess
import sys
from pathlib import Path

from part_finder.search import search_part_numbers


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "part_numbers.csv"
PART_NUMBER_RE = re.compile(r"^P\d{7}$")


def ensure_dummy_data():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_dummy_data.py"), "--force"],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def load_rows():
    ensure_dummy_data()
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_dummy_data_has_500_rows_and_valid_unique_part_numbers():
    rows = load_rows()
    part_numbers = [row["part_number"] for row in rows]
    assert len(rows) == 500
    assert all(PART_NUMBER_RE.match(part_number) for part_number in part_numbers)
    assert len(part_numbers) == len(set(part_numbers))


def test_seed_data_is_included():
    rows = load_rows()
    seed_pairs = {
        ("윈도우쿼츠", "P2155479"),
        ("피엠킷", "P2100958"),
        ("피엠키트", "P2100957"),
        ("PM Kit", "P2100555"),
        ("라이너쿼츠", "P2155480"),
        ("Window Quartz", "P2100452"),
    }
    actual_pairs = {(row["part_name"], row["part_number"]) for row in rows}
    assert seed_pairs <= actual_pairs


def test_dummy_search_seed_queries():
    load_rows()
    window_results = search_part_numbers("윈도우쿼츠 파트넘버 알려줘")
    assert any(result["part_number"] == "P2155479" for result in window_results)

    pm_results = search_part_numbers("피엠킷 파트넘버 알려줘")
    assert any(result["part_number"] in {"P2100958", "P2100957", "P2100555"} for result in pm_results)

    wq_results = search_part_numbers("W/Q의 파트넘버 알려줘")
    assert wq_results
    assert wq_results[0]["matched_query"] == "Window Quartz"

    liner_results = search_part_numbers("Liner Quartz 의 파트넘버 알려줘")
    assert any("Liner Quartz" in str(result["description"]) for result in liner_results)

    o_ring_results = search_part_numbers("오링 파트넘버 알려줘")
    assert any("O-ring" in str(result["part_name"]) for result in o_ring_results)
