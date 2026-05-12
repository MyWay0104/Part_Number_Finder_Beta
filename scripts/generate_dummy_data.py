from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "part_numbers.csv"

FIELDNAMES = [
    "part_number",
    "part_name",
    "description",
    "equipment_module",
    "vendor_part_number",
    "vendor",
]

SEED_ROWS = [
    {
        "part_number": "P2155479",
        "part_name": "윈도우쿼츠",
        "description": "Window Quartz",
        "equipment_module": "Vantage Radox",
        "vendor_part_number": "AMAT-WQ-2155479",
        "vendor": "AMAT",
    },
    {
        "part_number": "P2100958",
        "part_name": "피엠킷",
        "description": "PM Kit for Helios XP",
        "equipment_module": "Helios XP",
        "vendor_part_number": "AMAT-PMK-2100958",
        "vendor": "AMAT",
    },
    {
        "part_number": "P2100957",
        "part_name": "피엠키트",
        "description": "PM Kit for Helios XP",
        "equipment_module": "Helios XP",
        "vendor_part_number": "TEL-PMK-2100957",
        "vendor": "TEL",
    },
    {
        "part_number": "P2100555",
        "part_name": "PM Kit",
        "description": "PM Kit for Helios XP",
        "equipment_module": "Helios XP",
        "vendor_part_number": "LAM-PMK-2100555",
        "vendor": "Lam Research",
    },
    {
        # Same physical parts can be represented by Korean/English aliases in real
        # systems, but the default dummy data keeps part_number unique so search
        # result dedupe is easy to validate.
        "part_number": "P2155480",
        "part_name": "라이너쿼츠",
        "description": "Liner Quartz",
        "equipment_module": "Vantage Radox",
        "vendor_part_number": "ASML-LQ-2155480",
        "vendor": "ASML",
    },
    {
        "part_number": "P2100452",
        "part_name": "Window Quartz",
        "description": "Window Quartz for Vantage Radox",
        "equipment_module": "Vantage Radox",
        "vendor_part_number": "TEL-WQ-2100452",
        "vendor": "TEL",
    },
]

PART_NAMES = [
    "Window Quartz",
    "Liner Quartz",
    "PM Kit",
    "Throttle Valve",
    "Ceramic Ring",
    "Focus Ring",
    "Shower Head",
    "ESC Chuck",
    "O-ring",
    "Bellows",
    "RF Match",
    "Gas Line Filter",
    "Quartz Tube",
    "Edge Ring",
    "Clamp Ring",
    "Slit Valve",
    "Gate Valve",
    "Heater Block",
    "Susceptor",
    "Robot Blade",
    "Vacuum Gauge",
    "MFC",
    "Turbo Pump",
    "Dry Pump",
    "Cooling Plate",
    "Ceramic Plate",
    "Ion Source",
    "Chamber Wall",
    "Nozzle",
    "Diffuser",
]

MODULES = [
    "Vantage Radox",
    "Helios XP",
    "Centura",
    "Endura",
    "Producer",
    "P5000",
    "Etch Module",
    "CVD Module",
    "RTP Module",
    "Implant Module",
    "Loadlock Module",
    "Transfer Chamber",
]

VENDORS = ["AMAT", "TEL", "Lam Research", "ASML"]


def generate_rows(count: int = 500) -> list[dict[str, str]]:
    rows = list(SEED_ROWS)
    used_part_numbers = {row["part_number"] for row in rows}
    next_number = 2200000

    while len(rows) < count:
        part_name = PART_NAMES[len(rows) % len(PART_NAMES)]
        module = MODULES[(len(rows) * 3) % len(MODULES)]
        vendor = VENDORS[(len(rows) * 5) % len(VENDORS)]
        part_number = f"P{next_number:07d}"
        next_number += 1
        if part_number in used_part_numbers:
            continue
        used_part_numbers.add(part_number)

        rows.append(
            {
                "part_number": part_number,
                "part_name": f"{part_name} {len(rows):03d}",
                "description": f"{part_name} for {module}",
                "equipment_module": module,
                "vendor_part_number": f"{vendor[:3].upper().replace(' ', '')}-{part_name[:3].upper().replace('-', '')}-{len(rows):04d}",
                "vendor": vendor,
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic dummy part-number data.")
    parser.add_argument("--force", action="store_true", help="기존 data/part_numbers.csv를 덮어씁니다.")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        print(f"경고: {OUTPUT_PATH} 파일이 이미 존재합니다. 덮어쓰려면 --force 옵션을 사용하세요.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(500)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"생성 완료: {OUTPUT_PATH} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
