from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CANONICAL_FIELDS = ["part_number", "part_name", "description", "equipment_module", "vendor_part_number", "vendor"]
PART_NUMBER_RE = re.compile(r"\bP\d{7}\b")


def _split_line(line: str) -> list[str]:
    if "\t" in line:
        return [part.strip() for part in line.split("\t")]
    if "|" in line:
        return [part.strip() for part in line.split("|")]
    if "," in line:
        return [part.strip() for part in line.split(",")]
    return re.split(r"\s{2,}", line.strip())


def parse_txt_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("part_number"):
            continue
        fields = _split_line(stripped)
        part_number = next((value for value in fields if PART_NUMBER_RE.fullmatch(value)), "")
        if not part_number:
            match = PART_NUMBER_RE.search(stripped)
            part_number = match.group(0) if match else ""
        if not part_number:
            continue
        values = [value for value in fields if value != part_number]
        row = {
            "part_number": part_number,
            "part_name": values[0] if len(values) > 0 else "",
            "description": values[1] if len(values) > 1 else values[0] if values else "",
            "equipment_module": values[2] if len(values) > 2 else "",
            "vendor_part_number": values[3] if len(values) > 3 else "",
            "vendor": values[4] if len(values) > 4 else "",
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess company TXT part data into canonical CSV.")
    parser.add_argument("txt_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/part_numbers.csv"))
    args = parser.parse_args()

    rows = parse_txt_rows(args.txt_path)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
