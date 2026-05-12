from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from part_finder.config import ALIASES_PATH, DEFAULT_CSV_PATH, DEFAULT_DB_PATH, PART_NUMBER_PATTERN, PROJECT_ROOT


PART_NUMBER_RE = re.compile(f"^{PART_NUMBER_PATTERN}$")

FIELD_ALIASES = {
    "part_number": ["part_number", "part no", "part_no", "pn", "p/n", "partnumber"],
    "part_name": ["part_name", "part name", "name", "item_name", "item name"],
    "description": ["description", "desc", "part_description", "part description"],
    "vendor_part_number": ["vendor_part_number", "vendor part number", "vpn", "vendor_pn", "vendor pn"],
    "vendor": ["vendor", "maker", "manufacturer", "supplier"],
    "equipment_module": ["equipment_module", "equipment module", "module", "tool_module"],
}


def _clean_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def _build_column_map(headers: list[str]) -> dict[str, str]:
    """Map flexible source column names into the canonical schema used by search."""
    cleaned = {_clean_header(header): header for header in headers}
    column_map: dict[str, str] = {}

    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            source = cleaned.get(_clean_header(alias))
            if source:
                column_map[canonical] = source
                break

    return column_map


def _canonicalize_row(row: dict[str, Any], column_map: dict[str, str]) -> dict[str, str] | None:
    canonical = {
        "part_number": "",
        "part_name": "",
        "description": "",
        "equipment_module": "",
        "vendor_part_number": "",
        "vendor": "",
    }
    for target, source in column_map.items():
        canonical[target] = str(row.get(source, "") or "").strip()

    if not PART_NUMBER_RE.match(canonical["part_number"]):
        return None
    if not canonical["part_name"] and not canonical["description"]:
        return None
    return canonical


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """Load alias config from JSON, returning an empty map if the file is absent."""
    alias_path = path or ALIASES_PATH
    if not alias_path.exists():
        return {}
    with alias_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(target): [str(alias) for alias in aliases] for target, aliases in data.items()}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return []
        column_map = _build_column_map(reader.fieldnames)
        if "part_number" not in column_map or ("part_name" not in column_map and "description" not in column_map):
            return []
        rows = [_canonicalize_row(row, column_map) for row in reader]
    return [row for row in rows if row is not None]


def _load_db(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        table_names = [
            item[0]
            for item in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        for table_name in table_names:
            cursor = connection.execute(f'SELECT * FROM "{table_name}"')
            headers = [description[0] for description in cursor.description or []]
            column_map = _build_column_map(headers)
            if "part_number" not in column_map or ("part_name" not in column_map and "description" not in column_map):
                continue
            for db_row in cursor.fetchall():
                canonical = _canonicalize_row(dict(db_row), column_map)
                if canonical:
                    rows.append(canonical)
            if rows:
                break
    return rows


def load_part_data(path: Path | None = None) -> list[dict[str, str]]:
    """Load part rows by priority: explicit path, Part_Number.db, CSV, then TXT.

    TXT support is intentionally simple for the PoC: it expects delimited content
    readable as CSV. More unstructured TXT should move to a later RAG workflow.
    """
    candidates: list[Path]
    if path:
        candidates = [path]
    else:
        candidates = []
        if DEFAULT_DB_PATH.exists():
            candidates.append(DEFAULT_DB_PATH)
        if DEFAULT_CSV_PATH.exists():
            candidates.append(DEFAULT_CSV_PATH)
        candidates.extend(sorted(PROJECT_ROOT.glob("*.csv")))
        candidates.extend(sorted(PROJECT_ROOT.glob("*.txt")))

    for candidate in candidates:
        if not candidate.exists():
            continue
        suffix = candidate.suffix.lower()
        if suffix == ".db":
            rows = _load_db(candidate)
        elif suffix in {".csv", ".txt"}:
            rows = _load_csv(candidate)
        else:
            continue
        if rows:
            return rows

    return []


def load_part_catalog(limit: int = 80) -> list[str]:
    """Return unique searchable part names for LLM routing prompts.

    Generated dummy rows often append numeric suffixes such as "O-ring 014".
    Removing only a trailing numeric suffix gives the router a concise catalog
    while keeping real source data untouched.
    """
    names: list[str] = []
    seen: set[str] = set()
    for row in load_part_data():
        for field in ["part_name", "description"]:
            value = re.sub(r"\s+\d{3,}$", "", row.get(field, "")).strip()
            value = re.sub(r"\s+for\s+.+$", "", value, flags=re.IGNORECASE).strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                names.append(value)
        if len(names) >= limit:
            break
    return names
