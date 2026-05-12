from __future__ import annotations

from pathlib import Path


# Keep all paths relative to the repository root so the CLI and tests behave the
# same way from a local checkout or a closed network machine.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ALIASES_PATH = DATA_DIR / "aliases.json"
DEFAULT_CSV_PATH = DATA_DIR / "part_numbers.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "Part_Number.db"

PART_NUMBER_PATTERN = r"P\d{7}"

