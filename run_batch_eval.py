from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from part_finder.agent import answer_query_result


DEFAULT_INPUT_CSV = DATA / "langfuse_partfinder_test_questions_50cases.csv"
DEFAULT_OUTPUT_CSV = DATA / "langfuse_batch_result.csv"


def _string_value(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return str(value)


def _result_answer(result: Any) -> str:
    return str(getattr(result, "answer", result))


def _result_field(result: Any, field: str, default: Any = "") -> Any:
    if is_dataclass(result):
        return asdict(result).get(field, default)
    return getattr(result, field, default)


def run_batch(input_csv: Path, output_csv: Path, top_k: int, limit: int | None = None) -> Path:
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)
    results: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        test_id = _string_value(row, "Test_ID")
        question = _string_value(row, "Question")
        validation_target = _string_value(row, "Validation_Target")
        evaluation_point = _string_value(row, "Evaluation_Point")

        metadata = {
            "test_id": test_id,
            "validation_target": validation_target,
            "evaluation_point": evaluation_point,
            "source": "csv_batch_eval",
            "run_time": datetime.now().isoformat(),
        }

        base_record = {
            "Test_ID": test_id,
            "Question": question,
            "Validation_Target": validation_target,
            "Evaluation_Point": evaluation_point,
        }

        try:
            result = answer_query_result(
                question,
                top_k=top_k,
                metadata=metadata,
            )
            rows = _result_field(result, "rows", []) or []
            results.append(
                {
                    **base_record,
                    "Answer": _result_answer(result),
                    "Best_Score": _result_field(result, "best_score", ""),
                    "Needs_Retry": _result_field(result, "needs_retry", ""),
                    "Intent": _result_field(result, "intent", ""),
                    "Row_Count": len(rows) if isinstance(rows, list) else "",
                    "Status": "success",
                    "Error": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    **base_record,
                    "Answer": "",
                    "Best_Score": "",
                    "Needs_Retry": "",
                    "Intent": "",
                    "Row_Count": "",
                    "Status": "error",
                    "Error": str(exc),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CSV batch evaluation for Part Number Finder.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output result CSV path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to retrieve per question.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of questions to run.")
    args = parser.parse_args()

    output_csv = run_batch(args.input_csv, args.output_csv, args.top_k, args.limit)
    print(f"Batch test complete: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
