from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from part_finder.agent import answer_query_result


def _print_answer_with_optional_retry(query: str, top_k: int) -> dict[str, object] | None:
    result = answer_query_result(query, top_k=top_k)
    print(result.answer)
    if result.pending_confirmation:
        return result.pending_confirmation
    if not result.needs_retry or not sys.stdin.isatty():
        return None

    retry_query = input("파트 영문명 입력: ").strip()
    if not retry_query:
        return None
    retry_result = answer_query_result(retry_query, top_k=top_k)
    print(retry_result.answer)
    return retry_result.pending_confirmation


def main() -> int:
    parser = argparse.ArgumentParser(description="Part Number Finder Agent PoC")
    parser.add_argument("query", nargs="*", help="검색할 Part Name, 약어, 한글명 또는 영문명")
    parser.add_argument("--top-k", type=int, default=3, help="반환할 최대 Part Number 개수")
    args = parser.parse_args()

    if args.query:
        _print_answer_with_optional_retry(" ".join(args.query), args.top_k)
        return 0

    pending_confirmation: dict[str, object] | None = None
    try:
        while True:
            query = input("질문을 입력하세요: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                return 0
            if not query:
                continue
            result = answer_query_result(query, top_k=args.top_k, pending_confirmation=pending_confirmation)
            print(result.answer)
            pending_confirmation = result.pending_confirmation
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
