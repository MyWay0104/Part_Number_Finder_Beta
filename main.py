from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from part_finder.agent import answer_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Part Number Finder Agent PoC")
    parser.add_argument("query", nargs="*", help="검색할 Part Name, 약어, 한글명 또는 영어명")
    parser.add_argument("--top-k", type=int, default=3, help="반환할 최대 Part Number 개수")
    args = parser.parse_args()

    if args.query:
        print(answer_query(" ".join(args.query), top_k=args.top_k))
        return 0

    try:
        while True:
            query = input("질문을 입력하세요: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                return 0
            if not query:
                continue
            print(answer_query(query, top_k=args.top_k))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
