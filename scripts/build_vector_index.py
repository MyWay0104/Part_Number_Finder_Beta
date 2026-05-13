from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from part_finder.rag_index import build_rag_index, load_embedding_provider


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a row-chunk RAG index from a part CSV.")
    parser.add_argument("csv_path", type=Path, help="Canonical CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "part_vector_index.json",
        help="Output JSON index path",
    )
    parser.add_argument(
        "--embedding-provider",
        default="",
        help="Optional internal embedding callable, e.g. company_embeddings:embed_texts",
    )
    args = parser.parse_args()

    provider = load_embedding_provider(args.embedding_provider) if args.embedding_provider else None
    payload = build_rag_index(args.csv_path, args.output, provider)
    print(f"Wrote {len(payload['chunks'])} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
