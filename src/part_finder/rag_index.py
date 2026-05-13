from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from part_finder.data_loader import load_part_data
from part_finder.vector_index import PartChunk, build_part_chunks


EmbeddingProvider = Callable[[list[str]], list[list[float]]]


def load_embedding_provider(dotted_path: str) -> EmbeddingProvider:
    """Load an internal embedding function from module:function."""
    module_name, separator, function_name = dotted_path.partition(":")
    if not separator:
        module_name, _, function_name = dotted_path.rpartition(".")
    if not module_name or not function_name:
        raise ValueError("Embedding provider must be 'module:function' or 'module.function'.")
    module = importlib.import_module(module_name)
    provider = getattr(module, function_name)
    if not callable(provider):
        raise TypeError(f"Embedding provider is not callable: {dotted_path}")
    return provider


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False)) / (left_norm * right_norm)


def chunk_payload(chunk: PartChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "row": chunk.row,
    }


def build_rag_index(
    csv_path: Path,
    output_path: Path,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Build a row-chunk RAG index from a canonical part CSV.

    Without an embedding provider this writes searchable chunk text only. The
    existing TF-IDF vector_search remains the local fallback. In the company
    environment, pass a provider that returns one embedding vector per chunk.
    """
    rows = load_part_data(csv_path)
    chunks = build_part_chunks(rows)
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_provider(texts) if embedding_provider else None
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError("Embedding provider must return one vector per chunk.")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_csv": str(csv_path),
        "embedding_model": getattr(embedding_provider, "__name__", "") if embedding_provider else "",
        "chunks": [
            {
                **chunk_payload(chunk),
                **({"embedding": embeddings[index]} if embeddings is not None else {}),
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_rag_index(index_path: Path) -> dict[str, Any]:
    return json.loads(index_path.read_text(encoding="utf-8"))


def search_rag_index(
    index_path: Path,
    query: str,
    embedding_provider: EmbeddingProvider,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search an embedding-backed JSON RAG index."""
    index = load_rag_index(index_path)
    query_vector = embedding_provider([query])[0]
    ranked: list[dict[str, Any]] = []
    for chunk in index.get("chunks", []):
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list):
            continue
        score = _cosine([float(value) for value in query_vector], [float(value) for value in embedding])
        ranked.append({**chunk.get("row", {}), "score": round(score * 100, 2), "chunk_id": chunk.get("chunk_id", "")})
    ranked.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("part_number", ""))))
    return ranked[:top_k]
