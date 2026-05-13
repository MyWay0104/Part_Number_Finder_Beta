from __future__ import annotations

import math
import re
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from part_finder.config import SEMANTIC_HINTS_PATH
from part_finder.data_loader import load_aliases, load_part_data
from part_finder.normalizer import simplify_text


DEFAULT_SEMANTIC_HINTS = {
    "Robot Blade": [
        "robot arm pick transfer wafer handler end effector blade",
        "robot blade arm pick place part",
    ],
    "ESC Chuck": ["wafer chuck electrostatic chuck holding stage"],
    "MFC": ["gas flow mass flow controller flow control"],
    "Vacuum Gauge": ["pressure gauge vacuum sensor pressure measurement"],
    "Gate Valve": ["isolation valve gate chamber valve"],
    "Slit Valve": ["slot valve slit door transfer opening valve"],
    "Throttle Valve": ["pressure control valve throttle valve"],
    "Shower Head": ["gas distribution plate showerhead gas injection"],
}


def _load_semantic_hints() -> dict[str, list[str]]:
    hints = {key: list(values) for key, values in DEFAULT_SEMANTIC_HINTS.items()}
    if SEMANTIC_HINTS_PATH.exists():
        try:
            with SEMANTIC_HINTS_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        hints.setdefault(str(key), [])
                        hints[str(key)].extend(str(value) for value in values)
        except Exception:
            pass
    return hints


SEMANTIC_HINTS = _load_semantic_hints()


@dataclass(frozen=True)
class PartChunk:
    chunk_id: str
    text: str
    row: dict[str, str]


@dataclass(frozen=True)
class VectorIndex:
    chunks: tuple[PartChunk, ...]
    idf: dict[str, float]
    vectors: tuple[dict[str, float], ...]
    norms: tuple[float, ...]


def _base_part_name(row: dict[str, str]) -> str:
    value = row.get("description", "") or row.get("part_name", "")
    return value.split(" for ", 1)[0].strip()


def _tokenize(text: str) -> list[str]:
    simple = simplify_text(text)
    words = re.findall(r"[a-z0-9]+", simple)
    char_grams: list[str] = []
    compact = re.sub(r"[^a-z0-9]+", "", simple)
    for size in (3, 4):
        char_grams.extend(compact[index : index + size] for index in range(max(0, len(compact) - size + 1)))
    return words + char_grams


def build_part_chunks(rows: Iterable[dict[str, str]] | None = None) -> list[PartChunk]:
    """Create RAG chunks from catalog rows.

    The dummy CSV is already row-structured, so one row is the safest chunk
    size. Each chunk combines identifying fields, searchable descriptions, and
    a small controlled semantic expansion for domain terms.
    """
    source_rows = list(rows if rows is not None else load_part_data())
    chunks: list[PartChunk] = []
    aliases = load_aliases()
    for index, row in enumerate(source_rows):
        base_name = _base_part_name(row)
        normalized_tokens = " ".join(sorted(set(_tokenize(" ".join(row.values())))))
        alias_text = " ".join(aliases.get(base_name, []))
        hint_text = " ".join(SEMANTIC_HINTS.get(base_name, []))
        fields = [
            "part_number",
            row.get("part_number", ""),
            "part_name",
            row.get("part_name", ""),
            "description",
            row.get("description", ""),
            "equipment_module",
            row.get("equipment_module", ""),
            "vendor_part_number",
            row.get("vendor_part_number", ""),
            "vendor",
            row.get("vendor", ""),
            "aliases",
            alias_text,
            "normalized_tokens",
            normalized_tokens,
            "semantic_hints",
            hint_text,
            base_name,
        ]
        text = " | ".join(field for field in fields if field)
        chunks.append(PartChunk(chunk_id=f"part-row-{index}", text=text, row=row))
    return chunks


def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    if not counts:
        return {}
    total = sum(counts.values())
    return {token: (count / total) * idf.get(token, 0.0) for token, count in counts.items()}


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


@lru_cache(maxsize=1)
def build_vector_index() -> VectorIndex:
    chunks = tuple(build_part_chunks())
    documents = [_tokenize(chunk.text) for chunk in chunks]
    document_count = len(documents)
    doc_freq: Counter[str] = Counter()
    for tokens in documents:
        doc_freq.update(set(tokens))

    idf = {
        token: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for token, frequency in doc_freq.items()
    }
    vectors = tuple(_tfidf(tokens, idf) for tokens in documents)
    norms = tuple(_norm(vector) for vector in vectors)
    return VectorIndex(chunks=chunks, idf=idf, vectors=vectors, norms=norms)


def _cosine(left: dict[str, float], left_norm: float, right: dict[str, float], right_norm: float) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    return dot / (left_norm * right_norm)


def vector_search(query: str, top_k: int = 3) -> list[dict[str, object]]:
    index = build_vector_index()
    query_vector = _tfidf(_tokenize(query), index.idf)
    query_norm = _norm(query_vector)
    ranked: list[dict[str, object]] = []
    for chunk, vector, norm in zip(index.chunks, index.vectors, index.norms, strict=True):
        similarity = _cosine(query_vector, query_norm, vector, norm)
        if similarity <= 0.0:
            continue
        ranked.append(
            {
                **chunk.row,
                "score": round(similarity * 100, 2),
                "matched_query": _base_part_name(chunk.row),
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.text,
                "semantic_match": True,
                "vector_match": True,
            }
        )

    ranked.sort(key=lambda item: (-float(item["score"]), str(item["part_number"])))
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in ranked:
        part_number = str(item["part_number"])
        if part_number in seen:
            continue
        seen.add(part_number)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped
