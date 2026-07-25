"""Local LanceDB store for T²-RAGBench chunk retrieval (vector / BM25 / hybrid)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import lancedb
from rank_bm25 import BM25Okapi

from foundry_rag.mechanisms.pgvector_store import (
    RRF_K,
    ChunkHit,
    ranked_context_ids,
    reciprocal_rank_fusion,
    tokenize,
)
from foundry_rag.paths import DATA_ROOT

DEFAULT_DB_URI = str(DATA_ROOT / "t2_ragbench" / "lancedb")
TABLE_NAME = "t2_chunks"


def db_uri() -> str:
    return os.environ.get("LANCEDB_URI", DEFAULT_DB_URI)


def connect(uri: str | None = None) -> lancedb.DBConnection:
    path = uri or db_uri()
    Path(path).mkdir(parents=True, exist_ok=True)
    return lancedb.connect(path)


def open_table(db: lancedb.DBConnection | None = None):
    database = db or connect()
    if TABLE_NAME not in database.table_names():
        raise RuntimeError(
            f"LanceDB table '{TABLE_NAME}' not found at {db_uri()}. "
            "Run: uv run python scripts/index_t2_ragbench_lancedb.py --reset"
        )
    return database.open_table(TABLE_NAME)


def upsert_chunks(
    rows: Iterable[dict[str, Any]],
    *,
    reset: bool = True,
    uri: str | None = None,
) -> int:
    """Write chunk rows into LanceDB (replaces table when ``reset``)."""
    records = [
        {
            "id": row["id"],
            "context_id": row["context_id"],
            "subset": row.get("subset", ""),
            "file_name": row.get("file_name", ""),
            "chunk_index": int(row.get("chunk_index", 0)),
            "content": row["content"],
            "vector": [float(v) for v in row["embedding"]],
        }
        for row in rows
    ]
    if not records:
        return 0

    database = connect(uri)
    if reset and TABLE_NAME in database.table_names():
        database.drop_table(TABLE_NAME)
    database.create_table(TABLE_NAME, data=records, mode="overwrite")
    clear_bm25_cache()
    return len(records)


def chunk_count(uri: str | None = None) -> int:
    database = connect(uri)
    if TABLE_NAME not in database.table_names():
        return 0
    return int(database.open_table(TABLE_NAME).count_rows())


def search(
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
    uri: str | None = None,
) -> list[ChunkHit]:
    table = open_table(connect(uri))
    rows = table.search(list(query_embedding)).metric("cosine").limit(top_k).to_list()
    hits: list[ChunkHit] = []
    for row in rows:
        distance = float(row.get("_distance", 0.0))
        hits.append(
            ChunkHit(
                chunk_id=row["id"],
                context_id=row["context_id"],
                subset=row.get("subset", ""),
                file_name=row.get("file_name", ""),
                chunk_index=int(row.get("chunk_index", 0)),
                content=row["content"],
                score=1.0 - distance,
            )
        )
    return hits


def fetch_all_chunks(uri: str | None = None) -> list[dict[str, Any]]:
    table = open_table(connect(uri))
    frame = table.to_pandas()
    cols = ["id", "context_id", "subset", "file_name", "chunk_index", "content"]
    return frame[cols].to_dict(orient="records")


@lru_cache(maxsize=4)
def _bm25_bundle(uri: str) -> tuple[BM25Okapi, tuple[dict[str, Any], ...]]:
    rows = tuple(fetch_all_chunks(uri))
    if not rows:
        return BM25Okapi([["__empty__"]]), ()
    tokenized = [tokenize(r["content"]) or ["__empty__"] for r in rows]
    return BM25Okapi(tokenized), rows


def clear_bm25_cache() -> None:
    _bm25_bundle.cache_clear()


def search_bm25(query: str, *, top_k: int = 3, uri: str | None = None) -> list[ChunkHit]:
    bm25, rows = _bm25_bundle(uri or db_uri())
    if not rows:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        ChunkHit(
            chunk_id=rows[i]["id"],
            context_id=rows[i]["context_id"],
            subset=rows[i].get("subset", ""),
            file_name=rows[i].get("file_name", ""),
            chunk_index=int(rows[i].get("chunk_index", 0)),
            content=rows[i]["content"],
            score=float(scores[i]),
        )
        for i in ranked_idx
    ]


def search_hybrid(
    query: str,
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
    candidate_k: int | None = None,
    uri: str | None = None,
    rrf_k: int = RRF_K,
) -> list[ChunkHit]:
    pool = candidate_k or max(top_k * 4, 20)
    vector_hits = search(query_embedding, top_k=pool, uri=uri)
    bm25_hits = search_bm25(query, top_k=pool, uri=uri)
    return reciprocal_rank_fusion([vector_hits, bm25_hits], top_k=top_k, rrf_k=rrf_k)


__all__ = [
    "ChunkHit",
    "chunk_count",
    "clear_bm25_cache",
    "connect",
    "db_uri",
    "fetch_all_chunks",
    "open_table",
    "ranked_context_ids",
    "search",
    "search_bm25",
    "search_hybrid",
    "upsert_chunks",
]
