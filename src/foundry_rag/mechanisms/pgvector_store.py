"""Postgres + pgvector store for T²-RAGBench chunk retrieval.

Supports pure vector search, Okapi BM25 (via ``rank_bm25``), and hybrid RRF.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from rank_bm25 import BM25Okapi

DEFAULT_DATABASE_URL = "postgresql://foundry:foundry@localhost:5432/t2_ragbench"
EMBEDDING_DIM = int(os.environ.get("FOUNDRY_EMBEDDING_DIMENSIONS", "1536"))
RRF_K = 60
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    context_id: str
    subset: str
    file_name: str
    chunk_index: int
    content: str
    score: float


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect() -> psycopg.Connection:
    return psycopg.connect(database_url(), row_factory=dict_row)


def init_schema(conn: psycopg.Connection, *, embedding_dim: int = EMBEDDING_DIM) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS t2_chunks (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                subset TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL DEFAULT '',
                chunk_index INT NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                embedding vector({embedding_dim}) NOT NULL,
                meta JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS t2_chunks_context_id_idx ON t2_chunks (context_id)"
        )
        # HNSW needs rows; create if missing (safe on empty table in recent pgvector).
        cur.execute(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 't2_chunks_embedding_hnsw'
              ) THEN
                CREATE INDEX t2_chunks_embedding_hnsw
                  ON t2_chunks USING hnsw (embedding vector_cosine_ops);
              END IF;
            END $$;
            """
        )
    conn.commit()


def clear_chunks(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE t2_chunks")
    conn.commit()


def upsert_chunks(
    conn: psycopg.Connection,
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int = 64,
) -> int:
    sql = """
        INSERT INTO t2_chunks (
            id, context_id, subset, file_name, chunk_index, content, embedding, meta
        ) VALUES (
            %(id)s, %(context_id)s, %(subset)s, %(file_name)s, %(chunk_index)s,
            %(content)s, %(embedding)s::vector, %(meta)s
        )
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            meta = EXCLUDED.meta,
            context_id = EXCLUDED.context_id,
            subset = EXCLUDED.subset,
            file_name = EXCLUDED.file_name,
            chunk_index = EXCLUDED.chunk_index
    """
    count = 0
    batch: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        for row in rows:
            payload = {
                **row,
                "embedding": _vector_literal(row["embedding"]),
                "meta": Jsonb(row.get("meta") or {}),
            }
            batch.append(payload)
            if len(batch) >= batch_size:
                cur.executemany(sql, batch)
                count += len(batch)
                batch.clear()
        if batch:
            cur.executemany(sql, batch)
            count += len(batch)
    conn.commit()
    return count


def search(
    conn: psycopg.Connection,
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
) -> list[ChunkHit]:
    """Pure dense vector search (cosine)."""
    sql = """
        SELECT
            id,
            context_id,
            subset,
            file_name,
            chunk_index,
            content,
            1 - (embedding <=> %(embedding)s::vector) AS score
        FROM t2_chunks
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(top_k)s
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            {"embedding": _vector_literal(query_embedding), "top_k": top_k},
        )
        rows = cur.fetchall()
    return [_row_to_hit(row) for row in rows]


def fetch_all_chunks(conn: psycopg.Connection) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, context_id, subset, file_name, chunk_index, content
            FROM t2_chunks
            ORDER BY id
            """
        )
        return list(cur.fetchall())


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@lru_cache(maxsize=4)
def _bm25_bundle(database: str) -> tuple[tuple[str, ...], BM25Okapi, tuple[dict[str, Any], ...]]:
    """Cache BM25 index keyed by DSN (invalidate by calling ``clear_bm25_cache``)."""
    with psycopg.connect(database, row_factory=dict_row) as conn:
        rows = fetch_all_chunks(conn)
    if not rows:
        empty = BM25Okapi([["__empty__"]])
        return ((), empty, ())
    tokenized = [tokenize(r["content"]) or ["__empty__"] for r in rows]
    bm25 = BM25Okapi(tokenized)
    ids = tuple(r["id"] for r in rows)
    return ids, bm25, tuple(rows)


def clear_bm25_cache() -> None:
    _bm25_bundle.cache_clear()


def search_bm25(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 3,
) -> list[ChunkHit]:
    """Okapi BM25 over chunk text (corpus loaded from Postgres)."""
    _ = conn  # connection ensures DB is up; corpus is cached from DSN
    ids, bm25, rows = _bm25_bundle(database_url())
    if not rows:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits: list[ChunkHit] = []
    for i in ranked_idx:
        row = rows[i]
        hits.append(
            ChunkHit(
                chunk_id=row["id"],
                context_id=row["context_id"],
                subset=row["subset"],
                file_name=row["file_name"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                score=float(scores[i]),
            )
        )
    return hits


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ChunkHit]],
    *,
    top_k: int = 3,
    rrf_k: int = RRF_K,
) -> list[ChunkHit]:
    """Merge ranked hit lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    best: dict[str, ChunkHit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            prev = best.get(hit.chunk_id)
            if prev is None or hit.score > prev.score:
                best[hit.chunk_id] = hit
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    fused: list[ChunkHit] = []
    for chunk_id, score in ordered:
        base = best[chunk_id]
        fused.append(
            ChunkHit(
                chunk_id=base.chunk_id,
                context_id=base.context_id,
                subset=base.subset,
                file_name=base.file_name,
                chunk_index=base.chunk_index,
                content=base.content,
                score=score,
            )
        )
    return fused


def search_hybrid(
    conn: psycopg.Connection,
    query: str,
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
    candidate_k: int | None = None,
) -> list[ChunkHit]:
    """BM25 + dense vector fused with RRF."""
    pool = candidate_k or max(top_k * 4, 20)
    vector_hits = search(conn, query_embedding, top_k=pool)
    bm25_hits = search_bm25(conn, query, top_k=pool)
    return reciprocal_rank_fusion([vector_hits, bm25_hits], top_k=top_k)


def chunk_count(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM t2_chunks")
        return int(cur.fetchone()["n"])


def ranked_context_ids(hits: Sequence[ChunkHit], *, k: int = 3) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.context_id in seen:
            continue
        seen.add(hit.context_id)
        ranked.append(hit.context_id)
        if len(ranked) >= k:
            break
    return ranked


def _row_to_hit(row: dict[str, Any]) -> ChunkHit:
    return ChunkHit(
        chunk_id=row["id"],
        context_id=row["context_id"],
        subset=row["subset"],
        file_name=row["file_name"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        score=float(row["score"]),
    )


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"
