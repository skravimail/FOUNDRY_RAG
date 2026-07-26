"""Azure AI Search store for T²-RAGBench chunk retrieval.

Hybrid queries use Search ``search_text`` (BM25) + ``VectorizedQuery`` (RRF
fusion inside Azure AI Search). Vector-only / BM25-only modes are also exposed
for ablations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from foundry_rag.clients import get_search_client, get_search_index_client
from foundry_rag.mechanisms.pgvector_store import ChunkHit, ranked_context_ids

DEFAULT_INDEX_NAME = "t2-ragbench-chunks"
EMBEDDING_DIM = int(os.environ.get("FOUNDRY_EMBEDDING_DIMENSIONS", "1536"))
VECTOR_PROFILE = "t2-hnsw-profile"
VECTOR_ALGO = "t2-hnsw"
RetrievalMode = Literal["vector", "bm25", "hybrid"]

# Azure Search document keys allow letters, digits, underscore, dash, equal.
_KEY_SAFE = re.compile(r"[^A-Za-z0-9_\-=]+")


def index_name() -> str:
    return os.environ.get("T2_AZURE_SEARCH_INDEX", DEFAULT_INDEX_NAME)


def document_key(chunk_id: str) -> str:
    """Map ``context_id::chunk_index`` to a Search-safe key."""
    return _KEY_SAFE.sub("_", chunk_id)


def ensure_index(*, embedding_dim: int = EMBEDDING_DIM, name: str | None = None) -> SearchIndex:
    """Create or update the T² chunk index (BM25 text + HNSW vectors)."""
    idx = name or index_name()
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="context_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="subset", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGO)],
        profiles=[
            VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=VECTOR_ALGO)
        ],
    )
    index = SearchIndex(name=idx, fields=fields, vector_search=vector_search)
    return get_search_index_client().create_or_update_index(index)


def delete_index(*, name: str | None = None) -> None:
    idx = name or index_name()
    client = get_search_index_client()
    try:
        client.delete_index(idx)
    except Exception as exc:  # noqa: BLE001 — ignore missing index on reset
        if "not found" not in str(exc).lower() and "ResourceNotFound" not in type(exc).__name__:
            raise


def upsert_chunks(
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int = 64,
    name: str | None = None,
) -> int:
    client = get_search_client(name or index_name())
    batch: list[dict[str, Any]] = []
    count = 0
    for row in rows:
        batch.append(
            {
                "id": document_key(row["id"]),
                "chunk_id": row["id"],
                "context_id": row["context_id"],
                "subset": row.get("subset") or "",
                "file_name": row.get("file_name") or "",
                "chunk_index": int(row.get("chunk_index") or 0),
                "content": row["content"],
                "content_vector": list(row["embedding"]),
            }
        )
        if len(batch) >= batch_size:
            count += _upload_batch(client, batch)
            batch.clear()
    if batch:
        count += _upload_batch(client, batch)
    return count


def _upload_batch(client: Any, batch: list[dict[str, Any]]) -> int:
    result = client.upload_documents(documents=batch)
    failed = [r for r in result if not r.succeeded]
    if failed:
        raise RuntimeError(f"Azure Search upload failed for {len(failed)} docs: {failed[:3]}")
    return len(batch)


def chunk_count(*, name: str | None = None) -> int:
    client = get_search_client(name or index_name())
    # get_document_count is available on SearchClient
    return int(client.get_document_count())


def search(
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
    name: str | None = None,
) -> list[ChunkHit]:
    """Pure dense vector search."""
    client = get_search_client(name or index_name())
    vector_query = VectorizedQuery(
        vector=list(query_embedding),
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )
    results = client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["chunk_id", "context_id", "subset", "file_name", "chunk_index", "content"],
        top=top_k,
    )
    return [_doc_to_hit(doc) for doc in results]


def search_bm25(
    query: str,
    *,
    top_k: int = 3,
    name: str | None = None,
) -> list[ChunkHit]:
    """Full-text BM25 over ``content``."""
    client = get_search_client(name or index_name())
    results = client.search(
        search_text=query or "*",
        vector_queries=None,
        select=["chunk_id", "context_id", "subset", "file_name", "chunk_index", "content"],
        top=top_k,
        query_type="simple",
        search_fields=["content"],
    )
    return [_doc_to_hit(doc) for doc in results]


def search_hybrid(
    query: str,
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
    name: str | None = None,
) -> list[ChunkHit]:
    """BM25 + vector hybrid (Azure Search RRF)."""
    client = get_search_client(name or index_name())
    vector_query = VectorizedQuery(
        vector=list(query_embedding),
        k_nearest_neighbors=max(top_k, 8),
        fields="content_vector",
    )
    results = client.search(
        search_text=query or "*",
        vector_queries=[vector_query],
        select=["chunk_id", "context_id", "subset", "file_name", "chunk_index", "content"],
        top=top_k,
        query_type="simple",
        search_fields=["content"],
    )
    return [_doc_to_hit(doc) for doc in results]


def _doc_to_hit(doc: dict[str, Any]) -> ChunkHit:
    return ChunkHit(
        chunk_id=str(doc.get("chunk_id") or doc.get("id") or ""),
        context_id=str(doc.get("context_id") or ""),
        subset=str(doc.get("subset") or ""),
        file_name=str(doc.get("file_name") or ""),
        chunk_index=int(doc.get("chunk_index") or 0),
        content=str(doc.get("content") or ""),
        score=float(doc.get("@search.score") or 0.0),
    )


__all__ = [
    "ChunkHit",
    "DEFAULT_INDEX_NAME",
    "RetrievalMode",
    "chunk_count",
    "delete_index",
    "document_key",
    "ensure_index",
    "index_name",
    "ranked_context_ids",
    "search",
    "search_bm25",
    "search_hybrid",
    "upsert_chunks",
]
