"""Unit tests for BM25+vector RRF fusion."""

from foundry_rag.mechanisms.pgvector_store import ChunkHit, reciprocal_rank_fusion, tokenize


def _hit(chunk_id: str, context_id: str, score: float = 1.0) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        context_id=context_id,
        subset="FinQA",
        file_name="x.pdf",
        chunk_index=0,
        content="text",
        score=score,
    )


def test_tokenize_splits_words():
    assert tokenize("Entergy Mississippi, Inc. 2017") == [
        "entergy",
        "mississippi",
        "inc",
        "2017",
    ]


def test_rrf_prefers_agreement():
    vector = [_hit("a", "ctx_a"), _hit("b", "ctx_b"), _hit("c", "ctx_c")]
    bm25 = [_hit("b", "ctx_b"), _hit("a", "ctx_a"), _hit("d", "ctx_d")]
    fused = reciprocal_rank_fusion([vector, bm25], top_k=3)
    # a and b appear in both lists → should outrank unique c/d.
    top_ids = [h.chunk_id for h in fused]
    assert top_ids[0] in {"a", "b"}
    assert "a" in top_ids and "b" in top_ids
