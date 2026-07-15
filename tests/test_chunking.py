"""Unit tests for FixedSizeChunker (no Azure required)."""

from foundry_rag.chunking.fixed import FixedSizeChunker


def test_fixed_size_chunker_overlap_and_stride():
    chunker = FixedSizeChunker()
    # ~100 tokens of repetitive text
    text = " ".join(["alpha"] * 200)
    chunks = chunker.create_chunks(text, chunk_size=50, overlap_percentage=20.0)
    assert len(chunks) >= 2
    # Each chunk should decode to something containing alpha
    assert all("alpha" in c for c in chunks)


def test_fixed_size_rejects_full_overlap():
    chunker = FixedSizeChunker()
    try:
        chunker.create_chunks("hello world", chunk_size=10, overlap_percentage=100.0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "stride" in str(exc).lower() or "overlap" in str(exc).lower()


def test_sentence_split_primitive():
    from foundry_rag.chunking.semantic_base import SemanticChunkerBase

    sentences = SemanticChunkerBase.split_into_sentences(
        "Hello world. How are you? Fine!"
    )
    assert sentences == ["Hello world.", "How are you?", "Fine!"]
