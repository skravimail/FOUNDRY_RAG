"""Embedding-similarity sentence grouping chunker.

Port of `06_ChunkingStrategies/SemanticChunker.cs`.
"""

from __future__ import annotations

from foundry_rag.chunking.semantic_base import SemanticChunkerBase
from foundry_rag.llm import embed_texts


class SemanticChunker(SemanticChunkerBase):
    def create_chunks(
        self,
        text: str,
        *,
        similarity_threshold: float = 0.75,
        max_tokens_per_chunk: int = 1024,
    ) -> list[str]:
        sentences = self.split_into_sentences(text)
        if len(sentences) <= 1:
            return sentences

        embeddings = embed_texts(sentences)
        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_tokens = self.count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            sentence_tokens = self.count_tokens(sentences[i])
            similarity = self.cosine_similarity(embeddings[i - 1], embeddings[i])
            if (
                similarity < similarity_threshold
                or current_tokens + sentence_tokens > max_tokens_per_chunk
            ):
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentences[i])
            current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks
