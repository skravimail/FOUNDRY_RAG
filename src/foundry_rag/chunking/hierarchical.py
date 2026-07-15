"""Parent-child hierarchical chunker.

Port of `06_ChunkingStrategies/HierarchicalChunker.cs`.
"""

from __future__ import annotations

from foundry_rag.chunking.semantic_base import SemanticChunkerBase
from foundry_rag.llm import embed_texts


class HierarchicalChunker(SemanticChunkerBase):
    def create_parent_child_chunks(
        self,
        text: str,
        *,
        similarity_threshold: float = 0.75,
        max_tokens_per_parent_chunk: int = 1024,
        max_tokens_per_child_chunk: int = 256,
    ) -> list[tuple[str, str]]:
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        embeddings = embed_texts(sentences)

        parent_groups: list[list[int]] = []
        current_parent = [0]
        current_parent_tokens = self.count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            sentence_tokens = self.count_tokens(sentences[i])
            similarity = self.cosine_similarity(embeddings[i - 1], embeddings[i])
            if (
                similarity < similarity_threshold
                or current_parent_tokens + sentence_tokens > max_tokens_per_parent_chunk
            ):
                parent_groups.append(current_parent)
                current_parent = []
                current_parent_tokens = 0
            current_parent.append(i)
            current_parent_tokens += sentence_tokens

        if current_parent:
            parent_groups.append(current_parent)

        result: list[tuple[str, str]] = []
        for parent_indices in parent_groups:
            parent_text = " ".join(sentences[i] for i in parent_indices)
            current_child = [parent_indices[0]]
            current_child_tokens = self.count_tokens(sentences[parent_indices[0]])

            for j in range(1, len(parent_indices)):
                idx = parent_indices[j]
                prev_idx = parent_indices[j - 1]
                sentence_tokens = self.count_tokens(sentences[idx])
                similarity = self.cosine_similarity(embeddings[prev_idx], embeddings[idx])
                if (
                    similarity < similarity_threshold
                    or current_child_tokens + sentence_tokens > max_tokens_per_child_chunk
                ):
                    child_text = " ".join(sentences[i] for i in current_child)
                    result.append((parent_text, child_text))
                    current_child = []
                    current_child_tokens = 0
                current_child.append(idx)
                current_child_tokens += sentence_tokens

            if current_child:
                child_text = " ".join(sentences[i] for i in current_child)
                result.append((parent_text, child_text))

        return result
