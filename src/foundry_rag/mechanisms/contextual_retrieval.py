"""Contextual retrieval: LLM situating context prepended before embedding.

Port of `07_ContextualRetrieval/ContextualRetrievalExample.cs`.
"""

from __future__ import annotations

from dataclasses import dataclass

from foundry_rag.chunking.fixed import FixedSizeChunker
from foundry_rag.llm import chat_complete, embed_text


@dataclass(frozen=True)
class EnrichedChunk:
    context: str
    chunk: str

    @property
    def enriched_text(self) -> str:
        return f"{self.context} \n {self.chunk}"


def get_context_enrichment_prompt(document_content: str, chunk_content: str) -> str:
    # Exact prompt wording from the reference ContextualRetrievalExample.cs
    return f"""### System Instructions
You are a retrieval-augmentation specialist. Your goal is to prepend a 1-sentence situational context to a text chunk.
Focus on: 
1. Who/What/Where (Entities).
2. Document Subject (The 'Global' context).
3. Critical IDs or Dates.
4. Versions/References

### Document
{document_content}

### Chunk
{chunk_content}

### Output Requirement
Provide ONLY the 1-sentence context. Do not include 'The context is...' or any preamble. 
Goal: Improve keyword and vector match for search.
"""


def enrich_chunk(document_content: str, chunk_content: str) -> EnrichedChunk:
    prompt = get_context_enrichment_prompt(document_content, chunk_content)
    context = chat_complete(user=prompt, max_output_tokens=200)
    return EnrichedChunk(context=context, chunk=chunk_content)


def enrich_document_chunks(
    document_content: str,
    *,
    chunk_size: int = 512,
    overlap_percentage: float = 25.0,
    max_chunks: int | None = None,
) -> list[tuple[EnrichedChunk, list[float]]]:
    chunker = FixedSizeChunker()
    chunks = chunker.create_chunks(
        document_content,
        chunk_size=chunk_size,
        overlap_percentage=overlap_percentage,
    )
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    results: list[tuple[EnrichedChunk, list[float]]] = []
    for chunk in chunks:
        enriched = enrich_chunk(document_content, chunk)
        vector = embed_text(enriched.enriched_text)
        results.append((enriched, vector))
    return results
