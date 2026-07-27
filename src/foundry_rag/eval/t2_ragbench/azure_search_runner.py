"""Azure AI Search RAG runner for T²-RAGBench eval."""

from __future__ import annotations

from typing import Any, Literal

from foundry_rag.llm import chat_complete, embed_text
from foundry_rag.mechanisms.azure_search_store import (
    ranked_context_ids,
    search,
    search_bm25,
    search_hybrid,
)

RetrievalMode = Literal["vector", "bm25", "hybrid"]

METHOD_NAMES = {
    "vector": "Azure AI Search (vector)",
    "bm25": "Azure AI Search (BM25)",
    "hybrid": "Azure AI Search (BM25+vector)",
}

METHOD_NAME = METHOD_NAMES["hybrid"]

SYSTEM_PROMPT = (
    "You answer financial questions using ONLY the provided document excerpts. "
    "Reply with a short answer that includes the final numeric value when applicable. "
    "Do not invent figures that are not supported by the context."
)


def run_azure_search(
    question: str,
    *,
    top_k: int = 3,
    retrieval: RetrievalMode = "hybrid",
    generate: bool = True,
) -> dict[str, Any]:
    fetch_k = max(top_k * 4, 8)
    if retrieval == "vector":
        hits = search(embed_text(question), top_k=fetch_k)
    elif retrieval == "bm25":
        hits = search_bm25(question, top_k=fetch_k)
    else:
        hits = search_hybrid(question, embed_text(question), top_k=fetch_k)

    ranked = ranked_context_ids(hits, k=top_k)
    chosen = []
    seen: set[str] = set()
    for hit in hits:
        if hit.context_id in ranked and hit.context_id not in seen:
            chosen.append(hit)
            seen.add(hit.context_id)
        if len(chosen) >= top_k:
            break

    answer = ""
    if generate:
        context_blocks = [
            f"[Doc {i} | context_id={hit.context_id} | {hit.file_name}]\n{hit.content}"
            for i, hit in enumerate(chosen, start=1)
        ]
        user = (
            "Context:\n"
            + ("\n\n".join(context_blocks) if context_blocks else "(no retrieved context)")
            + f"\n\nQuestion: {question}\n\nFinal answer:"
        )
        answer = chat_complete(system=SYSTEM_PROMPT, user=user, max_output_tokens=400)
    return {
        "method": METHOD_NAMES[retrieval],
        "answer": answer,
        "ranked_context_ids": ranked,
        "raw": {
            "retrieval": retrieval,
            "generate": generate,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "context_id": h.context_id,
                    "file_name": h.file_name,
                    "score": h.score,
                    "content_preview": h.content[:240],
                }
                for h in hits[:fetch_k]
            ],
        },
    }
