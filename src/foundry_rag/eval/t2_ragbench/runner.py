"""Run Foundry IQ (KB retrieve) and Oracle Context baselines for T²-RAGBench."""

from __future__ import annotations

from typing import Any

from foundry_rag.eval.t2_ragbench.parse import extract_answer_text, ranked_context_ids_from_result
from foundry_rag.eval.t2_ragbench.azure_search_runner import METHOD_NAME as AZURE_SEARCH_METHOD
from foundry_rag.eval.t2_ragbench.azure_search_runner import run_azure_search
from foundry_rag.eval.t2_ragbench.lancedb_runner import METHOD_NAME as LANCEDB_METHOD
from foundry_rag.eval.t2_ragbench.lancedb_runner import run_lancedb
from foundry_rag.eval.t2_ragbench.pgvector_runner import METHOD_NAME as PGVECTOR_METHOD
from foundry_rag.eval.t2_ragbench.pgvector_runner import run_pgvector
from foundry_rag.llm import chat_complete
from foundry_rag.mechanisms.foundryiq import retrieve_foundryiq

__all__ = [
    "AZURE_SEARCH_METHOD",
    "LANCEDB_METHOD",
    "PGVECTOR_METHOD",
    "run_azure_search",
    "run_foundryiq",
    "run_lancedb",
    "run_oracle",
    "run_pgvector",
]


def run_foundryiq(question: str, *, top_k: int = 3) -> dict[str, Any]:
    """KB retrieve + answer synthesis; return answer and ranked context_ids."""
    payload = retrieve_foundryiq(question)
    raw = payload.get("raw")
    answer = payload.get("answer") or extract_answer_text(raw)
    ranked = ranked_context_ids_from_result(raw, k=top_k)
    return {
        "method": "FoundryIQ / Knowledge Base",
        "answer": answer,
        "ranked_context_ids": ranked,
        "raw": raw,
    }


def run_oracle(question: str, context: str, gold_context_id: str) -> dict[str, Any]:
    """Answer from gold context only (retrieval ceiling: MRR/R = 1.0)."""
    system = (
        "You answer financial questions using ONLY the provided document context. "
        "Reply with a short answer that includes the final numeric value when applicable. "
        "Do not invent figures that are not supported by the context."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    answer = chat_complete(system=system, user=user, max_output_tokens=400)
    return {
        "method": "Oracle Context",
        "answer": answer,
        "ranked_context_ids": [gold_context_id],
        "raw": None,
    }
