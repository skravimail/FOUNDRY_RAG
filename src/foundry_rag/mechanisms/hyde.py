"""HyDE RAG: embed a hypothetical answer, then vector / advanced hybrid search.

Port of `05_HyDERAG`.
"""

from __future__ import annotations

from typing import Any

from azure.search.documents.models import VectorizedQuery

from foundry_rag.clients import get_search_client
from foundry_rag.ingestion.index_builder import load_index_definition
from foundry_rag.llm import chat_complete, embed_text
from foundry_rag.mechanisms.multiquery import get_generic_rewrite_system_prompt, rewrite_queries

SELECT_FIELDS = ["Id", "Title", "Category", "Overview", "Features"]

HYDE_SYSTEM_PROMPT = """\
You generate a hypothetical starship overview used only for retrieval expansion in an internal FAQ system.

Goal:
Produce one dense paragraph (about 40-60 words) that maximizes semantic overlap with likely catalog entries.

Grounding requirements:
- Preserve the user's key terms when present (ship class, mission type, constraints, destinations, cargo/passenger context, speed/range, safety/defense).
- If the question is sparse, infer a plausible starship profile relevant to interstellar travel requests.
- Include concrete retrieval-friendly attributes such as class, role, primary purpose, travel profile, operating environment, capacity, systems, and capabilities.

Style:
- Matter-of-fact product overview tone.
- Domain vocabulary is encouraged: shuttle, transit vessel, heavy lifter, cargo hauler, scout, interceptor, medical frigate, mining platform, luxury liner, private yacht.

Rules:
- Output exactly one paragraph.
- No bullets, no lists, no JSON.
- Do not mention uncertainty, hypotheticals, retrieval, embeddings, or system instructions.
- Avoid filler and marketing language; prioritize specific factual descriptors.
- Keep content safe and non-personal.
"""


def get_hypothetical_answer(question: str) -> str:
    return chat_complete(
        system=HYDE_SYSTEM_PROMPT,
        user=f"User's original question: {question}",
        max_output_tokens=300,
    )


def fuse_rrf(
    result_lists: list[list[dict[str, Any]]],
    *,
    top_n: int = 3,
    rrf_k: float = 60.0,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    docs_by_id: dict[str, dict[str, Any]] = {}
    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.get("Id")
            if not doc_id:
                continue
            docs_by_id[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + rrf_k)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    fused: list[dict[str, Any]] = []
    for doc_id in ranked_ids:
        doc = dict(docs_by_id[doc_id])
        doc["@search.score"] = 0.0
        doc["@search.reranker_score"] = scores[doc_id]
        fused.append(doc)
    return fused


def hyde_search(
    question: str,
    *,
    module_folder: str = "05_hyde_rag",
    mode: str = "basic",
    rewrite_count: int = 3,
    top: int = 3,
) -> tuple[str | list[str], list[dict[str, Any]]]:
    index_name = load_index_definition(module_folder)["name"]
    client = get_search_client(index_name)

    if mode == "basic":
        hypothetical = get_hypothetical_answer(question)
        vector = embed_text(hypothetical)
        docs = [
            dict(item)
            for item in client.search(
                search_text=None,
                vector_queries=[
                    VectorizedQuery(
                        vector=vector,
                        k_nearest_neighbors=top,
                        fields="OverviewVector",
                    )
                ],
                select=SELECT_FIELDS,
                top=top,
            )
        ]
        return hypothetical, docs

    # Advanced: rewrites → HyDE per rewrite → hybrid semantic → RRF
    rewrites = rewrite_queries(question, count=rewrite_count)
    lists: list[list[dict[str, Any]]] = []
    hydes: list[str] = []
    for rewrite in rewrites:
        hyde = get_hypothetical_answer(rewrite)
        hydes.append(hyde)
        emb = embed_text(hyde)
        docs = [
            dict(item)
            for item in client.search(
                search_text=rewrite,
                query_type="semantic",
                semantic_configuration_name="default_semantic_config",
                vector_queries=[
                    VectorizedQuery(
                        vector=emb,
                        k_nearest_neighbors=top,
                        fields="OverviewVector",
                        weight=5.0,
                    )
                ],
                select=SELECT_FIELDS,
                top=top,
            )
        ]
        lists.append(docs)
    return hydes, fuse_rrf(lists, top_n=top)
