"""Multi-query RAG: LLM query rewrites + hybrid/vector search fusion.

Port of `04_MultiQueryRAG` (custom rewriter path with generic prompt).
"""

from __future__ import annotations

import json
from typing import Any

from azure.search.documents.models import VectorizedQuery

from foundry_rag.clients import get_search_client
from foundry_rag.ingestion.index_builder import load_index_definition
from foundry_rag.llm import chat_complete, embed_text

SELECT_FIELDS = ["Id", "Title", "Category", "Overview", "Features"]


def get_generic_rewrite_system_prompt(count: int) -> str:
    return f"""\
You are a query rewriting assistant for Retrieval-Augmented Generation (RAG).
Your task is to take a user’s original question and generate {count} alternative versions of that query.
Each rewritten query should preserve the user’s intent while exploring different phrasings, clarifications, or interpretations that might help retrieve more relevant FAQ answers.
Return the rewrites in a structured JSON array called "Rewrites".

The JSON must follow this exact structure:
{{
    "Rewrites": [
        "rewrite 1",
        "rewrite 2",
        "rewrite 3"
    ]
}}

The "Rewrites" array must contain exactly {count} items.
"""


def rewrite_queries(question: str, count: int = 5) -> list[str]:
    raw = chat_complete(
        system=get_generic_rewrite_system_prompt(count),
        user=f"User's original question: {question}",
        max_output_tokens=500,
    )
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    rewrites = payload.get("Rewrites") or payload.get("rewrites") or []
    return [str(r) for r in rewrites][:count]


def _search_once(
    *,
    client,
    text: str | None,
    vector: list[float] | None,
    top: int,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "search_text": text,
        "select": SELECT_FIELDS,
        "top": top,
        "query_type": "simple",
    }
    if vector is not None:
        kwargs["vector_queries"] = [
            VectorizedQuery(
                vector=vector,
                k_nearest_neighbors=top,
                fields="OverviewVector",
            )
        ]
    return [dict(item) for item in client.search(**kwargs)]


def fuse_by_max_score(result_lists: list[list[dict[str, Any]]], top_n: int = 3) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for docs in result_lists:
        for doc in docs:
            doc_id = doc.get("Id")
            if not doc_id:
                continue
            score = float(doc.get("@search.score") or 0.0)
            prev = best.get(doc_id)
            if prev is None or score > float(prev.get("@search.score") or 0.0):
                best[doc_id] = doc
    ranked = sorted(
        best.values(),
        key=lambda d: float(d.get("@search.score") or 0.0),
        reverse=True,
    )
    return ranked[:top_n]


def multiquery_search(
    question: str,
    *,
    module_folder: str = "04_multiquery_rag",
    mode: str = "custom_hybrid",
    rewrite_count: int = 5,
    top: int = 3,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Modes: none_full_text, none_vector, custom_full_text, custom_hybrid."""
    index_name = load_index_definition(module_folder)["name"]
    client = get_search_client(index_name)

    if mode.startswith("none_"):
        rewrites = [question]
    else:
        rewrites = rewrite_queries(question, count=rewrite_count)

    lists: list[list[dict[str, Any]]] = []
    for rewrite in rewrites:
        if mode.endswith("full_text") or mode == "none_full_text":
            lists.append(_search_once(client=client, text=rewrite, vector=None, top=top))
        else:
            vector = embed_text(rewrite)
            # hybrid-ish: text + vector when custom_hybrid; vector-only for none_vector
            text = rewrite if "hybrid" in mode or "full_text" in mode else None
            if mode == "none_vector":
                text = None
            if mode == "custom_full_text":
                vector = None
            lists.append(_search_once(client=client, text=text, vector=vector, top=top))

    return rewrites, fuse_by_max_score(lists, top_n=top)
