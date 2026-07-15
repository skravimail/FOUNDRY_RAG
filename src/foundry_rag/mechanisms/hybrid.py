"""Hybrid RAG: BM25 + vector search on Azure AI Search.

Port of `02_HybridRAG` SearchService / HybridRagExample behavior.
"""

from __future__ import annotations

from typing import Any

from azure.search.documents.models import VectorizedQuery

from foundry_rag.clients import get_search_client
from foundry_rag.ingestion.index_builder import load_index_definition
from foundry_rag.llm import chat_complete, embed_text

SYSTEM_PROMPT = """\
You are a helpful assistant for the Galactic Voyages travel agency.
Answer questions using only the information provided in the documents.
Keep your answers clear and friendly.
"""

SELECT_FIELDS = [
    "Id",
    "Title",
    "ProductId",
    "Category",
    "Overview",
    "TopSpeed",
    "Fuel",
    "Seats",
    "ArtificialGravity",
    "Features",
    "Notes",
]


def _format_doc(doc: dict[str, Any], index: int) -> str:
    features = doc.get("Features") or []
    feature_text = ", ".join(features) if isinstance(features, list) else str(features)
    return "\n".join(
        [
            f"[Document {index}]",
            f"Id: {doc.get('Id')}",
            f"Title: {doc.get('Title')}",
            f"ProductId: {doc.get('ProductId')}",
            f"Category: {doc.get('Category')}",
            f"Overview: {doc.get('Overview')}",
            f"TopSpeed: {doc.get('TopSpeed')}",
            f"Fuel: {doc.get('Fuel')}",
            f"Seats: {doc.get('Seats')}",
            f"ArtificialGravity: {doc.get('ArtificialGravity')}",
            f"Features: {feature_text}",
            f"Notes: {doc.get('Notes')}",
        ]
    )


def create_user_prompt(question: str, docs: list[dict[str, Any]]) -> str:
    parts = [
        "Here are the documents related to the question.",
        "Use only the information in these documents when answering.",
        "",
        "=== Retrieved Documents ===",
        "",
    ]
    for i, doc in enumerate(docs, start=1):
        parts.append(_format_doc(doc, i))
        parts.append("")
    parts.extend(["=== User Question ===", question])
    return "\n".join(parts)


def hybrid_search(
    question: str,
    *,
    module_folder: str = "02_hybrid_rag",
    top: int = 3,
    mode: str = "hybrid_overview",
) -> list[dict[str, Any]]:
    """Run one of the reference search modes.

    Modes:
      - full_text
      - vector_overview
      - hybrid_overview  (BM25 + OverviewVector, weight=2.0)
      - vector_both
      - hybrid_both
    """
    index_name = load_index_definition(module_folder)["name"]
    client = get_search_client(index_name)
    query_vector = embed_text(question)

    search_text: str | None = question
    vector_queries: list[VectorizedQuery] = []

    if mode == "full_text":
        search_text = question
    elif mode == "vector_overview":
        search_text = None
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="OverviewVector",
            )
        ]
    elif mode == "hybrid_overview":
        search_text = question
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="OverviewVector",
                weight=2.0,
            )
        ]
    elif mode == "vector_both":
        search_text = None
        vector_queries = [
            VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top, fields="OverviewVector"
            ),
            VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top, fields="NotesVector"
            ),
        ]
    elif mode == "hybrid_both":
        search_text = question
        vector_queries = [
            VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top, fields="OverviewVector"
            ),
            VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top, fields="NotesVector"
            ),
        ]
    else:
        raise ValueError(f"Unknown hybrid search mode: {mode}")

    results = client.search(
        search_text=search_text,
        vector_queries=vector_queries or None,
        select=SELECT_FIELDS,
        top=top,
        query_type="simple",
        search_mode="any",
    )
    docs: list[dict[str, Any]] = []
    for item in results:
        doc = dict(item)
        doc["@search.score"] = item.get("@search.score")
        docs.append(doc)
    return docs


def answer_question(
    question: str,
    *,
    module_folder: str = "02_hybrid_rag",
    mode: str = "hybrid_overview",
) -> tuple[str, list[dict[str, Any]]]:
    docs = hybrid_search(question, module_folder=module_folder, mode=mode)
    answer = chat_complete(
        system=SYSTEM_PROMPT,
        user=create_user_prompt(question, docs),
        max_output_tokens=600,
    )
    return answer, docs
