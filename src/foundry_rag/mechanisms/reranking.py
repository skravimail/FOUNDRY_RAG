"""Re-ranking RAG: hybrid search + Azure AI Search semantic ranker.

Port of `03_ReRankingRAG/ReRankingRAGExample.cs`.
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

SELECT_FIELDS = ["Id", "Title", "Category", "Overview", "Features"]


def create_user_prompt(question: str, docs: list[dict[str, Any]]) -> str:
    parts = [
        "Here are the documents related to the question.",
        "Use only the information in these documents when answering.",
        "",
        "=== Retrieved Documents ===",
        "",
    ]
    for i, doc in enumerate(docs, start=1):
        features = doc.get("Features") or []
        feature_text = ", ".join(features) if isinstance(features, list) else str(features)
        parts.extend(
            [
                f"[Document {i}]",
                f"Id: {doc.get('Id')}",
                f"Title: {doc.get('Title')}",
                f"Category: {doc.get('Category')}",
                f"Overview: {doc.get('Overview')}",
                f"Features: {feature_text}",
                "",
            ]
        )
    parts.extend(["=== User Question ===", question])
    return "\n".join(parts)


def rerank_search(
    question: str,
    *,
    module_folder: str = "03_reranking_rag",
    mode: str = "hybrid_semantic",
    request_size: int = 10,
    keep: int = 3,
) -> list[dict[str, Any]]:
    """Modes mirror the reference menu:

    - full_text
    - hybrid
    - hybrid_semantic
    - hybrid_semantic_scoring
    """
    index_name = load_index_definition(module_folder)["name"]
    client = get_search_client(index_name)
    query_vector = embed_text(question)

    use_semantic = "semantic" in mode
    use_vector = mode != "full_text"
    use_scoring = "scoring" in mode

    kwargs: dict[str, Any] = {
        "search_text": question,
        "select": SELECT_FIELDS,
        "top": request_size,
        "include_total_count": True,
        "query_type": "semantic" if use_semantic else "simple",
    }
    if use_semantic:
        kwargs["semantic_configuration_name"] = "default_semantic_config"
        kwargs["query_caption"] = "extractive"
        kwargs["query_answer"] = "extractive"
    if use_vector:
        kwargs["vector_queries"] = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=request_size,
                fields="OverviewVector",
            )
        ]
    if use_scoring:
        kwargs["scoring_profile"] = "boost_category_field"
        kwargs["scoring_parameters"] = ["tagBoostCategory-Luxury"]

    results = client.search(**kwargs)
    docs: list[dict[str, Any]] = []
    for item in results:
        doc = dict(item)
        docs.append(doc)
        if len(docs) >= keep:
            break
    return docs


def answer_question(
    question: str,
    *,
    module_folder: str = "03_reranking_rag",
    mode: str = "hybrid_semantic",
) -> tuple[str, list[dict[str, Any]]]:
    docs = rerank_search(question, module_folder=module_folder, mode=mode)
    answer = chat_complete(
        system=SYSTEM_PROMPT,
        user=create_user_prompt(question, docs),
        max_output_tokens=600,
    )
    return answer, docs
