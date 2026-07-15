"""FoundryIQ / Knowledge Base agentic retrieval.

Port of `09_FoundryIQ/FoundryIQExample.cs` using Azure AI Search Knowledge Bases.
Falls back to documenting missing KB configuration clearly when env vars are unset.
"""

from __future__ import annotations

import os
from typing import Any

from azure.identity import DefaultAzureCredential

from foundry_rag import config

DEFAULT_QUERY = (
    "What are the requirements for the CTO position in Contoso and what is the data "
    "retention policy for the low business value data in Contoso and can I visit a dentist for free?"
)

KNOWLEDGE_SOURCES = [
    "knowledge-source-contoso-cloud",
    "knowledge-source-health-plans",
    "knowledge-source-job-roles",
]


def retrieve_foundryiq(question: str = DEFAULT_QUERY) -> dict[str, Any]:
    """Call Azure AI Search Knowledge Base retrieval (AnswerSynthesis)."""
    endpoint = config.AZURE_AI_SEARCH_ENDPOINT or os.environ.get("AZURE_AI_SEARCH_URI")
    kb_name = os.environ.get("AZURE_AI_SEARCH_KNOWLEDGE_BASE")
    missing = []
    if not endpoint:
        missing.append("AZURE_AI_SEARCH_ENDPOINT (or AZURE_AI_SEARCH_URI)")
    if not kb_name:
        missing.append("AZURE_AI_SEARCH_KNOWLEDGE_BASE")
    if missing:
        raise RuntimeError(
            "FoundryIQ missing env: "
            + ", ".join(missing)
            + f". Provision a Knowledge Base with sources {KNOWLEDGE_SOURCES}."
        )

    # Prefer the dedicated KnowledgeBases client when available in the installed SDK.
    try:
        from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient
        from azure.search.documents.knowledgebases.models import (
            AzureBlobKnowledgeSourceParams,
            KnowledgeBaseMessage,
            KnowledgeBaseMessageTextContent,
            KnowledgeBaseRetrievalRequest,
            KnowledgeRetrievalMediumReasoningEffort,
            KnowledgeRetrievalOutputMode,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "azure-search-documents Knowledge Bases API is not available in this SDK build. "
            "Upgrade azure-search-documents or provision the KB and use the REST API."
        ) from exc

    client = KnowledgeBaseRetrievalClient(
        endpoint=endpoint,
        knowledge_base_name=kb_name,
        credential=DefaultAzureCredential(),
    )
    request = KnowledgeBaseRetrievalRequest(
        include_activity=True,
        knowledge_source_params=[
            AzureBlobKnowledgeSourceParams(knowledge_source_name=name, always_query_source=False)
            for name in KNOWLEDGE_SOURCES
        ],
        messages=[
            KnowledgeBaseMessage(
                role="user",
                content=[KnowledgeBaseMessageTextContent(text=question)],
            )
        ],
        output_mode=KnowledgeRetrievalOutputMode.ANSWER_SYNTHESIS,
        retrieval_reasoning_effort=KnowledgeRetrievalMediumReasoningEffort(),
    )
    result = client.retrieve(retrieval_request=request)
    text = ""
    response = getattr(result, "response", None) or []
    if response:
        content = getattr(response[0], "content", None) or []
        if content:
            text = getattr(content[0], "text", "") or ""
    return {"answer": text, "raw": result}
