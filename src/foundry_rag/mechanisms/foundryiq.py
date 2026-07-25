"""FoundryIQ / Knowledge Base agentic retrieval.

Uses the Azure AI Search Knowledge Bases ``retrieve`` REST API (preview),
with an optional SDK path when the installed ``azure-search-documents`` build
exposes the matching models.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from azure.identity import DefaultAzureCredential

from foundry_rag import config
from foundry_rag.mechanisms.foundryiq_refs import extract_answer_text, ranked_context_ids_from_result

DEFAULT_QUERY = (
    "What are the requirements for the CTO position in Contoso and what is the data "
    "retention policy for the low business value data in Contoso and can I visit a dentist for free?"
)

# Preview API version that supports knowledge base retrieve.
API_VERSION = os.environ.get("AZURE_AI_SEARCH_API_VERSION", "2025-11-01-preview")


def _knowledge_sources() -> list[str]:
    """Resolve blob knowledge source names from env (comma-separated)."""
    raw = os.environ.get("AZURE_AI_SEARCH_KNOWLEDGE_SOURCE", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _resolve_config() -> tuple[str, str, list[str]]:
    endpoint = (config.AZURE_AI_SEARCH_ENDPOINT or os.environ.get("AZURE_AI_SEARCH_URI") or "").rstrip(
        "/"
    )
    kb_name = os.environ.get("AZURE_AI_SEARCH_KNOWLEDGE_BASE") or ""
    knowledge_sources = _knowledge_sources()
    missing = []
    if not endpoint:
        missing.append("AZURE_AI_SEARCH_ENDPOINT (or AZURE_AI_SEARCH_URI)")
    if not kb_name:
        missing.append("AZURE_AI_SEARCH_KNOWLEDGE_BASE")
    if not knowledge_sources:
        missing.append("AZURE_AI_SEARCH_KNOWLEDGE_SOURCE")
    if missing:
        raise RuntimeError(
            "FoundryIQ missing env: "
            + ", ".join(missing)
            + ". Example: AZURE_AI_SEARCH_KNOWLEDGE_BASE=t2-ragbench-kb, "
            "AZURE_AI_SEARCH_KNOWLEDGE_SOURCE=t2-ragbench-ks."
        )
    return endpoint, kb_name, knowledge_sources


def _retrieve_via_rest(
    *,
    endpoint: str,
    kb_name: str,
    knowledge_sources: list[str],
    question: str,
) -> dict[str, Any]:
    token = DefaultAzureCredential().get_token("https://search.azure.com/.default").token
    body = {
        "includeActivity": True,
        "knowledgeSourceParams": [
            {
                "knowledgeSourceName": name,
                "kind": "azureBlob",
                "alwaysQuerySource": False,
            }
            for name in knowledge_sources
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ],
        "outputMode": "answerSynthesis",
        "retrievalReasoningEffort": {"kind": "medium"},
    }
    url = (
        f"{endpoint}/knowledgebases/{quote(kb_name)}/retrieve"
        f"?api-version={API_VERSION}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Knowledge Base retrieve HTTP {exc.code}: {detail}") from exc


def _retrieve_via_sdk(
    *,
    endpoint: str,
    kb_name: str,
    knowledge_sources: list[str],
    question: str,
) -> Any | None:
    """Return SDK result, or None if this SDK build cannot construct the request."""
    try:
        from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient
        from azure.search.documents.knowledgebases.models import (
            AzureBlobKnowledgeSourceParams,
            KnowledgeBaseMessage,
            KnowledgeBaseMessageTextContent,
            KnowledgeBaseRetrievalRequest,
            KnowledgeRetrievalOutputMode,
        )
    except ImportError:
        return None

    # Reasoning-effort model names vary across preview SDKs; omit if unavailable.
    effort = None
    try:
        from azure.search.documents.knowledgebases.models import (  # type: ignore
            KnowledgeRetrievalMediumReasoningEffort,
        )

        effort = KnowledgeRetrievalMediumReasoningEffort()
    except ImportError:
        try:
            from azure.search.documents.knowledgebases.models import (  # type: ignore
                KnowledgeRetrievalReasoningEffort,
            )

            effort = KnowledgeRetrievalReasoningEffort(kind="medium")
        except Exception:
            effort = None

    client = KnowledgeBaseRetrievalClient(
        endpoint=endpoint,
        knowledge_base_name=kb_name,
        credential=DefaultAzureCredential(),
    )
    kwargs: dict[str, Any] = {
        "include_activity": True,
        "knowledge_source_params": [
            AzureBlobKnowledgeSourceParams(knowledge_source_name=name, always_query_source=False)
            for name in knowledge_sources
        ],
        "messages": [
            KnowledgeBaseMessage(
                role="user",
                content=[KnowledgeBaseMessageTextContent(text=question)],
            )
        ],
        "output_mode": KnowledgeRetrievalOutputMode.ANSWER_SYNTHESIS,
    }
    if effort is not None:
        kwargs["retrieval_reasoning_effort"] = effort
    request = KnowledgeBaseRetrievalRequest(**kwargs)
    return client.retrieve(retrieval_request=request)


def retrieve_foundryiq(question: str = DEFAULT_QUERY) -> dict[str, Any]:
    """Call Azure AI Search Knowledge Base retrieval (AnswerSynthesis)."""
    endpoint, kb_name, knowledge_sources = _resolve_config()

    result: Any = _retrieve_via_sdk(
        endpoint=endpoint,
        kb_name=kb_name,
        knowledge_sources=knowledge_sources,
        question=question,
    )
    if result is None:
        result = _retrieve_via_rest(
            endpoint=endpoint,
            kb_name=kb_name,
            knowledge_sources=knowledge_sources,
            question=question,
        )

    text = extract_answer_text(result)
    return {
        "answer": text,
        "ranked_context_ids": ranked_context_ids_from_result(result, k=3),
        "raw": result,
    }
