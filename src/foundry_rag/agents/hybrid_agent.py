"""Hybrid RAG as a Foundry prompt agent (Foundry Agent Service SDK).

This is the agent-native counterpart of :mod:`foundry_rag.mechanisms.hybrid`.

Instead of querying Azure AI Search from the client and stuffing the results
into a chat prompt, we provision a Foundry *prompt agent* whose only tool is the
native **Azure AI Search** tool. The agent performs retrieval (keyword / vector /
hybrid, server-side) and grounds its answer on the results in a single call.

Requirements beyond module 02:
  - ``AZURE_AI_SEARCH_CONNECTION_NAME`` must name an Azure AI Search *connection*
    registered in the Foundry project (Management center -> Connected resources).
  - The target index (``02_hybrid_rag`` -> ``load_index_definition``) must already
    exist in that Search service.
"""

from __future__ import annotations

from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AISearchIndexResource,
    AzureAISearchQueryType,
    AzureAISearchTool,
    AzureAISearchToolResource,
    PromptAgentDefinition,
)

from foundry_rag import config
from foundry_rag.clients import get_project_client
from foundry_rag.ingestion.index_builder import load_index_definition
from foundry_rag.llm import require_chat_deployment

AGENT_NAME = "hybrid-rag-agent"

INSTRUCTIONS = """\
You are a helpful assistant for the Galactic Voyages travel agency.
Always call the Azure AI Search tool to retrieve starship information before
answering, and answer using only the information it returns.
Keep your answers clear and friendly. If the retrieved documents do not contain
the answer, say that you don't have that information.
"""

# Friendly mode -> native Azure AI Search query type.
#   full_text        -> BM25 keyword search only            [works today]
#   vector           -> pure vector similarity              [needs vectorizer]
#   hybrid           -> keyword + vector (RRF fusion)        [needs vectorizer]
#   hybrid_semantic  -> keyword + vector + semantic reranker [needs vectorizer + Basic+ tier]
#
# IMPORTANT: Unlike foundry_rag.mechanisms.hybrid (which computes query
# embeddings client-side and passes them via VectorizedQuery), the native
# Azure AI Search agent tool performs *integrated vectorization* — the Search
# service embeds the query itself at run time. That requires the index to have
# an azureOpenAI vectorizer bound to the vector field's profile, plus the Search
# service's managed identity granted "Cognitive Services OpenAI User" on the
# embedding model's resource. This index currently has no vectorizer
# (`_sanitize_index_definition` strips them), so only `full_text` works until
# integrated vectorization is configured.
QUERY_TYPES: dict[str, str] = {
    "full_text": AzureAISearchQueryType.SIMPLE,
    "vector": AzureAISearchQueryType.VECTOR,
    "hybrid": AzureAISearchQueryType.VECTOR_SIMPLE_HYBRID,
    "hybrid_semantic": AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
}


def _resolve_search_connection_id(client: AIProjectClient) -> str:
    if not config.AZURE_AI_SEARCH_CONNECTION_NAME:
        raise RuntimeError(
            "AZURE_AI_SEARCH_CONNECTION_NAME is required for the native Azure AI "
            "Search agent tool. Register the Search service as a connection in the "
            "Foundry project and set AZURE_AI_SEARCH_CONNECTION_NAME in .env."
        )
    connection = client.connections.get(config.AZURE_AI_SEARCH_CONNECTION_NAME)
    return connection.id


def create_hybrid_agent(
    client: AIProjectClient,
    *,
    module_folder: str = "02_hybrid_rag",
    mode: str = "hybrid",
    top_k: int = 3,
) -> str:
    """Create (or add a version to) the hybrid RAG prompt agent.

    Returns the agent name, which is used to reference the agent at invocation.
    """
    if mode not in QUERY_TYPES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(QUERY_TYPES)}")

    index_name = load_index_definition(module_folder)["name"]
    connection_id = _resolve_search_connection_id(client)

    search_tool = AzureAISearchTool(
        azure_ai_search=AzureAISearchToolResource(
            indexes=[
                AISearchIndexResource(
                    project_connection_id=connection_id,
                    index_name=index_name,
                    query_type=QUERY_TYPES[mode],
                    top_k=top_k,
                )
            ]
        )
    )

    definition = PromptAgentDefinition(
        model=require_chat_deployment(),
        instructions=INSTRUCTIONS,
        tools=[search_tool],
    )

    version = client.agents.create_version(agent_name=AGENT_NAME, definition=definition)
    return version.name


def _extract_citations(response: Any) -> list[str]:
    """Pull url citation annotations out of a Responses API result, if any."""
    citations: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            for annotation in getattr(content, "annotations", []) or []:
                url = getattr(annotation, "url", None)
                title = getattr(annotation, "title", None)
                if url or title:
                    citations.append(title or url)
    return citations


def answer_question(
    question: str,
    *,
    module_folder: str = "02_hybrid_rag",
    mode: str = "hybrid",
    top_k: int = 3,
) -> tuple[str, list[str]]:
    """Provision the hybrid agent and answer one question.

    Returns ``(answer_text, citations)``.
    """
    client = get_project_client()
    agent_name = create_hybrid_agent(
        client, module_folder=module_folder, mode=mode, top_k=top_k
    )

    openai_client = client.get_openai_client()
    response = openai_client.responses.create(
        input=question,
        tool_choice="required",
        extra_body={"agent_reference": {"name": agent_name, "type": "agent_reference"}},
    )

    answer = getattr(response, "output_text", None) or "(empty response)"
    return answer.strip(), _extract_citations(response)
