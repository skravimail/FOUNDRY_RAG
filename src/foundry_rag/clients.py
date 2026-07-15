"""Foundry / Azure AI Search client construction.

Mirrors the reference repo's `DefaultAzureCredential`-only auth pattern —
no API keys anywhere.

Note: project-scoped OpenAI (`.../api/projects/.../openai/v1`) supports chat /
responses, but embeddings currently 404 on that path. Embeddings therefore use
the account-level `.../openai/v1` endpoint derived from the project URI.
"""

from __future__ import annotations

from urllib.parse import urlparse

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import OpenAI

from foundry_rag import config


def get_project_client() -> AIProjectClient:
    if not config.FOUNDRY_PROJECT_ENDPOINT:
        raise RuntimeError("Missing required environment variable: FOUNDRY_PROJECT_ENDPOINT")

    return AIProjectClient(
        endpoint=config.FOUNDRY_PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )


def get_openai_client() -> OpenAI:
    """Project-scoped OpenAI client (chat / responses)."""
    return get_project_client().get_openai_client()


def get_embeddings_client() -> OpenAI:
    """Account-scoped OpenAI client used for embeddings."""
    if not config.FOUNDRY_PROJECT_ENDPOINT:
        raise RuntimeError("Missing required environment variable: FOUNDRY_PROJECT_ENDPOINT")

    parsed = urlparse(config.FOUNDRY_PROJECT_ENDPOINT)
    base_url = f"{parsed.scheme}://{parsed.netloc}/openai/v1/"
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return OpenAI(base_url=base_url, api_key=token_provider)


def get_search_client(index_name: str) -> SearchClient:
    if not config.AZURE_AI_SEARCH_ENDPOINT:
        raise RuntimeError("Missing required environment variable: AZURE_AI_SEARCH_ENDPOINT")

    return SearchClient(
        endpoint=config.AZURE_AI_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=DefaultAzureCredential(),
    )


def get_search_index_client() -> SearchIndexClient:
    if not config.AZURE_AI_SEARCH_ENDPOINT:
        raise RuntimeError("Missing required environment variable: AZURE_AI_SEARCH_ENDPOINT")

    return SearchIndexClient(
        endpoint=config.AZURE_AI_SEARCH_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
