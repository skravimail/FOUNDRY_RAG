"""Foundry / Azure AI Search client construction.

Mirrors the reference repo's `DefaultAzureCredential`-only auth pattern —
no API keys anywhere.
"""

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from foundry_rag import config


def get_project_client() -> AIProjectClient:
    if not config.FOUNDRY_PROJECT_ENDPOINT:
        raise RuntimeError("Missing required environment variable: FOUNDRY_PROJECT_ENDPOINT")

    return AIProjectClient(
        endpoint=config.FOUNDRY_PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )


def get_openai_client():
    return get_project_client().get_openai_client()


def get_search_client(index_name: str) -> SearchClient:
    if not config.AZURE_AI_SEARCH_ENDPOINT:
        raise RuntimeError("Missing required environment variable: AZURE_AI_SEARCH_ENDPOINT")

    return SearchClient(
        endpoint=config.AZURE_AI_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=DefaultAzureCredential(),
    )
