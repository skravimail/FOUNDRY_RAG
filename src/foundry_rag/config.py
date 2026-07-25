"""Environment-variable configuration for Foundry RAG mechanisms.
Python-named equivalents of the reference `deployed-in-azure/RAG` repo's
.NET environment variables (see plans/00-rag-foundry-python.md, Phase 1).
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, *, required: bool = False, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# Foundry project + model deployments
FOUNDRY_PROJECT_ENDPOINT = _get_env("FOUNDRY_PROJECT_ENDPOINT")
FOUNDRY_CHAT_DEPLOYMENT = _get_env("FOUNDRY_CHAT_DEPLOYMENT")
FOUNDRY_EMBEDDING_DEPLOYMENT = _get_env("FOUNDRY_EMBEDDING_DEPLOYMENT")

# Azure AI Search (modules 02 HybridRAG, 09 FoundryIQ)
AZURE_AI_SEARCH_ENDPOINT = _get_env("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_CONNECTION_NAME = _get_env("AZURE_AI_SEARCH_CONNECTION_NAME")
