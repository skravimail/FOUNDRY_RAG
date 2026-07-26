"""Thin wrappers around Foundry chat + embedding deployments."""

from __future__ import annotations

import os
import re

from foundry_rag import config
from foundry_rag.clients import get_embeddings_client, get_openai_client

# gpt-5* Responses deployments accept reasoning.effort; gpt-4.x generally do not.
_REASONING_MODEL_RE = re.compile(r"gpt-5", re.IGNORECASE)


def require_chat_deployment() -> str:
    # Prefer live env so evals can override without reloading config module.
    name = os.environ.get("FOUNDRY_CHAT_DEPLOYMENT") or config.FOUNDRY_CHAT_DEPLOYMENT
    if not name:
        raise RuntimeError("Missing required environment variable: FOUNDRY_CHAT_DEPLOYMENT")
    return name


def require_embedding_deployment() -> str:
    name = os.environ.get("FOUNDRY_EMBEDDING_DEPLOYMENT") or config.FOUNDRY_EMBEDDING_DEPLOYMENT
    if not name:
        raise RuntimeError("Missing required environment variable: FOUNDRY_EMBEDDING_DEPLOYMENT")
    return name


def _supports_reasoning_effort(deployment: str) -> bool:
    return bool(_REASONING_MODEL_RE.search(deployment))


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_embeddings_client()
    response = client.embeddings.create(
        model=require_embedding_deployment(),
        input=texts,
    )
    # API may not preserve order guarantees across all backends; sort by index.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def chat_complete(
    *,
    system: str | None = None,
    user: str,
    max_output_tokens: int = 800,
    json_mode: bool = False,
) -> str:
    """Chat via the Responses API (gpt-5* + older chat deployments)."""
    client = get_openai_client()
    model = require_chat_deployment()
    kwargs: dict = {
        "model": model,
        "input": user,
        "max_output_tokens": max_output_tokens,
    }
    if _supports_reasoning_effort(model):
        kwargs["reasoning"] = {"effort": "minimal"}
    if system:
        # Prefer `instructions` — role="system" message items are rejected by
        # some Foundry Responses deployments.
        kwargs["instructions"] = system
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    response = client.responses.create(**kwargs)
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    # Fallback: concatenate text parts from output items
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip() or "Empty response"
