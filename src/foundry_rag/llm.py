"""Thin wrappers around Foundry chat + embedding deployments."""

from __future__ import annotations

from foundry_rag import config
from foundry_rag.clients import get_embeddings_client, get_openai_client


def require_chat_deployment() -> str:
    if not config.FOUNDRY_CHAT_DEPLOYMENT:
        raise RuntimeError("Missing required environment variable: FOUNDRY_CHAT_DEPLOYMENT")
    return config.FOUNDRY_CHAT_DEPLOYMENT


def require_embedding_deployment() -> str:
    if not config.FOUNDRY_EMBEDDING_DEPLOYMENT:
        raise RuntimeError("Missing required environment variable: FOUNDRY_EMBEDDING_DEPLOYMENT")
    return config.FOUNDRY_EMBEDDING_DEPLOYMENT


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
    """Chat via the Responses API (works with gpt-5-mini reasoning models)."""
    client = get_openai_client()
    kwargs: dict = {
        "model": require_chat_deployment(),
        "input": user,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "minimal"},
    }
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
