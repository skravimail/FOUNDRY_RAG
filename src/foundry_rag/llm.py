"""Thin wrappers around Foundry chat + embedding deployments."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

from foundry_rag import config
from foundry_rag.clients import get_embeddings_client, get_openai_client

# gpt-5* Responses deployments accept reasoning.effort; gpt-4.x generally do not.
_REASONING_MODEL_RE = re.compile(r"gpt-5", re.IGNORECASE)
# Smaller / specialized gpt-5* variants stay on minimal by default.
_REASONING_MINIMAL_RE = re.compile(r"(mini|nano|chat|codex)", re.IGNORECASE)


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


def _default_reasoning_effort(deployment: str) -> str | None:
    """Return Responses API reasoning.effort for gpt-5* deployments.

    Full gpt-5 / gpt-5.x use ``high``; mini/nano/chat/codex stay ``minimal``.
    """
    if not _supports_reasoning_effort(deployment):
        return None
    if _REASONING_MINIMAL_RE.search(deployment):
        return "minimal"
    return "high"


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
    reasoning_effort: str | None = None,
) -> str:
    """Chat via the Responses API (gpt-5* + older chat deployments)."""
    client = get_openai_client()
    model = require_chat_deployment()
    kwargs: dict = {
        "model": model,
        "input": user,
        "max_output_tokens": max_output_tokens,
    }
    effort = reasoning_effort if reasoning_effort is not None else _default_reasoning_effort(model)
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    if system:
        # Prefer `instructions` — role="system" message items are rejected by
        # some Foundry Responses deployments.
        kwargs["instructions"] = system
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    response = client.responses.create(**kwargs)
    return _response_text(response)


def chat_complete_pdf(
    *,
    pdf_path: str | Path,
    user: str,
    system: str | None = None,
    max_output_tokens: int = 800,
) -> str:
    """Chat via Responses API with an inline PDF file attachment.

    Uses official ``input_file`` + base64 ``file_data`` (extracted text + page
    images are placed in the model context by the service).
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    client = get_openai_client()
    model = require_chat_deployment()
    kwargs: dict = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": path.name,
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                    {"type": "input_text", "text": user},
                ],
            }
        ],
    }
    effort = _default_reasoning_effort(model)
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    if system:
        kwargs["instructions"] = system

    response = client.responses.create(**kwargs)
    return _response_text(response)


def _response_text(response: object) -> str:
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
