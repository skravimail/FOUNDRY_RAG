"""Parse Knowledge Base retrieve payloads into answer text + ranked context IDs."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlparse

_CONTEXT_IN_PATH = re.compile(
    r"/t2rag/[^/]+/(?P<context_id>[^/]+)/",
    re.IGNORECASE,
)


def context_id_from_blob_url(url: str | None) -> str | None:
    if not url:
        return None
    path = unquote(urlparse(url).path)
    match = _CONTEXT_IN_PATH.search(path)
    if match:
        return match.group("context_id")
    for part in path.strip("/").split("/"):
        if "_ctx_" in part:
            return part
    return None


def _as_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "as_dict"):
        try:
            return dict(obj.as_dict())
        except Exception:
            pass
    out: dict[str, Any] = {}
    for key in (
        "type",
        "id",
        "blob_url",
        "blobUrl",
        "reranker_score",
        "rerankerScore",
        "activity_source",
        "activitySource",
        "source_data",
        "sourceData",
        "references",
        "response",
        "activity",
        "content",
        "text",
    ):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


def extract_answer_text(result: Any) -> str:
    data = _as_mapping(result)
    response = data.get("response") or getattr(result, "response", None) or []
    if not response:
        return ""
    first = response[0]
    content = _as_mapping(first).get("content") or getattr(first, "content", None) or []
    if not content:
        return ""
    part = content[0]
    text = _as_mapping(part).get("text") or getattr(part, "text", None) or ""
    return str(text)


def ranked_context_ids_from_result(result: Any, *, k: int = 3) -> list[str]:
    """Unique context_ids from retrieve ``references``, citation order preserved."""
    data = _as_mapping(result)
    refs = data.get("references") or getattr(result, "references", None) or []
    ranked: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        mapping = _as_mapping(ref)
        url = mapping.get("blobUrl") or mapping.get("blob_url")
        ctx = context_id_from_blob_url(url if isinstance(url, str) else None)
        if not ctx:
            source = mapping.get("sourceData") or mapping.get("source_data") or {}
            if isinstance(source, dict):
                ctx = source.get("context_id") or source.get("contextId")
        if ctx and ctx not in seen:
            seen.add(ctx)
            ranked.append(ctx)
        if len(ranked) >= k:
            break
    return ranked
