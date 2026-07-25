"""Simple token chunker for T² PDF text (tiktoken)."""

from __future__ import annotations

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    tokens = _ENC.encode(text)
    if len(tokens) <= chunk_tokens:
        return [text]

    chunks: list[str] = []
    step = max(chunk_tokens - overlap_tokens, 1)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_tokens]
        if not window:
            break
        chunks.append(_ENC.decode(window).strip())
        if start + chunk_tokens >= len(tokens):
            break
    return [c for c in chunks if c]
