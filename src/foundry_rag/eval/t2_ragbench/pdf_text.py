"""Extract text from local T²-RAGBench PDFs."""

from __future__ import annotations

from pathlib import Path

import pymupdf4llm


def extract_pdf_text(path: Path | str) -> str:
    """Return markdown text via pymupdf4llm for LLM/RAG chunking."""
    md = pymupdf4llm.to_markdown(str(path))
    if isinstance(md, list):
        # page_chunks=True path (not used by default); join page texts
        parts = []
        for item in md:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("md") or ""))
            else:
                parts.append(str(item))
        return "\n\n".join(p.strip() for p in parts if p and str(p).strip()).strip()
    return str(md or "").strip()
