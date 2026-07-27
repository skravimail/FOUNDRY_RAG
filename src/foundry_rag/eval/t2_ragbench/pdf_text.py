"""Extract text from local T²-RAGBench PDFs."""

from __future__ import annotations

from pathlib import Path


def _join_page_chunks(items: list) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            parts.append(str(item.get("text") or item.get("md") or ""))
        else:
            parts.append(str(item))
    return "\n\n".join(p.strip() for p in parts if p and str(p).strip()).strip()


def _fitz_get_text(path: Path | str) -> str:
    import pymupdf

    doc = pymupdf.open(str(path))
    try:
        return "\n".join(page.get_text() for page in doc).strip()
    finally:
        doc.close()


def extract_pdf_text(path: Path | str) -> str:
    """Return plain text for LLM/RAG chunking (no markdown tables/headers).

    Prefers ``pymupdf4llm.to_text`` when available; falls back to PyMuPDF
    ``page.get_text()`` joined across pages.
    """
    path_str = str(path)
    try:
        import pymupdf4llm

        to_text = getattr(pymupdf4llm, "to_text", None)
        if callable(to_text):
            text = to_text(path_str)
            if isinstance(text, list):
                text = _join_page_chunks(text)
            else:
                text = str(text or "").strip()
            if text:
                return text
    except Exception:
        # Layout-only to_text, missing OCR deps, or other extract failures → fitz
        pass

    return _fitz_get_text(path)
