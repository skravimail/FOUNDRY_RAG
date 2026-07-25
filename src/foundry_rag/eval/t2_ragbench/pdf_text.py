"""Extract text from local T²-RAGBench PDFs."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(path: Path | str) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts).strip()
