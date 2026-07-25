#!/usr/bin/env python3
"""Index local T²-RAGBench pilot PDFs into Postgres/pgvector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundry_rag.eval.t2_ragbench.chunking import chunk_text  # noqa: E402
from foundry_rag.eval.t2_ragbench.pdf_text import extract_pdf_text  # noqa: E402
from foundry_rag.llm import embed_texts  # noqa: E402
from foundry_rag.mechanisms.pgvector_store import (  # noqa: E402
    chunk_count,
    clear_bm25_cache,
    clear_chunks,
    connect,
    init_schema,
    upsert_chunks,
)
from foundry_rag.paths import DATA_ROOT  # noqa: E402

console = Console()
DEFAULT_GOLD = DATA_ROOT / "t2_ragbench" / "gold_docs.json"
DEFAULT_PDF_ROOT = DATA_ROOT / "t2_ragbench" / "pdfs"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--chunk-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--embed-batch", type=int, default=16)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate t2_chunks before indexing",
    )
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    console.print(f"gold docs: {len(gold)}")

    prepared: list[dict] = []
    empty_pdfs = 0
    with Progress() as progress:
        task = progress.add_task("extract+chunk", total=len(gold))
        for doc in gold:
            local = REPO_ROOT / doc["local_path"]
            if not local.is_file():
                # Fallback layout: pdfs/{subset}/{context_id}/*.pdf
                alt = args.pdf_root / doc["subset"] / doc["context_id"]
                pdfs = sorted(alt.glob("*.pdf")) if alt.is_dir() else []
                if not pdfs:
                    raise FileNotFoundError(f"Missing PDF for {doc['context_id']}: {local}")
                local = pdfs[0]
            text = extract_pdf_text(local)
            if not text:
                empty_pdfs += 1
                progress.advance(task)
                continue
            chunks = chunk_text(
                text,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
            )
            for idx, content in enumerate(chunks):
                prepared.append(
                    {
                        "id": f"{doc['context_id']}::{idx}",
                        "context_id": doc["context_id"],
                        "subset": doc.get("subset", ""),
                        "file_name": doc.get("file_name", ""),
                        "chunk_index": idx,
                        "content": content,
                        "meta": {"blob_name": doc.get("blob_name"), "local_path": str(local)},
                    }
                )
            progress.advance(task)

    console.print(f"chunks to embed: {len(prepared)} (empty_pdfs={empty_pdfs})")
    if not prepared:
        console.print("[red]No chunks to index[/red]")
        return 1

    # Embed in batches
    embeddings: list[list[float]] = []
    t0 = time.perf_counter()
    with Progress() as progress:
        task = progress.add_task("embed", total=len(prepared))
        for i in range(0, len(prepared), args.embed_batch):
            batch = prepared[i : i + args.embed_batch]
            vectors = embed_texts([row["content"] for row in batch])
            embeddings.extend(vectors)
            progress.advance(task, len(batch))
    console.print(f"embedded in {time.perf_counter() - t0:.1f}s")

    for row, vector in zip(prepared, embeddings, strict=True):
        row["embedding"] = vector

    with connect() as conn:
        init_schema(conn)
        if args.reset:
            clear_chunks(conn)
            console.print("truncated t2_chunks")
        n = upsert_chunks(conn, prepared)
        total = chunk_count(conn)
    clear_bm25_cache()
    console.print(f"upserted={n} table_total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
