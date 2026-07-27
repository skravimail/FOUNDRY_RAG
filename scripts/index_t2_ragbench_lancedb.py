#!/usr/bin/env python3
"""Index local T²-RAGBench pilot PDFs into LanceDB."""

from __future__ import annotations

import argparse
import json
import os
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
from foundry_rag.mechanisms.lancedb_store import (  # noqa: E402
    chunk_count,
    db_uri,
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
        help="Drop and recreate the LanceDB table (recommended)",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="",
        help=f"LanceDB directory (default: {db_uri()})",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Store zero vectors (BM25-only indexing; no Foundry embeddings call)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=int(os.environ.get("FOUNDRY_EMBEDDING_DIMENSIONS", "1536")),
        help="Embedding dimension when using --skip-embed (default: 1536)",
    )
    args = parser.parse_args()
    uri = args.uri or None

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    console.print(f"gold docs: {len(gold)}")
    console.print(f"lancedb uri: {uri or db_uri()}")

    prepared: list[dict] = []
    empty_pdfs = 0
    with Progress() as progress:
        task = progress.add_task("extract+chunk", total=len(gold))
        for doc in gold:
            local = REPO_ROOT / doc["local_path"]
            if not local.is_file():
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
                    }
                )
            progress.advance(task)

    console.print(f"chunks to embed: {len(prepared)} (empty_pdfs={empty_pdfs})")
    if not prepared:
        console.print("[red]No chunks to index[/red]")
        return 1

    if args.skip_embed:
        console.print(f"[yellow]skip-embed: zero vectors dim={args.embed_dim} (BM25-only)[/yellow]")
        for row in prepared:
            row["embedding"] = [0.0] * args.embed_dim
    else:
        embeddings: list[list[float]] = []
        t0 = time.perf_counter()
        with Progress() as progress:
            task = progress.add_task("embed", total=len(prepared))
            for i in range(0, len(prepared), args.embed_batch):
                batch = prepared[i : i + args.embed_batch]
                embeddings.extend(embed_texts([row["content"] for row in batch]))
                progress.advance(task, len(batch))
        console.print(f"embedded in {time.perf_counter() - t0:.1f}s")
        for row, vector in zip(prepared, embeddings, strict=True):
            row["embedding"] = vector

    n = upsert_chunks(prepared, reset=args.reset or True, uri=uri)
    console.print(f"upserted={n} table_total={chunk_count(uri)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
