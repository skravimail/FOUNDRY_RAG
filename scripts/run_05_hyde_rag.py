#!/usr/bin/env python3
"""Interactive CLI for HyDE RAG (module 05)."""

from __future__ import annotations

import argparse
import time

from rich.console import Console

from foundry_rag.ingestion.index_builder import index_starships
from foundry_rag.mechanisms.hyde import hyde_search

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE RAG demo")
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", default="basic", choices=["basic", "advanced"])
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    if not args.skip_index:
        console.print("Indexing...")
        for attempt in range(6):
            try:
                console.print(f"[green]Indexed {index_starships('05_hyde_rag')} docs[/green]")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                console.print(f"[yellow]retry {attempt + 1}: {exc}[/yellow]")
                time.sleep(10)

    hypo, docs = hyde_search(args.question, mode=args.mode)
    console.print("[bold]Hypothetical answer(s):[/bold]")
    if isinstance(hypo, list):
        for h in hypo:
            console.print(f"  - {h}")
    else:
        console.print(hypo)
    console.print("\n[bold]Top documents:[/bold]")
    for doc in docs:
        console.print(
            f"  score={doc.get('@search.score')} rrf={doc.get('@search.reranker_score')}  "
            f"{doc.get('Title')}"
        )


if __name__ == "__main__":
    main()
