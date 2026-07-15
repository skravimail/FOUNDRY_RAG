#!/usr/bin/env python3
"""Interactive CLI for Multi-Query RAG (module 04)."""

from __future__ import annotations

import argparse
import time

from rich.console import Console

from foundry_rag.ingestion.index_builder import index_starships
from foundry_rag.mechanisms.multiquery import multiquery_search

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Query RAG demo")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--mode",
        default="custom_hybrid",
        choices=["none_full_text", "none_vector", "custom_full_text", "custom_hybrid"],
    )
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    if not args.skip_index:
        console.print("Indexing...")
        for attempt in range(6):
            try:
                console.print(f"[green]Indexed {index_starships('04_multiquery_rag')} docs[/green]")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                console.print(f"[yellow]retry {attempt + 1}: {exc}[/yellow]")
                time.sleep(10)

    rewrites, docs = multiquery_search(args.question, mode=args.mode)
    console.print("[bold]Rewrites:[/bold]")
    for r in rewrites:
        console.print(f"  - {r}")
    console.print("\n[bold]Top documents:[/bold]")
    for doc in docs:
        console.print(f"  score={doc.get('@search.score')}  {doc.get('Title')}")


if __name__ == "__main__":
    main()
