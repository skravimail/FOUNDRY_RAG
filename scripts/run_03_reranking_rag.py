#!/usr/bin/env python3
"""Interactive CLI for Re-ranking RAG (module 03)."""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.prompt import Confirm, Prompt

from foundry_rag.ingestion.index_builder import index_starships
from foundry_rag.mechanisms import reranking

console = Console()

MODES = {
    "1": ("full_text", "FullText (BM25)"),
    "2": ("hybrid", "Hybrid: FullText + Vector"),
    "3": ("hybrid_semantic", "Hybrid + Semantic"),
    "4": ("hybrid_semantic_scoring", "Hybrid + Semantic + Scoring Profile"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ranking RAG demo")
    parser.add_argument("--question", help="Non-interactive question")
    parser.add_argument(
        "--mode", default="hybrid_semantic", choices=[m for m, _ in MODES.values()]
    )
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    if not args.skip_index:
        console.print("Indexing starships (semantic index)...")
        for attempt in range(6):
            try:
                count = index_starships("03_reranking_rag")
                console.print(f"[green]Indexed {count} documents[/green]")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                console.print(f"[yellow]Index attempt {attempt + 1} failed ({exc}); retrying...[/yellow]")
                time.sleep(10)

    if args.question:
        answer, docs = reranking.answer_question(args.question, mode=args.mode)
        for doc in docs:
            console.print(
                f"score={doc.get('@search.score')} reranker={doc.get('@search.reranker_score')}  "
                f"{doc.get('Title')}"
            )
        console.print("\n[bold green]Chat response:[/bold green]")
        console.print(answer)
        return

    while True:
        console.print("\n[bold blue]Select search method:[/bold blue]")
        for key, (_, label) in MODES.items():
            console.print(f"  {key}. {label}")
        pick = Prompt.ask("Choice", choices=list(MODES), default="3")
        mode, _ = MODES[pick]
        question = Prompt.ask("Question")
        answer, docs = reranking.answer_question(question, mode=mode)
        for doc in docs:
            console.print(
                f"score={doc.get('@search.score')} reranker={doc.get('@search.reranker_score')}  "
                f"{doc.get('Title')}"
            )
        console.print("\n[bold green]Chat response:[/bold green]")
        console.print(answer)
        if not Confirm.ask("Continue?", default=True):
            break


if __name__ == "__main__":
    main()
