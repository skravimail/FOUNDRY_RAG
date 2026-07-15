#!/usr/bin/env python3
"""Interactive CLI for Hybrid RAG (module 02)."""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.prompt import Confirm, Prompt

from foundry_rag.ingestion.index_builder import index_starships
from foundry_rag.mechanisms import hybrid

console = Console()

MODES = {
    "1": ("full_text", "FullText (BM25)"),
    "2": ("vector_overview", "Vector (Overview)"),
    "3": ("hybrid_overview", "Hybrid: FullText + Overview vector"),
    "4": ("vector_both", "Vector (Overview + Notes)"),
    "5": ("hybrid_both", "Hybrid: FullText + Overview + Notes"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG demo")
    parser.add_argument("--question", help="Non-interactive question")
    parser.add_argument("--mode", default="hybrid_overview", choices=[m for m, _ in MODES.values()])
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    if not args.skip_index:
        console.print("Indexing starships...")
        # RBAC propagation can lag briefly after role assignment.
        for attempt in range(6):
            try:
                count = index_starships("02_hybrid_rag")
                console.print(f"[green]Indexed {count} documents[/green]")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                console.print(f"[yellow]Index attempt {attempt + 1} failed ({exc}); retrying...[/yellow]")
                time.sleep(10)

    if args.question:
        answer, docs = hybrid.answer_question(args.question, mode=args.mode)
        for doc in docs:
            console.print(
                f"score={doc.get('@search.score')}  {doc.get('Title')}  ({doc.get('Id')})"
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
        answer, docs = hybrid.answer_question(question, mode=mode)
        for doc in docs:
            console.print(
                f"score={doc.get('@search.score')}  {doc.get('Title')}  ({doc.get('Id')})"
            )
        console.print("\n[bold green]Chat response:[/bold green]")
        console.print(answer)
        if not Confirm.ask("Continue?", default=True):
            break


if __name__ == "__main__":
    main()
