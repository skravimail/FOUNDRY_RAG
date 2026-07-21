#!/usr/bin/env python3
"""Interactive CLI for Hybrid RAG as a Foundry prompt agent (module 02, agent port).

Unlike ``run_02_hybrid_rag.py`` (which queries Azure AI Search directly and then
calls the chat model), this runner provisions a Foundry prompt agent with the
native Azure AI Search tool and lets the agent retrieve + answer in one call.

Prerequisites:
  - AZURE_AI_SEARCH_CONNECTION_NAME set to a Search connection in the Foundry project.
  - The 02_hybrid_rag index present (use without --skip-index to (re)build it).
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.prompt import Confirm, Prompt

from foundry_rag.agents import hybrid_agent
from foundry_rag.ingestion.index_builder import index_starships

console = Console()

MODES = {
    "1": ("full_text", "FullText (BM25)"),
    "2": ("vector", "Vector similarity [needs integrated vectorizer]"),
    "3": ("hybrid", "Hybrid: keyword + vector (RRF) [needs integrated vectorizer]"),
    "4": ("hybrid_semantic", "Hybrid + semantic reranker [needs vectorizer + Basic+ tier]"),
}


def _ask(question: str, mode: str) -> None:
    answer, citations = hybrid_agent.answer_question(question, mode=mode)
    if citations:
        console.print("[dim]Citations:[/dim]")
        for citation in citations:
            console.print(f"  - {citation}")
    console.print("\n[bold green]Agent response:[/bold green]")
    console.print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG demo (Foundry agent port)")
    parser.add_argument("--question", help="Non-interactive question")
    # NOTE: only full_text works today. The vector/hybrid/hybrid_semantic modes
    # require the index to have an integrated vectorizer (the native Search tool
    # embeds the query server-side); that isn't configured yet. See
    # foundry_rag.agents.hybrid_agent for details.
    parser.add_argument(
        "--mode",
        default="full_text",
        choices=[m for m, _ in MODES.values()],
    )
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
                console.print(
                    f"[yellow]Index attempt {attempt + 1} failed ({exc}); retrying...[/yellow]"
                )
                time.sleep(10)

    if args.question:
        _ask(args.question, args.mode)
        return

    while True:
        console.print("\n[bold blue]Select search method:[/bold blue]")
        for key, (_, label) in MODES.items():
            console.print(f"  {key}. {label}")
        pick = Prompt.ask("Choice", choices=list(MODES), default="1")
        mode, _ = MODES[pick]
        question = Prompt.ask("Question")
        _ask(question, mode)
        if not Confirm.ask("Continue?", default=True):
            break


if __name__ == "__main__":
    main()
