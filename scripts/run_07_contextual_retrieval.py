#!/usr/bin/env python3
"""Interactive CLI for contextual retrieval (module 07)."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from foundry_rag.mechanisms.contextual_retrieval import enrich_document_chunks
from foundry_rag.paths import data_file

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual retrieval demo")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of chunks (useful for smoke validation)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process once and exit (non-interactive)",
    )
    args = parser.parse_args()

    markdown = data_file("07_contextual_retrieval", "remote_work_policy_pl.md").read_text(
        encoding="utf-8"
    )

    while True:
        results = enrich_document_chunks(markdown, max_chunks=args.max_chunks)
        for index, (enriched, vector) in enumerate(results, start=1):
            body = (
                f"[bold yellow]Context[/bold yellow]\n{enriched.context}\n\n"
                f"[bold steel_blue]Chunk[/bold steel_blue]\n{enriched.chunk}\n\n"
                f"[dim]Embedding dims: {len(vector)}[/dim]"
            )
            console.print(Panel(body, title=f"Chunk {index}", border_style="green"))
        if args.once or not Confirm.ask("Continue?", default=False):
            break


if __name__ == "__main__":
    main()
