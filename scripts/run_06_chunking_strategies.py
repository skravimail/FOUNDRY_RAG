#!/usr/bin/env python3
"""Interactive CLI for chunking strategy comparison (module 06)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from foundry_rag.chunking import FixedSizeChunker, HierarchicalChunker, SemanticChunker
from foundry_rag.paths import data_file

console = Console()


def _render_chunks(chunks: list[str]) -> None:
    for i, chunk in enumerate(chunks, start=1):
        console.print(Panel(chunk, title=f"Chunk {i}", border_style="steel_blue"))
    console.print(f"[dim]Total: {len(chunks)} chunk(s)[/dim]")


def _render_parent_child(pairs: list[tuple[str, str]]) -> None:
    current_parent = None
    parent_count = 0
    child_in_parent = 0
    for parent, child in pairs:
        if parent != current_parent:
            parent_count += 1
            child_in_parent = 0
            current_parent = parent
            console.print(Panel(parent, title=f"Parent {parent_count}", border_style="yellow"))
        child_in_parent += 1
        console.print(
            Panel(child, title=f"Child {parent_count}.{child_in_parent}", border_style="steel_blue")
        )
    console.print(f"[dim]Total: {parent_count} parent(s), {len(pairs)} child(ren)[/dim]")


def run_once(strategy: str, markdown: str) -> None:
    if strategy == "fixed":
        chunks = FixedSizeChunker().create_chunks(markdown, chunk_size=512, overlap_percentage=25.0)
        _render_chunks(chunks)
    elif strategy == "semantic":
        chunks = SemanticChunker().create_chunks(
            markdown, similarity_threshold=0.75, max_tokens_per_chunk=512
        )
        _render_chunks(chunks)
    elif strategy == "hierarchical":
        pairs = HierarchicalChunker().create_parent_child_chunks(markdown)
        _render_parent_child(pairs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "hierarchical"],
        help="Run a single strategy non-interactively (for validation)",
    )
    args = parser.parse_args()

    markdown = data_file("06_chunking_strategies", "grounding-data-design.md").read_text(
        encoding="utf-8"
    )

    if args.strategy:
        console.print(f"[bold]Strategy:[/bold] {args.strategy}")
        run_once(args.strategy, markdown)
        return

    choices = {
        "1": ("fixed", "Fixed-size with overlap"),
        "2": ("semantic", "Semantic chunking"),
        "3": ("hierarchical", "Hierarchical (Parent-Child)"),
    }
    while True:
        console.print("\n[bold blue]Select a chunking strategy:[/bold blue]")
        for key, (_, label) in choices.items():
            console.print(f"  {key}. {label}")
        pick = Prompt.ask("Choice", choices=list(choices), default="1")
        strategy, label = choices[pick]
        console.print(f"Selected strategy: [bold italic blue]{label}[/bold italic blue]\n")
        run_once(strategy, markdown)
        if not Confirm.ask("Continue?", default=True):
            break


if __name__ == "__main__":
    main()
