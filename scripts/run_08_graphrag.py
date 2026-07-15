#!/usr/bin/env python3
"""CLI for GraphRAG (module 08)."""

from __future__ import annotations

import argparse

from rich.console import Console

from foundry_rag.mechanisms.graphrag.extraction import extract_graph_from_document
from foundry_rag.mechanisms.graphrag.graph_db import GraphDb
from foundry_rag.mechanisms.graphrag.query import answer_question
from foundry_rag.paths import data_file

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG demo")
    parser.add_argument("--index", action="store_true", help="Extract + ingest graph")
    parser.add_argument("--question", help="Ask a question against the graph")
    parser.add_argument("--max-chunks", type=int, default=3, help="Limit chunks when indexing")
    args = parser.parse_args()

    if args.index:
        text = data_file("08_graphrag", "grounding-data-design.md").read_text(encoding="utf-8")
        console.print(f"Extracting knowledge graph (max_chunks={args.max_chunks})...")
        graph = extract_graph_from_document(text, max_chunks=args.max_chunks)
        console.print(
            f"Merged graph: {len(graph.entities)} entities, {len(graph.relationships)} relationships"
        )
        with GraphDb() as db:
            db.ensure_vector_index()
            db.ingest_graph(graph)
            nodes, edges = db.count_graph()
        console.print(f"[green]Neo4j now has {nodes} entities and {edges} relationships[/green]")

    if args.question:
        answer, result = answer_question(args.question)
        console.print(f"Seeds: {[e['name'] for e in result.get('seeds', [])]}")
        console.print(f"Rels: {len(result.get('relationships', []))}")
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(answer)

    if not args.index and not args.question:
        parser.error("Provide --index and/or --question")


if __name__ == "__main__":
    main()
