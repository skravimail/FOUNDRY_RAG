#!/usr/bin/env python3
"""CLI for FoundryIQ Knowledge Base retrieval (module 09)."""

from __future__ import annotations

import argparse

from rich.console import Console

from foundry_rag.mechanisms.foundryiq import DEFAULT_QUERY, retrieve_foundryiq

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="FoundryIQ demo")
    parser.add_argument("--question", default=DEFAULT_QUERY)
    args = parser.parse_args()
    try:
        result = retrieve_foundryiq(args.question)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]FoundryIQ not runnable yet:[/yellow] {exc}")
        console.print(
            "Provision an Azure AI Search Knowledge Base with Contoso blob sources, "
            "set AZURE_AI_SEARCH_KNOWLEDGE_BASE in .env, then re-run."
        )
        raise SystemExit(2) from exc
    console.print("[bold green]Synthesized answer:[/bold green]")
    console.print(result["answer"] or "(empty)")


if __name__ == "__main__":
    main()
