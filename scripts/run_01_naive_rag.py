#!/usr/bin/env python3
"""Interactive CLI for Naive RAG (module 01)."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.prompt import Confirm, Prompt

from foundry_rag.mechanisms import naive

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive RAG demo")
    parser.add_argument("--question", help="Ask a single question non-interactively")
    parser.add_argument(
        "--system",
        choices=["default", "kid", "marketing"],
        default="default",
        help="System prompt variant for --question mode",
    )
    args = parser.parse_args()

    console.print("*** Source data vectorization started ***")
    db = naive.build_vector_db()
    console.print(f"[green]*** {len(naive.DATA_SOURCE)} text chunks were vectorized successfully ***[/green]")

    prompts = {
        "default": naive.DEFAULT_SYSTEM_PROMPT,
        "kid": naive.KID_FRIENDLY_SYSTEM_PROMPT,
        "marketing": naive.MARKETING_SYSTEM_PROMPT,
    }

    if args.question:
        answer, results = naive.answer_question(
            db, args.question, system_prompt=prompts[args.system]
        )
        for result in results:
            console.print(f"Similarity: [blue]{result.similarity:.2f}[/blue]  Id: {result.id}")
        console.print("\n[bold green]Chat response:[/bold green]")
        console.print(answer)
        return

    while True:
        console.print("\n[bold blue]Select a system prompt:[/bold blue]")
        for i, text in enumerate(naive.system_prompts(), start=1):
            preview = text.strip().splitlines()[0][:80]
            console.print(f"  {i}. {preview}...")
        prompt_idx = int(Prompt.ask("Choice", choices=["1", "2", "3"], default="1"))
        system_prompt = naive.system_prompts()[prompt_idx - 1]

        console.print("\n[bold blue]Select a question:[/bold blue]")
        choices = naive.SAMPLE_QUESTIONS + ["Custom"]
        for i, q in enumerate(choices, start=1):
            console.print(f"  {i}. {q}")
        q_idx = int(Prompt.ask("Choice", choices=[str(i) for i in range(1, len(choices) + 1)], default="1"))
        question = choices[q_idx - 1]
        if question == "Custom":
            question = Prompt.ask("Provide a custom question")

        answer, results = naive.answer_question(db, question, system_prompt=system_prompt)
        console.print(f"*** Top {len(results)} most similar vectors were found ***")
        for result in results:
            console.print(f"Similarity score: [blue]{result.similarity:.2f}[/blue], Id: {result.id}")
        console.print("\n[bold green]Chat response:[/bold green]")
        console.print(answer)
        if not Confirm.ask("Continue?", default=True):
            break


if __name__ == "__main__":
    main()
