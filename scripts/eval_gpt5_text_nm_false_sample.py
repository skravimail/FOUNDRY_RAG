#!/usr/bin/env python3
"""Re-answer NM=false pilot rows with gpt-5 text RAG (pgvector hybrid)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundry_rag.eval.t2_ragbench.metrics import number_match  # noqa: E402
from foundry_rag.eval.t2_ragbench.pgvector_runner import (  # noqa: E402
    SYSTEM_PROMPT,
    run_pgvector,
)
from foundry_rag.llm import require_chat_deployment  # noqa: E402
from foundry_rag.paths import DATA_ROOT  # noqa: E402

console = Console()

DEFAULT_INPUT = DATA_ROOT / "t2_ragbench" / "results" / "pgvector-hybrid-pypdf-rerun-pilot50.jsonl"
DEFAULT_OUT = DATA_ROOT / "t2_ragbench" / "results" / "gpt5-text-nm-false-sample.jsonl"

SAMPLE_IDS = [
    "finqa_test_22",
    "finqa_test_48",
    "finqa_test_81",
    "finqa_test_148",
    "finqa_test_213",
    "finqa_test_231",
    "finqa_test_361",
    "finqa_test_482",
    "finqa_test_681",
    "finqa_test_800",
    "finqa_test_834",
    "finqa_test_858",
    "finqa_test_885",
    "finqa_test_947",
    "finqa_test_1011",
    "finqa_test_1042",
]


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["id"]] = row
    return done


def _short(text: str, n: int = 80) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4000,
        help="gpt-5 high reasoning needs headroom beyond answer text",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Overwrite prior out file instead of resuming",
    )
    args = parser.parse_args()

    model = require_chat_deployment()
    by_id: dict[str, dict[str, Any]] = {}
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("id") in SAMPLE_IDS and row.get("nm") is False:
                by_id[row["id"]] = row

    missing = [i for i in SAMPLE_IDS if i not in by_id]
    if missing:
        console.print(f"[red]Missing sample rows: {missing}[/red]")
        return 1

    rows = [by_id[i] for i in SAMPLE_IDS]
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and args.out.exists():
        args.out.unlink()
    done = _load_done(args.out)
    # Treat empty/invalid answers as incomplete so a fixed-token re-run can overwrite.
    done = {
        qid: row
        for qid, row in done.items()
        if (row.get("answer") or "").strip()
        and (row.get("answer") or "").strip() != "Empty response"
        and not row.get("error")
    }
    console.print(
        f"model={model} sample={len(rows)} already_done={len(done)} "
        f"top_k={args.top_k} max_output_tokens={args.max_output_tokens} out={args.out}"
    )
    console.print(f"system_prompt={SYSTEM_PROMPT[:60]}…")

    results: list[dict[str, Any]] = []
    # Rewrite file from scratch each resume so skipped+new rows stay unique.
    pending = [src for src in rows if src["id"] not in done]
    with args.out.open("w", encoding="utf-8") as out_fh:
        for src in rows:
            qid = src["id"]
            if qid in done:
                out_fh.write(json.dumps(done[qid], ensure_ascii=False) + "\n")
                results.append(done[qid])
                console.print(f"[dim]skip {qid}[/dim]")
                continue

            t0 = time.perf_counter()
            console.print(f"→ {qid}")
            try:
                result = run_pgvector(
                    src["question"],
                    top_k=args.top_k,
                    retrieval="hybrid",
                    max_output_tokens=args.max_output_tokens,
                )
                answer = result.get("answer") or ""
                err = None
            except Exception as exc:  # noqa: BLE001 — record per-row failures
                result = {}
                answer = ""
                err = f"{type(exc).__name__}: {exc}"
                console.print(f"[red]{qid} failed: {err}[/red]")

            latency = time.perf_counter() - t0
            nm_new = number_match(answer, src["program_answer"]) if answer else False
            out_row = {
                "id": qid,
                "subset": src.get("subset", "FinQA"),
                "method": "pgvector-hybrid-text-gpt5",
                "model": model,
                "question": src["question"],
                "program_answer": src["program_answer"],
                "gold_context_id": src.get("gold_context_id"),
                "prev_answer": src.get("answer"),
                "prev_nm": False,
                "prev_model": src.get("model"),
                "answer": answer,
                "nm": nm_new,
                "flipped": bool(nm_new),
                "ranked_context_ids": result.get("ranked_context_ids"),
                "raw": result.get("raw"),
                "latency_s": round(latency, 3),
                "error": err,
                "max_output_tokens": args.max_output_tokens,
                "reasoning_effort": "high",
            }
            out_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_fh.flush()
            results.append(out_row)
            console.print(
                f"  nm={nm_new} flipped={nm_new} latency={latency:.1f}s answer={_short(answer)}"
            )
    _ = pending  # used for clarity in logs above

    final_by_id = {r["id"]: r for r in results}
    ordered = [final_by_id[i] for i in SAMPLE_IDS if i in final_by_id]

    n = len(ordered)
    n_flip = sum(1 for r in ordered if r.get("flipped"))
    n_nm = sum(1 for r in ordered if r.get("nm"))
    summary = {
        "n": n,
        "nm_rate": (n_nm / n) if n else 0.0,
        "flipped": n_flip,
        "model": model,
        "method": "pgvector-hybrid-text-gpt5",
        "reasoning_effort": "high",
        "input": str(args.input.relative_to(REPO_ROOT)),
        "out": str(args.out.relative_to(REPO_ROOT)),
        "vs_gpt5_mini_text": {
            "baseline_nm": 0,
            "baseline_flipped": 0,
            "note": "All 16 were NM=false under gpt-5-mini text RAG",
        },
        "vs_vision_pdf_sample": {
            "vision_flipped": 3,
            "vision_nm_rate": 0.1875,
        },
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    table = Table(title="gpt-5 text RAG NM=false sample")
    table.add_column("id")
    table.add_column("gold")
    table.add_column("answer_short")
    table.add_column("nm_new")
    table.add_column("flipped?")
    for r in ordered:
        table.add_row(
            r["id"],
            str(r["program_answer"]),
            _short(r.get("answer") or r.get("error") or "", 72),
            str(bool(r.get("nm"))),
            str(bool(r.get("flipped"))),
        )
    console.print(table)
    console.print(
        f"summary: n={n} nm={n_nm}/{n} ({summary['nm_rate']:.2f}) flipped={n_flip} → {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
