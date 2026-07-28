#!/usr/bin/env python3
"""Vision/PDF sample: re-answer NM=false pilot rows with gold PDF via Responses API."""

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
from foundry_rag.llm import chat_complete_pdf, require_chat_deployment  # noqa: E402
from foundry_rag.paths import DATA_ROOT  # noqa: E402

console = Console()

DEFAULT_INPUT = DATA_ROOT / "t2_ragbench" / "results" / "pgvector-hybrid-pypdf-rerun-pilot50.jsonl"
DEFAULT_GOLD = DATA_ROOT / "t2_ragbench" / "gold_docs.json"
DEFAULT_PDF_ROOT = DATA_ROOT / "t2_ragbench" / "pdfs"
DEFAULT_OUT = DATA_ROOT / "t2_ragbench" / "results" / "vision-pdf-nm-false-sample.jsonl"
# gpt-5 high reasoning needs headroom beyond answer text.
DEFAULT_MAX_OUTPUT_TOKENS = 4000

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

SYSTEM_PROMPT = (
    "You answer financial questions using ONLY the attached PDF document. "
    "Reply with a short answer that includes the final numeric value when applicable. "
    "Do not invent figures that are not supported by the document."
)


def _resolve_pdf(doc: dict[str, Any], pdf_root: Path) -> Path:
    local = REPO_ROOT / doc["local_path"]
    if local.is_file():
        return local
    alt = pdf_root / doc["subset"] / doc["context_id"]
    pdfs = sorted(alt.glob("*.pdf")) if alt.is_dir() else []
    if not pdfs:
        raise FileNotFoundError(f"Missing PDF for {doc['context_id']}: {local}")
    return pdfs[0]


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
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4000,
        help="Responses max_output_tokens (raise for gpt-5 high reasoning)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    args = parser.parse_args()

    model = require_chat_deployment()
    gold_docs = {d["context_id"]: d for d in json.loads(args.gold.read_text(encoding="utf-8"))}

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
    done = _load_done(args.out)
    console.print(
        f"model={model} sample={len(rows)} already_done={len(done)} "
        f"max_output_tokens={args.max_output_tokens} reasoning_effort=high out={args.out}"
    )

    results: list[dict[str, Any]] = []
    with args.out.open("a", encoding="utf-8") as out_fh:
        for src in rows:
            qid = src["id"]
            if qid in done:
                results.append(done[qid])
                console.print(f"[dim]skip {qid}[/dim]")
                continue

            gold_cid = src["gold_context_id"]
            doc = gold_docs.get(gold_cid)
            if not doc:
                console.print(f"[red]No gold_docs entry for {gold_cid}[/red]")
                return 1
            pdf_path = _resolve_pdf(doc, args.pdf_root)
            user = (
                f"Question: {src['question']}\n\n"
                "Using only the attached PDF, give a short final numeric answer when applicable.\n"
                "Final answer:"
            )

            t0 = time.perf_counter()
            console.print(f"→ {qid} pdf={pdf_path.relative_to(REPO_ROOT)} ({pdf_path.stat().st_size} bytes)")
            try:
                answer = chat_complete_pdf(
                    pdf_path=pdf_path,
                    user=user,
                    system=SYSTEM_PROMPT,
                    max_output_tokens=args.max_output_tokens,
                )
                err = None
            except Exception as exc:  # noqa: BLE001 — record per-row failures
                answer = ""
                err = f"{type(exc).__name__}: {exc}"
                console.print(f"[red]{qid} failed: {err}[/red]")

            latency = time.perf_counter() - t0
            nm_new = number_match(answer, src["program_answer"]) if answer else False
            out_row = {
                "id": qid,
                "subset": src.get("subset", "FinQA"),
                "method": "vision-pdf-gold",
                "model": model,
                "question": src["question"],
                "program_answer": src["program_answer"],
                "gold_context_id": gold_cid,
                "pdf_path": str(pdf_path.relative_to(REPO_ROOT)),
                "prev_answer": src.get("answer"),
                "prev_nm": False,
                "answer": answer,
                "nm": nm_new,
                "flipped": bool(nm_new),
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

    # Prefer file order if resuming mixed; rebuild from SAMPLE_IDS when complete.
    final_by_id = {r["id"]: r for r in results}
    ordered = [final_by_id[i] for i in SAMPLE_IDS if i in final_by_id]

    n = len(ordered)
    n_flip = sum(1 for r in ordered if r.get("flipped"))
    n_nm = sum(1 for r in ordered if r.get("nm"))
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    summary = {
        "n": n,
        "nm_rate": (n_nm / n) if n else 0.0,
        "flipped": n_flip,
        "model": model,
        "method": "vision-pdf-gold",
        "reasoning_effort": "high",
        "max_output_tokens": args.max_output_tokens,
        "input": _rel(args.input),
        "out": _rel(args.out),
        "vs_gpt5_mini_vision": {
            "vision_flipped": 3,
            "vision_nm_rate": 0.1875,
        },
        "vs_gpt5_text": {
            "text_flipped": 0,
            "text_nm_rate": 0.0,
        },
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    table = Table(title="Vision PDF NM=false sample")
    table.add_column("id")
    table.add_column("gold")
    table.add_column("vision_answer_short")
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
