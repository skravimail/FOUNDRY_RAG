#!/usr/bin/env python3
"""Answer prebuilt text-RAG prompts with gpt-5 (no re-embed)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(os.environ.get("FOUNDRY_RAG_ROOT", Path.cwd()))
load_dotenv(REPO / ".env")

os.environ["FOUNDRY_CHAT_DEPLOYMENT"] = os.environ.get("FOUNDRY_CHAT_DEPLOYMENT") or "gpt-5"

from foundry_rag.eval.t2_ragbench.metrics import number_match  # noqa: E402
from foundry_rag.llm import chat_complete, require_chat_deployment  # noqa: E402

PROMPTS = REPO / "data/t2_ragbench/results/gpt5-text-nm-false-prompts.json"
OUT = REPO / "data/t2_ragbench/results/gpt5-text-nm-false-sample.jsonl"
SUMMARY = OUT.with_suffix(".summary.json")
MAX_TOKENS = int(os.environ.get("GPT5_MAX_OUTPUT_TOKENS", "4000"))


def main() -> int:
    model = require_chat_deployment()
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
    OUT.write_text("", encoding="utf-8")
    results = []
    print(f"model={model} n={len(prompts)} max_output_tokens={MAX_TOKENS}", flush=True)

    for p in prompts:
        qid = p["id"]
        print(f"→ {qid}", flush=True)
        t0 = time.perf_counter()
        err = None
        try:
            answer = chat_complete(
                system=p["system_prompt"],
                user=p["user_prompt"],
                max_output_tokens=MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            answer = ""
            err = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED {err}", flush=True)
        latency = time.perf_counter() - t0
        nm = number_match(answer, p["program_answer"]) if answer else False
        row = {
            "id": qid,
            "subset": p.get("subset", "FinQA"),
            "method": "pgvector-hybrid-text-gpt5",
            "model": model,
            "question": p["question"],
            "program_answer": p["program_answer"],
            "gold_context_id": p.get("gold_context_id"),
            "prev_answer": p.get("prev_answer"),
            "prev_nm": False,
            "prev_model": p.get("prev_model"),
            "answer": answer,
            "nm": nm,
            "flipped": bool(nm),
            "ranked_context_ids": p.get("ranked_context_ids"),
            "latency_s": round(latency, 3),
            "error": err,
            "max_output_tokens": MAX_TOKENS,
            "reasoning_effort": "high",
            "context_source": "replay-hybrid-ranked-ids",
        }
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.append(row)
        short = " ".join((answer or err or "").split())[:80]
        print(f"  nm={nm} flipped={nm} latency={latency:.1f}s answer={short}", flush=True)

    n = len(results)
    n_nm = sum(1 for r in results if r["nm"])
    n_flip = sum(1 for r in results if r["flipped"])
    summary = {
        "n": n,
        "nm_rate": (n_nm / n) if n else 0.0,
        "nm_true": n_nm,
        "flipped": n_flip,
        "model": model,
        "method": "pgvector-hybrid-text-gpt5",
        "reasoning_effort": "high",
        "max_output_tokens": MAX_TOKENS,
        "deployment": model,
        "out": str(OUT.relative_to(REPO)),
        "vs_gpt5_mini_text": {
            "baseline_nm": 0,
            "baseline_flipped": 0,
            "note": "All 16 were NM=false under gpt-5-mini text RAG",
        },
        "vs_vision_pdf_sample": {
            "vision_flipped": 3,
            "vision_nm_rate": 0.1875,
        },
        "per_id": [
            {"id": r["id"], "nm": r["nm"], "flipped": r["flipped"], "answer": r["answer"][:200]}
            for r in results
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"summary: nm={n_nm}/{n} ({summary['nm_rate']:.2%}) flipped={n_flip} → {SUMMARY}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
