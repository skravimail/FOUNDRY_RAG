#!/usr/bin/env python3
"""Evaluate RAG methods against T²-RAGBench pilot (NM + MRR@3 + R@3)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Allow `uv run python scripts/...` without editable install edge cases.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundry_rag.eval.t2_ragbench.load import (  # noqa: E402
    DEFAULT_ORACLE,
    DEFAULT_PILOT,
    load_oracle_by_id,
    load_pilot,
)
from foundry_rag.eval.t2_ragbench.metrics import (  # noqa: E402
    mrr_at_k,
    number_match,
    recall_at_k,
)
from foundry_rag.eval.t2_ragbench.azure_search_runner import (  # noqa: E402
    METHOD_NAMES as AZURE_SEARCH_METHODS,
)
from foundry_rag.eval.t2_ragbench.lancedb_runner import METHOD_NAMES as LANCEDB_METHODS  # noqa: E402
from foundry_rag.eval.t2_ragbench.pgvector_runner import METHOD_NAMES as PGVECTOR_METHODS  # noqa: E402
from foundry_rag.eval.t2_ragbench.runner import (  # noqa: E402
    run_azure_search,
    run_foundryiq,
    run_lancedb,
    run_oracle,
    run_pgvector,
)
from foundry_rag.paths import DATA_ROOT  # noqa: E402

console = Console()
RESULTS_DIR = DATA_ROOT / "t2_ragbench" / "results"

METHOD_LABELS = {
    "foundryiq": "FoundryIQ / Knowledge Base",
    "oracle": "Oracle Context",
    "pgvector": PGVECTOR_METHODS["hybrid"],
    "lancedb": LANCEDB_METHODS["hybrid"],
    "azure_search": AZURE_SEARCH_METHODS["hybrid"],
}


def _load_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = f"{row.get('method')}|{row.get('id')}"
            done.add(key)
    return done


def _serialize_raw(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "as_dict"):
        try:
            return raw.as_dict()
        except Exception:
            pass
    return {"repr": repr(raw)[:2000]}


def _summarize(rows: list[dict[str, Any]], *, run_id: str, model: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["subset"], row["method"])].append(row)

    summary: list[dict[str, Any]] = []
    for (subset, method), items in sorted(groups.items()):
        n = len(items)
        nm_vals = [r["nm"] for r in items if r.get("nm") is not None]
        summary.append(
            {
                "subset": subset,
                "NM": (sum(1 for v in nm_vals if v) / len(nm_vals)) if nm_vals else None,
                "MRR@3": sum(r["mrr_at_3"] for r in items) / n if n else 0.0,
                "R@3": sum(r["r_at_3"] for r in items) / n if n else 0.0,
                "n": n,
                "model": model,
                "method": method,
                "run_id": run_id,
            }
        )
    return summary


def _write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    fields = ["subset", "NM", "MRR@3", "R@3", "n", "model", "method", "run_id"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    **row,
                    "NM": "" if row["NM"] is None else f"{row['NM']:.4f}",
                    "MRR@3": f"{row['MRR@3']:.4f}",
                    "R@3": f"{row['R@3']:.4f}",
                }
            )


def _resolve_methods(choice: str) -> list[str]:
    if choice == "all":
        return ["foundryiq", "oracle", "pgvector", "lancedb", "azure_search"]
    if choice == "both":
        return ["foundryiq", "oracle"]
    return [choice]


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument(
        "--method",
        choices=("foundryiq", "oracle", "pgvector", "lancedb", "azure_search", "both", "all"),
        default="pgvector",
        help="Which baseline(s) to run (default: pgvector)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retrieval",
        choices=("hybrid", "vector", "bm25"),
        default="hybrid",
        help=(
            "Retrieval mode for pgvector/lancedb/azure_search "
            "(default: hybrid = BM25 + vector RRF)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question/method pairs already present in the results JSONL",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds between calls (useful for Foundry IQ rate limits)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Score MRR@3 / R@3 only (skip chat generation / NM). "
            "Useful for pure BM25 ablations without Foundry credentials."
        ),
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    model = os.environ.get("FOUNDRY_CHAT_DEPLOYMENT", "unknown")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = RESULTS_DIR / f"{run_id}.jsonl"
    out_summary = RESULTS_DIR / f"{run_id}.summary.csv"

    questions = load_pilot(args.pilot)
    if args.offset:
        questions = questions[args.offset :]
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    methods = _resolve_methods(args.method)
    METHOD_LABELS["pgvector"] = PGVECTOR_METHODS[args.retrieval]
    METHOD_LABELS["lancedb"] = LANCEDB_METHODS[args.retrieval]
    METHOD_LABELS["azure_search"] = AZURE_SEARCH_METHODS[args.retrieval]
    oracle_by_id = load_oracle_by_id(args.oracle) if "oracle" in methods else {}
    done = _load_done_ids(out_jsonl) if args.resume else set()

    console.print(
        f"[bold]T²-RAGBench eval[/bold] run_id={run_id} n={len(questions)} "
        f"methods={methods} retrieval={args.retrieval} "
        f"retrieval_only={args.retrieval_only}"
    )
    console.print(f"results → {out_jsonl}")

    scored_rows: list[dict[str, Any]] = []
    if args.resume and out_jsonl.is_file():
        with out_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    scored_rows.append(json.loads(line))

    generate = not args.retrieval_only
    if args.retrieval_only and any(m in methods for m in ("foundryiq", "oracle")):
        console.print(
            "[yellow]--retrieval-only ignores foundryiq/oracle (they require generation)[/yellow]"
        )
        methods = [m for m in methods if m not in ("foundryiq", "oracle")]

    errors = 0
    for i, qrow in enumerate(questions, start=1):
        qid = qrow["id"]
        for method in methods:
            label = METHOD_LABELS[method]
            key = f"{label}|{qid}"
            if key in done:
                console.print(f"[dim]skip[/dim] {method} {qid}")
                continue

            t0 = time.perf_counter()
            try:
                if method == "foundryiq":
                    result = run_foundryiq(qrow["question"], top_k=args.top_k)
                elif method == "pgvector":
                    result = run_pgvector(
                        qrow["question"],
                        top_k=args.top_k,
                        retrieval=args.retrieval,
                        generate=generate,
                    )
                elif method == "lancedb":
                    result = run_lancedb(
                        qrow["question"],
                        top_k=args.top_k,
                        retrieval=args.retrieval,
                        generate=generate,
                    )
                elif method == "azure_search":
                    result = run_azure_search(
                        qrow["question"],
                        top_k=args.top_k,
                        retrieval=args.retrieval,
                        generate=generate,
                    )
                else:
                    oracle = oracle_by_id.get(qid)
                    if not oracle:
                        raise KeyError(f"Missing oracle context for {qid}")
                    result = run_oracle(
                        qrow["question"],
                        oracle["context"],
                        qrow["context_id"],
                    )
                latency_s = time.perf_counter() - t0
                nm = (
                    None
                    if args.retrieval_only
                    else number_match(result["answer"], qrow["program_answer"])
                )
                mrr = mrr_at_k(result["ranked_context_ids"], qrow["context_id"], k=args.top_k)
                r_at = recall_at_k(result["ranked_context_ids"], qrow["context_id"], k=args.top_k)
                if method == "oracle":
                    mrr, r_at = 1.0, 1.0

                record = {
                    "run_id": run_id,
                    "id": qid,
                    "subset": qrow.get("subset", "FinQA"),
                    "method": result["method"],
                    "model": model if generate else "retrieval-only",
                    "question": qrow["question"],
                    "program_answer": qrow["program_answer"],
                    "gold_context_id": qrow["context_id"],
                    "answer": result["answer"],
                    "ranked_context_ids": result["ranked_context_ids"],
                    "nm": nm,
                    "mrr_at_3": mrr,
                    "r_at_3": r_at,
                    "latency_s": round(latency_s, 3),
                    "error": None,
                    "raw": _serialize_raw(result.get("raw")),
                }
            except Exception as exc:  # noqa: BLE001 — persist per-question failures
                errors += 1
                latency_s = time.perf_counter() - t0
                record = {
                    "run_id": run_id,
                    "id": qid,
                    "subset": qrow.get("subset", "FinQA"),
                    "method": label,
                    "model": model if generate else "retrieval-only",
                    "question": qrow["question"],
                    "program_answer": qrow["program_answer"],
                    "gold_context_id": qrow["context_id"],
                    "answer": "",
                    "ranked_context_ids": [],
                    "nm": None if args.retrieval_only else False,
                    "mrr_at_3": 0.0,
                    "r_at_3": 0.0,
                    "latency_s": round(latency_s, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw": None,
                }

            with out_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            scored_rows.append(record)
            status = "OK" if not record["error"] else "ERR"
            nm_disp = "n/a" if record["nm"] is None else str(record["nm"])
            console.print(
                f"[{i}/{len(questions)}] {status} {method} {qid} "
                f"NM={nm_disp} MRR={record['mrr_at_3']:.2f} "
                f"R={record['r_at_3']:.0f} {record['latency_s']:.1f}s"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    summary = _summarize(scored_rows, run_id=run_id, model=model if generate else "retrieval-only")
    _write_summary_csv(out_summary, summary)

    table = Table(title=f"T²-RAGBench summary ({run_id})")
    for col in ("subset", "method", "n", "NM", "MRR@3", "R@3", "model"):
        table.add_column(col)
    for row in summary:
        table.add_row(
            row["subset"],
            row["method"],
            str(row["n"]),
            "n/a" if row["NM"] is None else f"{row['NM']:.3f}",
            f"{row['MRR@3']:.3f}",
            f"{row['R@3']:.3f}",
            row["model"],
        )
    console.print(table)
    console.print(f"wrote {out_summary} (errors={errors})")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
