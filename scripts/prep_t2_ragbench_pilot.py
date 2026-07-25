#!/usr/bin/env python3
"""Phase 1: build a local T2-RAGBench FinQA pilot set and download only needed PDFs.

Outputs under data/t2_ragbench/:
  - pilot.jsonl
  - gold_docs.json
  - oracle_contexts.jsonl
  - pdfs/  (local copies of pilot PDFs)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "G4KMU/t2-ragbench"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "t2_ragbench"


def _row_record(row: dict, *, subset: str) -> dict:
    return {
        "id": row["id"],
        "subset": subset,
        "split": row.get("split") or "test",
        "question": row["question"],
        "program_answer": row["program_answer"],
        "original_answer": row.get("original_answer"),
        "context_id": row["context_id"],
        "file_name": row["file_name"],
        "company_symbol": row.get("company_symbol"),
        "company_name": row.get("company_name"),
        "report_year": row.get("report_year"),
        "page_number": row.get("page_number"),
    }


def _hf_pdf_path(subset: str, file_name: str, *, split: str) -> str:
    # FinQA/TAT-DQA HF layout: data/FinQA/test/pdf/ETR/2016/page_23.pdf
    # ConvFinQA layout: data/ConvFinQA/pdf/...
    # Dataset file_name: pdf/ETR/2016/page_23.pdf
    rel = file_name.lstrip("/")
    if not rel.startswith("pdf/"):
        rel = f"pdf/{rel}"
    if subset == "ConvFinQA":
        return f"data/{subset}/{rel}"
    return f"data/{subset}/{split}/{rel}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare T2-RAGBench FinQA pilot")
    parser.add_argument("--subset", default="FinQA", choices=["FinQA", "ConvFinQA", "TAT-DQA"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=50, help="Pilot sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pdfs", action="store_true", help="Only write JSONL/JSON artifacts")
    args = parser.parse_args()

    # ConvFinQA uses turn_0 in the HF viewer; keep override simple.
    split = args.split
    if args.subset == "ConvFinQA" and split == "test":
        split = "turn_0"

    print(f"Loading {REPO_ID} config={args.subset} split={split} ...")
    ds = load_dataset(REPO_ID, args.subset, split=split)
    n = min(args.n, len(ds))
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = sorted(indices[:n])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dir = OUT_DIR / "pdfs" / args.subset
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pilot_path = OUT_DIR / "pilot.jsonl"
    oracle_path = OUT_DIR / "oracle_contexts.jsonl"
    gold_path = OUT_DIR / "gold_docs.json"

    gold_docs: OrderedDict[str, dict] = OrderedDict()
    downloaded = 0
    missing_pdfs: list[str] = []

    with pilot_path.open("w", encoding="utf-8") as pilot_f, oracle_path.open(
        "w", encoding="utf-8"
    ) as oracle_f:
        for i in indices:
            row = ds[i]
            rec = _row_record(row, subset=args.subset)
            pilot_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            oracle_f.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "question": rec["question"],
                        "program_answer": rec["program_answer"],
                        "context_id": rec["context_id"],
                        "context": row.get("context"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            cid = rec["context_id"]
            hf_path = _hf_pdf_path(args.subset, rec["file_name"], split=split)
            if cid not in gold_docs:
                gold_docs[cid] = {
                    "context_id": cid,
                    "file_name": rec["file_name"],
                    "subset": args.subset,
                    "split": split,
                    "hf_path": hf_path,
                    "local_path": None,
                    "question_ids": [],
                }
            gold_docs[cid]["question_ids"].append(rec["id"])

            if args.skip_pdfs:
                continue

            # Stable local path for Blob upload: pdfs/{subset}/{context_id}/{basename}
            local_name = Path(rec["file_name"]).name
            local_path = pdf_dir / cid / local_name
            if local_path.exists() and local_path.stat().st_size > 0:
                gold_docs[cid]["local_path"] = str(local_path.relative_to(ROOT))
                continue

            try:
                cached = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=hf_path,
                )
                local_path.parent.mkdir(parents=True, exist_ok=True)
                data = Path(cached).read_bytes()
                local_path.write_bytes(data)
                gold_docs[cid]["local_path"] = str(local_path.relative_to(ROOT))
                downloaded += 1
                print(f"  PDF {downloaded}: {hf_path} -> {local_path.relative_to(ROOT)}")
            except Exception as exc:  # noqa: BLE001 — keep pilot build going
                missing_pdfs.append(f"{hf_path}: {exc}")
                print(f"  WARN missing PDF {hf_path}: {exc}")

    gold_path.write_text(json.dumps(list(gold_docs.values()), indent=2), encoding="utf-8")

    meta = {
        "repo_id": REPO_ID,
        "subset": args.subset,
        "split": split,
        "n_requested": args.n,
        "n_pilot": n,
        "n_unique_docs": len(gold_docs),
        "seed": args.seed,
        "pdfs_downloaded": downloaded,
        "missing_pdfs": missing_pdfs,
        "artifacts": {
            "pilot": str(pilot_path.relative_to(ROOT)),
            "oracle_contexts": str(oracle_path.relative_to(ROOT)),
            "gold_docs": str(gold_path.relative_to(ROOT)),
            "pdfs_dir": str((pdf_dir).relative_to(ROOT)),
        },
    }
    (OUT_DIR / "pilot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("Phase 1 prep complete.")


if __name__ == "__main__":
    main()
