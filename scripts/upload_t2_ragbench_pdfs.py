#!/usr/bin/env python3
"""Phase 2: upload local T2-RAGBench pilot PDFs to Azure Blob Storage.

Blob path layout (stable IDs for MRR@3 mapping):
  t2rag/{subset}/{context_id}/{file_basename}.pdf

Uses DefaultAzureCredential (az login).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Upload T2-RAGBench pilot PDFs to Blob")
    parser.add_argument(
        "--gold-docs",
        type=Path,
        default=ROOT / "data" / "t2_ragbench" / "gold_docs.json",
    )
    parser.add_argument(
        "--account",
        default=os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "foudryragstorageacct"),
    )
    parser.add_argument(
        "--container",
        default=os.environ.get("AZURE_STORAGE_CONTAINER", "t2-ragbench"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    docs = json.loads(args.gold_docs.read_text(encoding="utf-8"))
    account_url = f"https://{args.account}.blob.core.windows.net"
    client = BlobServiceClient(account_url, credential=DefaultAzureCredential())
    container = client.get_container_client(args.container)

    uploaded = 0
    skipped = 0
    missing = []
    manifest = []

    for doc in docs:
        local_rel = doc.get("local_path")
        if not local_rel:
            missing.append(doc["context_id"])
            continue
        local_path = ROOT / local_rel
        if not local_path.is_file():
            missing.append(doc["context_id"])
            continue

        basename = Path(doc["file_name"]).name
        blob_name = f"t2rag/{doc['subset']}/{doc['context_id']}/{basename}"
        manifest.append(
            {
                "context_id": doc["context_id"],
                "subset": doc["subset"],
                "file_name": doc["file_name"],
                "blob_name": blob_name,
                "local_path": local_rel,
            }
        )

        if args.dry_run:
            print(f"DRY {local_rel} -> {blob_name}")
            uploaded += 1
            continue

        blob = container.get_blob_client(blob_name)
        if blob.exists():
            skipped += 1
            print(f"SKIP exists {blob_name}")
            continue

        with local_path.open("rb") as fh:
            blob.upload_blob(
                fh,
                overwrite=False,
                content_settings=ContentSettings(content_type="application/pdf"),
                metadata={
                    "context_id": doc["context_id"],
                    "subset": doc["subset"],
                    "source_file": basename,
                },
            )
        uploaded += 1
        print(f"OK  {blob_name}")

    out = ROOT / "data" / "t2_ragbench" / "blob_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "account": args.account,
        "container": args.container,
        "uploaded_or_dry": uploaded,
        "skipped_existing": skipped,
        "missing_local": missing,
        "manifest": str(out.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
