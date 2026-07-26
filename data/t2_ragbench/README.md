# T²-RAGBench local eval artifacts

Built from [G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench) for Foundry IQ evaluation.

## Regenerate pilot (Phase 1)

```bash
uv run python scripts/prep_t2_ragbench_pilot.py --subset FinQA --split test --n 50 --seed 42
```

## Contents

| Artifact | Purpose |
|----------|---------|
| `pilot.jsonl` | 50 FinQA `test` questions + gold fields (`program_answer`, `context_id`, …) |
| `gold_docs.json` | Unique gold documents for MRR@3 / Blob path mapping |
| `oracle_contexts.jsonl` | Same questions with gold `context` text (Oracle NM baseline) |
| `pilot_meta.json` | Sample seed/size/download stats |
| `pdfs/FinQA/<context_id>/…` | Local pilot PDFs only (not full HF corpus) |

Gold answers stay here for the eval harness — **do not** upload `pilot.jsonl` / `oracle_contexts.jsonl` into the Knowledge Base. Phase 2 uploads **PDFs only** to Azure Blob.

## Upload PDFs to Blob (Phase 2)

```bash
uv run python scripts/upload_t2_ragbench_pdfs.py
```

Blobs land at `t2rag/FinQA/<context_id>/<page_*.pdf>` in container `t2-ragbench`. See `blob_manifest.json`.

## Local Postgres / pgvector RAG

```bash
docker compose up -d
uv run python scripts/index_t2_ragbench_pgvector.py --reset
# hybrid = Okapi BM25 + dense vector (RRF); also: --retrieval vector|bm25
uv run python scripts/eval_foundryiq_t2_ragbench.py --method pgvector --retrieval hybrid --run-id pgvector-hybrid-pilot50
```

Uses Foundry embeddings + chat; stores chunks in local `pgvector`. Default retrieval is **BM25 + vector** (RRF). Results under `results/`.

## Local LanceDB RAG (embedded, no Docker)

```bash
uv run python scripts/index_t2_ragbench_lancedb.py --reset
uv run python scripts/eval_foundryiq_t2_ragbench.py --method lancedb --retrieval hybrid --run-id lancedb-hybrid-pilot50
```

Index lands in `data/t2_ragbench/lancedb/` (gitignored). Same `--retrieval hybrid|vector|bm25` options as pgvector.

## Azure AI Search hybrid RAG (Basic)

```bash
# Requires AZURE_AI_SEARCH_ENDPOINT → Basic service (e.g. foundryiq-search-basic)
uv run python scripts/index_t2_ragbench_azure_search.py --reset
uv run python scripts/eval_foundryiq_t2_ragbench.py \
  --method azure_search --retrieval hybrid --run-id azure-search-hybrid-pilot50
```

Uses the same chunking/embeddings as local stores; hybrid fusion is Azure Search RRF (`search_text` + `VectorizedQuery`). Index name defaults to `t2-ragbench-chunks` (`T2_AZURE_SEARCH_INDEX`).
