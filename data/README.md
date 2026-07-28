# Data — Microsoft Foundry RAG evaluation

This directory holds corpora and eval artifacts for **FOUNDRY_RAG**: a RAG evaluation framework on **Microsoft Foundry** (chat + embeddings via Foundry deployments, with Azure AI Search / Foundry IQ, plus local baselines).

Primary eval track: [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench) FinQA pilot (Number Match, MRR@3, R@3). Mechanism demos also use small synthetic starship corpora ported from [deployed-in-azure/RAG](https://github.com/deployed-in-azure/RAG).

Full write-up: [`docs/findings.md`](../docs/findings.md). Per-corpus details: [`t2_ragbench/README.md`](t2_ragbench/README.md).

## Layout

| Path | Role |
|------|------|
| `t2_ragbench/` | FinQA pilot Q&A, gold PDFs, oracle contexts, indexed stores, **results/** |
| `01_naive_rag/` | Synthetic markdown docs (module 01) |
| `02_hybrid_rag/` | `starships.json` + search index definition (module 02) |
| `09_foundryiq/` | Foundry IQ notes only (no reference `Data/` corpus) |

Rebuild the T² pilot with:

```bash
uv run python scripts/prep_t2_ragbench_pilot.py
```

## Findings summary (FinQA pilot, N=50)

**Setup:** FinQA `test`, seed **42**, **48** gold PDFs. Generator default `gpt-5-mini`; embeddings `text-embedding-3-small`. Metrics: Number Match (ε=1e−2), MRR@3, R@3.

| Method | NM | MRR@3 | R@3 |
|--------|-----|-------|-----|
| Oracle Context (gold HF text, no retrieval) | **0.76** | 1.00 | 1.00 |
| pgvector / Azure BM25 (best RAG NM) | **0.72** | ~0.94–0.95 | 1.00 |
| pgvector hybrid (BM25+vector RRF) | **0.72** | 0.95 | 1.00 |
| LanceDB hybrid | 0.68 | 0.95 | 1.00 |
| Azure AI Search hybrid (Basic) | 0.66 | **0.97** | 1.00 |
| Foundry IQ KB (Standard Search) | 0.66 | 1.00 | 1.00 |
| Foundry IQ KB (Basic Search) | 0.64 | 1.00 | 1.00 |
| Vector-only (pgvector / LanceDB / Search) | 0.66 | ~0.92 | ~0.98 |

**Takeaways**

- **Retrieval is near the ceiling:** best RAG NM (0.72) is within ~4 pts of Oracle (0.76). Remaining misses are mostly numeric extraction/computation, not “wrong PDF.”
- **Hybrid / BM25 beat vector-only** on NM for local stores and Azure Search BM25-only.
- **Foundry IQ finds well, answers no better:** agentic KB hit perfect MRR@3/R@3 but NM stayed ~0.64–0.66 — orchestration helped ranking, not Number Match vs direct BM25/hybrid.
- **Basic Search is enough for the KB path** on this pilot (0.64 vs 0.66 Standard); SKU upgrade did not fix answer quality.
- **PDF extract matters:** `pypdf` hybrid ~0.68–0.72 NM; `pymupdf4llm` markdown dropped to 0.60 — cleaner-looking tables are not automatically better for FinQA numbers.
- **Generator swap** (`gpt-4.1-mini` vs `gpt-5-mini` on the same hybrid index) matched NM at 0.72.

Raw run JSONL/CSV: `t2_ragbench/results/`. Full dataset (ConvFinQA / TAT-DQA) not yet evaluated.
