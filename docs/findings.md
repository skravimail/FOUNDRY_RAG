# T²-RAGBench findings

Shared notes from Foundry IQ and local RAG evals on [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench).  
Plan: [`docs/plans/2026-07-24-foundryiq-t2-ragbench-eval.md`](plans/2026-07-24-foundryiq-t2-ragbench-eval.md).  
Raw runs: `data/t2_ragbench/results/`.

---

## Pilot

**Scope:** FinQA `test`, **N=50** questions (seed **42**), **48** unique gold PDFs.  
**Metrics:** Number Match (ε=1e−2), MRR@3, R@3 vs gold `context_id`.  
**Date:** 2026-07-25.

| Method | Retrieval | Embedding model | Generative model | NM | MRR@3 | R@3 | n | Run id |
|--------|-----------|-----------------|------------------|-----|-------|-----|---|--------|
| Foundry IQ / Knowledge Base (Standard Search) | Agentic KB (answer synthesis) | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 1.00 | 1.00 | 50 | `pilot50` |
| Foundry IQ / Knowledge Base (Basic Search) | Agentic KB (answer synthesis) | `text-embedding-3-small` | `gpt-5-mini` | 0.64 | 1.00 | 1.00 | 50 | `foundryiq-basic-kb-pilot50` |
| Postgres / pgvector | Vector only | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 0.92 | 0.98 | 50 | `pgvector-pilot50` |
| Postgres / pgvector | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-5-mini` | **0.72** | 0.95 | 1.00 | 50 | `pgvector-hybrid-pilot50` |
| Postgres / pgvector | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-4.1-mini` | **0.72** | 0.92 | 0.96 | 50 | `pgvector-hybrid-gpt41mini-pilot50` |
| Postgres / pgvector (pymupdf4llm extract) | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-5-mini` | 0.60 | 0.96 | 0.98 | 50 | `pgvector-hybrid-pymupdf4llm-pilot50` |
| LanceDB | Vector only | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 0.92 | 0.98 | 50 | `lancedb-vector-pilot50` |
| LanceDB | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-5-mini` | 0.68 | 0.95 | 1.00 | 50 | `20260725T234144Z-0dcd33` |
| Azure AI Search (Basic) | Vector only | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 0.92 | 0.98 | 50 | `azure-search-vector-pilot50` |
| Azure AI Search (Basic) | BM25 only | `text-embedding-3-small` | `gpt-5-mini` | **0.72** | 0.94 | 1.00 | 50 | `azure-search-bm25-pilot50` |
| Azure AI Search (Basic) | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | **0.97** | 1.00 | 50 | `azure-search-hybrid-pilot50` |

### Notes

- Pilot corpus is gold-only (48 PDFs). R@3 can still be &lt; 1 if the wrong gold PDF ranks above the true one; MRR@3 &lt; 1 when gold is not rank 1.
- Hybrid BM25+vector improved NM vs vector-only on both local stores; Foundry IQ led on MRR@3 (perfect rank-1 citations on this pilot).
- Generator ablation (same hybrid index): `gpt-4.1-mini` matched `gpt-5-mini` on NM (0.72); MRR@3/R@3 slightly lower (0.92 / 0.96 vs 0.95 / 1.00).
- PDF extract swap (same hybrid + `gpt-5-mini`): `pymupdf4llm` markdown tables look cleaner than `pypdf`, but NM dropped **0.72 → 0.60** (MRR@3 0.95→0.96, R@3 1.00→0.98). Markdown/OCR noise may hurt numeric extraction more than it helps layout.
- Azure AI Search Basic ablation (same index, `gpt-5-mini`): BM25-only led NM (0.72); hybrid led MRR@3 (0.97) with perfect R@3; vector-only matched Foundry IQ NM (0.66) with one R@3 miss (0.98). Service `foundryiq-search-basic` left running (~$74/mo SU).
- Key takeaway: Foundry IQ agentic KB orchestration helped **finding** the right chunk (MRR@3=1.00 vs Azure Search hybrid 0.97) but did **not** improve Number Match (NM=0.66 same as Search hybrid; BM25-only got NM=0.72). Report line: "orchestration helped finding the right chunk; it didn’t make the model compute/extract the answer better."
- Foundry IQ on Basic Search (`foundryiq-search-basic`, KB `t2-ragbench-knowledge-base`): NM **0.64** (vs 0.66 Standard IQ / Azure hybrid); MRR@3/R@3 still perfect (1.00). Confirms Basic is enough for agentic KB; answer quality not improved vs direct hybrid.
- Oracle Context baseline (NM ceiling) not yet run on this pilot.

### Full dataset

*TBD — evaluate full FinQA / ConvFinQA / TAT-DQA after pilot.*
