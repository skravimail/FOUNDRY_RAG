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
| Foundry IQ / Knowledge Base | Agentic KB (answer synthesis) | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 1.00 | 1.00 | 50 | `pilot50` |
| Postgres / pgvector | Vector only | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 0.92 | 0.98 | 50 | `pgvector-pilot50` |
| Postgres / pgvector | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-5-mini` | **0.72** | 0.95 | 1.00 | 50 | `pgvector-hybrid-pilot50` |
| Postgres / pgvector | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-4.1-mini` | **0.72** | 0.92 | 0.96 | 50 | `pgvector-hybrid-gpt41mini-pilot50` |
| LanceDB | Vector only | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 0.92 | 0.98 | 50 | `lancedb-vector-pilot50` |
| LanceDB | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-5-mini` | 0.68 | 0.95 | 1.00 | 50 | `20260725T234144Z-0dcd33` |
| Azure AI Search (Basic) | Vector only | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | 0.92 | 0.98 | 50 | `azure-search-vector-pilot50` |
| Azure AI Search (Basic) | BM25 only | `text-embedding-3-small` | `gpt-5-mini` | **0.72** | 0.94 | 1.00 | 50 | `azure-search-bm25-pilot50` |
| Azure AI Search (Basic) | BM25 + vector (RRF) | `text-embedding-3-small` | `gpt-5-mini` | 0.66 | **0.97** | 1.00 | 50 | `azure-search-hybrid-pilot50` |

### Notes

- Pilot corpus is gold-only (48 PDFs). R@3 can still be &lt; 1 if the wrong gold PDF ranks above the true one; MRR@3 &lt; 1 when gold is not rank 1.
- Hybrid BM25+vector improved NM vs vector-only on both local stores; Foundry IQ led on MRR@3 (perfect rank-1 citations on this pilot).
- Generator ablation (same hybrid index): `gpt-4.1-mini` matched `gpt-5-mini` on NM (0.72); MRR@3/R@3 slightly lower (0.92 / 0.96 vs 0.95 / 1.00).
- Azure AI Search Basic ablation (same index, `gpt-5-mini`): BM25-only led NM (0.72); hybrid led MRR@3 (0.97) with perfect R@3; vector-only matched Foundry IQ NM (0.66) with one R@3 miss (0.98). Service `foundryiq-search-basic` left running (~$74/mo SU).
- Foundry IQ further eval remains **deferred** (needs Standard Search for KB). Local stores + Basic Search are the active managed/local paths.
- Oracle Context baseline (NM ceiling) not yet run on this pilot.

### Full dataset

*TBD — evaluate full FinQA / ConvFinQA / TAT-DQA after pilot.*
