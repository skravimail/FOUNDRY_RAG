# FOUNDRY_RAG — Microsoft Foundry RAG evaluation

RAG evaluation framework on **Microsoft Foundry**: chat + embeddings via Foundry deployments, with **Foundry IQ / Azure AI Search** knowledge bases and local baselines (pgvector, LanceDB).

Primary benchmark: [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench) FinQA (Number Match, MRR@3, R@3). Also includes mechanism demos ported from [deployed-in-azure/RAG](https://github.com/deployed-in-azure/RAG).

| Doc | Purpose |
|-----|---------|
| [`docs/findings.md`](docs/findings.md) | Full pilot results table + notes |
| [`data/README.md`](data/README.md) | Corpora, pilot artifacts, results paths |
| [`data/t2_ragbench/README.md`](data/t2_ragbench/README.md) | How to prep / index / eval T²-RAGBench |

Branch: [`master_maf`](https://github.com/skravimail/FOUNDRY_RAG/tree/master_maf).

## Findings summary (FinQA pilot, N=50)

**Setup:** FinQA `test`, seed **42**, **48** gold PDFs. Generator `gpt-5-mini`; embeddings `text-embedding-3-small`. Metrics: Number Match (ε=1e−2), MRR@3, R@3.

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

- **Retrieval is near the ceiling:** best RAG NM (0.72) is within ~4 pts of Oracle (0.76). Remaining misses are mostly numeric extraction/computation, not wrong-document retrieval.
- **Hybrid / BM25 beat vector-only** on NM for local stores and Azure Search BM25-only.
- **Foundry IQ finds well, answers no better:** agentic KB hit perfect MRR@3/R@3 but NM stayed ~0.64–0.66 — orchestration helped ranking, not Number Match vs direct BM25/hybrid.
- **Basic Search is enough for the KB path** on this pilot (0.64 vs 0.66 Standard).
- **PDF extract matters:** `pypdf` hybrid ~0.68–0.72 NM; `pymupdf4llm` markdown dropped to 0.60.
- **Generator swap** (`gpt-4.1-mini` vs `gpt-5-mini` on the same hybrid index) matched NM at 0.72.

Raw runs: `data/t2_ragbench/results/`. Full FinQA / ConvFinQA / TAT-DQA not yet evaluated.

## Setup

```bash
uv sync
cp .env.example .env   # then fill values
az login
```

| Variable | Purpose |
|----------|---------|
| `FOUNDRY_PROJECT_ENDPOINT` | Foundry project URI |
| `FOUNDRY_CHAT_DEPLOYMENT` | Chat model (e.g. `gpt-5-mini`) |
| `FOUNDRY_EMBEDDING_DEPLOYMENT` | Embedding model (e.g. `text-embedding-3-small`) |
| `AZURE_AI_SEARCH_ENDPOINT` | Search service (Hybrid RAG / Foundry IQ) |

Module 09 / KB eval needs a provisioned Knowledge Base (`AZURE_AI_SEARCH_KNOWLEDGE_BASE`).

### Quick T²-RAGBench eval

```bash
uv run python scripts/prep_t2_ragbench_pilot.py
docker compose up -d   # local pgvector
uv run python scripts/index_t2_ragbench_pgvector.py --reset
uv run python scripts/eval_foundryiq_t2_ragbench.py --method pgvector --retrieval hybrid --run-id pgvector-hybrid-pilot50
```

See [`data/t2_ragbench/README.md`](data/t2_ragbench/README.md) for Foundry IQ, LanceDB, and Azure AI Search paths.

## Mechanism demos

| # | Doc | Script |
|---|-----|--------|
| 01 | [Naive RAG](docs/rag/01_naive_rag.md) | `uv run python scripts/run_01_naive_rag.py` |
| 02 | [Hybrid RAG](docs/rag/02_hybrid_rag.md) | `uv run python scripts/run_02_hybrid_rag.py` |
| 09 | [FoundryIQ](docs/rag/09_foundryiq.md) | `uv run python scripts/run_09_foundryiq.py` |

Hybrid RAG also has a Foundry Agent Service CLI: `uv run python scripts/run_02_hybrid_rag_agent.py`.

## Parity notes (C# → Python / Foundry)

| Topic | Port choice |
|-------|-------------|
| Spectre.Console → `rich` | Interactive CLIs |
| Azure OpenAI SDK → Foundry `AIProjectClient.get_openai_client()` | Chat via Responses API (`instructions=`) |
| Embeddings | Account-level `/openai/v1` (project-scoped embeddings currently 404) |
| Tokenizers | `tiktoken` `cl100k_base` |
| Search indexes | Reference `index_definition.json` sanitized for current API |
| FoundryIQ | Knowledge Base client; sources provisioned separately |

## Tests

```bash
uv run pytest -q
```
