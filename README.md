# Foundry RAG — Python port of deployed-in-azure/RAG

Nine RAG-mechanism demos on **Microsoft Foundry**, using `uv`, `DefaultAzureCredential`,
and the synthetic corpora under `data/`.

## Setup

```bash
uv sync
cp .env.example .env   # then fill values
az login
```

Required for most modules:

| Variable | Purpose |
|----------|---------|
| `FOUNDRY_PROJECT_ENDPOINT` | Foundry project URI |
| `FOUNDRY_CHAT_DEPLOYMENT` | Chat model (e.g. `gpt-5-mini`) |
| `FOUNDRY_EMBEDDING_DEPLOYMENT` | Embedding model (e.g. `text-embedding-3-small`) |
| `AZURE_AI_SEARCH_ENDPOINT` | Search service for modules 02–05 |

Module 08 also needs Neo4j (`docker compose up -d`) and `NEO4J_*` vars.
Module 09 needs a provisioned Knowledge Base (`AZURE_AI_SEARCH_KNOWLEDGE_BASE`).

## Mechanism docs

Each mechanism has a dedicated README (purpose, problem, figure, how to run):

| # | Doc | Script |
|---|-----|--------|
| 01 | [Naive RAG](docs/rag/01_naive_rag.md) | `uv run python scripts/run_01_naive_rag.py` |
| 02 | [Hybrid RAG](docs/rag/02_hybrid_rag.md) | `uv run python scripts/run_02_hybrid_rag.py` |
| 03 | [Re-ranking RAG](docs/rag/03_reranking_rag.md) | `uv run python scripts/run_03_reranking_rag.py` |
| 04 | [Multi-Query RAG](docs/rag/04_multiquery_rag.md) | `uv run python scripts/run_04_multiquery_rag.py` |
| 05 | [HyDE RAG](docs/rag/05_hyde_rag.md) | `uv run python scripts/run_05_hyde_rag.py` |
| 06 | [Chunking Strategies](docs/rag/06_chunking_strategies.md) | `uv run python scripts/run_06_chunking_strategies.py` |
| 07 | [Contextual Retrieval](docs/rag/07_contextual_retrieval.md) | `uv run python scripts/run_07_contextual_retrieval.py` |
| 08 | [GraphRAG](docs/rag/08_graphrag.md) | `uv run python scripts/run_08_graphrag.py` |
| 09 | [FoundryIQ](docs/rag/09_foundryiq.md) | `uv run python scripts/run_09_foundryiq.py` |

## Parity notes (C# → Python / Foundry)

| Topic | Port choice |
|-------|-------------|
| Spectre.Console → `rich` | Interactive CLIs |
| Azure OpenAI SDK → Foundry `AIProjectClient.get_openai_client()` | Chat via Responses API (`instructions=`) |
| Embeddings | Account-level `/openai/v1` (project-scoped embeddings currently 404) |
| Tokenizers | `tiktoken` `cl100k_base` |
| Search indexes | Reference `index_definition.json` sanitized for current API (drops `flightingOptIn`, unused vectorizers) |
| FoundryIQ | Knowledge Base client; Contoso sources must be provisioned separately (no `Data/` in reference) |

## Tests

```bash
uv run pytest -q
```
