# Foundry RAG — Python port of deployed-in-azure/RAG

Focused RAG demos on **Microsoft Foundry**, using `uv`, `DefaultAzureCredential`,
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
| `AZURE_AI_SEARCH_ENDPOINT` | Search service for module 02 Hybrid RAG |

Module 09 needs a provisioned Knowledge Base (`AZURE_AI_SEARCH_KNOWLEDGE_BASE`).

## Mechanism docs

Each mechanism has a dedicated README (purpose, problem, figure, how to run):

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
| Search indexes | Reference `index_definition.json` sanitized for current API (drops `flightingOptIn`, unused vectorizers) |
| FoundryIQ | Knowledge Base client; Contoso sources must be provisioned separately (no `Data/` in reference) |

## Tests

```bash
uv run pytest -q
```
