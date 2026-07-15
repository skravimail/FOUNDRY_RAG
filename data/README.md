# Synthetic data ported from `deployed-in-azure/RAG`

Source: https://github.com/deployed-in-azure/RAG (copied as-is from each module's `Data/`).

| This repo | Reference module | Contents |
|-----------|------------------|----------|
| `01_naive_rag/` | `01_NaiveRAG/Data` | 5 starship markdown docs |
| `02_hybrid_rag/` | `02_HybridRAG/Data` | `starships.json` + `index_definition.json` |
| `03_reranking_rag/` | `03_ReRankingRAG/Data` | `starships.json` + `index_definition.json` |
| `04_multiquery_rag/` | `04_MultiQueryRAG/Data` | `starships.json` + `index_definition.json` |
| `05_hyde_rag/` | `05_HyDERAG/Data` | `starships.json` + `index_definition.json` |
| `06_chunking_strategies/` | `06_ChunkingStrategies/Data` | `grounding-data-design.md` |
| `07_contextual_retrieval/` | `07_ContextualRetrieval/Data` | `remote_work_policy_pl.md` |
| `08_graphrag/` | `08_GraphRAG/Data` | `grounding-data-design.md` |
| `09_foundryiq/` | `09_FoundryIQ` | **No `Data/` in reference** (KB sample only) |

## Shared vs module-specific

- `starships.json`: identical across modules 03, 04, 05; module 02 has a **different** (larger) copy.
- `index_definition.json`: **distinct** per search-backed module (02, 03, 04, 05) — not interchangeable.
- `grounding-data-design.md`: identical between modules 06 and 08.
- Module 01 uses per-ship markdown files rather than `starships.json`.
- Module 07 uses a remote-work policy doc (not the starship corpus).
