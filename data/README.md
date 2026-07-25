# Synthetic data ported from `deployed-in-azure/RAG`

Source: https://github.com/deployed-in-azure/RAG (copied as-is from each module's `Data/`).

| This repo | Reference module | Contents |
|-----------|------------------|----------|
| `01_naive_rag/` | `01_NaiveRAG/Data` | 5 starship markdown docs |
| `02_hybrid_rag/` | `02_HybridRAG/Data` | `starships.json` + `index_definition.json` |
| `09_foundryiq/` | `09_FoundryIQ` | **No `Data/` in reference** (KB sample only) |
| `t2_ragbench/` | [G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench) | FinQA pilot Q&A + PDFs for Foundry IQ eval |

## Notes

- Module 01 uses per-ship markdown files rather than `starships.json`.
- Module 02 has a search-backed `starships.json` + module-specific `index_definition.json`.
- Rebuild the T² pilot with `uv run python scripts/prep_t2_ragbench_pilot.py`.
