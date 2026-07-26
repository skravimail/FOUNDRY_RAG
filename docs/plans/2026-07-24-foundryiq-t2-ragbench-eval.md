# Plan: T²-RAGBench eval (Foundry IQ deferred; local Postgres/LanceDB done)

**Date:** 2026-07-24 (updated 2026-07-25)  
**Goal:** Produce leaderboard-comparable **Number Match (NM)** and **MRR@3** (plus R@3) on [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench) using Microsoft Foundry embeddings + chat.

**Current direction (2026-07-25):**

| Track | Status | Notes |
|-------|--------|-------|
| **Local Postgres / pgvector** | **Done (pilot)** | Docker; BM25+vector RRF; FinQA n=50 |
| **Local LanceDB** | **Done (pilot)** | Embedded; BM25+vector RRF; FinQA n=50 |
| **Azure AI Search hybrid (Basic)** | **Done (pilot)** | Direct hybrid index (not Foundry IQ KB); FinQA n=50 |
| **Foundry IQ / Azure AI Search KB** | **Deferred** | Needs Standard Search; Basic used for direct hybrid only |

---

## Problem frame

T²-RAGBench scoring needs custom **NM** (ε=1e−2) and **MRR@3 / R@3** — Foundry built-in evaluators do not replace these.

**Foundry IQ path** (`foundryiq.py` → Knowledge Base `ANSWER_SYNTHESIS`) was proven on a 50-question pilot, then **deferred** because Standard Azure AI Search (~$7–8/day) dominates cost vs token spend.

**Active path:** local vector stores (pgvector / LanceDB) + Foundry `text-embedding-3-small` + `gpt-5-mini` for answer generation.

---

## Recommended architecture

### A) Local stores (active)

```mermaid
flowchart LR
  HF[T2-RAGBench pilot PDFs] --> Prep[Local prep + chunk + embed]
  Prep --> PG[(Postgres / pgvector)]
  Prep --> Lance[(LanceDB)]
  PG --> Harness[Eval CLI]
  Lance --> Harness
  Prep --> Harness
  Harness --> Metrics[NM, MRR@3, R@3]
```

Retrieval modes: `vector` | `bm25` | `hybrid` (Okapi BM25 + dense cosine, RRF).

### B) Foundry IQ (deferred)

```mermaid
flowchart LR
  HF[T2-RAGBench PDFs] --> Blob[Azure Blob]
  Blob --> KS[AI Search Knowledge Source]
  KS --> KB[Knowledge Base / Foundry IQ]
  KB --> Harness[Eval CLI]
```

**To resume Foundry IQ:** recreate Standard Search → re-index Blob KS → set `.env` KB/KS → `--method foundryiq`.

---

## Key design decisions

| Decision | Recommendation | Why |
|----------|-----------------|-----|
| Active SUT | **pgvector / LanceDB hybrid** | Cheap, stoppable, leaderboard metrics without Search S1 |
| Foundry IQ | **Deferred** after pilot | Search SKU cost, not model tokens |
| Corpus form | Pilot PDFs → extract text → chunks | Same PDFs for all methods |
| Document identity | `context_id` on chunks / blob path | MRR@3 |
| Embeddings / chat | Foundry `text-embedding-3-small` + `gpt-5-mini` | Shared generator across methods |
| Scope | FinQA `test` pilot N=50 (seed 42) | Done; full corpus = later |

---

## Phased steps

### Phase 0 — Prerequisites (Foundry + Search)

#### Phase 0 status (2026-07-25)

| Item | Status | Notes |
|------|--------|-------|
| Subscription | Done | `ab031c7f-87a9-41c0-87af-9c45f5a9d571` |
| Foundry project | Done | `foundry-rag` / `gpt-5-mini` / `text-embedding-3-small` |
| Storage + pilot Blob upload | Done | `foudryragstorageacct` / `t2-ragbench` |
| Standard Search for Foundry IQ | **Torn down** | `foundryiq-knowledge-resource` deleted 2026-07-25 (cost) |
| Basic Search (direct hybrid) | **Running** | `foundryiq-search-basic` / North Central US / ~$74/mo per SU |
| Free Search | Optional / unused | Not used for T² pilot |

### Phase 1 — Dataset prep (local) — **Done**

| Item | Status | Notes |
|------|--------|-------|
| Pilot N=50 FinQA test | Done | seed 42; 48 unique PDFs |
| Artifacts | Done | `data/t2_ragbench/pilot.jsonl`, `gold_docs.json`, `oracle_contexts.jsonl`, `pdfs/` |
| Prep script | Done | `scripts/prep_t2_ragbench_pilot.py` |

### Phase 2 — Ingest into Foundry IQ — **Done (then deferred)**

| Item | Status | Notes |
|------|--------|-------|
| Blob upload + KS/KB + indexer | Done | `t2-ragbench-ks` / `t2-ragbench-kb`; 48/48 indexed |
| Smoke retrieve + citation map | Done | Search Index Data Reader; `context_id` from blob URL |
| Keep Standard Search running | **Stopped** | Service deleted; Foundry IQ eval **deferred** until recreated |

### Phase 3 — Eval harness — **Done**

| Item | Status | Notes |
|------|--------|-------|
| Loader / metrics / parse | Done | `src/foundry_rag/eval/t2_ragbench/` |
| Foundry IQ runner | Done | REST fallback; `--method foundryiq` |
| Oracle runner | Done | gold context → Foundry chat |
| CLI | Done | `scripts/eval_foundryiq_t2_ragbench.py` |
| Unit tests | Done | `tests/test_t2_ragbench_metrics.py`, `test_pgvector_hybrid.py` |
| Foundry IQ pilot (n=50) | Done | `pilot50`: NM=0.66, MRR@3=1.00, R@3=1.00 |

### Phase 3a — Local Postgres / pgvector — **Done**

| Item | Status | Notes |
|------|--------|-------|
| Docker Compose | Done | `docker-compose.yml` → `foundry-rag-pgvector` |
| Store + BM25 + RRF hybrid | Done | `mechanisms/pgvector_store.py` |
| Index script | Done | `scripts/index_t2_ragbench_pgvector.py` (77 chunks) |
| Eval method | Done | `--method pgvector --retrieval hybrid\|vector\|bm25` |
| Vector-only pilot | Done | `pgvector-pilot50`: NM=0.66, MRR@3=0.92, R@3=0.98 |
| Hybrid pilot | Done | `pgvector-hybrid-pilot50`: NM=0.72, MRR@3=0.95, R@3=1.00 |

```bash
docker compose up -d
uv run python scripts/index_t2_ragbench_pgvector.py --reset
uv run python scripts/eval_foundryiq_t2_ragbench.py --method pgvector --retrieval hybrid
```

Connect: `postgresql://foundry:foundry@localhost:5432/t2_ragbench` (`DATABASE_URL`).

### Phase 3b — Local LanceDB — **Done**

| Item | Status | Notes |
|------|--------|-------|
| Embedded store + hybrid | Done | `mechanisms/lancedb_store.py` (no Docker) |
| Index script | Done | `scripts/index_t2_ragbench_lancedb.py` → `data/t2_ragbench/lancedb/` (gitignored) |
| Eval method | Done | `--method lancedb --retrieval hybrid\|vector\|bm25` |
| Hybrid pilot | Done | `20260725T234144Z-0dcd33`: NM=0.68, MRR@3=0.95, R@3=1.00 |

```bash
uv run python scripts/index_t2_ragbench_lancedb.py --reset
uv run python scripts/eval_foundryiq_t2_ragbench.py --method lancedb --retrieval hybrid
```

### Phase 3c — Azure AI Search Basic hybrid — **Done**

| Item | Status | Notes |
|------|--------|-------|
| Basic Search service | Done | `foundryiq-search-basic` (SKU basic, North Central US) |
| Store + hybrid query | Done | `mechanisms/azure_search_store.py` (BM25 + vector RRF) |
| Index script | Done | `scripts/index_t2_ragbench_azure_search.py` (77 chunks) |
| Eval method | Done | `--method azure_search --retrieval hybrid\|vector\|bm25` |
| Hybrid pilot | Done | `azure-search-hybrid-pilot50`: NM=0.66, MRR@3=0.97, R@3=1.00 |

```bash
uv run python scripts/index_t2_ragbench_azure_search.py --reset
uv run python scripts/eval_foundryiq_t2_ragbench.py --method azure_search --retrieval hybrid
```

### Pilot scoreboard (FinQA n=50, `gpt-5-mini`)

| Method | NM | MRR@3 | R@3 | Run id |
|--------|-----|-------|-----|--------|
| Foundry IQ / Knowledge Base | 0.66 | 1.00 | 1.00 | `pilot50` |
| Postgres / pgvector (vector) | 0.66 | 0.92 | 0.98 | `pgvector-pilot50` |
| Postgres / pgvector (BM25+vector) | **0.72** | 0.95 | 1.00 | `pgvector-hybrid-pilot50` |
| LanceDB (BM25+vector) | 0.68 | 0.95 | 1.00 | `20260725T234144Z-0dcd33` |
| Azure AI Search Basic (BM25+vector) | 0.66 | **0.97** | 1.00 | `azure-search-hybrid-pilot50` |

Results under `data/t2_ragbench/results/`.

### Phase 4 — Scale & report — **Deferred for Foundry IQ; optional for local**

| Item | Status | Notes |
|------|--------|-------|
| Foundry IQ full FinQA / all subsets | **Deferred** | Recreate Standard Search only for a timed eval window |
| Local scale (pgvector / LanceDB) | Optional next | Download full T² PDFs → re-index → same CLI |
| Leaderboard-shaped summary | Partial | Pilot table above; full subsets TBD |
| Cost note | Observed | Search S1 ≫ Foundry tokens (~cents for emb+chat MTD) |

### Phase 5 — Hardening — **Not started**

1. Resume failed runs (idempotent JSONL append) — partial (`--resume`).  
2. Rate limiting / concurrency caps.  
3. Ablations: chunk size, retrieval mode, reasoning effort (Foundry when resumed).

---

## What “same metrics” means (implementation contract)

**Number Match (NM)**  
Parse numeric prediction from model answer; compare to `program_answer` with relative tolerance **ε = 1e−2**. Non-numeric → incorrect.

**MRR@3**  
If gold `context_id` at rank `r` in top-3 retrieved docs → `1/r`, else `0`. Average over questions.  
(Not Oracle: even a gold-only corpus can rank the wrong PDF higher.)

**R@3**  
`1` if gold in top-3, else `0`. Average over questions.

**Oracle**  
MRR@3 / R@3 = 1.0 by definition; NM is reasoning ceiling for the chosen generator.

---

## Effort & cost expectations

| Stage | Status | Cost note |
|-------|--------|-----------|
| Phase 0–2 Foundry IQ setup | Done | Search S1 was the expensive line item |
| Phase 3 harness + Foundry pilot | Done | ~50 KB retrieve calls |
| Phase 3a/3b local pilots | Done | Docker pgvector or LanceDB disk; Foundry tokens only |
| Foundry IQ scale (Phase 4) | **Deferred** | Recreate Search only when needed |
| Full ~23k all subsets | Later | Prefer local hybrid first |

---

## Out of scope (unless requested)

- Submitting to public T²-RAGBench leaderboard  
- Retraining / fine-tuning models  
- Keeping Standard Search always-on for Foundry IQ  
- Exact replication of paper embedding ablations  

---

## Success criteria

| Criterion | Status |
|-----------|--------|
| Pilot NM / MRR@3 / R@3 for at least one SUT | **Met** (Foundry IQ + pgvector + LanceDB) |
| Citation / chunk → `context_id` mapping | **Met** (blob URL / chunk metadata) |
| One-command CLI reproduce summary CSV | **Met** |
| Foundry IQ production-scale eval | **Deferred** |
| Local hybrid path without Search S1 | **Met** |

---

## Implementation units

1. Dataset prep scripts — **done**  
2. Blob upload + Foundry IQ KS/KB — **done**; Search torn down  
3. Metrics module + tests — **done**  
4. Foundry IQ adapter — **done**; further eval **deferred**  
5. Eval CLI — **done** (`foundryiq` \| `oracle` \| `pgvector` \| `lancedb` \| `azure_search`)  
6. Postgres/pgvector + LanceDB stores — **done**  
7. Azure AI Search Basic hybrid store — **done** (direct index; Foundry IQ still deferred)  

---

## Open choices (when resuming Foundry IQ)

1. Recreate Standard Search only for a short eval window, then delete again?  
2. Scale local hybrid to full FinQA test before touching Foundry IQ again?  
3. Add Oracle Context baseline run for NM ceiling on the same 50?
