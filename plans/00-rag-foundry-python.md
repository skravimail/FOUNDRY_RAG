# Plan: Python Port of `deployed-in-azure/RAG` on Microsoft Foundry

## Goal

Port the 9 RAG-mechanism demos from [`deployed-in-azure/RAG`](https://github.com/deployed-in-azure/RAG) (C#/.NET)
to Python running on Microsoft/Azure AI Foundry — **as-is**, using the same synthetic "starship" data each module
already ships in its `Data/` folder. This is a faithful port: same mechanisms, same sample data, same interactive
demo behavior, different language and different LLM/search hosting surface (Foundry instead of raw Azure OpenAI).

Evaluating these mechanisms against `G4KMU/t2-ragbench` is **deferred** — out of scope for this plan, to be
picked up as a follow-on plan once the port is working end-to-end.

**Dependency management:** use [`uv`](https://github.com/astral-sh/uv) for all Python dependency management in this
project — `uv init`/`uv add` to manage `pyproject.toml`, `uv run` to execute scripts/entry points. Don't use plain
`pip` or `poetry`.

## Phase 0: Documentation Discovery (consolidated findings)

### Reference repo (`deployed-in-azure/RAG`) — 9 mechanisms

| # | Module | Technique | Key dependency |
|---|--------|-----------|-----------------|
| 01 | NaiveRAG | Embed docs → in-memory cosine-similarity store → top-k → prompt stuffing | none (in-memory) |
| 02 | HybridRAG | Azure AI Search: BM25 + vector (`exhaustiveKnn`, cosine) combined | Azure AI Search |
| 03 | ReRankingRAG | Hybrid results reranked by Azure AI Search semantic ranker (cross-encoder) | Azure AI Search (semantic config) |
| 04 | MultiQueryRAG | LLM generates query rewrites; Search's semantic query-rewrite feature | Azure AI Search + chat model |
| 05 | HyDERAG | LLM writes a hypothetical answer, embeds *that*, searches with it | chat + embedding model |
| 06 | ChunkingStrategies | `FixedSizeChunker`, `SemanticChunker`, `HierarchicalChunker` (parent-child) | tokenizer (cl100k_base) |
| 07 | ContextualRetrieval | LLM prepends situating context to each chunk before embedding (Anthropic-style) | chat model |
| 08 | GraphRAG | LLM extracts entities/relations → Neo4j knowledge graph → graph queries | Neo4j |
| 09 | FoundryIQ | Azure AI Search "Knowledge Base" agentic/multi-step retrieval | Azure AI Search Knowledge Base |

Each module in the reference repo is a standalone `.csproj` console app with its own `Program.cs` (entry point +
Spectre.Console interactive prompts), an `XxxExample.cs` (actual RAG logic), and a `Data/` folder of small synthetic
markdown/JSON documents (a fictional "starship" corpus — e.g. `aurora-class.md`, `starships.json`). Search-backed
modules also ship an `index_definition.json` describing the Azure AI Search index schema.

Auth pattern throughout: `DefaultAzureCredential`, no API keys anywhere. Python equivalents: `azure-identity`,
`azure-search-documents` (near-identical API surface to the .NET SDK), `openai`/`azure-ai-projects` for
chat+embeddings, `neo4j` Python driver, `tiktoken` for `cl100k_base` (drop-in for `Microsoft.ML.Tokenizers`), `rich`
for interactive CLI prompts (drop-in for Spectre.Console).

**Known gap:** exact prompt text and Cypher queries for GraphRAG (`Prompts.cs`, `GraphDb.cs`,
`KnowledgeGraphAggregator.cs`), and the full source for MultiQueryRAG/HyDERAG's example files, were not fetched in
Phase 0 — only `NaiveRagExample.cs`, `InMemoryVectorDb.cs`, and `02_HybridRAG/Data/index_definition.json` were
read in full. Each phase below that touches these modules must fetch the actual `.cs` source first and port it
directly — do not invent prompts or graph logic.

### Foundry Python SDK — allowed APIs

Source: `~/.claude/skills/microsoft-foundry/references/sdk/foundry-sdk-py.md`

```
uv add azure-ai-projects azure-identity azure-ai-inference azure-search-documents openai azure-ai-evaluation python-dotenv neo4j tiktoken rich
```

- Client: `AIProjectClient(endpoint=..., credential=DefaultAzureCredential())`
- Chat/embeddings: `project_client.get_openai_client()` → standard `openai` SDK surface (`.chat.completions.create`,
  `.embeddings.create`).
- RAG-via-agent pattern (used for the FoundryIQ module, mechanism 09):
  `AzureAISearchToolDefinition` / `AzureAISearchToolResource` / `AISearchIndexResource` /
  `AzureAISearchQueryType.HYBRID`, wired onto `project_client.agents.create_agent(...)`.
- Search connection lookup: `project_client.connections.get(<connection_name>)`.
- Knowledge Index inspection (MCP, optional/debugging): `foundry_knowledge_index_list`,
  `foundry_knowledge_index_schema`.

**Anti-pattern guard:** do not invent Foundry SDK methods. Every call in mechanism code must trace back to this
file or to `azure-search-documents`/`neo4j`/`openai` upstream docs fetched during the relevant phase.

---

## Phase 1: Project Scaffolding & Shared Infrastructure

**Implement:**
- `pyproject.toml` with the dependency list above.
- `src/foundry_rag/config.py` — env var loading (`python-dotenv`), Python-named equivalents of the reference repo's
  vars: `FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_CHAT_DEPLOYMENT`, `FOUNDRY_EMBEDDING_DEPLOYMENT`,
  `AZURE_AI_SEARCH_ENDPOINT`, `AZURE_AI_SEARCH_CONNECTION_NAME`, `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` (module 08
  only).
- `src/foundry_rag/clients.py` — copy the connection pattern from `foundry-sdk-py.md` lines 44–59 verbatim
  (`AIProjectClient` + `DefaultAzureCredential`); expose `get_project_client()`, `get_openai_client()`,
  `get_search_client(index_name)` (via `azure.search.documents.SearchClient`, same credential).
- `.env.example` documenting every var above.
- `src/foundry_rag/shared/vector_db.py` — port `Shared/InMemoryVectorDb.cs` directly: in-memory store of
  `(id, text, embedding)` records, cosine-similarity top-k search. Used by Naive RAG and as a building block
  elsewhere.

**Verification checklist:**
- [ ] `python -c "from foundry_rag.clients import get_project_client; get_project_client()"` succeeds against a real
      `FOUNDRY_PROJECT_ENDPOINT`.
- [ ] No hardcoded credentials anywhere (`grep -rn "api_key\s*=" src/` returns nothing suspicious).
- [ ] `InMemoryVectorDb` port has a unit test with 3 toy vectors confirming top-k ordering matches expected cosine
      similarity ranking.

---

## Phase 2: Port Synthetic Data

**Implement:**
- Fetch every `Data/` folder from the reference repo (`RAG/01_NaiveRAG/Data/`, `RAG/02_HybridRAG/Data/`, … through
  `09_FoundryIQ`) and copy the markdown/JSON starship documents as-is into `data/<module_number>_<module_name>/`
  in this repo, preserving filenames.
- Fetch each search-backed module's `index_definition.json` and keep alongside its data for Phase 5/8 to consume.
- Note where modules share the same underlying documents vs. have module-specific data (confirm during the fetch —
  Phase 0 didn't enumerate every `Data/` folder's exact contents, only sampled the repo tree).

**Verification checklist:**
- [ ] Every module's `Data/` folder has a Python-side counterpart with identical file content (diff the fetched
      files against what's in `data/`).
- [ ] `index_definition.json` files present for modules 02, 03, 04, 09 (the Search-backed ones).

---

## Phase 3: Chunking Strategies + Contextual Retrieval

Corresponds to reference modules **06_ChunkingStrategies** and **07_ContextualRetrieval**. Built early since later
mechanisms' ingestion depends on these chunkers.

**Implement:**
- Fetch full source for `RAG/06_ChunkingStrategies/*` and `RAG/07_ContextualRetrieval/*` (not yet pulled in
  Phase 0) to get exact chunk-size defaults, overlap logic, and the contextual-retrieval prompt wording.
- `src/foundry_rag/chunking/fixed.py` — port `FixedSizeChunker`: token-bounded fixed-size chunks with overlap,
  using `tiktoken` `cl100k_base` (drop-in for `Microsoft.ML.Tokenizers`).
- `src/foundry_rag/chunking/semantic.py` — port `SemanticChunker`: embedding-similarity-based splitting.
- `src/foundry_rag/chunking/hierarchical.py` — port `HierarchicalChunker`: parent-child chunk structure.
- `src/foundry_rag/mechanisms/contextual_retrieval.py` — port `07_ContextualRetrieval`'s logic: chat model
  generates situating context per chunk, prepended before embedding.
- `scripts/run_06_chunking_strategies.py` and `scripts/run_07_contextual_retrieval.py` — interactive CLI entry
  points using `rich`, mirroring each module's `Program.cs` prompt flow (pick a question, print retrieved
  chunks/answer) against the module's own `Data/` folder from Phase 2.

**Verification checklist:**
- [ ] Each chunker run against the module's sample docs produces chunk sets consistent with the ported
      size/overlap parameters.
- [ ] `run_06_chunking_strategies.py` runs interactively and lets a user compare the 3 chunkers' outputs on the
      same document, matching the original's comparison intent.
- [ ] `run_07_contextual_retrieval.py` output visibly includes the generated situating context per chunk.

---

## Phase 4: Naive RAG

Corresponds to **01_NaiveRAG**.

**Implement:**
- `src/foundry_rag/mechanisms/naive.py` — port `NaiveRagExample.cs` directly: embed all docs from
  `data/01_naive_rag/` into Phase 1's `InMemoryVectorDb`, cosine-similarity top-k at query time, stuff into a chat
  prompt via `get_openai_client()`, generate.
- `scripts/run_01_naive_rag.py` — interactive CLI entry point (rich-based prompt), same interaction shape as the
  original (`Program.cs`).

**Verification checklist:**
- [ ] Running the script interactively against the module's own sample questions produces on-topic answers
      referencing the starship data.
- [ ] Output structurally matches the original's console flow (question prompt → retrieved context shown →
      generated answer).

---

## Phase 5: Hybrid RAG + Re-Ranking RAG

Corresponds to **02_HybridRAG** and **03_ReRankingRAG** (reranking is a query-time flag on the same index).

**Implement:**
- Fetch full source for `RAG/02_HybridRAG/*` and `RAG/03_ReRankingRAG/*`.
- `src/foundry_rag/ingestion/index_builder.py` — build an Azure AI Search index matching Phase 2's
  `index_definition.json` (BM25 similarity config + `exhaustiveKnn` vector profile) via
  `azure.search.documents.indexes.SearchIndexClient`. Upload module 02's starship docs (chunked via Phase 3's
  fixed chunker, matching what the original module does).
- `src/foundry_rag/mechanisms/hybrid.py` — `SearchClient.search()` with `search_text` (BM25) + `vector_queries`
  set (hybrid), per `azure-search-documents` docs (fetch method signatures during this phase — not covered by the
  Foundry SDK reference, which only documents the agent-tool path to Search).
- `src/foundry_rag/mechanisms/reranking.py` — same search call with semantic ranking enabled
  (`query_type="semantic"` + semantic configuration), per Azure AI Search semantic ranking docs.
- `scripts/run_02_hybrid_rag.py`, `scripts/run_03_reranking_rag.py` — interactive CLI entry points.

**Verification checklist:**
- [ ] Index created successfully; `SearchIndexClient.get_index()` confirms field schema matches
      `index_definition.json`.
- [ ] Hybrid query on a sample question returns relevant starship-doc chunks in top-5.
- [ ] Reranked ordering visibly differs from plain hybrid for at least one logged example.

---

## Phase 6: Multi-Query RAG + HyDE RAG

Corresponds to **04_MultiQueryRAG** and **05_HyDERAG** — both query-transformation techniques over the Hybrid index
from Phase 5.

**Implement:**
- Fetch full source for `RAG/04_MultiQueryRAG/*` and `RAG/05_HyDERAG/*` (not yet pulled in Phase 0) — port the
  exact rewrite-generation and hypothetical-answer prompts, don't invent them.
- `src/foundry_rag/mechanisms/multiquery.py` — chat model generates N query rewrites; run each through Phase 5's
  hybrid search; merge/dedupe results before generation.
- `src/foundry_rag/mechanisms/hyde.py` — chat model writes a hypothetical answer; embed it; use as the vector
  query against Phase 5's index.
- `scripts/run_04_multiquery_rag.py`, `scripts/run_05_hyde_rag.py` — interactive CLI entry points.

**Verification checklist:**
- [ ] Multi-query produces distinct rewrites (not near-duplicates) for a sample question — log them.
- [ ] HyDE's hypothetical answer is visibly different in style/length from the final generated answer.

---

## Phase 7: GraphRAG

Corresponds to **08_GraphRAG**. Fetch `Prompts.cs`, `GraphDb.cs`, `KnowledgeGraphAggregator.cs`, and the example
file first — this is the module with the least Phase 0 coverage.

**Implement:**
- `src/foundry_rag/mechanisms/graphrag/extraction.py` — port the entity/relation extraction prompt + JSON schema
  exactly as found in the reference repo.
- `src/foundry_rag/mechanisms/graphrag/graph_db.py` — Neo4j Python driver wrapper matching the reference's node/
  edge model (`neo4j` package).
- `src/foundry_rag/mechanisms/graphrag/query.py` — port the reference's graph query/traversal logic.
- `docker-compose.yml` for running Neo4j locally as a sidecar (matches the reference's external dependency; no
  substitution).
- `scripts/run_08_graphrag.py` — interactive CLI entry point.

**Verification checklist:**
- [ ] Entity/relation extraction on the module's sample docs produces a non-trivial graph (>0 nodes, >0 edges) in
      Neo4j (verify via a Cypher `MATCH` count query).
- [ ] A sample question resolves through a graph traversal path into the final answer.

---

## Phase 8: FoundryIQ (Agentic Retrieval)

Corresponds to **09_FoundryIQ** — Foundry-native rather than a straight port, since it uses Azure AI Search's
"Knowledge Base" agentic retrieval feature directly through a Foundry agent.

**Implement:**
- Fetch `RAG/09_FoundryIQ/*` source to confirm exact reasoning-effort configuration and query shape used.
- `src/foundry_rag/mechanisms/foundryiq.py` — port the pattern from `foundry-sdk-py.md` lines 61–95:
  `AzureAISearchToolDefinition` / `AzureAISearchToolResource` / `AISearchIndexResource(query_type=HYBRID)` wired
  into `project_client.agents.create_agent(...)`, queried via
  `project_client.get_openai_client().responses.create(...)`.
- Confirm against Azure AI Search "Knowledge Base" docs (fetch during this phase) whether it needs a dedicated
  Knowledge-Base-configured index distinct from Phase 5's, or can reuse it.
- `scripts/run_09_foundryiq.py` — interactive CLI entry point.

**Verification checklist:**
- [ ] Agent created successfully (`project_client.agents.get_agent(...)` returns it).
- [ ] Streaming query against a sample question yields citations (`url_citation` annotations).
- [ ] Agent + resources cleaned up after test run (context-manager pattern from `foundry-sdk-py.md`).

---

## Phase 9: Final Verification

1. Run every `scripts/run_0N_*.py` entry point interactively end-to-end against its own module's starship data,
   confirming each produces sensible, on-topic answers.
2. Grep for anti-patterns: no hardcoded API keys (`grep -rn "sk-\|api_key\s*=\s*[\"']" src/`), no invented SDK
   methods (spot-check each `azure.ai.*`/`azure.search.*`/`neo4j.*` import against installed package docs).
3. Confirm parity notes: for each of the 9 modules, a short entry in `README.md` stating what was ported 1:1 vs.
   what had to be adapted for Python/Foundry (e.g. Spectre.Console → rich, connection-string differences).
4. Produce a top-level `README.md`: setup steps (env vars, Azure AI Search + Neo4j provisioning), how to run each
   mechanism standalone, and the parity notes from step 3.

---

## Deferred (follow-on plan, not in scope here)

- Evaluation harness against `G4KMU/t2-ragbench` (financial-document QA, numeric-match grading) — was originally
  discussed for this plan but pushed to a follow-on plan once the straight port is working. When picked up, revisit
  the dataset research already gathered: 3 subsets (FinQA/ConvFinQA/TAT-DQA), 23,088 QA pairs total, CC-BY-4.0,
  `program_answer` field designed for numeric-match evaluation.

## Open items to revisit mid-plan

- Fetch full source for modules 04, 05, 06, 07, 08, 09 before their respective phases — Phase 0 only read 01 and
  the Shared vector DB in full, plus one `index_definition.json`.
- Enumerate every module's `Data/` folder contents precisely during Phase 2 (Phase 0 only sampled the repo tree,
  not full folder listings).
