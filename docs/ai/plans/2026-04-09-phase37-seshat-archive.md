# Phase 37 — Seshat's Archive: Shared Corpus Library with RAG + pgvector

> Date: 2026-04-09
> Status: Planned
> Phase: 37
> Depends on: Phase 9a (Microtheories), Phase 11 (Goal System), Phase 12g (Multi-Agent Protocol)
> Enhances: Phase 32 (Knowledge Extraction), Phase 36 (Delegation)
> ADR: [047-seshat-archive](../decisions/047-seshat-archive.md)

## Motivation

Akh-medu's current ingest pipeline (`src/library/ingest.rs`) parses documents
locally, chunks them, and embeds them as 10K-bit binary hypervectors in item
memory. This works for individual documents but fails at corpus scale: the HNSW
index is in-memory (lost on restart), there's no way for multiple akh instances
to share a corpus, and rule-based extraction yields ~1 triple per 28 sentences.

Users have corpora of reference material (textbooks, papers, manuals in PDF,
EPUB, MOBI, plain text) they want indexed and retrievable. The existing
CloudNativePG PostgreSQL cluster provides persistent vector storage via pgvector.

Seshat's Archive — named after the Egyptian goddess of writing, wisdom, and
knowledge — introduces a dedicated MCP service that mediates access to a shared
pgvector-backed corpus for multiple akh-medu instances.

## Architecture

```
 ┌──────────────────┐
 │ seshat-ingest CLI │  (reference pipeline, replaceable)
 │ PDF/EPUB/MOBI/txt │
 └────────┬─────────┘
          │ INSERT pa_documents, pa_chunks (direct to postgres)
          ▼
 ┌─────────────────────────┐
 │   PostgreSQL / pgvector  │  (CloudNativePG cluster)
 │   pa_documents           │
 │   pa_chunks              │
 │   pa_embeddings          │
 │   pa_microtheory_cache   │
 └─────────────┬───────────┘
               │ SeaORM
               ▼
 ┌─────────────────────────┐
 │   Seshat Service (seshd) │  ← crates/seshat/
 │   ┌───────────────────┐ │
 │   │  MCP Server (rmcp) │ │  ← exposes tools via MCP protocol
 │   └───────────────────┘ │
 │   ┌───────────────────┐ │
 │   │  Embed Worker      │ │  ← polls unembedded chunks, runs ONNX model
 │   └───────────────────┘ │
 │   ┌───────────────────┐ │
 │   │  LLM Synthesis     │ │  ← optional: Candle (GGUF), target: T5 Knowledge
 │   └───────────────────┘ │
 └─────────────────────────┘
        ↑ MCP calls              ↑ MCP calls
 ┌──────┴──────┐          ┌──────┴──────┐
 │  akh-medu A  │          │  akh-medu B  │
 │  (engine)    │          │  (engine)    │
 │              │◄────────►│              │  (multi-agent sharing)
 └──────────────┘          └──────────────┘
```

**Key insight:** Akh-medu never connects directly to PostgreSQL. All pgvector
access is mediated by the Seshat service via MCP.

## Sub-phases

### 37a — Workspace Crate & pgvector Schema (~700 lines)

New workspace crate `crates/seshat/` with:
- SeaORM entities for 4 tables: `pa_documents`, `pa_chunks`, `pa_embeddings`,
  `pa_microtheory_cache`
- Sea-ORM migrations with pgvector extension setup
- Config parsing (`seshat.toml`)
- Error types with miette diagnostics

**Schema:** See ADR 047 for full DDL.

### 37b — Embedding Service (~400 lines)

- ONNX inference for `all-MiniLM-L6-v2` (Apache 2.0, ~80MB, 384-dim)
- Uses `ort` + `tokenizers` (same pattern as akh-medu's nlu-ml NER tier)
- Async embed worker: polls unembedded chunks, embeds in batches, writes to
  `pa_embeddings`
- Auto-downloads model on first use with SHA-256 validation

**Note:** Embedding uses ONNX/ort (lightweight sentence encoder). Synthesis uses
Candle (GGUF generative models). Intentionally different frameworks for different
purposes.

### 37c — MCP Server & Tools (~500 lines)

- rmcp-based MCP server with streamable HTTP transport
- 7 tools: `seshat_search`, `seshat_wish`, `seshat_documents`, `seshat_status`,
  `seshat_ingest`, `seshat_embed_now`, `seshat_cache_list`
- pgvector cosine similarity search via SeaORM
- Context assembly with surrounding chunk retrieval

### 37d — LLM Synthesis in Service (optional) (~400 lines)

- Feature-gated behind `synthesis` (Candle deps)
- Candle loads GGUF models (aligned with Phase 26b `CandleLlmBackend` pattern)
- Model progression:
  1. Interim: Qwen2.5-1.5B or Phi-3-mini GGUF (general-purpose)
  2. Target: T5 Knowledge (`t5-kg-wikidata.gguf`, ~250MB, Phase 35)
  3. Future: Burn LoRA adapters merged into T5 Knowledge (Phase 27)
- Results cached in `pa_microtheory_cache` by query hash

### 37e — Akh-Medu MCP Client Integration (~400 lines)

- `SeshatClient` wraps rmcp MCP client calls to Seshat service
- Feature-gated `seshat` on akh-medu side
- `Engine` gains optional `seshat_client` field
- Config: `[seshat] service_url`, `synthesize_in_service`

### 37f — Wish Mechanism (Goal Integration) (~350 lines)

- New `GoalSource::SeshatWish` variant
- Curiosity drive generates wishes for unknown topics
- OODA Act phase routes SeshatWish goals to `execute_wish()`
- Flow: MCP wish → receive triples or chunks → synthesize if needed → compartment

### 37g — Compartment Bridge (~350 lines)

- Creates `CompartmentKind::Library` compartments from synthesized triples
- Writes `compartment.toml` + `triples.json` to compartment directory
- Creates `Microtheory` KG entity with `ctx:specializes` → root seshat context
- New `DerivationKind::SeshatSynthesis` provenance variant

### 37h — Local LLM Synthesis Path in Akh-Medu (~350 lines)

For `synthesize_in_service = false`, akh-medu runs synthesis locally:
- **Path A — Candle backend** (primary, after Phase 26b): Same model/prompt as 37d
- **Path B — DelegationManager** (Phase 36): Claude subprocess for high quality
- **Path C — Legacy llama-cpp-2** (interim): Existing nlu-llm feature
- Progression: C → A → A+T5 Knowledge → A+T5+LoRA

### 37i — Multi-Agent Sharing (~250 lines)

- New `WishRequest`/`WishResponse` protocol messages
- New `CapabilityScope::WishSeshat`
- Akhs without Seshat can request microtheories from peers
- Seshat's internal cache benefits all requesting akhs

### 37j — Library Module Migration (~300 lines)

- Mark `ingest_document()`, `ingest_file()`, `ingest_url()` as `#[deprecated]`
- LibraryCatalog search delegates to SeshatClient when configured
- Old local catalog stays readable for backward compat

### 37k — Reference CLI: `seshat-ingest` (~500 lines)

- Binary in `crates/seshat/` (or separate `crates/seshat-ingest/`)
- Commands: `add`, `list`, `status`, `remove`
- Writes directly to PostgreSQL (bulk loading, not through service)
- Reuses parsers from `src/library/parser/` + `src/library/chunker.rs`
- Adds MOBI support
- Chunk strategy: 256-512 tokens (MiniLM's 512-token context window)

### 37l — Akh-Medu MCP Tools & CLI (~250 lines)

- Proxy tools in `src/mcp/mod.rs`: `seshat_wish`, `seshat_search`, etc.
- CLI subcommands: `akh seshat wish`, `search`, `status`, `documents`

## Dependency Graph

```
37a → 37b → 37c → 37d (Seshat service complete)
37k (after 37a, parallel with service work)
37e → 37f → 37g → 37h (akh-medu integration)
37i, 37j, 37l (parallel, after 37f)
```

## Estimated Lines

| Sub-phase | Description | Lines |
|-----------|-------------|-------|
| 37a | Workspace crate + schema | ~700 |
| 37b | Embedding service | ~400 |
| 37c | MCP server + query | ~500 |
| 37d | LLM synthesis (service) | ~400 |
| 37e | MCP client in akh-medu | ~400 |
| 37f | Wish mechanism | ~350 |
| 37g | Compartment bridge | ~350 |
| 37h | Local LLM synthesis | ~350 |
| 37i | Multi-agent sharing | ~250 |
| 37j | Library migration | ~300 |
| 37k | seshat-ingest CLI | ~500 |
| 37l | MCP tools + CLI | ~250 |
| **Total** | | **~4,750** |

## Verification

1. `seshd migrate` creates tables on test PostgreSQL
2. `seshat-ingest add test.pdf` → rows in pa_documents + pa_chunks
3. Embed worker populates pa_embeddings
4. MCP `seshat_search("Egyptian math", 5)` → relevant chunks
5. MCP `seshat_wish` with `synthesize_here=true` → triples returned
6. `akh seshat wish "Egyptian mathematics"` → compartment loaded with triples
7. Repeat wish → cache hit, no LLM call
8. Akh A sends WishRequest to Akh B → WishResponse → compartment loaded
9. Old `ingest_document()` emits deprecation warning
