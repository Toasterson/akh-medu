# ADR 047 — Seshat's Archive: Shared Corpus Library with RAG + pgvector

> Date: 2026-04-09
> Status: Proposed
> Phase: 37
> Depends on: Phase 9a (Microtheories), Phase 11 (Goal System), Phase 12g (Multi-Agent Protocol),
>             ADR-029 (MCP server), ADR-037 (Candle migration)
> Enhances: Phase 32 (Knowledge Extraction), Phase 36 (Delegation)

## Context

Akh-medu's current document ingest pipeline (`src/library/ingest.rs`) operates
entirely within a single engine instance: parse → chunk → extract concepts →
embed as 10K-bit binary hypervectors → catalog. This has several limitations:

1. **No persistent vector search** — the HNSW index lives in-memory and is lost
   on restart.
2. **No corpus-scale support** — designed for individual documents, not a library
   of hundreds of PDFs, MOBI files, and EPUBs.
3. **No cross-akh sharing** — each akh maintains its own ingested knowledge with
   no way for multiple engines to share a corpus.
4. **Conservative extraction** — rule-based extraction yields ~1 triple per 28
   sentences; LLM-assisted extraction is needed for quality.

Users have corpora of reference material (textbooks, papers, manuals) they want
indexed and retrievable. The existing PostgreSQL cluster (CloudNativePG) provides
a natural home for persistent vector storage via pgvector.

## Decision

### Introduce Seshat's Archive as a dedicated MCP service

Named after **Seshat**, the Egyptian goddess of writing, wisdom, and knowledge.

**Architecture: Service-mediated access via MCP protocol.**

A new workspace crate `crates/seshat/` runs as an independent service (`seshd`)
that:

1. Owns the pgvector-backed PostgreSQL schema (documents, chunks, embeddings,
   microtheory cache)
2. Runs an embedding worker (all-MiniLM-L6-v2 via ONNX/ort, 384-dim vectors)
3. Exposes MCP tools for search, wish, ingest, and status
4. Optionally runs LLM synthesis via Candle (aligned with Phase 26b architecture)

**Akh-medu never connects directly to PostgreSQL.** All corpus access is mediated
by the Seshat service via MCP. This allows multiple akh instances to share one
corpus and keeps akh-medu focused on symbolic reasoning.

### Wish mechanism

Akhs request knowledge from the archive through a "wish" — a goal with
`GoalSource::SeshatWish`. The OODA loop picks up the wish, calls the Seshat
service via MCP, and either:

- Receives pre-synthesized triples from the service (when `synthesize_here=true`
  and the service has Candle configured), or
- Receives raw context chunks and runs local LLM synthesis (Candle backend after
  Phase 26b, DelegationManager after Phase 36, or legacy llama-cpp as interim)

Synthesized triples are assembled into a `CompartmentKind::Library` compartment
with a `Microtheory` entity in the KG, full provenance tracking, and caching in
`pa_microtheory_cache` for cross-akh reuse.

### LLM alignment with Candle/Burn/T5 roadmap

- **Embedding:** ONNX/ort (lightweight sentence encoder, same pattern as nlu-ml NER)
- **Synthesis (interim):** Candle loads Qwen2.5/Phi-3 GGUF for prompted triple extraction
- **Synthesis (target):** T5 Knowledge model from Phase 35 (~250MB, purpose-built concept→triples)
- **Synthesis (future):** Burn-trained LoRA adapters (Phase 27) merged into T5 Knowledge

### Full replacement of local ingest pipeline

The existing `src/library/ingest.rs` pipeline is deprecated. All document
ingestion flows through Seshat's pgvector path. A reference CLI (`seshat-ingest`)
ships for parsing PDF/EPUB/MOBI/text into pgvector.

### Multi-agent sharing via existing protocol

The multi-agent protocol (ADR-014) gains `WishRequest`/`WishResponse` message
variants and a `WishSeshat` capability scope. Akhs without a Seshat connection
can request microtheories from peers that have one.

## Alternatives Considered

### Direct pgvector connection from akh-medu

Rejected. Multiple akh instances would each need their own connection pool and
embedding worker, leading to resource duplication and coordination complexity.
The service-mediated approach centralizes embedding, caching, and schema
migration.

### In-memory vector database (e.g., keep existing HNSW)

Rejected. Does not survive restarts, cannot scale to corpus-level document
counts, and cannot be shared across akh instances.

### Dedicated vector database (Qdrant, Milvus, Weaviate)

Rejected. Adds another infrastructure dependency when pgvector on the existing
PostgreSQL cluster is sufficient for the scale (~100K chunks). FLOSS sovereignty
principle: fewer moving parts.

### RAG without microtheory synthesis

Rejected. Raw chunk retrieval is useful but doesn't integrate with akh-medu's
symbolic reasoning. The microtheory synthesis step converts unstructured text
into structured KG triples that can participate in inference, analogy, and
OODA loops.

## Consequences

### Positive

- Persistent, shareable vector search across restarts and akh instances
- Corpus-scale ingestion (hundreds of documents, millions of chunks)
- LLM-quality triple extraction from retrieved context
- Microtheory caching eliminates redundant LLM calls
- Clean service boundary — akh-medu stays focused on symbolic reasoning
- Reference CLI lowers barrier for corpus population

### Negative

- New infrastructure dependency: PostgreSQL with pgvector extension
- New service to deploy and monitor (`seshd`)
- SeaORM is a new dependency in the project (first PostgreSQL integration)
- Embedding model download (~80MB) on first run

### Risks

- **pgvector index performance:** IVFFlat requires periodic REINDEX at scale.
  Mitigated: at <100K chunks this is not a concern; pgvector 0.5+ supports HNSW.
- **SeaORM learning curve:** First use in the project. Mitigated: feature-gated,
  isolated in `crates/seshat/`.
- **LLM synthesis quality (interim):** General-purpose GGUF models produce lower
  quality triples than the target T5 Knowledge model. Mitigated: caching
  prevents repeated low-quality synthesis; `force_refresh` allows re-synthesis
  when better models ship.

## Schema

Four PostgreSQL tables: `pa_documents`, `pa_chunks`, `pa_embeddings`,
`pa_microtheory_cache`. See phase plan for full DDL.

## Implementation

12 sub-phases (37a–37l), ~4,750 lines estimated.
See `docs/ai/plans/2026-04-09-phase37-seshat-archive.md` for detailed breakdown.
