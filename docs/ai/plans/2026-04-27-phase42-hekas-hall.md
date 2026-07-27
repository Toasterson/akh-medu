# Phase 42 — Heka's Hall: Long-Term Memory with pgvector

**Status:** In progress (Phase 42a — foundation)
**Date:** 2026-04-27
**Parent:** Phase 38 — Relocatable Akh (`019dc599-81c5`)
**ADR:** [docs/ai/decisions/052-hekas-hall.md](../decisions/052-hekas-hall.md)

## Overview

Heka is the long-term-memory subsystem for akh-medu. It stores timestamped "drawers" (memory entries) per akh, with project-scoped bindings and pgvector-backed semantic retrieval. Named after the Egyptian god of magic and writing — the personification of the creative power of words.

This plan mirrors the Phase 37 Seshat pattern: SeaORM 1.x entities, raw-SQL migrations for pgvector types, miette diagnostics, and a daemon CLI (`hekad`).

## Phase 42a — Foundation (this slice)

### Deliverables

- `crates/heka/Cargo.toml`, `heka.toml.example`, `migrations/20260427_000001_init.sql`.
- SeaORM 1.x entities for: `heka_akhs`, `heka_project_bindings`, `heka_drawers`, `heka_drawer_chunks`, `heka_drawer_embeddings`.
- pgvector(384) + HNSW index on `heka_drawer_embeddings` (`vector_cosine_ops`).
- `HekaError` enum with miette diagnostics.
- `HekaConfig` parser (`heka.toml` — service URL, postgres URL, embed model path, eval data path).
- `hekad migrate` CLI binary.
- ADR 052.

### Acceptance

- `cargo test -p heka` passes
- `psql -c "\dt heka_*"` shows 5 tables
- HNSW index visible in `\di+`
- `cargo run --bin hekad migrate` creates schema on test DB

### Reuses

- `crates/seshat/Cargo.toml` (template for deps)
- `crates/seshat/src/error.rs` (miette pattern)
- `crates/seshat/src/migration/` (raw-SQL migration pattern)
- `crates/seshat/src/entity/` (SeaORM entity pattern)

### Dependencies

None — this is the foundation slice.

## Future phases (out of scope for 42a)

- **42b:** `hekad` MCP server (HTTP API, drawer CRUD, semantic search)
- **42c:** Embedding worker (ONNX sentence transformer, background chunk embedding)
- **42d:** Evaluation harness (recall/precision scoring against `eval.data_path`)
- **42e:** Memory consolidation (drawer dedup, summarization)
- **42f:** Cross-akh knowledge sharing (project-scoped drawer visibility)
- **42g:** Forgetting curves (timestamp-based decay, drawer pruning)
- **42h:** Integration with akh OODA loop (memory recall in `observe` phase)
- **42i:** `heka_contradict` tool (Phase 18 ACH + Phase 17 Dempster-Shafer)

## Schema

See `crates/heka/migrations/20260427_000001_init.sql` for the full DDL.

```
heka_akhs
  ├─ heka_project_bindings (akh_id FK)
  └─ heka_drawers (akh_id FK)
       └─ heka_drawer_chunks (drawer_id FK)
            └─ heka_drawer_embeddings (chunk_id FK, vector(384), HNSW index)
```

## Configuration

See `crates/heka/heka.toml.example`. Key sections:

- `[database]` — PostgreSQL URL, max connections
- `[server]` — hekad bind address (reserved for 42b)
- `[embedding]` — model name, model directory
- `[eval]` — ground-truth dataset path (for 42d)