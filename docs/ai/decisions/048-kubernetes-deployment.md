# ADR 048 — Kubernetes Deployment Architecture

**Status:** Accepted  
**Date:** 2026-04-14  
**Deciders:** Till Wegmueller

## Context

Akh-medu needs a production deployment path. The project has two independent services:

1. **akhomed** — the main daemon (REST + WebSocket + MCP server, knowledge graph, NLU pipeline)
2. **seshd** — Seshat's Archive (pgvector RAG service, MCP server, embedding pipeline)

Both need ONNX Runtime at runtime for ML inference (`load-dynamic` mode). Seshat requires PostgreSQL with the `pgvector` extension.

## Decision

### Container Images

Two multi-arch Docker images (`linux/amd64` + `linux/arm64`), built from the workspace root:

| Image | Dockerfile | Binary | Features | Base |
|-------|-----------|--------|----------|------|
| `code.aopc.cloud/toasty/akhomed` | `docker/akhomed.Dockerfile` | `akhomed` | `server,mcp,nlu-ml,nlu-candle,delegation,email,calendar` | `debian:bookworm-slim` |
| `code.aopc.cloud/toasty/seshd` | `docker/seshd.Dockerfile` | `seshd` | `embedding,synthesis` | `debian:bookworm-slim` |

Both use a 3-stage build:
1. **Builder** — Rust compilation with platform-specific cargo caches
2. **ORT fetch** — downloads ONNX Runtime for the target arch
3. **Runtime** — minimal Debian with `tini` init, non-root user

### Helm Chart

A single umbrella chart at `charts/akh-medu/` deploys both services:

- **akhomed**: Deployment (Recreate strategy — single writer to KG volumes) + Service + PVCs (data + models) + optional Ingress
- **seshd**: Deployment + Service + ConfigMap (seshat.toml) + PVC (models) + optional Ingress + init container for DB migrations
- **PostgreSQL**: Optional Bitnami subchart dependency (`postgresql.enabled`), using `pgvector/pgvector:pg16` image with `CREATE EXTENSION vector` in initdb. When disabled, `externalPostgresql.*` values point at an existing instance

The `DATABASE_URL` is constructed as a Kubernetes Secret and injected as an env var. The `SeshatConfig::from_file()` method was extended to honour the `DATABASE_URL` env var, overriding the toml value — this keeps credentials out of ConfigMaps.

### Key Design Choices

1. **pgvector/pgvector image** instead of vanilla postgres — pgvector extension pre-compiled, no custom initContainer needed
2. **Bitnami PostgreSQL subchart optional** — battle-tested HA options, backup hooks, well-maintained; can be disabled in favour of an external instance via `externalPostgresql.*`
3. **Recreate strategy for akhomed** — the knowledge graph (oxigraph + redb + HNSW) is a single-writer store; no rolling update possible
4. **Separate model PVCs** — ML models can be pre-populated or shared; keeps data volume focused on KG state
5. **ORT_DYLIB_PATH baked in** — `load-dynamic` mode needs the env var; set in Dockerfile ENV
6. **Migration init container** for seshd — `seshd migrate` runs SeaORM migrations before the service starts
7. **Full feature builds** — akhomed includes NLU tiers 1-2 (rule parser + ONNX NER) and Candle backend for tier 3 LLM (replacing llama-cpp-2 — pure Rust, no cmake/C++ needed, hidden state access for Phase 26d), plus MCP, delegation, email, calendar; seshd includes embedding + Candle synthesis
8. **Features configurable via ARG** — `FEATURES` build arg allows building slimmer images if needed

## Consequences

- CI/CD must build two Docker images on release
- PostgreSQL can be provided externally (`--set postgresql.enabled=false --set externalPostgresql.host=...`) or via the bundled subchart
- NLU models (NER ONNX, LLM GGUF, Candle models) and embedding models need to be populated into PVCs post-deploy (future: init container that downloads them)
- akhomed is limited to 1 replica due to single-writer KG constraint
