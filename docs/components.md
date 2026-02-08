# akh-medu Component Status

| Component | Status | Module | Crates Used | Notes |
|-----------|--------|--------|-------------|-------|
| Error types | Implemented | `error` | miette, thiserror | Rich diagnostics with error codes and help text |
| Symbol system | Implemented | `symbol` | serde | SymbolId (NonZeroU64), SymbolKind, SymbolMeta, AtomicSymbolAllocator |
| SIMD kernels | Implemented | `simd` | (std::arch) | VsaKernel trait, Generic fallback, AVX2 acceleration |
| Memory store | Implemented | `store::mem` | dashmap | Hot-tier concurrent in-memory storage |
| Mmap store | Implemented | `store::mmap` | memmap2 | Warm-tier memory-mapped file storage with header/index |
| Durable store | Implemented | `store::durable` | redb | Cold-tier ACID key-value storage |
| Tiered store | Implemented | `store::mod` | — | Composes hot/warm/cold with auto-promotion |
| HyperVec | Implemented | `vsa` | — | Configurable-dimension hypervector type |
| VSA operations | Implemented | `vsa::ops` | — | bind, unbind, bundle, permute, similarity, cosine |
| Symbol encoding | Implemented | `vsa::encode` | rand | Deterministic symbol→vector, sequence, role-filler |
| Item memory | Implemented | `vsa::item_memory` | hnsw_rs, anndists, dashmap | ANN search with HNSW, concurrent access |
| Knowledge graph | Implemented | `graph::index` | petgraph, dashmap | In-memory digraph with dual indexing |
| SPARQL store | Implemented | `graph::sparql` | oxigraph | Persistent RDF with SPARQL queries |
| Graph traversal | Implemented | `graph::traverse` | — | BFS with depth/predicate/confidence filtering |
| Reasoning (egg) | Implemented | `reason` | egg | AkhLang, built-in rewrite rules, equality saturation |
| Engine facade | Implemented | `engine` | — | Top-level API, owns all subsystems |
| CLI | Implemented | `main` | clap, miette | init, ingest, query, info subcommands |
| Provenance | Stub | `provenance` | — | Types defined, ledger pending (Phase 2) |
| Inference | Stub | `infer` | — | Query/result types defined, engine pending (Phase 2) |
| Pipeline | Stub | `pipeline` | — | Stage types defined, DAG execution pending (Phase 2) |
| Skills | Stub | `skills` | — | Manifest/state types defined, manager pending (Phase 3) |
