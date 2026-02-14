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
| Provenance | Implemented | `provenance` | redb, bincode | Persistent ledger with 4 redb tables, multi-index (derived/source/kind), batch ops |
| Inference | Implemented | `infer` | egg | Spreading activation, backward chaining, superposition reasoning, VSA recovery, e-graph verification |
| Pipeline | Implemented | `pipeline` | egg | Linear stage pipeline (Retrieve → Infer → Reason → Extract), built-in query/ingest pipelines |
| Skills | Implemented | `skills` | egg, serde_json | MoE-style skillpacks with Cold/Warm/Hot lifecycle, memory budgets, dynamic rule compilation |
| Graph analytics | Implemented | `graph::analytics` | petgraph | Degree centrality, PageRank, strongly connected components |
| Agent | Implemented | `agent` | — | OODA loop, 17 tools, planning/reflection, session persistence, REPL mode |
| Autonomous cycle | Implemented | `autonomous` | — | Symbol grounding, superposition inference, confidence fusion, KG commit |
| TUI idle learning | Implemented | `agent::idle` | — | Background consolidation, reflection, equivalence learning, rule inference during TUI idle time |
| Agent daemon | Implemented | `agent::daemon` | tokio (feature-gated) | Long-running background process with 8 scheduled tasks, session persistence, graceful shutdown |
