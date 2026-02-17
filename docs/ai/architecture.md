# Akh-medu Architecture

> Last updated: 2026-02-17 (Phase 10 expanded to 8 sub-phases after deep research)

## Overview

Akh-medu is a neuro-symbolic AI engine that runs entirely on CPU with no LLM dependency. It hybridizes:

- **Vector Symbolic Architecture (VSA)** — 10,000-bit binary hypervectors for distributed representation
- **Knowledge Graphs** — dual-indexed (petgraph + oxigraph/SPARQL) for structured symbolic reasoning
- **E-graph Reasoning** — equality saturation via `egg` for symbolic rewriting
- **Autonomous Agent** — OODA-loop agent with 15+ tools, working/episodic memory, planning, reflection
- **Multilingual Grammar** — GF-inspired abstract/concrete syntax split for 5 languages
- **Content Library** — document ingestion (PDF, EPUB, HTML) with chunking and semantic enrichment
- **Tiered Storage** — hot (DashMap) → warm (mmap) → cold (redb) for scalability

## Module Map

```
src/
├── agent/              24 modules — OODA loop, tools, memory, goals, planning, psyche
├── autonomous/          6 modules — background learning, confidence fusion, grounding
├── compartment/         3 modules — knowledge isolation (core/skill/project), Jungian psyche
├── grammar/            20 modules — GF-inspired parsing/generation, entity resolution
├── graph/               3 modules — KG (petgraph), SPARQL (oxigraph), analytics
├── infer/               3 modules — spreading activation, backward chaining, superposition
├── library/            12 modules — document parsing, chunking, concept extraction
├── reason/              1 module  — e-graph language (AkhLang), rewrite rules
├── simd/                5 modules — runtime SIMD kernel dispatch (AVX2 / generic)
├── skills/              1 module  — skillpack lifecycle (Cold/Warm/Hot)
├── store/               3 modules — tiered storage (hot/warm/cold)
├── tui/                 6 modules — ratatui terminal UI, WebSocket remote
├── vsa/                 4 modules — HyperVec, VsaOps, encoding, item memory (HNSW)
├── engine.rs                      — facade composing all subsystems
├── error.rs                       — miette + thiserror rich diagnostics
├── provenance.rs                  — persistent explanation ledger (redb, multi-index)
├── symbol.rs                      — SymbolId (NonZeroU64), SymbolKind, allocator
├── pipeline.rs                    — composable stage pipelines
└── main.rs                        — CLI (clap) with 50+ subcommands
```

## Core Data Model

### Symbols
- **SymbolId**: `NonZeroU64` (niche-optimized for `Option` packing)
- **SymbolKind**: `Entity`, `Relation`, `Composite`, `Glyph(codepoint)`
- **AtomicSymbolAllocator**: thread-safe monotonic ID generator

### Triples
- `(subject: SymbolId, predicate: SymbolId, object: SymbolId)` with confidence, timestamp, provenance_id, compartment_id
- Stored in both petgraph (in-memory graph ops) and oxigraph (SPARQL queries)
- Each triple carries provenance linking back to how it was derived

### Hypervectors (VSA)
- 10,000-bit binary vectors, SIMD-accelerated (AVX2 with generic fallback)
- Operations: bind (XOR), unbind, bundle (majority vote), permute, similarity (Hamming)
- Item Memory: HNSW approximate nearest-neighbor search

## Reasoning Systems

| System | Strategy | Use Case |
|--------|----------|----------|
| Spreading Activation | Seeds → expand frontier via graph edges + VSA recovery | Forward inference, "what follows from X?" |
| Backward Chaining | Goal ← find supporting evidence recursively | Why-questions, evidence chains |
| Superposition | Parallel competing hypotheses, constructive/destructive interference | Multi-path exploration |
| E-graph Rewriting | `AkhLang` + `egg` equality saturation | Symbolic simplification, equivalence |
| Confidence Fusion | Noisy-OR and consensus across multi-source evidence | Combining evidence |

## Agent Architecture

OODA loop (synchronous, no async runtime):
1. **Observe** — scan KG for active goals, recall episodic memories
2. **Orient** — assess working memory, build context
3. **Decide** — utility-based tool scoring with recency penalty, novelty bonus, episodic hints
4. **Act** — execute selected tool, evaluate goal progress

Supporting infrastructure: working memory (ephemeral), episodic memory (consolidated), goal management with decomposition, multi-step planning with backtracking, periodic reflection, Jungian psyche model.

## Storage Architecture

```
Hot  (DashMap)     — sub-microsecond, volatile
Warm (mmap)        — memory-mapped files, persistent, fast reads
Cold (redb)        — ACID transactions, durable, slower writes
Provenance (redb)  — multi-index ledger (derived/source/kind)
SPARQL (oxigraph)  — persistent RDF store for structured queries
```

## Provenance

Every inference, agent decision, and knowledge derivation creates a `ProvenanceRecord`:
- Derived symbol, derivation kind (20+ variants), confidence, depth, source symbols, metadata
- Full traceback from any result to its original sources
- Indices by derived symbol, source symbol, and kind for fast lookup

## Development Phases

Phases 1–7: Engine foundation (VSA, KG, reasoning, storage, provenance, inference, pipeline, skills)
Phase 8a–8f: Agent evolution (wiring, goals, decision-making, persistence, external tools, planning)
Phase 9a–9o: Cyc-inspired HOL enhancements (15 sub-phases):
- **High**: 9a microtheories, 9b predicate hierarchy, 9c TMS, 9d defeasibility, 9e argumentation, 9f reasoner dispatch
- **Medium**: 9g rule macros, 9h skolem functions, 9i arg-based truth, 9j arity/types, 9k temporal projection, 9l contradiction detection
- **Lower**: 9m circumscription/CWA, 9n second-order quantification, 9o NARTs
Phase 10a–10h: Rust code generation (8 sub-phases):
- **Core**: 10a RustCodeGrammar, 10b code_gen tool, 10c code-aware planning, 10d iterative refinement, 10e templates
- **Pattern learning**: 10f VSA code pattern encoding, 10g pattern mining from examples, 10h library learning cycle
Phase 11a–11g: Autonomous task system (7 sub-phases — goal generation, intelligent decomposition, priority reasoning, projects, world monitoring, self-evaluation, resource awareness)
