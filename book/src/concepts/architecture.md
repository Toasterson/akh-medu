# Architecture

akh-medu is built on three complementary foundations: hyperdimensional
computing (VSA), knowledge graphs, and symbolic reasoning. Each handles a
different aspect of intelligence, and the engine unifies them under a single
API.

## The Three Pillars

### 1. Vector Symbolic Architecture (VSA)

VSA encodes symbols as high-dimensional binary vectors (default: 10,000 bits).
These hypervectors support algebraic operations:

| Operation | Function | Effect |
|-----------|----------|--------|
| **Bind** (XOR) | `bind(A, B)` | Creates a composite dissimilar to both inputs. Reversible: `unbind(bind(A, B), B) = A` |
| **Bundle** (majority vote) | `bundle(A, B, C)` | Creates a vector similar to all inputs. Set-like union. |
| **Permute** (bit shift) | `permute(A, n)` | Preserves structure but shifts representation. Encodes sequence position. |
| **Similarity** (Hamming) | `similarity(A, B)` | 0.5 = random, 1.0 = identical. Measures semantic closeness. |

VSA provides:
- **Fuzzy matching**: Misspellings, spelling variants, and near-synonyms
  discovered automatically via similarity search.
- **Analogy**: "A is to B as C is to ?" computed as `unbind(bind(A, B), C)`.
- **Implicit knowledge**: Relationships not explicit in the graph exist in the
  vector space and can be recovered via unbinding.

### 2. Knowledge Graph

A directed graph of `(subject, predicate, object)` triples with confidence
scores. Backed by two stores:

- **petgraph**: In-memory directed graph with dual indexing (symbol-to-node,
  node-to-symbol) for fast traversal.
- **Oxigraph**: Persistent RDF store with SPARQL query support and named
  graphs for compartment isolation.

The graph provides:
- **Explicit knowledge**: Direct lookup of facts.
- **Traversal**: BFS with depth, predicate, and confidence filtering.
- **Analytics**: Degree centrality, PageRank, strongly connected components.

### 3. Symbolic Reasoning (egg)

The `egg` e-graph library performs equality saturation -- applying rewrite
rules until no more simplifications are possible. This provides:

- **Algebraic simplification**: Expressions like `unbind(bind(A, B), B)`
  simplify to `A`.
- **Forward-chaining inference**: Rules derive new facts from existing ones.
- **Verification**: VSA-recovered inferences can be checked against the
  e-graph for mathematical consistency.

## Subsystem Map

```
┌──────────────────────────────────────────────────────────┐
│              Daemon / TUI Idle Scheduler                  │
│  Equivalence · Reflection · Rules · Schema · Gaps        │
├──────────────────────────────────────────────────────────┤
│                      Engine API                          │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│   VSA    │Knowledge │Reasoning │Inference │   Agent      │
│  Ops     │  Graph   │  (egg)   │ Engine   │  OODA Loop   │
├──────────┼──────────┼──────────┼──────────┼──────────────┤
│ HyperVec │petgraph  │ rewrite  │spreading │ 17 tools     │
│ SIMD     │oxigraph  │ rules    │backward  │ planning     │
│ ItemMem  │SPARQL    │ e-graphs │superpos. │ psyche       │
├──────────┴──────────┴──────────┴──────────┴──────────────┤
│             Grammar · Skills · Provenance                │
├──────────────────────────────────────────────────────────┤
│                   Tiered Storage                         │
│  Hot (DashMap) · Warm (mmap) · Durable (redb)            │
└──────────────────────────────────────────────────────────┘
```

The top layer -- the daemon and TUI idle scheduler -- drives autonomous
background learning. It sits above the Engine API and periodically calls
into subsystems (equivalence learning, rule inference, schema discovery)
without user prompting. See [Autonomous Background Learning](../agent/autonomy.md).

## Data Flow

A typical query flows through multiple subsystems:

1. **Input**: Natural language parsed by the grammar framework into an `AbsTree`.
2. **Resolution**: Entity names resolved to `SymbolId` values via the registry
   (exact match) or item memory (fuzzy VSA match).
3. **Graph lookup**: Direct triples retrieved from the knowledge graph.
4. **Inference**: Spreading activation discovers related symbols. VSA
   bind/unbind recovers implicit relationships in parallel.
5. **Reasoning**: E-graph rules derive new facts and verify VSA results.
6. **Provenance**: Every derivation step is recorded with full traceability.
7. **Output**: Results linearized back to prose via the grammar framework.

## Tiered Storage

Data flows through three tiers based on access pattern:

| Tier | Backend | Purpose | Access Speed |
|------|---------|---------|--------------|
| **Hot** | DashMap (concurrent HashMap) | Active working set | Nanoseconds |
| **Warm** | memmap2 (memory-mapped files) | Large read-heavy data | Microseconds |
| **Durable** | redb (ACID key-value store) | Persistent data that survives restarts | Milliseconds |

The `TieredStore` composes all three with automatic promotion: data accessed
from the durable tier is promoted to the hot tier for subsequent reads.

## Configuration

The `EngineConfig` struct controls all subsystem parameters:

```rust
EngineConfig {
    dimension: Dimension::DEFAULT,    // 10,000 bits
    encoding: Encoding::Bipolar,      // +1/-1 interpretation
    data_dir: Some(path),             // None = memory-only
    max_memory_mb: 1024,              // hot tier budget
    max_symbols: 1_000_000,           // symbol registry limit
    language: Language::Auto,         // grammar language
}
```
