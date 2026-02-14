# Symbols and Triples

Symbols and triples are the fundamental data model in akh-medu. Every piece
of knowledge is represented as symbols connected by triples.

## Symbols

A symbol is a unique entity in the knowledge base. Each symbol has:

- **SymbolId**: A `NonZeroU64` identifier, globally unique within an engine.
- **SymbolKind**: The type of thing the symbol represents.
- **Label**: A human-readable name (e.g., "Dog", "is-a", "mammal").
- **Confidence**: How certain we are this symbol is meaningful (0.0-1.0).

### Symbol Kinds

| Kind | Purpose | Examples |
|------|---------|---------|
| `Entity` | A concrete or abstract thing | "Dog", "Paris", "mitochondria" |
| `Relation` | A relationship type | "is-a", "has-part", "causes" |
| `Concept` | An abstract idea | "freedom", "recursion" |
| `Agent` | An autonomous agent | The agent itself |

### Creating Symbols

```rust
use akh_medu::symbol::SymbolKind;

// Create specific symbols
let dog = engine.create_symbol(SymbolKind::Entity, "Dog")?;
let is_a = engine.create_symbol(SymbolKind::Relation, "is-a")?;

// Resolve-or-create (idempotent)
let mammal = engine.resolve_or_create_entity("mammal")?;
let causes = engine.resolve_or_create_relation("causes")?;
```

### Symbol Resolution

The engine resolves names to IDs through multiple strategies:

1. **Exact match**: `engine.lookup_symbol("Dog")` -- direct registry lookup.
2. **ID parse**: `engine.resolve_symbol("42")` -- tries parsing as a numeric ID.
3. **Fuzzy match**: The lexer encodes unknown tokens as hypervectors and
   searches item memory for similar known symbols (threshold >= 0.60).

### Hypervector Encoding

Every symbol is encoded as a 10,000-bit binary hypervector using deterministic
seeded random generation. The same `SymbolId` always produces the same vector:

```
SymbolId(42) -> deterministic random HyperVec (seeded with 42)
```

Multi-word labels are encoded by bundling per-word vectors:

```
"big red dog" -> bundle(encode("big"), encode("red"), encode("dog"))
```

The resulting vector is similar to each component but identical to none,
capturing set-like semantics.

## Triples

A triple is a fact: `(subject, predicate, object)` with a confidence score.

```rust
use akh_medu::graph::Triple;

let triple = Triple::new(dog_id, is_a_id, mammal_id, 0.95);
engine.add_triple(triple)?;
```

### Triple Fields

| Field | Type | Description |
|-------|------|-------------|
| `subject` | `SymbolId` | The entity being described |
| `predicate` | `SymbolId` | The relationship type |
| `object` | `SymbolId` | The related entity |
| `confidence` | `f32` | Certainty score (0.0-1.0) |

### Confidence

Confidence scores flow through the system:

- **Input confidence**: Set when triples are created (default: 0.8 for seed
  packs, 0.90 for grammar-parsed facts).
- **Propagation**: During inference, confidence propagates multiplicatively:
  `C(node) = C(parent) * edge_confidence`.
- **Fusion**: When multiple paths reach the same node, confidences merge:
  `C = max(C_graph, C_vsa)`.

### Querying Triples

```rust
// All facts about Dog
let from_dog = engine.triples_from(dog_id);

// All facts pointing to mammal
let to_mammal = engine.triples_to(mammal_id);

// Check existence
let exists = engine.has_triple(dog_id, is_a_id, mammal_id);

// All triples in the KG
let all = engine.all_triples();
```

### SPARQL

For complex queries, use SPARQL against the Oxigraph store:

```bash
akh-medu sparql "SELECT ?animal WHERE {
  ?animal <https://akh-medu.dev/sym/is-a> <https://akh-medu.dev/sym/mammal>
}"
```

Symbol IRIs follow the pattern `https://akh-medu.dev/sym/{label}`.

## Provenance

Every triple and derived fact carries provenance -- a record of how it was
created:

| Derivation Kind | Description |
|-----------------|-------------|
| `Fact { source }` | External input (ingested file, user assertion) |
| `InferenceRule { rule_id }` | Derived by e-graph rewrite rule |
| `VsaBind { factors }` | Created via VSA binding operation |
| `VsaUnbind { factors }` | Recovered via VSA unbinding |
| `GraphEdge` | Found via knowledge graph traversal |
| `AgentConsolidation` | Created during agent memory consolidation |
| `AgentDecision` | Created by agent tool execution |
| `CompartmentLoaded` | Loaded from a compartment's triples file |

```rust
// Get the derivation trail for a symbol
let records = engine.provenance_of(dog_id)?;
for record in &records {
    println!("  derived via {:?} (confidence {:.2})",
        record.derivation_kind, record.confidence);
}
```

## The Symbol Registry

The `SymbolRegistry` is a bidirectional map between labels and IDs, backed
by the tiered store:

- Labels are case-sensitive.
- Duplicate labels are rejected -- each label maps to exactly one ID.
- The `AtomicSymbolAllocator` generates IDs via atomic increment (lock-free).

## Item Memory

The `ItemMemory` stores hypervector encodings for approximate nearest-neighbor
(ANN) search using the HNSW algorithm with Hamming distance:

```rust
// Find symbols similar to Dog
let results = engine.search_similar_to(dog_id, 5)?;
for r in &results {
    println!("  {} (similarity {:.2})", engine.resolve_label(r.id), r.similarity);
}
```

Search is sub-linear: O(log n) even with millions of vectors.
