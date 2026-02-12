# Inference Engine

The akh-medu inference engine discovers implicit knowledge from existing symbol
associations. It combines three complementary strategies — spreading activation,
backward chaining, and superposition reasoning — all operating on the same
hypervector (VSA) substrate. Every inference produces a full provenance trail
so results can be explained, audited, and verified.

**Module**: `src/infer/` (4 files, ~1,500 lines, 17 tests)

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │       Engine API         │
                    │  infer()  infer_analogy  │
                    │  recover_filler()        │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐ ┌─────▼──────┐ ┌──────▼──────────┐
    │   Spreading    │ │  Backward  │ │  Superposition  │
    │   Activation   │ │  Chaining  │ │   Reasoning     │
    │  (engine.rs)   │ │(backward.rs│ │(superposition.rs│
    └───────┬────────┘ └─────┬──────┘ └───────┬─────────┘
            │                │                │
    ┌───────▼────────────────▼────────────────▼─────────┐
    │              Shared Infrastructure                 │
    │  VsaOps · ItemMemory · KnowledgeGraph · Provenance│
    └───────────────────────────────────────────────────┘
```

Each strategy accesses:
- **VsaOps** — bind, unbind, bundle, similarity (SIMD-accelerated)
- **ItemMemory** — HNSW-based approximate nearest neighbor search
- **KnowledgeGraph** — directed graph of `(subject, predicate, object)` triples
- **ProvenanceLedger** — persistent record of how each result was derived

---

## Strategy 1: Spreading Activation

**File**: `src/infer/engine.rs` (623 lines, 10 tests)

The primary inference strategy. Starting from seed symbols, activation spreads
outward along knowledge graph edges. At each hop, VSA bind/unbind recovery
runs in parallel, catching implicit relationships that the graph edges alone
would miss.

### Algorithm

1. **Seed activation**: Each seed symbol gets confidence 1.0. Their hypervectors
   are bundled into an initial interference pattern.

2. **Frontier expansion** (repeated for `max_depth` iterations):
   - For each activated but unexpanded symbol, retrieve outgoing triples
   - **Graph-direct activation**: Activate the triple's object with
     `confidence = parent_confidence * edge_confidence`
   - **VSA recovery**: Compute `unbind(subject_vec, predicate_vec)`, search
     item memory for the nearest match. If similarity >= threshold and the
     match differs from the graph-direct object, activate it too
   - Bundle newly activated vectors into the interference pattern

3. **E-graph verification** (optional): For VSA-recovered results, build an
   `egg` expression `(bind (bind from predicate) symbol)` and check if the
   e-graph can simplify it. Non-simplifiable expressions get a 10% confidence
   penalty.

4. **Result assembly**: Filter out seed symbols, sort by confidence, truncate
   to `top_k`.

### Confidence Model

Confidence propagates **multiplicatively** along the graph path:

```
C(node) = C(parent) * edge_confidence
```

For VSA recovery, confidence is capped by both the graph path and the vector
similarity:

```
C_vsa(node) = C(parent) * min(edge_confidence, similarity)
C(node) = max(C_graph, C_vsa)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seeds` | `Vec<SymbolId>` | `[]` | Starting symbols (required, non-empty) |
| `top_k` | `usize` | `10` | Maximum results to return |
| `max_depth` | `usize` | `1` | Number of expansion hops |
| `min_confidence` | `f32` | `0.1` | Discard activations below this |
| `min_similarity` | `f32` | `0.6` | VSA recovery similarity threshold |
| `verify_with_egraph` | `bool` | `false` | Enable e-graph verification |
| `predicate_filter` | `Option<Vec<SymbolId>>` | `None` | Only follow these predicates |

### API

```rust
// Builder pattern for queries
let query = InferenceQuery::default()
    .with_seeds(vec![dog_id, cat_id])
    .with_max_depth(3)
    .with_min_confidence(0.2)
    .with_egraph_verification();

let result: InferenceResult = engine.infer(&query)?;
// result.activations: Vec<(SymbolId, f32)> — sorted by confidence
// result.pattern: Option<HyperVec> — combined interference pattern
// result.provenance: Vec<ProvenanceRecord> — full derivation trail
```

### Additional Operations

**Analogy** — "A is to B as C is to ?":

```rust
// Computes bind(A, B) to capture the A→B relation,
// then unbind(relation, C) to recover D, and searches item memory.
let results: Vec<(SymbolId, f32)> = engine.infer_analogy(a, b, c, top_k)?;
```

Requires three distinct symbols. The relational vector `bind(A, B)` captures
the abstract relationship, which is then applied to C via `unbind`.

**Role-filler recovery** — "What is the object of (subject, predicate)?":

```rust
// unbind(subject_vec, predicate_vec) → search item memory
let fillers: Vec<(SymbolId, f32)> = engine.recover_filler(subject, predicate, 5)?;
```

---

## Strategy 2: Backward Chaining

**File**: `src/infer/backward.rs` (245 lines, 3 tests)

Reasons **from a goal backward** to find supporting evidence. Given a target
symbol, finds all triples where it appears as the object, then recursively
finds support for each subject. This answers the question: "What evidence
supports this conclusion?"

### Algorithm

1. Find all incoming triples `(?, ?, goal)` where the goal is the object
2. For each triple, optionally verify via VSA:
   `similarity(unbind(subject_vec, predicate_vec), goal_vec)`
3. Compute chain confidence: `parent_confidence * edge_confidence * vsa_similarity`
4. Prune chains below `min_confidence`
5. Recursively find support for each subject (up to `max_depth`)
6. Record leaf chains (where no further support exists)

### Types

```rust
pub struct BackwardChain {
    pub goal: SymbolId,                  // Target symbol
    pub supporting_triples: Vec<Triple>, // Evidence chain
    pub confidence: f32,                 // Product of step confidences
    pub depth: usize,                    // Deepest step
}

pub struct BackwardConfig {
    pub max_depth: usize,       // Default: 3
    pub min_confidence: f32,    // Default: 0.1
    pub vsa_verify: bool,       // Default: true
}
```

### API

```rust
use akh_medu::infer::backward::{infer_backward, BackwardConfig};

let chains = infer_backward(&engine, goal_symbol, &BackwardConfig::default())?;

for chain in &chains {
    println!("Support chain (confidence {:.2}, depth {}):",
        chain.confidence, chain.depth);
    for triple in &chain.supporting_triples {
        println!("  {} --{}--> {}", triple.subject, triple.predicate, triple.object);
    }
}
```

### VSA Verification

When `vsa_verify` is enabled (default), each step in the chain is checked:

```
recovered = unbind(subject_vec, predicate_vec)
similarity = cosine(recovered, goal_vec)
```

This acts as a plausibility check — if the VSA substrate doesn't "agree" that
the relationship holds, confidence is reduced proportionally.

---

## Strategy 3: Superposition Reasoning

**File**: `src/infer/superposition.rs` (517 lines, 4 tests)

Implements "computing in superposition" — multiple competing hypotheses
processed simultaneously in the same vector substrate. At branch points,
hypotheses fork. Constructive interference merges similar hypotheses;
destructive interference collapses contradicted ones.

### Core Concept

Unlike spreading activation which maintains a single global activation map,
superposition maintains **multiple independent hypotheses**. Each hypothesis
is a separate hypervector pattern with its own confidence and provenance.
This enables the engine to explore contradictory interpretations in parallel
and let the mathematics of interference determine the winner.

### Algorithm

1. **Seed**: Create initial hypothesis from bundled seed vectors (confidence 1.0)
2. **Expand** (repeated for `max_depth` iterations):
   - For each hypothesis, expand each activated symbol's outgoing triples
   - At branch points (multiple outgoing edges), **fork** new hypotheses
   - Each fork bundles the parent's pattern with the new symbol's vector
3. **Constructive interference**: Merge hypotheses whose patterns are similar
   (similarity > `merge_threshold`). Merged confidence uses noisy-OR:
   `(C_a + C_b) * 0.6`
4. **Destructive interference**: Compare each hypothesis against the seed
   evidence pattern. Low similarity reduces confidence:
   `interference = (similarity - 0.5) * 2.0`
   Negative interference decays confidence; hypotheses below `min_confidence`
   are pruned.
5. **Result**: Return all surviving hypotheses sorted by confidence, with the
   dominant (highest-confidence) hypothesis highlighted.

### Types

```rust
pub struct Hypothesis {
    pub pattern: HyperVec,                 // Superposition vector
    pub confidence: f32,                   // Current confidence
    pub provenance: Vec<ProvenanceRecord>, // How this hypothesis formed
    pub activated: Vec<(SymbolId, f32)>,   // Symbols in this hypothesis
}

pub struct SuperpositionConfig {
    pub max_hypotheses: usize,   // Default: 8
    pub merge_threshold: f32,    // Default: 0.65
    pub min_confidence: f32,     // Default: 0.1
    pub max_depth: usize,        // Default: 3
}

pub struct SuperpositionResult {
    pub dominant: Option<Hypothesis>,  // Highest-confidence survivor
    pub hypotheses: Vec<Hypothesis>,   // All survivors, sorted
    pub merges: usize,                 // Number of constructive merges
    pub collapses: usize,              // Number of destructive collapses
}
```

### API

```rust
use akh_medu::infer::superposition::{infer_with_superposition, SuperpositionConfig};

let config = SuperpositionConfig {
    max_hypotheses: 16,
    merge_threshold: 0.7,
    ..Default::default()
};

let result = infer_with_superposition(&[seed1, seed2], &engine, &config)?;

println!("Surviving hypotheses: {}", result.hypotheses.len());
println!("Merges: {}, Collapses: {}", result.merges, result.collapses);

if let Some(dominant) = &result.dominant {
    println!("Dominant hypothesis (confidence {:.2}):", dominant.confidence);
    for (sym, conf) in &dominant.activated {
        println!("  {:?} ({:.2})", sym, conf);
    }
}
```

### State Management

`SuperpositionState` manages the hypothesis population:

| Method | Description |
|--------|-------------|
| `fork()` | Create new hypothesis from parent + new symbol |
| `merge_constructive()` | Merge similar hypotheses (constructive interference) |
| `collapse_destructive()` | Prune contradicted hypotheses (destructive interference) |
| `dominant()` | Get highest-confidence hypothesis |
| `into_result()` | Consume state into final `SuperpositionResult` |

---

## Provenance

Every inference operation produces `ProvenanceRecord` entries that explain
exactly how each result was derived. Records are persisted to the provenance
ledger (redb) when available.

### Derivation Kinds

| Kind | Description | Fields |
|------|-------------|--------|
| `Seed` | Starting point of inference | — |
| `GraphEdge` | Activated via knowledge graph triple | `from`, `predicate` |
| `VsaRecovery` | Recovered via VSA unbind + item memory search | `from`, `predicate`, `similarity` |
| `RuleInference` | Derived by e-graph rewrite rule | `rule_name`, `from_symbols` |
| `FusedInference` | Produced by confidence fusion in autonomous cycle | `path_count`, `interference_score` |

Each record also carries:
- `derived_id: SymbolId` — the symbol this record is about
- `confidence: f32` — confidence at this derivation step
- `depth: usize` — how many hops from the seed
- `sources: Vec<SymbolId>` — symbols that contributed to this derivation

---

## Integration Points

### Engine (`src/engine.rs`)

The `Engine` type exposes three top-level inference methods:

```rust
impl Engine {
    /// Spreading activation with all active rules (built-in + skills).
    /// Automatically persists provenance to ledger.
    pub fn infer(&self, query: &InferenceQuery) -> AkhResult<InferenceResult>;

    /// Analogy: A:B :: C:?
    pub fn infer_analogy(&self, a: SymbolId, b: SymbolId, c: SymbolId, top_k: usize)
        -> AkhResult<Vec<(SymbolId, f32)>>;

    /// Role-filler recovery for (subject, predicate) → object.
    pub fn recover_filler(&self, subject: SymbolId, predicate: SymbolId, top_k: usize)
        -> AkhResult<Vec<(SymbolId, f32)>>;
}
```

### Pipeline (`src/pipeline/mod.rs`)

The `Infer` stage runs spreading activation as part of the linear pipeline:

```
Retrieve → Infer → Reason → Extract
```

The infer stage accepts a `query_template` that is cloned and populated with
seeds from the retrieve stage's output. Custom depth and confidence can be
set via the template.

### Autonomous Cycle (`src/autonomous/integration.rs`)

The autonomous cycle uses superposition reasoning as step 3:

1. Forward-chaining rules (e-graph rewrite)
2. Symbol grounding (re-encode all symbols into VSA)
3. **Superposition inference** — forking hypotheses from seed symbols
4. Confidence fusion — merge rule-derived and superposition-derived paths
5. KG commit — insert high-confidence triples into the knowledge graph

### Agent (`src/agent/ooda.rs`)

The agent's Orient phase runs spreading activation to find relevant knowledge
for the current goal, feeding inferences into the Decide phase for tool
selection.

### CLI (`src/main.rs`)

```bash
# Spreading activation
akh-medu query --seeds "Dog,Cat" --depth 3 --top-k 20

# Analogy
akh-medu analogy --a "King" --b "Man" --c "Queen" --top-k 5

# Forward-chaining inference rules
akh-medu infer

# Pipeline with custom inference depth
akh-medu pipeline run --infer-depth 3 --stages retrieve,infer,reason
```

---

## Error Handling

All inference errors are reported via `InferError` with miette diagnostics:

| Error | Code | Description |
|-------|------|-------------|
| `NoSeeds` | `akh::infer::no_seeds` | Inference query has empty seeds list |
| `InvalidAnalogy` | `akh::infer::analogy` | Analogy requires 3 distinct symbols |
| `MaxDepthExceeded` | `akh::infer::depth` | Inference depth limit reached |
| `VsaError` | (transparent) | Underlying VSA operation failure |

---

## Design Rationale

**Why three strategies?** Each addresses a different reasoning need:

- **Spreading activation** is the workhorse — fast, breadth-first, good for
  "what's related to X?" queries. It finds direct and indirect associations.
- **Backward chaining** answers "why?" questions — given a conclusion, find
  the evidence chain that supports it. Essential for explainability.
- **Superposition** handles ambiguity — when the graph has multiple
  contradictory paths, it explores them all simultaneously and lets
  constructive/destructive interference pick the winner.

**Why VSA recovery alongside graph traversal?** The knowledge graph captures
explicit relationships, but VSA encodes distributional similarity. A triple
`(Dog, is-a, Mammal)` is explicit in the graph, but the implicit analogy
"Dog is to Puppy as Cat is to Kitten" lives in the vector space. Running both
in parallel catches knowledge that either alone would miss.

**Why e-graph verification?** The `egg` e-graph engine provides algebraic
simplification of VSA expressions. If `unbind(bind(A, B), B)` doesn't simplify
to `A`, it suggests the recovery was noisy. This provides a cheap
mathematical sanity check on VSA-recovered inferences.

---

## Tests

The inference module has 17 tests across the three strategy files:

**Spreading Activation** (`engine.rs`, 10 tests):
- `infer_no_seeds_returns_error` — empty seeds produce `NoSeeds` error
- `single_hop_inference` — one-step graph traversal finds correct target
- `multi_hop_inference` — depth 1 vs depth 2 reaches different nodes
- `confidence_propagates_multiplicatively` — confidence decays along path
- `role_filler_recovery` — unbind(subject, predicate) recovers filler
- `analogy_inference` — A:B::C:? returns results
- `analogy_requires_three_distinct` — duplicate symbols rejected
- `provenance_records_generated` — Seed and GraphEdge records present
- `empty_graph_no_activations` — isolated symbol produces empty results
- `predicate_filter_respected` — only specified predicates are followed

**Backward Chaining** (`backward.rs`, 3 tests):
- `find_support_chain` — Dog→Mammal→Animal finds multi-step evidence
- `confidence_decreases_with_depth` — deeper chains have lower confidence
- `no_support_for_isolated_symbol` — no incoming triples means no chains

**Superposition** (`superposition.rs`, 4 tests):
- `fork_creates_multiple_hypotheses` — branch points produce multiple hypotheses
- `constructive_merge_combines_similar` — identical patterns merge
- `destructive_collapse_removes_contradicted` — dissimilar patterns are pruned
- `dominant_hypothesis_has_highest_confidence` — dominant picks the right one
