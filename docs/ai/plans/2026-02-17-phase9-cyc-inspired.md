# Phase 9 — Cyc-Inspired Enhancements

- **Date**: 2026-02-17
- **Updated**: 2026-02-20 (Production wiring: all 15 sub-systems integrated into engine lifecycle)
- **Status**: Complete (including production wiring — see ADR-007)
- **Motivation**: ADR-001 (Cyc paper analysis)
- **Depends on**: Phases 1–8f (all complete)

## Goal

Incorporate ideas from Cyc's 40-year symbolic AI project and CycL's higher-order logic features into akh-medu's neuro-symbolic architecture. Ordered by priority: foundational capabilities first (microtheories, predicate hierarchy, TMS), then reasoning enhancements (defeasibility, argumentation, dispatch), then knowledge-quality features (rule macros, skolem, type constraints), then advanced logic (circumscription, second-order quantification, NARTs).

---

## HIGH PRIORITY

---

## Phase 9a — Microtheories (`ist` + `genlMt` + Lifting Rules)

**Problem**: Compartments are opaque containers. You can't reason about them, inherit between them, assert context-relative truth, or propagate entailments across contexts.

**Cyc mechanism**: The `ist(context, sentence)` modal operator asserts that a sentence is true in a given context. `genlMt` specifies inheritance (multiple inheritance allowed). Lifting rules govern when entailments propagate from one microtheory to another. Each microtheory carries domain assumptions as implicit conjuncts — assertions within are terser because shared assumptions are factored out. Each microtheory is internally contradiction-free even if the KB as a whole contains contradictions across contexts.

**Design**:
- Promote `Compartment` to a first-class `Entity` symbol with well-known relation predicates (prefixed `ctx:`)
  - `ctx:specializes` (`genlMt`) — inheritance link with multiple inheritance
  - `ctx:assumes` — domain assumption triples factored out of in-context assertions
  - `ctx:domain` — what kind of context (temporal, cultural, belief, fictional, task, etc.)
  - `ctx:disjoint` — two contexts cannot both apply (StarWars vs RealWorld)
- `ist` operator: a triple can assert "in context C, (S, P, O) holds" — reifying context-relative truth
- Context inheritance: when querying in context C, also search ancestor contexts via `ctx:specializes` chains (transitive closure)
- Lifting rules: configurable rules for when entailments propagate between sibling/cousin contexts
- Domain assumption factoring: triples in a context implicitly carry the context's `ctx:assumes` as preconditions
- Internal consistency: contradiction detection (9l) scoped per-microtheory

**Key types**:
```rust
Microtheory { id: SymbolId, domain: ContextDomain, ancestors: Vec<SymbolId> }
ContextDomain: Temporal | Cultural | Belief | Fictional | Task | General
LiftingRule { from: SymbolId, to: SymbolId, condition: LiftCondition }
LiftCondition: Always | IfConsistent | IfNotOverridden
```

**Changes**:
- [x] `compartment/microtheory.rs` — `Microtheory`, `ContextDomain`, `LiftingRule`, `LiftCondition`, `ContextPredicates` (6 well-known ctx: predicates), context-scoped queries, ancestry cache, 9 tests
- [x] `compartment/error.rs` — `DisjointConflict`, `ContextCycle` error variants
- [x] `graph/` — context-aware triple queries (triples_in_context, all_triples_in_context, objects_in_context)
- [x] `engine.rs` — `create_context()`, `add_context_assumption()`, `query_in_context()`, `query_all_in_context()`, `add_lifting_rule()`, `context_ancestors()`, `contexts_are_disjoint()`, `apply_lifting_rules()` APIs
- [x] `provenance.rs` — new `DerivationKind::ContextInheritance` (tag 22), `DerivationKind::ContextLifting` (tag 23)
- [ ] Agent — goal contexts (a goal can specify which microtheory it operates in)

**Implemented scope**: ~420 lines

---

## Phase 9b — Predicate Hierarchy (`genlPreds` + `genlInverse`)

**Problem**: Relations are flat SymbolIds with no subsumption. Can't infer `parent(X,Y)` from `biologicalMother(X,Y)`. No inverse relations.

**Cyc mechanism**: `genlPreds` declares that one predicate is a specialization of another, forming a predicate subsumption lattice. If `biologicalMother(X,Y)` is asserted and `genlPreds(biologicalMother, mother)` and `genlPreds(mother, parent)` exist, then `parent(X,Y)` is automatically inferable. `genlInverse` declares argument-swapped equivalence: `parent(X,Y) ↔ child(Y,X)`.

**Design**:
- Well-known predicates: `rel:generalizes` (maps to `genlPreds`), `rel:inverse` (maps to `genlInverse`)
- Predicate hierarchy stored as triples in the KG, forming a lattice via `rel:generalizes`
- Inference integration: when searching for triples with predicate P, also search for triples with any predicate Q where `genlPreds(Q, P)` (Q is more specific)
- Inverse inference: when searching for `(?, P, X)`, also check `(X, P_inverse, ?)` where `genlInverse(P, P_inverse)`
- Cache the transitive closure of `rel:generalizes` for fast lookup (invalidate on predicate hierarchy changes)

**Key types**:
```rust
PredicateHierarchy { generalizes: HashMap<SymbolId, Vec<SymbolId>>, inverses: HashMap<SymbolId, SymbolId> }
```

**Changes**:
- [x] `graph/predicate_hierarchy.rs` — `PredicateHierarchy`, `HierarchyPredicates` (2 well-known `rel:` predicates), `HierarchyMatch`, `MatchVia`, hierarchy-aware queries, transitive closure cache, 7 tests
- [x] `graph/` — predicate-aware triple queries: `objects_with_hierarchy()`, `objects_with_hierarchy_and_inverse()`, `triples_with_hierarchy()`
- [x] `engine.rs` — `add_predicate_generalization()`, `add_predicate_inverse()`, `build_predicate_hierarchy()`, `query_with_hierarchy()` APIs
- [ ] `infer/` — spreading activation and backward chaining use predicate hierarchy
- [x] `provenance.rs` — `DerivationKind::PredicateGeneralization` (tag 24), `DerivationKind::PredicateInverse` (tag 25)

**Implemented scope**: ~390 lines

---

## Phase 9c — Truth Maintenance System (TMS)

**Problem**: Provenance records sources but there's no automatic retraction cascade. Removing a triple leaves downstream inferences orphaned. No re-evaluation of alternative justifications.

**Cyc mechanism**: Every deduction records its full list of supports (premises). When a support is retracted, the TMS automatically: (1) retracts all conclusions that depended solely on that support, (2) re-evaluates conclusions that had alternative justifications, (3) cascades through the entire dependency graph. This makes the KB self-healing.

**Design**:
- Each triple/inference carries a `SupportSet` — the set of premises that justify it
- A triple may have multiple support sets (alternative justifications)
- On retraction of triple T:
  1. Find all triples whose support sets include T (via provenance index by source)
  2. For each dependent, remove the support set containing T
  3. If the dependent has remaining support sets, keep it (re-evaluate confidence from surviving supports)
  4. If the dependent has no remaining support sets, retract it recursively
- Retraction cascade is transactional: either the full cascade succeeds or nothing changes
- Provenance ledger becomes bidirectional: not just recording derivation but driving retraction

**Key types**:
```rust
SupportSet { premises: Vec<SymbolId>, derivation: DerivationKind, confidence: f64 }
TruthMaintenanceSystem { supports: HashMap<SymbolId, Vec<SupportSet>> }
RetractionResult { retracted: Vec<SymbolId>, re_evaluated: Vec<(SymbolId, f64)>, cascade_depth: usize }
```

**Changes**:
- [x] `src/tms.rs` — `TruthMaintenanceSystem`, `SupportSet`, `RetractionResult`, BFS retraction cascade, re-evaluation of alternative justifications, 11 tests
- [ ] `provenance.rs` — extend to track support sets per derived symbol, index by source for fast reverse lookup
- [x] `graph/index.rs` — `remove_triple()` method added to `KnowledgeGraph`
- [x] `engine.rs` — `remove_triple()` public API with TMS integration
- [ ] `store/` — support sets persisted in durable store alongside provenance

**Implemented scope**: ~310 lines

---

## Phase 9d — Defeasible Reasoning

**Problem**: Conflicting triples are resolved by confidence alone. No specificity-based override, no exception hierarchy, no default-truth semantics.

**Cyc mechanism**: Almost all knowledge is default-true. When triples conflict, specificity-based override applies: more specific rules trump more general ones (`penguin.flies=false` overrides `bird.flies=true` because penguin ⊂ bird). Exceptions can be registered explicitly, and exceptions to exceptions, etc. Monotonic rules are always preferred over default rules.

**Design**:
- Compute **specificity** when triples conflict: check if one subject's type is more specific via `is-a` / `genls` chains
- Exception registration: `defeasible:except(general_rule_id, specific_override_id)` triples
- Override resolution order: monotonic > default; specific > general; then confidence as tiebreaker
- Integrate with argumentation (9e): each conflicting triple becomes a pro or con argument
- Triples can be marked as `defeasible:monotonic` (never overridden by defaults)

**Key types**:
```rust
DefeasibleResult { winner: Triple, losers: Vec<Triple>, reason: OverrideReason }
OverrideReason: Monotonic | Specificity { depth: usize } | Exception | Recency | Confidence
```

**Changes**:
- [x] `src/graph/defeasible.rs` — `DefeasiblePredicates` (4 well-known predicates), `OverrideReason` (5 variants: Monotonic, Specificity, Exception, Recency, Confidence), `DefeasibleResult`, `resolve_conflict()`, `find_conflicts()`, `find_hierarchy_conflicts()`, `query_defeasible()`, `mark_monotonic()`, `register_exception()`, `type_depth()`, BFS-based `is_subtype_of()`, 12 tests
- [x] `src/graph/mod.rs` — added `pub mod defeasible;`
- [x] `src/provenance.rs` — `DerivationKind::DefeasibleOverride { winner, loser, reason }` (tag 26)
- [x] `engine.rs` — `defeasible_predicates()`, `mark_monotonic()`, `register_exception()`, `query_defeasible()`, `resolve_conflict()` APIs
- [x] `main.rs` — `format_derivation_kind` match arm for DefeasibleOverride

**Depends on**: 9b (predicate hierarchy for specificity computation)

**Implemented scope**: ~440 lines, 12 tests

---

## Phase 9e — Pro/Con Argumentation

**Problem**: No structured way to collect, weigh, and present competing arguments for an answer. Competing hypotheses interact via vector interference, not structured arguments.

**Cyc mechanism**: When asked a question, Cyc gathers all pro and con arguments for each possible answer. Meta-level rules decide which arguments to prefer: monotonic over default, specific over general, recent over stale, expert over novice, constructive over non-constructive. The entire process produces not just an answer but a fully auditable argument structure.

**Design**:
- `ArgumentSet` collects all pro and con evidence chains for a query from the provenance ledger
- Each `Argument` wraps a provenance chain with strength metadata
- Meta-rules rank arguments (ordered by priority):
  1. **Monotonic > default**: monotonically-derived conclusions always win
  2. **Specificity**: more specific premises beat more general (from 9d)
  3. **Recency**: newer evidence beats older
  4. **Depth**: shorter chains beat longer (less default-truth decay per step)
  5. **Source quality**: provenance-tracked source reliability
  6. **Constructive > non-constructive**: arguments identifying instances beat pure existence claims
- `Verdict` summarizes: winning answer, confidence, top pro arguments, top con arguments, full reasoning chain

**Key types**:
```rust
Argument { conclusion: SymbolId, polarity: Pro | Con, chain: Vec<ProvenanceRecord>, strength: f64, monotonic: bool }
ArgumentSet { query: SymbolId, arguments: Vec<Argument>, verdict: Verdict }
Verdict { answer: SymbolId, confidence: f64, pro: Vec<Argument>, con: Vec<Argument>, reasoning: String }
MetaRule: Monotonicity | Specificity | Recency | Depth | SourceQuality | Constructiveness
```

**Changes**:
- [x] `src/argumentation/mod.rs` — `Polarity` (Pro/Con), `MetaRule` (6 rules with default priority order), `MetaRuleScores` (per-rule scores with exponentially-weighted total), `Argument` (conclusion, polarity, chain, strength, monotonic, scores), `ArgumentSet` (subject, predicate, candidates map, verdict), `Verdict` (answer, confidence, pro/con lists, reasoning string, decisive rule)
- [x] Scoring: `score_chain()` computes 6 dimensions (monotonicity, specificity via type_depth, recency via exponential decay, depth, source_quality, constructiveness), `is_monotonic_derivation()`, `is_constructive()`
- [x] Core logic: `argue()` and `argue_with_rules()` — collects candidates via triple query, builds pro arguments from provenance chains, generates con arguments from competing candidates (attenuated 0.8x), computes net strength verdict with decisive meta-rule detection
- [x] `src/provenance.rs` — `DerivationKind::ArgumentVerdict { winner, pro_count, con_count, decisive_rule }` (tag 28)
- [x] `engine.rs` — `argue()`, `argue_with_rules()` APIs
- [x] `main.rs` — `format_derivation_kind` match arm for ArgumentVerdict
- [x] `lib.rs` — `pub mod argumentation;`
- [x] `src/graph/defeasible.rs` — `type_depth()` made `pub` for cross-module use
- [x] 13 tests: meta-rules, scoring, polarity, single/competing candidates, confidence winner, empty verdict, custom meta-rule order, strength positivity, reasoning strings, constructiveness

**Depends on**: 9c (TMS for support chain traversal), 9d (defeasibility for specificity ranking)

**Implemented scope**: ~560 lines, 13 tests

---

## Phase 9f — Competitive Reasoner Dispatch

**Problem**: Inference strategies are called directly. No competitive selection, no resource budgets, no time-bounded interruption.

**Cyc mechanism**: 1,100+ specialized reasoners (Heuristic Level modules), each with its own redundant representation optimized for speed. A transitive-closure module pre-computes full closure. A linear-equation solver handles n-unknowns. They all "bid" on sub-problems and the fastest applicable one wins. Resource budgets enforce time limits; slow reasoners get interrupted.

**Design**:
- `ReasonerRegistry` holds registered reasoners, each implementing a `Reasoner` trait
- Each reasoner declares: `can_handle(&Problem) -> Option<Bid>` where `Bid` includes estimated cost/time
- Dispatcher sorts bids, executes the cheapest applicable reasoner, falls back on failure
- Resource budget: configurable per-query time limit; reasoner gets killed if exceeded
- Built-in specialized reasoners to add:
  - **TransitiveClosure**: pre-computes and caches full closure of `is-a`, `part-of`, `ctx:specializes`, and other transitive relations
  - **TypeHierarchy**: fast type-checking via cached ancestry chains
  - **PredicateHierarchy**: fast predicate subsumption via cached `rel:generalizes` closure (from 9b)
  - **ArithmeticEval**: handles numeric comparisons and basic arithmetic

**Key types**:
```rust
trait Reasoner: Send + Sync {
    fn name(&self) -> &str;
    fn can_handle(&self, problem: &Problem) -> Option<Bid>;
    fn solve(&self, problem: &Problem, budget: Duration) -> ReasonerResult;
}
Bid { estimated_cost: Duration, confidence: f64 }
ReasonerRegistry { reasoners: Vec<Box<dyn Reasoner>> }
```

**Changes**:
- [x] `src/dispatch/mod.rs` — `DispatchError` (4 variants with miette diagnostics), `Problem` (7 kinds: ForwardInference, BackwardChaining, Superposition, EGraphSimplify, TransitiveClosure, TypeCheck, PredicateSubsumption), `Bid` (estimated_cost + confidence, score-based sorting), `Reasoner` trait (name, can_handle, solve), `ReasonerRegistry` (register, dispatch, dispatch_with_budget, fallback-on-failure), `ReasonerOutput` (Inference, BooleanAnswer, Simplified), `DispatchTrace` (reasoner_name, all bids, elapsed, attempts)
- [x] Built-in wrappers: `SpreadingActivationReasoner`, `BackwardChainingReasoner`, `SuperpositionReasoner`, `EGraphReasoner`
- [x] Specialized reasoners: `TransitiveClosureReasoner` (BFS chain + TypeCheck fallback), `TypeHierarchyReasoner` (ancestor-set membership), `PredicateHierarchyReasoner` (rel:generalizes BFS)
- [x] `src/error.rs` — `AkhError::Dispatch` variant
- [x] `src/provenance.rs` — `DerivationKind::DispatchRoute { reasoner, problem_kind }` (tag 27)
- [x] `engine.rs` — `reasoner_registry()`, `dispatch()`, `dispatch_with_budget()` APIs
- [x] `main.rs` — `format_derivation_kind` match arm for DispatchRoute
- [x] `lib.rs` — `pub mod dispatch;`
- [x] 13 tests: registry, bid scoring, e-graph simplify, forward inference, transitive closure (positive + negative), type hierarchy, predicate subsumption, backward chaining, fallback, trace

**Implemented scope**: ~690 lines, 13 tests

---

## MEDIUM PRIORITY

---

## Phase 9g — Rule Macro Predicates

**Problem**: Common knowledge patterns require verbose raw triples. No compact higher-order shorthands. No specialized reasoner hooks for common patterns.

**Cyc mechanism**: Rule macro predicates (RMPs) are compact meta-predicates that expand into complex quantified formulas. `(relationAllExists biologicalMother Chordata FemaleAnimal)` expands to `∀x: Chordata(x) → ∃y: FemaleAnimal(y) ∧ biologicalMother(x,y)`. `genls` is itself an RMP: `(genls Dog Mammal)` expands to `∀x: Dog(x) → Mammal(x)`. Each RMP has its own specialized reasoner that handles it faster than expanding to raw quantified logic.

**Design**:
- `RuleMacro` trait: declares expansion from compact form to raw triple pattern
- Built-in rule macros:
  - `relationAllExists(R, C1, C2)` — every C1 instance has an R-related C2 instance
  - `relationExistsAll(R, C1, C2)` — there exists a C1 instance R-related to every C2 instance
  - `relationAllAll(R, C1, C2)` — every C1 instance is R-related to every C2 instance
  - `genls(C1, C2)` — C1 ⊂ C2 (already partially covered by `is-a`)
- Each rule macro registers a specialized removal module (reasoner) that answers queries matching the macro's pattern without expanding
- Rule macros are syntactic sugar at the engine API level — internally stored as annotated triples

**Key types**:
```rust
trait RuleMacro: Send + Sync {
    fn name(&self) -> &str;
    fn matches(&self, triple: &Triple) -> bool;
    fn expand(&self, triple: &Triple) -> Vec<ExpandedPattern>;
    fn can_answer(&self, query: &Query) -> bool;
    fn answer(&self, query: &Query, kg: &KnowledgeGraph) -> Vec<Triple>;
}
RuleMacroRegistry { macros: Vec<Box<dyn RuleMacro>> }
```

**Changes**:
- [x] New module: `src/rule_macro.rs` — `RuleMacro` trait, `RuleMacroRegistry`, `MacroPredicates`, `MacroInvocation`
- [x] `engine.rs` — `rule_macro_registry()`, `macro_predicates()` facade APIs
- [x] Built-in macros: `GenlsMacro`, `RelationAllExistsMacro`, `RelationExistsAllMacro`
- [x] 12 tests: matching, expansion, query answering, registry operations
- [x] `error.rs` — `AkhError::RuleMacro` variant
- [x] `provenance.rs` — `DerivationKind::RuleMacroExpansion` (tag 29)

**Estimated scope**: ~500–700 lines (actual: ~580 lines)

---

## Phase 9h — Skolem Functions (Existential Witnesses)

**Problem**: Existential inference produces abstract results. Can't query "which X satisfies ∃X: P(X)?" No concrete witness symbols.

**Cyc mechanism**: When `(thereExists ?M (mother ?A ?M))` is asserted, Cyc creates a Skolem function `(SkolemFn-12345 ?A)` — a named witness for the existential. This eliminates existentials from the KB, making inference purely universal-quantifier-based. The Skolem function is a first-class term.

**Design**:
- When the engine encounters an existential pattern (a rule macro `relationAllExists` or explicit), create a **Skolem symbol** — a placeholder entity representing "the thing that exists"
- Skolem symbols carry metadata: the existential they witness, the bound variables
- Skolem symbols are first-class `Entity` symbols, queryable and reifiable
- When a concrete entity later satisfies the existential, the Skolem symbol is **grounded** (linked to the real entity via provenance)

**Key types**:
```rust
SkolemSymbol { id: SymbolId, existential: SymbolId, bound_vars: Vec<SymbolId> }
```

**Changes**:
- [x] `src/skolem.rs` — `SkolemSymbol`, `SkolemRegistry` with deduplication, existential_index, create_skolem/ground/unground/check_grounding/auto_ground
- [x] `engine.rs` — `skolem_registry()` facade API
- [x] `error.rs` — `AkhError::Skolem` variant
- [x] `provenance.rs` — `DerivationKind::SkolemWitness` (tag 32), `DerivationKind::SkolemGrounding` (tag 33)
- [x] 10 tests: create, deduplication, grounding, double-grounding error, unground/reground, check_grounding from KG, auto_ground, find_for_existential, not_found error

**Depends on**: 9g (rule macros generate the existential patterns)

**Estimated scope**: ~300–500 lines (actual: ~330 lines)

---

## Phase 9i — Argumentation-Based Truth Values

**Problem**: Confidence is a single `f64` assigned at assertion time. Truth determination is lookup-based, not argumentation-based. No collection of supporting/opposing evidence before committing a truth value.

**Cyc mechanism**: To determine if an assertion is true or false, Cyc's inference engine uses argumentation — weighing various pro and con arguments to arrive at a truth value. This is the fundamental truth-determination mechanism, not a presentation layer.

**Design**:
- Extend triple storage: instead of a single `confidence: f64`, optionally store `ArgumentSet` (from 9e)
- On query, if multiple triples bear on the same question, dynamically compute truth via argumentation
- Cached truth value: store the verdict's confidence as the effective confidence, but retain the full argument set
- Re-evaluation: when new evidence arrives (new triples, TMS changes), re-run argumentation for affected questions
- Gradual adoption: opt-in per query via `query_with_argumentation()`, existing `confidence` field remains for simple cases

**Changes**:
- [x] `graph/argumentation_truth.rs` — `ArgumentationCache`, `CachedVerdict`, `ArgumentationKey`, `query_with_argumentation()`, `query_with_argumentation_rules()`, cache invalidation (by symbol, by (s,p) pair, full clear)
- [x] `engine.rs` — `argumentation_cache()`, `query_with_argumentation()` facade APIs
- [x] 5 tests: cache put/get, invalidate_for_symbol, invalidate_specific, clear, basic argumentation query

**Depends on**: 9e (argumentation framework), 9c (TMS for re-evaluation triggers)

**Estimated scope**: ~300–500 lines (actual: ~280 lines)

---

## Phase 9j — Arity and Type Constraints

**Problem**: Relations have no declared arity or argument-type constraints. Can assert malformed triples.

**Cyc mechanism**: Every relation declares its arity (`#$arity`) and argument types (`#$arg1Isa`, `#$arg2Isa`). The system enforces these at assertion time — you can't assert `(biologicalMother France BillClinton)` because France isn't an Animal. This catches errors early and improves KB quality.

**Design**:
- Well-known predicates: `onto:arity`, `onto:arg1type`, `onto:arg2type`, `onto:arg3type`
- On `add_triple()`, optionally check:
  - Predicate arity matches (binary predicates take exactly subject + object)
  - Subject type matches `onto:arg1type` via `is-a` chain
  - Object type matches `onto:arg2type` via `is-a` chain
- Violations produce diagnostic errors (miette), not silent failures
- Enforcement is opt-in (skip for bootstrap/migration scenarios)

**Key types**:
```rust
ArityConstraint { relation: SymbolId, arity: usize, arg_types: Vec<Option<SymbolId>> }
TypeViolation { relation: SymbolId, arg_position: usize, expected: SymbolId, actual: SymbolId }
```

**Changes**:
- [x] `graph/arity.rs` — `ConstraintRegistry`, `RelationConstraint`, `ConstraintViolation`, `ArityPredicates`, `ArityError`
- [x] `engine.rs` — `constraint_registry()`, `arity_predicates()` facade APIs
- [x] `is_instance_of()` BFS type checker through `is-a` chains
- [x] `check_triple()` / `check_triple_or_err()` with diagnostic errors
- [x] `error.rs` — `AkhError::Arity` variant
- [x] 8 tests: direct/transitive type checking, violations, arity mismatch

**Depends on**: 9b (predicate hierarchy for type checking via `is-a` chains)

**Estimated scope**: ~400–600 lines (actual: ~500 lines)

---

## Phase 9k — Temporal Projection

**Problem**: Triples have timestamps but no model of how confidence decays over time.

**Cyc mechanism**: Temporal projection models how truth changes over time. "I learn you own a house, from which I can infer how likely it was 2 years ago or 3 years from now." Each projection follows a decay curve (linear, Gaussian, etc.) with direction-specific parameters. Things change at boundaries (state lines) and interrupting events (selling your house).

**Design**:
- Relations can carry a `TemporalProfile` specifying how confidence decays over time
- Profiles: `Stable` (low decay — species, mathematical truths), `Decaying { half_life }` (ownership, employment), `Ephemeral { ttl }` (location, mood), `Periodic { period }` (seasonal facts)
- When querying, apply temporal decay to confidence based on triple timestamp vs query time
- Well-known relations get default profiles (configurable)
- Expired triples (confidence below threshold after decay) are soft-deleted, not hard-deleted

**Key types**:
```rust
TemporalProfile: Stable | Decaying { half_life: Duration } | Ephemeral { ttl: Duration } | Periodic { period: Duration }
```

**Changes**:
- [x] `src/temporal.rs` — `TemporalProfile` enum (Stable, Decaying, Ephemeral, Periodic), `TemporalRegistry`, `TemporalPredicates`, `TemporalError`
- [x] `apply_temporal_decay()` free function with 4 decay models
- [x] `engine.rs` — `temporal_registry()`, `set_temporal_profile()` facade APIs
- [x] Default profiles: `is-a` → Stable, `has-part` → Stable, `located-at` → Ephemeral(24h)
- [x] `filter_by_time()` for batch temporal filtering
- [x] `error.rs` — `AkhError::Temporal` variant
- [x] `provenance.rs` — `DerivationKind::TemporalDecay` (tag 30)
- [x] 16 tests: all decay models, registry, validation, display

**Estimated scope**: ~300–500 lines (actual: ~430 lines)

---

## Phase 9l — Contradiction Detection

**Problem**: New triples can silently contradict existing knowledge.

**Cyc mechanism**: Cyc actively checks new assertions against existing knowledge. Contradictions within a microtheory are not allowed (each microtheory is internally consistent). Contradictions across microtheories are expected and handled by the context system.

**Design**:
- On `add_triple()`, optionally check for contradictions:
  - **Functional violation**: same subject+predicate with conflicting object where the predicate is functional (one-to-one, declared via `onto:functional`)
  - **Disjointness violation**: new triple violates a disjointness constraint (`onto:disjoint_with`) — e.g., asserting something is both a Mouse and a Moose when Mice ∩ Moose = ∅
  - **Temporal conflict**: new triple contradicts a still-valid existing triple (using temporal profiles from 9k)
  - **Intra-microtheory**: contradiction within the same context (mandatory if microtheories are active)
- Contradictions are reported, not blocked — the caller decides (add anyway, replace, abort)
- Detected contradictions create `DerivationKind::ContradictionDetected` provenance records

**Key types**:
```rust
Contradiction { existing: Triple, incoming: Triple, kind: ContradictionKind, context: Option<SymbolId> }
ContradictionKind: FunctionalViolation | DisjointnessViolation | TemporalConflict | IntraMicrotheoryConflict
```

**Changes**:
- [x] `graph/contradiction.rs` — `check_contradictions()`, `Contradiction`, `ContradictionKind` (4 kinds), `FunctionalPredicates`, `DisjointnessConstraints`, `ContradictionPredicates`, `ContradictionError`
- [x] `engine.rs` — `contradiction_predicates()`, `check_contradictions()` facade APIs
- [x] Well-known predicates: `onto:functional`, `onto:disjoint_with`, `is-a`
- [x] Integration with temporal registry for temporal conflict detection
- [x] `error.rs` — `AkhError::Contradiction` variant
- [x] `provenance.rs` — `DerivationKind::ContradictionDetected` (tag 31)
- [x] 8 tests: functional, disjointness, temporal, symmetry, clean triples

**Depends on**: 9a (microtheories for intra-context checking), 9k (temporal profiles for temporal conflicts)

**Estimated scope**: ~400–600 lines

---

## LOWER PRIORITY

---

## Phase 9m — Circumscription and Closed World Assumption

**Problem**: The engine doesn't distinguish "unknown" from "false". No closed-world reasoning. No unique-names assumption.

**Cyc mechanism**: CycL uses circumscription (the only instances of a concept are those explicitly known or derivable), the unique names assumption (different constants refer to different things), and the closed world assumption (what cannot be proved is false) — all configurable per-context.

**Design**:
- **Closed World Assumption (CWA)**: per-microtheory toggle. When active in a context, failure to find a triple is treated as negation, not unknown.
- **Unique Names Assumption (UNA)**: per-microtheory toggle. When active, different SymbolIds are assumed to refer to different real-world entities unless explicitly linked via `owl:sameAs` or equivalent.
- **Circumscription**: per-collection toggle. When active for collection C, the only instances of C are those explicitly asserted or derivable — no more.
- These are context-level settings stored as `ctx:` triples on the microtheory symbol.

**Key types**:
```rust
ContextAssumptions { cwa: bool, una: bool, circumscribed_collections: HashSet<SymbolId> }
```

**Changes**:
- [x] `compartment/cwa.rs` — `ContextAssumptions` (CWA/UNA/circumscription), `AssumptionRegistry`, `CwaPredicates` (3 well-known ctx: predicates), `NafResult` (Found/NegatedByAbsence/Unknown), `query_naf()`, `circumscribed_instances()`, `una_same_entity()`
- [x] `engine.rs` — `assumption_registry()`, `cwa_predicates()` facade APIs
- [x] `error.rs` — `AkhError::Cwa` variant
- [x] `provenance.rs` — `DerivationKind::CwaQuery` (tag 34)
- [x] 12 tests: defaults, CWA, UNA, circumscription, NAF queries (found/unknown/negated), circumscribed instances, UNA same/different entities

**Depends on**: 9a (microtheories for per-context configuration)

**Estimated scope**: ~400–600 lines (actual: ~430 lines)

---

## Phase 9n — Second-Order Quantification

**Problem**: Can't write rules that quantify over predicates. Rules are first-order only.

**Cyc mechanism**: CycL allows `(forAll ?R (implies (isa ?R TransitiveRelation) ...))` — rules about entire classes of relations. This lets you define transitivity, symmetry, reflexivity, etc. once and have them apply to all matching relations. Predicates are first-class citizens subject to quantification and predication.

**Design**:
- Extend e-graph `AkhLang` with predicate variables: `(forall-pred ?R (implies (isa ?R transitive) (forall ?a ?b ?c ...)))`
- Predicate variables bind to relation SymbolIds during inference
- Built-in second-order rules:
  - Transitivity: `∀R: transitive(R) → ∀a,b,c: R(a,b) ∧ R(b,c) → R(a,c)`
  - Symmetry: `∀R: symmetric(R) → ∀a,b: R(a,b) → R(b,a)`
  - Reflexivity: `∀R: reflexive(R) → ∀a: type(a,domain(R)) → R(a,a)`
- These generate specialized reasoners (via 9f dispatch) for each qualifying relation

**Key types**:
```rust
SecondOrderRule { predicate_var: String, constraint: SymbolId, body: AkhLangExpr }
```

**Changes**:
- [x] `reason/second_order.rs` — `SecondOrderRule`, `RelationProperty` (5 variants: Transitive, Symmetric, Reflexive, Antisymmetric, Irreflexive), `SecondOrderRegistry` (3 built-in rules), `SecondOrderPredicates`, `GeneratedInference`, `qualifying_predicates()`, `apply_transitivity()`, `apply_symmetry()`, `instantiate_all()`
- [x] `engine.rs` — `second_order_registry()`, `second_order_predicates()` facade APIs
- [x] `error.rs` — `AkhError::SecondOrder` variant
- [x] `provenance.rs` — `DerivationKind::SecondOrderInstantiation` (tag 35)
- [x] 9 tests: property display/parse, registry with builtins, qualifying predicates, transitivity (basic + no redundant), symmetry (basic + already symmetric), instantiate_all, custom rule registration

**Depends on**: 9f (reasoner dispatch for generated specialized reasoners)

**Estimated scope**: ~600–900 lines (actual: ~470 lines)

---

## Phase 9o — Non-Atomic Reified Terms (NARTs)

**Problem**: No functional term construction. Can't represent computed entities like "the government of France" as first-class terms with structural unification.

**Cyc mechanism**: Non-atomic terms (NATs) are complex terms built from functions applied to arguments: `(GovernmentFn France)`, `(FruitFn AppleTree)`. They are reified as first-class entities (NARTs), can appear as arguments to predicates, and are unified structurally. They don't have their own constant names but are referenced by their functional expression.

**Design**:
- Extend `SymbolKind::Composite` to support functional term structure: `Composite { function: SymbolId, args: Vec<SymbolId> }`
- NART identity: two NARTs are the same if they have the same function and arguments (structural equality)
- NART deduplication: on creation, check if a NART with the same structure already exists
- NARTs can appear as subject, predicate, or object in triples
- Unification: when a query contains a NART pattern, match structurally against existing NARTs

**Key types**:
```rust
// Extend existing Composite variant:
SymbolKind::Composite { function: SymbolId, args: Vec<SymbolId> }
```

**Changes**:
- [x] `graph/nart.rs` — `NartDef` (id, function, args, label), `NartRegistry` with structural deduplication, `create_nart()`, `find_structural()`, `narts_for_function()`, `narts_with_arg()`, `unify()` with wildcard patterns, `is_nart()`
- [x] `engine.rs` — `nart_registry()` facade API
- [x] `error.rs` — `AkhError::Nart` variant
- [x] `provenance.rs` — `DerivationKind::NartCreation` (tag 36)
- [x] 10 tests: create basic, deduplication, different args, empty args error, find_structural, narts_for_function, narts_with_arg, structural unification, is_nart, multi-arg NART

**Estimated scope**: ~400–600 lines (actual: ~430 lines)

---

## Dependency Graph

```
9a (Microtheories) ──→ 9d (Defeasibility) ──→ 9e (Argumentation) ──→ 9i (Arg-based Truth)
      │                      ↑                        ↑
      │                 9b (Pred Hierarchy) ──→ 9j (Arity/Types)
      │                      ↑
      ├──→ 9m (CWA/UNA) ────┘
      ├──→ 9l (Contradiction) ←── 9k (Temporal)
      │
9c (TMS) ──→ 9e (Argumentation)
              9i (Arg-based Truth)

9f (Reasoner Dispatch) ←── 9n (Second-Order Quantification)
                            9g (Rule Macros) ──→ 9h (Skolem)

9o (NARTs) — independent
9k (Temporal) — independent
```

**Critical path**: 9a → 9b → 9d → 9e → 9i (microtheories → predicates → defeasibility → argumentation → truth values)

**Independent tracks**:
- 9c (TMS) can start in parallel with 9a/9b, feeds into 9e
- 9f (dispatch) can start in parallel with 9a–9d
- 9g–9h (rule macros + skolem) can start after 9f
- 9k (temporal), 9o (NARTs) can start at any time

## Implementation Order (Recommended)

**Wave 1** (foundations, can be parallelized):
1. 9a — Microtheories
2. 9b — Predicate hierarchy
3. 9c — TMS

**Wave 2** (reasoning, depends on wave 1):
4. 9d — Defeasible reasoning (needs 9b)
5. 9f — Competitive reasoner dispatch (independent but enhances everything)

**Wave 3** (argumentation, depends on wave 2):
6. 9e — Pro/Con argumentation (needs 9c, 9d)

**Wave 4** (knowledge quality, mixed dependencies):
7. 9g — Rule macro predicates
8. 9j — Arity and type constraints (needs 9b)
9. 9k — Temporal projection
10. 9l — Contradiction detection (needs 9a, optionally 9k)

**Wave 5** (advanced, depends on earlier waves):
11. 9h — Skolem functions (needs 9g)
12. 9i — Argumentation-based truth values (needs 9e, 9c)
13. 9m — Circumscription / CWA (needs 9a)
14. 9n — Second-order quantification (needs 9f)
15. 9o — NARTs (independent)

## Total Estimated Scope

~7,000–10,500 lines across 15 sub-phases.
