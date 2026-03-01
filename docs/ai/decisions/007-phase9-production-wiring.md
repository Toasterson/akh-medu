# ADR-007: Phase 9 Production Wiring

- **Date**: 2026-02-20
- **Status**: Accepted
- **Context**: Phase 9 (15 sub-phases, ~9,750 lines, 130+ tests) was fully implemented but only 5/15 systems were actively wired into engine production paths. The other 10 existed as standalone modules with factory methods creating fresh, empty registries per call.

## Decision

### Embed vs Factory

**Embedded in Engine struct** (persistent, accumulate over engine lifetime):
- TMS, TemporalRegistry, ConstraintRegistry, FunctionalPredicates, DisjointnessConstraints, SkolemRegistry, NartRegistry, PredicateHierarchy (lazy rebuild via dirty flag)

**Kept as factory methods** (stateless or non-Serialize):
- RuleMacroRegistry — contains `dyn RuleMacro` (not Serialize); created on demand
- AssumptionRegistry — per-context, opt-in CWA/UNA; not global state
- SecondOrderRegistry — stateless rule set; built-in rules are deterministic

### Configurable Enforcement

All enforcement is enabled by default via `Phase9Config` with 8 boolean/enum flags. Users can selectively disable features for performance tuning or testing. No backward-compatibility concerns — all features active out of the box.

`ContradictionPolicy` provides three levels: `Warn` (log), `Reject` (return error), `Replace` (remove conflicting triples before insertion).

### TMS Tracking

TMS maps derived SymbolId to SupportSets (not triples). When inference derives a new triple, the derived symbol's SymbolId anchors the TMS entry. On `remove_triple`, cascade via TMS retraction finds and removes dependent triples.

### Hierarchy Caching

PredicateHierarchy uses a dirty-flag pattern (`AtomicBool`). The hierarchy is rebuilt lazily when accessed after `rel:generalizes` or `rel:inverse` triples are modified. It is NOT persisted — always rebuilt from KG on startup.

### Inference Integration

`InferPhase9Context` carries optional references to PredicateHierarchy and TemporalRegistry. The spreading activation loop expands to include specialization predicates (genlPreds) and inverse predicates (genlInverse), with temporal decay applied to edge confidences.

### Reflexivity Completion

`instantiate_all()` now accepts an optional `&ConstraintRegistry`. The reflexivity rule uses `arg1_type` (domain type) to find entities via `is-a` BFS and generates R(a,a) self-triples.

## Consequences

- All Phase 9 features are active in every engine instance by default
- 7 registries persisted via bincode to TieredStore (restore on startup with graceful fallback)
- `add_triple()` has O(1) overhead from config flag checks when enforcement is enabled
- `remove_triple()` returns `AkhResult<RetractionResult>` instead of `bool` (richer return type)
- Inference queries automatically benefit from hierarchy and temporal decay

## Deferred Items

| Item | Reason | When |
|------|--------|------|
| CWA auto-enforcement | Per-context opt-in; needs microtheory lifecycle | Phase 12 |
| NART transparent query resolution | KG-level structural match in triples_from() | When NARTs are used in practice |
| Rule macro persistence | dyn RuleMacro not Serialize; needs invocation-log approach | When macros are used beyond one-shot |
| Goal microtheory contexts | Agent goal needs context: Option<SymbolId> field | Phase 12 or later |
