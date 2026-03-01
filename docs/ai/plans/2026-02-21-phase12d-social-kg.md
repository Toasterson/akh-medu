# Phase 12d — Social Knowledge Graph with Theory of Mind

> Date: 2026-02-21
> Status: Complete
> Depends on: Phase 12c (constraint checking)

## Objective

Add per-interlocutor social modeling with KG-backed profiles, theory-of-mind
microtheories, VSA interest vectors, and auto-registration on first interaction.

## Deliverables

### New files

1. **`src/agent/interlocutor.rs`** (~560 lines)
   - `InterlocutorError` miette diagnostic enum (NotFound, OperatorImmutable)
   - `InterlocutorPredicates` — 6 well-known KG relations
   - `InterlocutorProfile` struct with trust level, interests, knowledge microtheory
   - `InterlocutorRegistry` — HashMap-based profile store, VSA interest vectors
   - `register()` — creates KG entity + personal microtheory (ContextDomain::Belief)
   - `add_interest()` — records interest in KG, rebuilds VSA bundle
   - `set_trust_level()` — operator immutability enforced
   - `record_knowledge()` — compartment-scoped triples in personal microtheory
   - `find_similar()` — Hamming similarity search on interest vectors
   - `interest_overlap()` public function
   - 10 unit tests

### Modified files

2. **`src/agent/mod.rs`** — module declaration + re-exports
3. **`src/agent/error.rs`** — `Interlocutor` transparent variant
4. **`src/agent/agent.rs`** — `interlocutor_registry` field, initialization
   in both constructors with `init_predicates()`, accessor methods,
   `ensure_interlocutor()` convenience method, Debug impl
5. **`src/tui/mod.rs`** — auto-registration in `process_inbound_local()`
6. **`src/main.rs`** — auto-registration in headless chat path

## Verification

- `cargo build` — compiles with no new warnings
- `cargo test --lib` — 1082 tests pass (10 new)
