# Phase 15a — Causal World Model

- **Status**: Complete
- **Date**: 2026-02-22
- **Phase**: 15a of 15a–15c

## Summary

Implemented the core causal world model as `src/agent/causal.rs` (~700 lines + ~400 lines tests). This provides action schemas with preconditions and effects, pattern matching with variable binding, state transition prediction, outcome verification, and VSA-native causal encoding.

## Components

### Types & Error Handling
- `CausalError` — miette diagnostic enum (SchemaNotFound, PreconditionNotMet, InvalidPattern, Engine)
- `CausalRelation` — 5 causal relation types (Causes, Enables, Prevents, Inhibits, Correlates)
- `CausalPredicates` — 8 well-known KG relations in `causal:` namespace

### Pattern Matching
- `PatternElement` — Concrete(SymbolId) / Variable(String) / Wildcard with `matches_triple()` and `resolve()`
- `CausalPattern` — subject/predicate/object PatternElements with negation and confidence thresholds

### Action Schemas
- `ActionSchema` — action_id, name, preconditions, effects, success_rate, execution_count
- `is_applicable()` — pattern matching against current state with variable binding propagation
- `predict_effects()` — generate StateTransition (assertions, retractions, confidence changes)
- `EffectKind` — Assert / Retract / ModifyConfidence
- `CausalEffect` — kind + pattern elements + confidence

### Manager
- `CausalManager` — schema registry with lifecycle (default/new/ensure_init/restore/persist)
- Schema CRUD: register, get, list, remove
- Prediction: `predict_action_effects()`, `verify_outcome()`
- Learning: `update_schema_from_outcome()` adjusts success_rate via EMA
- Queries: `causal_strength()` with transitive 2-hop lookup
- Bootstrap: `bootstrap_from_kg()` discovers causal triples and creates schemas
- VSA: `CausalRoleVectors` with 4 role vectors, `encode_action()` for state-action similarity

### Provenance
- `DerivationKind::CausalSchemaLearned` (tag 58) — action_name, precondition_count, effect_count

## Cross-Cutting Integration
- `src/agent/mod.rs` — module registration + re-exports
- `src/agent/error.rs` — `AgentError::Causal` transparent variant
- `src/agent/agent.rs` — `causal_manager` field with full lifecycle
- `src/agent/nlp.rs` — `UserIntent::CausalQuery` with "causal" prefix
- `src/agent/explain.rs` — prose for CausalSchemaLearned
- `src/provenance.rs` — tag 58
- `src/main.rs` — CLI Commands::Causal (5 subcommands) + headless handler
- `src/tui/mod.rs` — TUI intent handler

## Tests
22 unit tests covering:
- Relation labels roundtrip, display, unknown fallback
- Predicate namespace verification
- Pattern element matching (concrete, variable, wildcard)
- Pattern negation
- Action schema applicability (with/without preconditions)
- Effect prediction (assert, retract)
- State transition construction
- Manager lifecycle (default, register, get, serialization roundtrip)
- Outcome learning (success_rate EMA update)
- Role vector distinctness
- Effect kind serialization

## Next Steps
- Phase 15b: Event Calculus Engine (Initiates/Terminates/HoldsAt/Clipped, temporal fluent reasoning)
- Phase 15c: Counterfactual Reasoning ("what if?" queries, prediction-outcome tracking)
