# Phase 15 — Causal World Model & Event Calculus

Status: **In Progress** (15a complete, 15b-15c pending)

Explicit causal model of the world: cause-and-effect predicates (causes, enables, prevents),
action schemas with preconditions and effects, event calculus engine (Initiates/Terminates/
HoldsAt/Clipped), temporal fluent reasoning and state projection, counterfactual reasoning
("what if I had done Y?"), prediction-outcome tracking with model refinement. E-graph rules
for causal transitivity and do-calculus. VSA encoding for state-action similarity lookup.
3 sub-phases (15a-15c). Foundation for all subsequent planning and epistemic reasoning.

- **Implementation plan**: `docs/ai/plans/2026-02-22-phase15-causal-world-model.md`
- **Research**: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

## Phase 15a — Causal World Model

- [x] `CausalError` miette diagnostic enum (4 variants: SchemaNotFound, PreconditionNotMet, InvalidPattern, Engine) with `CausalResult<T>`
- [x] `CausalRelation` enum: Causes, Enables, Prevents, Inhibits, Correlates — with as_label/from_label, Display, Serialize
- [x] `CausalPredicates` — 8 well-known KG relations in `causal:` namespace
- [x] `PatternElement` enum: Concrete(SymbolId), Variable(String), Wildcard — with matches_triple, resolve
- [x] `CausalPattern` — subject/predicate/object PatternElements + negated flag + confidence_threshold
- [x] `EffectKind` enum: Assert, Retract, ModifyConfidence — with Serialize/Deserialize
- [x] `CausalEffect` — kind + subject/predicate/object PatternElements + confidence
- [x] `ActionSchema` — action_id, name, preconditions (CausalPattern vec), effects (CausalEffect vec), success_rate, execution_count; `is_applicable()` with variable binding, `predict_effects()` with state transitions
- [x] `StateTransition` — assertions + retractions + confidence_changes from predicted effects
- [x] `CausalRoleVectors` — 4 deterministic VSA role vectors (action, precondition, effect, outcome)
- [x] `CausalManager` — schemas HashMap, predicates/roles lifecycle, register/get/list schemas, predict/verify outcomes, VSA encoding, causal strength queries, bootstrap from KG, provenance recording, persist/restore
- [x] `DerivationKind::CausalSchemaLearned` (tag 58) provenance variant
- [x] `UserIntent::CausalQuery` in NLP, wired into TUI + headless
- [x] CLI: `Commands::Causal` with 5 subcommands (Schemas, Schema, Predict, Applicable, Bootstrap)
- [x] `Agent.causal_manager` field with init/resume/persist lifecycle
- [x] `AgentError::Causal` transparent variant
- [x] 22 unit tests
