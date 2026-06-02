# Phase 15 — Causal World Model & Event Calculus

Status: **In Progress** (15a-15c complete, 15d pending)

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

## Phase 15b — Event Calculus Engine

- [x] `EventCalculusError` miette diagnostic enum (4 variants: FluentNotFound, EventNotFound, NoCausalManager, Engine) with `EventCalculusResult<T>`
- [x] `EventCalculusPredicates` — 8 well-known KG relations in `ec:` namespace (initiates, terminates, happens, holds-at, clipped, is-fluent, is-event, at-time)
- [x] `Event` struct — symbol_id, name, timestamp, initiates/terminates fluent lists
- [x] `Fluent` struct — symbol_id, label, current_value, last initiated/terminated by/at
- [x] `FluentHistoryEntry` — timestamp, initiated (bool), by_event
- [x] `StateProjection` — holding fluents, terminated fluents, events in interval
- [x] `SimulationResult` — action sequence, state trajectory, final state, confidence
- [x] `EventCalculusEngine` — events/fluents/history HashMaps, predicates lifecycle, persist/restore via bincode
- [x] `record_event()` — stores event + KG triples + updates fluent state + history
- [x] `holds_at(fluent, time)` — core EC axiom: initiated and not clipped
- [x] `is_clipped(fluent, t1, t2)` — termination between two timepoints
- [x] `project_state(from, to)` — what holds at future time
- [x] `simulate_actions(seq, causal_mgr, engine)` — multi-step prediction with 0.9^n confidence decay
- [x] `what_changed_since(time)` — temporal diff sorted by time
- [x] `fluent_history(fluent)` — full lifecycle with event attribution
- [x] `DerivationKind::EventCalculusProjection` (tag 78) provenance variant
- [x] `derivation_kind_prose` arm in `explain.rs`
- [x] `Agent.ec_engine` field with init/resume/persist lifecycle
- [x] E-graph rules in `reason/mod.rs`: ec-persist, ec-terminate, cause-trans, enable-cause
- [x] AkhLang extensions: Causes, Enables, Prevents, Initiates, Terminates, Happens, HoldsAt, Terminated
- [x] 6 MCP tools: record_event, holds_at, project_state, simulate_actions, what_changed_since, fluent_history
- [x] ADR 032: Event Calculus Engine
- [x] 13 unit tests (event_calculus) + 4 new reason tests (EC rules, causal transitivity)

## Phase 15c — Counterfactual Reasoning & Prediction Tracking

- [x] `CounterfactualError` miette diagnostic enum (3 variants: SchemaNotFound, PredictionNotFound, Engine) with `CfResult<T>`
- [x] `CounterfactualQuery` — actual_action, hypothetical_action, timestamp
- [x] `CounterfactualResult` — actual/hypothetical outcomes, divergent fluents, hypothetical_better, confidence
- [x] `PredictionTracker` — predictions_made/correct/incorrect, per-action accuracy HashMap, accuracy EMA, pending records
- [x] `PredictionRecord` — id, action_id, predicted_transition, timestamp, verified, correct
- [x] `log_prediction()` — record before execution, returns prediction ID
- [x] `verify_prediction()` — compare after execution, update EMA + per-action stats
- [x] `prediction_accuracy()` — overall accuracy (0.0–1.0)
- [x] `per_action_accuracy()` — per-action accuracy with uninformed prior
- [x] `refinement_suggestions()` — actions below threshold with min_samples
- [x] `counterfactual_query()` — Pearl Level 3: abduct → intervene → predict → compare
- [x] `record_counterfactual_provenance()` — provenance recording
- [x] `DerivationKind::CounterfactualReasoning` (tag 79) provenance variant
- [x] `derivation_kind_prose` arm in `explain.rs`
- [x] `Agent.prediction_tracker` field with init/resume/persist lifecycle
- [x] 2 MCP tools: counterfactual, prediction_accuracy
- [x] ADR 033: Counterfactual Reasoning
- [x] 8 unit tests
