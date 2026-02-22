# Phase 15 — Causal World Model & Event Calculus

> Date: 2026-02-22
> Research: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

- **Status**: Planned
- **Phase**: 15 (3 sub-phases: 15a–15c)
- **Depends on**: Phases 8–13 (all complete)
- **Provenance tags**: 58–62

## Goal

Give the agent an explicit causal model of its world — the ability to represent cause-and-effect relationships, reason about what actions *would* produce, track fluent persistence over time, and predict state transitions before acting. This is the foundation for all subsequent planning and epistemic reasoning.

## Architecture Overview

```
                 ┌───────────────────────┐
                 │  15a Causal Predicates │  causes, enables, prevents, initiates, terminates
                 │  + Transition Model    │  Action schemas: preconditions → effects
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  15b Event Calculus    │  HoldsAt, Initiates, Terminates, Clipped
                 │  Engine               │  E-graph rules for temporal projection
                 │                       │  State prediction: "what if I do X?"
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  15c Counterfactual    │  Pearl Level 3: "what if I had done Y?"
                 │  Reasoning            │  Abduct-Act-Predict pipeline
                 │                       │  Prediction-outcome tracking
                 └───────────────────────┘
```

## Sub-phases

### 15a — Causal Predicates & Action Transition Model (~500 lines)

**New file**: `src/agent/causal.rs`

**Input**: The agent's existing tool registry + KG with provenance

**Output**: Causal predicate infrastructure + action schemas with predicted effects

**Types**:
```rust
/// Causal relation types between entities/events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalRelation {
    /// A causes B (sufficient condition).
    Causes,
    /// A enables B (necessary but not sufficient).
    Enables,
    /// A prevents B (if A then not B).
    Prevents,
    /// A inhibits B (weakens but doesn't fully prevent).
    Inhibits,
}

/// Well-known KG predicates for causal reasoning.
pub struct CausalPredicates {
    pub causes: SymbolId,
    pub enables: SymbolId,
    pub prevents: SymbolId,
    pub inhibits: SymbolId,
    pub has_precondition: SymbolId,
    pub has_effect: SymbolId,
    pub has_expected_outcome: SymbolId,
    pub causal_strength: SymbolId,
}

/// An action schema describing what a tool/action does to world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSchema {
    /// Symbol for the action (usually a tool name entity).
    pub action_id: SymbolId,
    /// Human-readable name.
    pub name: String,
    /// Preconditions: triples that must hold for this action to be applicable.
    pub preconditions: Vec<CausalPattern>,
    /// Effects: triples that will be added/removed after execution.
    pub effects: Vec<CausalEffect>,
    /// Observed success rate (updated after each execution).
    pub success_rate: f32,
    /// Number of times this action has been executed.
    pub execution_count: u32,
}

/// A pattern for matching precondition triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPattern {
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
    pub negated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternElement {
    /// A specific SymbolId.
    Concrete(SymbolId),
    /// A named variable bound during matching.
    Variable(String),
    /// Match anything.
    Wildcard,
}

/// An effect that an action has on the KG state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEffect {
    pub kind: EffectKind,
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EffectKind {
    /// Triple is added to the KG.
    Assert,
    /// Triple is removed from the KG.
    Retract,
    /// Confidence of existing triple is modified by delta.
    ModifyConfidence { delta: f32 },
}

/// A predicted state transition: before → action → after.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// The action that caused this transition.
    pub action_id: SymbolId,
    /// Triples added by this action.
    pub assertions: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Triples removed by this action.
    pub retractions: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Confidence modifications.
    pub confidence_changes: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// Whether the prediction was verified against actual outcome.
    pub verified: Option<bool>,
    /// Timestamp.
    pub timestamp: u64,
}

/// VSA role vectors for encoding causal state-action pairs.
pub struct CausalRoleVectors {
    pub state: HyperVec,
    pub action: HyperVec,
    pub outcome: HyperVec,
    pub precondition: HyperVec,
    pub effect: HyperVec,
    pub strength: HyperVec,
}
```

**CausalManager** methods:
- `new(engine)` — init predicates, restore schemas from store
- `register_action_schema(schema)` — add/update an action schema
- `bootstrap_schemas_from_tools(tool_registry)` — auto-generate initial schemas from tool metadata
- `applicable_actions(engine) -> Vec<ActionSchema>` — check preconditions against current state
- `predict_effects(action, engine) -> StateTransition` — apply action schema to current state
- `verify_prediction(transition, engine) -> bool` — compare predicted vs actual state after execution
- `update_schema_from_outcome(action_id, predicted, actual)` — refine schema statistics
- `encode_state_action(ops, state_triples, action) -> HyperVec` — VSA encoding for similarity lookup
- `find_similar_transitions(state_action_vec, k) -> Vec<StateTransition>` — HNSW search for analogous past transitions
- `causal_strength(cause, effect, engine) -> f32` — compute causal link strength
- `persist(engine)` / `restore(engine)` — bincode serialization

**Provenance**: `DerivationKind::CausalSchemaLearned { action_name, precondition_count, effect_count }` (tag 58)

**Tests (~15)**:
1. causal_predicates_namespace
2. action_schema_precondition_matching
3. action_schema_effect_application
4. predict_effects_assert
5. predict_effects_retract
6. verify_prediction_correct
7. verify_prediction_incorrect
8. update_schema_success_rate
9. pattern_element_variable_binding
10. applicable_actions_filters_by_preconditions
11. encode_state_action_distinct
12. causal_strength_direct
13. causal_strength_transitive
14. serialization_roundtrip
15. bootstrap_schemas_creates_entries

### 15b — Event Calculus Engine (~400 lines)

**New file**: `src/agent/event_calculus.rs`

**Input**: Causal predicates from 15a + timestamped triples from KG

**Output**: Temporal fluent reasoning — what holds when, what changed when, state projection

**Types**:
```rust
/// Event calculus predicates for temporal reasoning.
pub struct EventCalculusPredicates {
    pub initiates: SymbolId,    // event initiates fluent at time
    pub terminates: SymbolId,   // event terminates fluent at time
    pub happens: SymbolId,      // event happens at time
    pub holds_at: SymbolId,     // fluent holds at time
    pub clipped: SymbolId,      // fluent is clipped between t1 and t2
}

/// An event in the event calculus sense.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub symbol_id: SymbolId,
    pub name: String,
    pub timestamp: u64,
    pub initiates: Vec<SymbolId>,   // fluents this event starts
    pub terminates: Vec<SymbolId>,  // fluents this event ends
}

/// A fluent (time-varying property).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fluent {
    pub symbol_id: SymbolId,
    pub label: String,
    pub current_value: bool,
    pub last_initiated_by: Option<SymbolId>,
    pub last_initiated_at: Option<u64>,
}

/// Result of projecting state forward in time.
#[derive(Debug, Clone)]
pub struct StateProjection {
    /// Fluents that hold at the projected time.
    pub holding: Vec<Fluent>,
    /// Fluents that were terminated since the reference time.
    pub terminated: Vec<(Fluent, SymbolId)>,
    /// Events that occurred in the interval.
    pub events: Vec<Event>,
}

/// Result of "what-if" simulation: project state after a hypothetical sequence of actions.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// The action sequence simulated.
    pub action_sequence: Vec<SymbolId>,
    /// Predicted state at each step.
    pub state_trajectory: Vec<Vec<Fluent>>,
    /// Final predicted state.
    pub final_state: Vec<Fluent>,
    /// Confidence in the prediction (decays with trajectory length).
    pub confidence: f32,
}
```

**EventCalculusEngine** methods:
- `new(engine)` — init predicates
- `record_event(event, engine)` — store event + Initiates/Terminates triples
- `holds_at(fluent, time, engine) -> bool` — core EC axiom: initiated and not clipped
- `project_state(from_time, to_time, engine) -> StateProjection` — compute what holds at future time
- `simulate_actions(action_sequence, engine) -> SimulationResult` — chain CausalManager.predict_effects for multi-step projection
- `what_changed_since(timestamp, engine) -> Vec<(Fluent, Event)>` — temporal diff
- `fluent_history(fluent, engine) -> Vec<(u64, bool, SymbolId)>` — when initiated/terminated and by what

**E-graph rules** (added to `src/reason/mod.rs`):
```
// EC persistence: if Initiates(e,f,t1) and Happens(e,t1) and NOT Clipped(f,t1,t2) then HoldsAt(f,t2)
rewrite!("ec-persist"; "(initiates ?e ?f ?t1)" "(happens ?e ?t1)" => "(holds-at ?f ?t1)")

// EC termination: if Terminates(e,f,t) and Happens(e,t) then NOT HoldsAt(f, t+)
rewrite!("ec-terminate"; "(terminates ?e ?f ?t)" "(happens ?e ?t)" => "(terminated ?f ?t)")

// Causal transitivity
rewrite!("cause-trans"; "(causes ?a ?b)" "(causes ?b ?c)" => "(causes ?a ?c)")

// Enable + cause = cause
rewrite!("enable-cause"; "(enables ?a ?b)" "(causes ?b ?c)" => "(enables ?a ?c)")
```

**Provenance**: `DerivationKind::EventCalculusProjection { fluent_count, event_count, interval_secs }` (tag 59)

**Tests (~12)**:
1. holds_at_after_initiation
2. holds_at_terminated
3. clipped_between_events
4. project_state_simple
5. project_state_multiple_events
6. simulate_single_action
7. simulate_action_sequence
8. simulation_confidence_decay
9. what_changed_since_empty
10. what_changed_since_with_events
11. fluent_history_tracks_toggles
12. ec_egraph_rules_fire

### 15c — Counterfactual Reasoning & Prediction Tracking (~350 lines)

**Input**: CausalManager + EventCalculusEngine + existing provenance

**Output**: "What if I had done Y instead?" + systematic prediction-outcome tracking

**Types**:
```rust
/// A counterfactual query: "What would have happened if action X instead of Y?"
#[derive(Debug, Clone)]
pub struct CounterfactualQuery {
    /// The actual action that was taken.
    pub actual_action: SymbolId,
    /// The hypothetical alternative action.
    pub hypothetical_action: SymbolId,
    /// The timestamp of the original action.
    pub timestamp: u64,
}

/// Result of counterfactual analysis.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// What actually happened.
    pub actual_outcome: StateTransition,
    /// What would have happened under the alternative.
    pub hypothetical_outcome: StateTransition,
    /// Fluents that differ between actual and hypothetical.
    pub divergent_fluents: Vec<(SymbolId, bool, bool)>,
    /// Whether the hypothetical would have been better (by goal progress).
    pub hypothetical_better: Option<bool>,
    /// Confidence in the counterfactual estimate.
    pub confidence: f32,
}

/// Tracks prediction accuracy over time for model refinement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionTracker {
    /// Total predictions made.
    pub predictions_made: u64,
    /// Predictions verified as correct.
    pub predictions_correct: u64,
    /// Predictions verified as incorrect.
    pub predictions_incorrect: u64,
    /// Per-action accuracy: action_id -> (correct, total).
    pub per_action_accuracy: HashMap<u64, (u32, u32)>,
    /// Running exponential moving average of accuracy.
    pub accuracy_ema: f32,
}

/// A logged prediction for later verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub id: u64,
    pub action_id: SymbolId,
    pub predicted_transition: StateTransition,
    pub timestamp: u64,
    pub verified: bool,
    pub correct: Option<bool>,
}
```

**Methods**:
- `counterfactual_query(query, causal_mgr, ec_engine, engine) -> CounterfactualResult` — Pearl Level 3: abduct latent state, apply hypothetical action, predict
- `log_prediction(action, predicted, tracker)` — record before execution
- `verify_prediction(prediction_id, actual_state, tracker)` — compare after execution
- `prediction_accuracy(tracker) -> f32` — overall model accuracy
- `per_action_accuracy(action_id, tracker) -> f32` — per-tool prediction accuracy
- `refinement_suggestions(tracker) -> Vec<String>` — suggest which action schemas need updating

**Provenance**: `DerivationKind::CounterfactualReasoning { actual_action, hypothetical_action, divergent_count }` (tag 60)

**Tests (~8)**:
1. counterfactual_basic
2. counterfactual_no_divergence
3. prediction_tracker_accuracy
4. prediction_tracker_per_action
5. log_and_verify_correct
6. log_and_verify_incorrect
7. accuracy_ema_updates
8. refinement_suggestions_low_accuracy

## Cross-Cutting Changes

### Agent (`src/agent/agent.rs`)
- New fields: `pub(crate) causal_manager: CausalManager`, `pub(crate) ec_engine: EventCalculusEngine`, `pub(crate) prediction_tracker: PredictionTracker`
- Init in `new()`, restore in `resume()`, persist in `persist_session()`
- Accessors for all three

### Module registry (`src/agent/mod.rs`)
- `pub mod causal;`, `pub mod event_calculus;`
- Re-exports

### Error (`src/agent/error.rs`)
- `Causal(#[from] super::causal::CausalError)` variant

### Provenance (`src/provenance.rs`)
- Tags 58–60: CausalSchemaLearned, EventCalculusProjection, CounterfactualReasoning

### OODA (`src/agent/ooda.rs`)
- In `decide()`: before selecting tool, predict effects via CausalManager and log prediction
- In `act()`: after tool execution, verify prediction and update tracker
- Prediction error feeds into tool selection scoring

### Explain (`src/agent/explain.rs`)
- 3 new `derivation_kind_prose()` arms

### E-graph rules (`src/reason/mod.rs`)
- `causal_rules()` function returning 4 rewrite rules
- Registered in Engine's rule set

### NLP (`src/agent/nlp.rs`)
- `UserIntent::CausalQuery { subject, action }` — "what would happen if...", "what caused..."

### CLI (`src/main.rs`)
- `Commands::Causal { action: CausalAction }` with subcommands: Schema, Predict, Simulate, Counterfactual, Accuracy

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all existing + ~35 new tests pass
3. `cargo clippy` — no new warnings
4. Manual: `akh causal schema` lists action schemas, `akh causal predict <tool>` shows predicted effects, `akh causal simulate <tool1> <tool2>` chains predictions, `akh causal accuracy` shows prediction tracking
