# Phase 20 — Active Inference OODA Enhancement

> Date: 2026-02-22
> Research: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

- **Status**: Planned
- **Phase**: 20 (2 sub-phases: 20a–20b)
- **Depends on**: Phase 15 (causal world model), Phase 16 (MCTS + value function), Phase 17 (belief intervals)
- **Provenance tags**: 71–72

## Goal

Reframe the OODA loop as an active inference cycle that minimizes expected free energy. Every action the agent takes should be evaluated by its combined pragmatic value (reaching goals) and epistemic value (reducing uncertainty). The agent should generate predictions before observing, compute surprise (prediction error), and use this to drive both learning and action selection. This unifies exploration and exploitation into a single principled framework.

## Architecture Overview

```
                 ┌───────────────────────┐
                 │  20a Generative Model  │  Predict observations before sensing
                 │  + Prediction Error   │  Compute surprise (prediction error)
                 │                       │  Precision-weighted belief updating
                 │                       │  Free energy decomposition
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  20b Expected Free    │  EFE = pragmatic + epistemic value
                 │  Energy Policy Select │  Policy = multi-step action sequence
                 │  + OODA Rewrite       │  Replaces simple utility scoring
                 │                       │  Integrates MCTS as tree policy search
                 └───────────────────────┘
```

## Sub-phases

### 20a — Generative Model & Prediction Error (~450 lines)

**New file**: `src/agent/active_inference.rs`

**Input**: CausalManager (Phase 15) + EventCalculusEngine (Phase 15) + ValueFunction (Phase 16) + EvidenceManager (Phase 17)

**Output**: Predictive model that generates expectations, computes surprise, updates beliefs

**Types**:
```rust
/// The agent's generative model: predicts observations from beliefs + actions.
///
/// A = likelihood mapping: P(observation | hidden state)
/// B = transition mapping: P(next state | current state, action)
/// C = preference mapping: preferred observations
/// D = prior: initial state belief
///
/// In akh-medu these are NOT neural nets — they are KG-derived symbolic structures:
/// - A: triples that should be observable given current state
/// - B: CausalManager action schemas (precond → effect)
/// - C: goal states (what the agent wants to observe)
/// - D: current epistemic state (what the agent currently believes)
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    /// Predicted observations: triples expected to be true given current beliefs.
    pub expected_observations: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// State transition model: action → predicted next state (from CausalManager).
    /// Stored as a reference, not owned.
    pub transition_model_ready: bool,
    /// Preferred states: goal descriptions as triple patterns.
    pub preferred_states: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Current belief state: triples the agent believes are true.
    pub belief_state: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
}

/// Prediction error: the difference between expected and actual observations.
#[derive(Debug, Clone)]
pub struct PredictionError {
    /// Triples that were expected but not observed (false expectations).
    pub false_positives: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// Triples that were observed but not expected (surprises).
    pub false_negatives: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// Total prediction error (scalar summary).
    pub total_error: f32,
    /// Surprise (negative log probability of observations under model).
    pub surprise: f32,
    /// Precision: inverse variance of prediction errors (how reliable is the model?).
    pub precision: f32,
}

/// Free energy decomposition for a single action/policy.
#[derive(Debug, Clone)]
pub struct FreeEnergy {
    /// Pragmatic value: how much does this action move toward preferred states?
    /// Negative expected distance to goal states.
    pub pragmatic_value: f32,
    /// Epistemic value: how much uncertainty does this action resolve?
    /// Expected information gain (KL divergence between prior and posterior).
    pub epistemic_value: f32,
    /// Combined expected free energy (lower is better).
    /// EFE = -(pragmatic_value + epistemic_value)
    pub expected_free_energy: f32,
    /// Breakdown for transparency.
    pub reasoning: String,
}

/// Configuration for active inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceConfig {
    /// Weight on pragmatic value (default 0.6).
    pub pragmatic_weight: f32,
    /// Weight on epistemic value (default 0.4).
    pub epistemic_weight: f32,
    /// Precision learning rate (how fast precision adapts).
    pub precision_learning_rate: f32,
    /// Minimum surprise threshold to trigger model update.
    pub surprise_threshold: f32,
    /// Maximum policies to evaluate per decision.
    pub max_policies: usize,
}

/// Precision-weighted belief update parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionWeights {
    /// Per-source precision: how much to trust each source's observations.
    pub source_precision: HashMap<u64, f32>,
    /// Overall model precision (self-assessed).
    pub model_precision: f32,
    /// Observation precision (how reliable is sensing?).
    pub observation_precision: f32,
}
```

**ActiveInferenceEngine** methods:
- `new(config)` — init with default weights
- `generate_predictions(beliefs, engine) -> GenerativeModel` — from current KG state, predict what should be observable
- `compute_prediction_error(model, actual_observations) -> PredictionError` — compare expected vs actual
- `update_beliefs(error, precision, engine)` — precision-weighted belief revision: high-precision errors cause large updates, low-precision errors are attenuated
- `compute_surprise(error) -> f32` — scalar surprise: sum of |predicted - actual| weighted by precision
- `should_update_model(surprise, config) -> bool` — threshold check
- `update_model_precision(error, precision)` — meta-learning: if predictions are consistently wrong, lower model precision; if right, raise it

**Free energy computation**:
- `pragmatic_value(action, goals, causal_mgr, engine) -> f32` — predict state after action, measure distance to goal states
- `epistemic_value(action, uncertain_claims, causal_mgr, engine) -> f32` — predict which uncertain claims this action would resolve
- `expected_free_energy(action, goals, uncertain_claims, causal_mgr, engine) -> FreeEnergy` — combine pragmatic + epistemic

**Provenance**: `DerivationKind::ActiveInferenceCycle { surprise, pragmatic, epistemic, efe }` (tag 71)

**Tests (~14)**:
1. generative_model_creates_predictions
2. prediction_error_false_positives
3. prediction_error_false_negatives
4. prediction_error_total
5. surprise_computation
6. precision_weighted_update_high
7. precision_weighted_update_low
8. pragmatic_value_toward_goal
9. pragmatic_value_away_from_goal
10. epistemic_value_resolves_uncertainty
11. epistemic_value_no_resolution
12. efe_combines_both
13. config_default_weights
14. model_precision_adapts

### 20b — EFE Policy Selection & OODA Rewrite (~400 lines)

**Changes to**: `src/agent/ooda.rs` (major restructure), `src/agent/agent.rs`

**Input**: ActiveInferenceEngine from 20a + MCTS (Phase 16) + existing OODA

**Output**: OODA loop rewritten as active inference cycle

**New OODA structure**:

```
┌─── OBSERVE (enhanced) ────────────────────────────────────────┐
│ 1. Generate predictions from generative model                  │
│ 2. Receive actual observations (sensor data, messages, KG)     │
│ 3. Compute prediction error (surprise)                         │
│ 4. If surprise > threshold: flag for model update              │
│ 5. Precision-weighted belief update                            │
└───────────────────────────────────────────────────────────────┘
           │
┌─── ORIENT (enhanced) ─────────────────────────────────────────┐
│ 1. Update causal model from prediction errors                  │
│ 2. Update epistemic states (who knows what changed)            │
│ 3. Identify most uncertain claims (highest m_ignorance)        │
│ 4. Build/update ToM models for active interlocutors            │
│ 5. Context assessment (same as before + new enrichments)       │
└───────────────────────────────────────────────────────────────┘
           │
┌─── DECIDE (rewritten) ────────────────────────────────────────┐
│ 1. Enumerate candidate policies (action sequences, len 1-3)    │
│ 2. For each policy, compute EFE:                               │
│    a. Pragmatic value via causal model state prediction         │
│    b. Epistemic value via uncertainty resolution prediction     │
│ 3. If policies > max_policies: use MCTS to prune search        │
│ 4. Select policy with lowest EFE (via softmax temperature)     │
│ 5. Store selected policy as current Plan                       │
│ 6. Log EFE breakdown for transparency                          │
└───────────────────────────────────────────────────────────────┘
           │
┌─── ACT (enhanced) ────────────────────────────────────────────┐
│ 1. Execute first action of selected policy                     │
│ 2. Record prediction (expected outcome)                        │
│ 3. Compute reward from outcome                                 │
│ 4. TD update on value function                                 │
│ 5. Verify prediction → update prediction tracker               │
│ 6. Check plan monitor for deviations                           │
│ 7. If critical deviation: re-plan via MCTS                     │
└───────────────────────────────────────────────────────────────┘
```

**Types**:
```rust
/// A policy: a sequence of actions to evaluate.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Action sequence (1-3 steps).
    pub actions: Vec<SymbolId>,
    /// Expected free energy.
    pub efe: FreeEnergy,
    /// Probability of selection (softmax over all policies).
    pub selection_probability: f32,
}

/// Result of the enhanced Decide phase.
#[derive(Debug, Clone)]
pub struct DecideResult {
    /// Selected policy.
    pub selected_policy: Policy,
    /// All evaluated policies with their EFE (for transparency).
    pub all_policies: Vec<Policy>,
    /// Whether MCTS was used for pruning.
    pub mcts_used: bool,
    /// Total policies evaluated.
    pub policies_evaluated: usize,
    /// Reasoning summary.
    pub reasoning: String,
}

/// Enhanced Observation result.
#[derive(Debug, Clone)]
pub struct ObservationResult {
    /// Raw observations (same as before).
    pub content: String,
    /// Prediction error from active inference.
    pub prediction_error: Option<PredictionError>,
    /// Whether the model needs updating.
    pub model_update_needed: bool,
    /// Recalled episodes (same as before).
    pub recalled_episodes: Vec<String>,
    /// JITIR suggestions (Phase 13g).
    pub jitir_summary: String,
}
```

**Policy enumeration**: For small action spaces (< max_policies), enumerate all length-1 to length-3 policies from applicable actions. For large action spaces, use MCTS (Phase 16b) to search the policy tree and return the top-K policies for EFE evaluation.

**Softmax selection**: Instead of argmin(EFE), use softmax with temperature:
```
P(policy_i) = exp(-EFE_i / temperature) / sum_j(exp(-EFE_j / temperature))
```
Low temperature → greedy (exploit). High temperature → exploratory. Temperature adapts:
- Low model precision → high temperature (explore when uncertain)
- High model precision → low temperature (exploit when confident)

**Backward compatibility**: The existing utility-based tool scoring is preserved as a fast heuristic fallback. When `ActiveInferenceConfig` is not active (e.g., first few cycles with no causal model), the agent falls back to the Phase 8c utility scoring. The transition is gradual: as the causal model learns, active inference increasingly dominates.

**Provenance**: `DerivationKind::PolicySelection { efe, pragmatic, epistemic, policy_length, mcts_used }` (tag 72)

**Tests (~12)**:
1. policy_enumeration_single_step
2. policy_enumeration_multi_step
3. softmax_selection_temperature
4. softmax_low_temp_exploits
5. softmax_high_temp_explores
6. decide_result_transparency
7. observation_with_prediction_error
8. model_update_triggered_by_surprise
9. fallback_to_utility_scoring
10. gradual_transition_to_efe
11. efe_prefers_goal_progress
12. efe_explores_when_uncertain

### 20c — Proprioception (Lifeform Engine Extension) (~400 lines)

> Added: 2026-02-24 (Lifeform Engine — self-awareness of resource state as felt signal)

Awareness of the agent's own resource state as a *felt* signal, not just a metric.
"I'm running low on memory" should change reasoning behavior automatically, like
hunger changes human behavior.

**Key types**:
```rust
struct ProprioceptiveState {
    memory_pressure: f32,      // 0.0 (plenty) to 1.0 (critical)
    kg_density: f32,           // How "full" the knowledge graph feels
    reasoning_load: f32,       // Current computational effort
    uptime_fatigue: f32,       // Increases with continuous operation
    storage_remaining: f32,    // Disk space awareness
}
```

**Effects on behavior**:
- High memory pressure → more aggressive consolidation, pause ingestion
- High reasoning load → defer non-urgent goals, simplify strategies
- High uptime fatigue → trigger consolidation cycle (like sleep drive)
- Low storage → stop ingesting new documents, warn operator

**Integration with Phase 23 (Affective System)**:
- Resource scarcity generates negative valence (stress)
- Recovering from pressure generates positive valence (relief)
- Proprioceptive signals contribute to somatic markers on resource-intensive operations

**Replaces**: The existing `resource.rs` (resource awareness, VOC) provides the raw
metrics. This extension adds the *felt* dimension — converting metrics into affective
signals that automatically modulate behavior without explicit threshold checks.

**Dependencies**: Phase 11g (resource awareness), Phase 23 (affective system)

## Cross-Cutting Changes

### Agent (`src/agent/agent.rs`)
- New field: `active_inference: ActiveInferenceEngine`
- `ActiveInferenceConfig` in agent config
- Existing `run_cycle()` calls the new OODA functions

### OODA (`src/agent/ooda.rs`)
- Major restructure: observe/orient/decide/act all gain active inference enrichment
- Backward-compatible: old behavior preserved when AI config is disabled

### Provenance (`src/provenance.rs`)
- Tags 71–72

### Reflect (`src/agent/reflect.rs`)
- Reflection gains `prediction_accuracy: f32` and `model_precision: f32` metrics
- Suggestion to increase exploration when model precision is low

### NLP (`src/agent/nlp.rs`)
- `UserIntent::InferenceQuery` — "why did you choose that action?", "what did you expect?"

### CLI (`src/main.rs`)
- `Commands::Inference { action: InferenceAction }` with subcommands: Status, Surprise, Precision, Policies

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all existing + ~26 new tests pass
3. `cargo clippy` — no new warnings
4. Manual: `akh inference status` shows current model precision/surprise, `akh inference policies` shows EFE-ranked policies
