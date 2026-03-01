# Phase 16 — Predictive Multi-Step Planning (MCTS + TD Learning)

> Date: 2026-02-22
> Research: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

- **Status**: Planned
- **Phase**: 16 (3 sub-phases: 16a–16c)
- **Depends on**: Phase 15 (causal world model, event calculus, prediction tracking)
- **Provenance tags**: 61–63

## Goal

Replace the agent's single-step utility-based tool selection with genuine multi-step look-ahead planning. The agent should be able to search through sequences of actions, evaluate their predicted cumulative value, and select the best plan — not just the best next action. Plans should be refined through experience using temporal-difference value learning.

## Architecture Overview

```
                 ┌───────────────────────┐
                 │  16a State Encoding    │  KG → compact state representation
                 │  + Value Function      │  TD-learned goal-state values
                 │                       │  VSA state encoding for generalization
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  16b MCTS Planner     │  Selection (UCT) → Expansion → Rollout → Backprop
                 │                       │  World model = CausalManager + EventCalcEngine
                 │                       │  Prior policy = existing utility scoring
                 │                       │  Reflection integration (R-MCTS)
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  16c OODA Integration  │  Decide phase uses MCTS for multi-step plans
                 │  + Plan Monitoring     │  GDA expectation monitoring on predicted effects
                 │                       │  Dynamic re-planning on prediction failures
                 └───────────────────────┘
```

## Sub-phases

### 16a — State Encoding & TD Value Function (~400 lines)

**New file**: `src/agent/state_value.rs`

**Input**: Current KG state + goal state + CausalManager from Phase 15

**Output**: Compact state representation, learned value estimates, reward signal

**Types**:
```rust
/// A compact representation of agent state for planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Active goal IDs and their progress estimates.
    pub goal_states: Vec<(SymbolId, f32)>,
    /// Number of active fluents (KG triples).
    pub fluent_count: usize,
    /// Key fluent fingerprint (VSA encoding of top-K relevant triples).
    pub state_vec: Option<HyperVec>,
    /// Timestamp of this snapshot.
    pub timestamp: u64,
}

/// Role vectors for state encoding.
pub struct StateRoleVectors {
    pub goal_progress: HyperVec,
    pub fluent_active: HyperVec,
    pub resource_available: HyperVec,
    pub memory_pressure: HyperVec,
}

/// Learned value estimate for a state (TD learning).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValueFunction {
    /// Per-goal-state value estimates: (goal_id, state_hash) -> value.
    pub state_values: HashMap<u64, f32>,
    /// Learning rate alpha.
    pub alpha: f32,
    /// Discount factor gamma.
    pub gamma: f32,
    /// Total TD updates performed.
    pub update_count: u64,
}

/// Reward signal after an action.
#[derive(Debug, Clone)]
pub struct Reward {
    /// Goal progress delta (positive = progress, negative = regression).
    pub goal_progress: f32,
    /// Knowledge gained (new triples asserted).
    pub knowledge_gain: f32,
    /// Prediction accuracy bonus.
    pub prediction_accuracy: f32,
    /// Combined scalar reward.
    pub total: f32,
}
```

**Methods**:
- `encode_state(goals, engine, ops, roles) -> AgentState` — snapshot current state with VSA fingerprint
- `state_hash(state) -> u64` — deterministic hash for value table lookup
- `compute_reward(before, after, prediction_correct) -> Reward` — multi-factor reward signal
- `td_update(value_fn, state, reward, next_state)` — V(s) += alpha * [r + gamma*V(s') - V(s)]
- `state_value(value_fn, state) -> f32` — lookup or estimate from VSA similarity to known states
- `generalize_value(value_fn, state, ops) -> f32` — when exact state not seen, use VSA similarity-weighted interpolation from k-nearest known states
- `persist(value_fn, engine)` / `restore(engine) -> ValueFunction`

**Tests (~10)**:
1. encode_state_captures_goals
2. state_hash_deterministic
3. compute_reward_goal_progress
4. compute_reward_knowledge_gain
5. td_update_increases_value
6. td_update_converges
7. state_value_unknown_returns_zero
8. generalize_value_interpolates
9. value_function_serialization
10. reward_combined_score

### 16b — MCTS Planner (~600 lines)

**New file**: `src/agent/mcts.rs`

**Input**: AgentState + applicable actions (from CausalManager) + ValueFunction + world model

**Output**: Best action sequence (plan) with expected value

**Types**:
```rust
/// Configuration for MCTS planning.
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Maximum number of MCTS iterations per decision.
    pub max_iterations: usize,
    /// Maximum tree depth (action sequence length).
    pub max_depth: usize,
    /// UCT exploration constant (typically sqrt(2)).
    pub exploration_constant: f32,
    /// Maximum rollout depth for simulation.
    pub rollout_depth: usize,
    /// Minimum visits before expanding a node.
    pub expansion_threshold: u32,
}

/// A node in the MCTS search tree.
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// The agent state at this node.
    pub state: AgentState,
    /// The action that led to this state (None for root).
    pub action: Option<SymbolId>,
    /// Visit count N(s).
    pub visits: u32,
    /// Cumulative value Q(s).
    pub total_value: f32,
    /// Children: action -> node.
    pub children: Vec<MctsNode>,
    /// Whether this is a terminal state (goal reached or no actions).
    pub terminal: bool,
}

/// Result of an MCTS planning session.
#[derive(Debug, Clone)]
pub struct MctsResult {
    /// Best action sequence found.
    pub best_plan: Vec<SymbolId>,
    /// Expected value of the best plan.
    pub expected_value: f32,
    /// Total iterations performed.
    pub iterations: usize,
    /// Tree depth explored.
    pub max_depth_reached: usize,
    /// All candidate first actions with their UCT scores (for transparency).
    pub first_action_scores: Vec<(SymbolId, f32, u32)>,
    /// Whether the plan reaches a goal state in simulation.
    pub reaches_goal: bool,
}

/// Reflection data for R-MCTS: past experience informs current search.
#[derive(Debug, Clone)]
pub struct MctsReflection {
    /// Past successful action sequences for similar states.
    pub successful_sequences: Vec<Vec<SymbolId>>,
    /// Past failed action sequences for similar states.
    pub failed_sequences: Vec<Vec<SymbolId>>,
    /// Warm-start bias from episodic memory.
    pub prior_bias: HashMap<u64, f32>,
}
```

**MctsPlanner** methods:
- `new(config) -> Self`
- `plan(state, causal_mgr, ec_engine, value_fn, reflection) -> MctsResult` — full MCTS loop
- `select(node, exploration_constant) -> &mut MctsNode` — UCT selection: Q/N + c*sqrt(ln(N_parent)/N)
- `expand(node, causal_mgr, ec_engine) -> &mut MctsNode` — add child for unexplored applicable action
- `rollout(state, causal_mgr, ec_engine, value_fn, depth) -> f32` — simulate using CausalManager.predict_effects + existing utility heuristic as rollout policy
- `backpropagate(path, value)` — update Q and N along the selection path
- `best_action_sequence(root) -> Vec<SymbolId>` — extract best path by visit count
- `inject_reflection(root, reflection)` — warm-start tree from past experience (R-MCTS)
- `prune_tree(root, max_size)` — limit memory usage

**Key algorithm**:
```
for iteration in 0..max_iterations:
    node = select(root)              // UCT traversal
    if node.visits >= threshold and not node.terminal:
        child = expand(node)         // add new action child
        value = rollout(child.state) // simulate forward using causal model
    else:
        value = value_fn(node.state) // use learned value estimate
    backpropagate(path, value)       // update Q/N along path
return best_action_sequence(root)
```

**Rollout policy**: Combines existing utility scoring (Phase 8c) with CausalManager predictions. At each rollout step:
1. Get applicable actions via `causal_mgr.applicable_actions()`
2. Score each using existing utility heuristic (base_score + recency_penalty + novelty_bonus)
3. Select stochastically proportional to score (softmax)
4. Apply predicted effects to simulated state via `ec_engine.simulate_actions()`

**Provenance**: `DerivationKind::MctsPlanning { iterations, depth, expected_value }` (tag 61)

**Tests (~15)**:
1. mcts_config_default
2. mcts_node_uct_formula
3. mcts_select_unexplored_first
4. mcts_select_balances_exploration
5. mcts_expand_creates_child
6. mcts_rollout_returns_value
7. mcts_backpropagate_updates
8. mcts_best_sequence_by_visits
9. mcts_single_action_trivial
10. mcts_reflection_warm_start
11. mcts_prune_limits_size
12. mcts_result_first_action_scores
13. mcts_reaches_goal_detection
14. mcts_max_depth_respected
15. mcts_plan_deterministic_same_seed

### 16c — OODA Integration & Plan Monitoring (~300 lines)

**Changes to**: `src/agent/ooda.rs`, `src/agent/plan.rs`, `src/agent/agent.rs`

**Input**: MctsPlanner + existing OODA cycle + GDA expectations (Phase 11e)

**Output**: Multi-step planning in Decide phase, prediction-monitored execution in Act phase

**Approach**:

1. **Decide phase enhancement**: When deciding next action, invoke MCTS planner instead of (or in addition to) single-step utility scoring. The MctsConfig.max_iterations is bounded (e.g., 100) to keep decisions fast. If MCTS returns a multi-step plan, store it as the current Plan (replacing Phase 8f's heuristic plan generation).

2. **Act phase enhancement**: After executing an action:
   - Verify prediction from CausalManager (Phase 15c)
   - Compute reward and run TD update (Phase 16a)
   - If prediction was wrong (surprise > threshold), trigger re-planning via MCTS
   - Log prediction error for model refinement

3. **Plan monitoring**: Each PlanStep now has `expected_effects` (Phase 11e) populated from CausalManager predictions. GDA expectation monitoring checks whether expected effects materialized. Deviation triggers:
   - If deviation is positive (unexpected progress): continue, update model
   - If deviation is negative (expected effect didn't happen): re-plan from current state
   - If deviation is critical (contradicts preconditions of remaining steps): abort plan

4. **R-MCTS reflection**: The existing reflect() function (Phase 8f) generates MctsReflection data: past successful/failed action sequences from episodic memory for states similar to the current one. This is injected into MCTS to bias early search toward promising paths.

**Types**:
```rust
/// Plan monitoring state for active plans.
#[derive(Debug, Clone)]
pub struct PlanMonitor {
    /// Expected state after each step (predicted by causal model).
    pub expected_states: Vec<AgentState>,
    /// Actual states observed after each step.
    pub actual_states: Vec<AgentState>,
    /// Per-step prediction accuracy.
    pub step_accuracy: Vec<f32>,
    /// Whether re-planning has been triggered.
    pub replanned: bool,
    /// The MCTS result that produced this plan.
    pub mcts_result: Option<MctsResult>,
}

/// Deviation severity for plan monitoring.
#[derive(Debug, Clone, Copy)]
pub enum DeviationSeverity {
    /// Minor deviation, plan can continue.
    Minor,
    /// Moderate deviation, consider re-planning.
    Moderate,
    /// Critical deviation, remaining steps may be invalid.
    Critical,
}
```

**Provenance**: `DerivationKind::PlanMonitorDeviation { step_index, severity, replanned }` (tag 62)

**Tests (~10)**:
1. mcts_integrated_decide
2. plan_monitor_no_deviation
3. plan_monitor_minor_deviation_continues
4. plan_monitor_critical_triggers_replan
5. td_update_after_act
6. reward_from_goal_progress
7. reflection_produces_mcts_reflection
8. replan_uses_current_state
9. expected_effects_populated_from_causal
10. prediction_error_triggers_model_update

## Cross-Cutting Changes

### Agent (`src/agent/agent.rs`)
- New fields: `value_function: ValueFunction`, `mcts_config: MctsConfig`, `plan_monitor: Option<PlanMonitor>`
- New method: `plan_with_mcts(goal) -> MctsResult`

### OODA (`src/agent/ooda.rs`)
- `decide()`: invoke MCTS planner when MctsConfig is active
- `act()`: verify predictions, compute rewards, TD update, plan monitoring

### Plan (`src/agent/plan.rs`)
- `Plan` gains optional `monitor: Option<PlanMonitor>` field
- `Plan::from_mcts_result(result, causal_mgr)` constructor

### Reflect (`src/agent/reflect.rs`)
- Generate `MctsReflection` from episodic memories of similar states

### Provenance (`src/provenance.rs`)
- Tags 61–62: MctsPlanning, PlanMonitorDeviation

### NLP (`src/agent/nlp.rs`)
- `UserIntent::PlanQuery { goal }` — "plan for...", "how would you achieve..."

### CLI (`src/main.rs`)
- `Commands::Plan { action: PlanAction }` with subcommands: Search { goal, iterations }, Monitor, Value { goal }

### Explain (`src/agent/explain.rs`)
- 2 new `derivation_kind_prose()` arms

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all existing + ~35 new tests pass
3. `cargo clippy` — no new warnings
4. Manual: `akh plan search "learn Rust"` runs MCTS planning, `akh plan monitor` shows current plan status, `akh plan value` shows learned state values
