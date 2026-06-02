# ADR 034 — MCTS Planning & TD Value Function (Phase 16a-b)

> Date: 2026-03-19
> Status: Accepted
> Phase: 16a-b
> Depends on: Phase 15a-c (causal world model, event calculus, counterfactual reasoning)

## Context

The agent's decision-making (Phase 8) uses single-step utility scoring — it evaluates each tool independently and picks the highest-scoring one. This works for simple tasks but cannot plan multi-step sequences like "first expand domain, then analyze prerequisites, then assess competence". Real planning requires search through action sequences.

## Decision

Implement Monte Carlo Tree Search (MCTS) with UCT selection and TD-learned value estimates, using the existing causal world model as the transition function.

### Phase 16a — State Encoding & TD Value Function
- `AgentState`: compact snapshot (active goals + progress, fluent count, optional VSA fingerprint)
- `ValueFunction`: TD(0) learning with configurable α=0.1, γ=0.99
- `Reward`: multi-factor signal (goal_progress * 1.0 + knowledge_gain * 0.3 + prediction_accuracy)
- State hashing quantizes goal progress (10 bins) and fluent count (100s) for stable lookup

### Phase 16b — MCTS Planner
- `MctsPlanner`: full MCTS loop (select → expand → rollout → backpropagate)
- UCT selection: Q/N + c·√(ln(N_parent)/N) with c=√2
- Expansion: one unexplored applicable action per iteration
- Rollout: simulate using CausalManager.predict_effects
- Backpropagation: update visit counts and cumulative values along selection path
- R-MCTS: warm-start from reflection data (successful/failed past sequences)
- Tree pruning to bound memory usage

### What we chose NOT to do
- **Neural network policy/value (AlphaZero-style)**: Would require training infrastructure and GPU. Our symbolic world model + TD values are sufficient and stay CPU-only.
- **Full VSA-indexed nearest-neighbor for value generalization**: Deferred to later — current average-prior fallback works for bootstrapping.
- **Parallel MCTS**: Single-threaded is simpler and fast enough at 100 iterations.

## Consequences

- The agent can now plan multi-step action sequences, not just pick the best single action
- TD learning improves value estimates over time through experience
- MCP tool count: 46 → **47** (mcts_plan)
- Phase 16c (OODA integration) will wire this into the actual decision cycle
- `DerivationKind::MctsPlanning` (tag 80)
- 23 new tests (12 state_value + 11 MCTS)

## Verification

- `cargo check` / `cargo check --features mcp` — clean
- 12 state_value + 11 MCTS unit tests pass
- UCT formula verified numerically in tests
