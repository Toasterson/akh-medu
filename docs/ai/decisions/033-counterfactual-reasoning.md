# ADR 033 — Counterfactual Reasoning & Prediction Tracking (Phase 15c)

> Date: 2026-03-19
> Status: Accepted
> Phase: 15c
> Depends on: Phase 15a (CausalManager), Phase 15b (EventCalculusEngine)

## Context

With causal action schemas (15a) and temporal fluent reasoning (15b), the agent can predict what will happen. But it cannot yet ask "what *would have* happened if I had chosen differently?" — Pearl's Level 3 (counterfactual) reasoning. Nor does it systematically track whether its predictions are accurate, which is essential for self-improving causal models.

## Decision

Implement counterfactual reasoning as a standalone module that composes CausalManager and EventCalculusEngine, plus a PredictionTracker for model refinement.

### Counterfactual Pipeline
1. **Abduct** — reconstruct state before the actual action (from current KG)
2. **Intervene** — substitute the hypothetical action's schema
3. **Predict** — compute effects via CausalManager.predict_effects
4. **Compare** — identify divergent fluents between actual and hypothetical outcomes

### Prediction Tracking
- `log_prediction()` before action execution
- `verify_prediction()` after, comparing predicted vs actual state
- Exponential moving average (alpha=0.1) for smooth accuracy signal
- Per-action accuracy breakdown for targeted schema refinement
- `refinement_suggestions()` identifies schemas below accuracy threshold

### What we chose NOT to do
- **Structural causal models (SCM)**: Full DAG-based do-calculus would be more principled but requires explicit causal graph construction. The schema-based approach is simpler and integrates naturally with the existing CausalManager.
- **Backtracking counterfactuals**: We don't unwind history — we compare predicted outcomes from the same starting state. True temporal backtracking would require an undo log.

## Consequences

- Completes Phase 15 (causal world model) — only 15d (temporal experience, lifeform extension) remains
- Enables Phase 16 (MCTS Planning) — prediction accuracy informs action selection
- The agent can now learn which action schemas are unreliable and need updating
- Total MCP tools: 44 → **46**
- `DerivationKind::CounterfactualReasoning` (tag 79)

## Verification

- `cargo check` / `cargo check --features mcp` — clean
- 8 counterfactual unit tests pass
- MCP tools: `counterfactual`, `prediction_accuracy`
