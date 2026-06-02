# ADR 032 — Event Calculus Engine (Phase 15b)

> Date: 2026-03-19
> Status: Accepted
> Phase: 15b
> Depends on: Phase 15a (Causal World Model)

## Context

Phase 15a gave the agent a causal world model — action schemas with preconditions and effects, state transition prediction, and outcome verification. But it has no concept of *time*: it can't answer "does this property still hold?", "what changed since yesterday?", or "if I do actions A then B, what's the final state?".

The Event Calculus (EC) is a well-established formalism for reasoning about change over time. It models:
- **Events**: things that happen at points in time
- **Fluents**: time-varying properties that are initiated/terminated by events
- **Axioms**: holds-at (a fluent is true if initiated and not clipped), clipped (terminated between two timepoints)

## Decision

Implement the Event Calculus as a separate engine (`EventCalculusEngine`) that:

1. **Stores events and fluents as KG entities** with `ec:` namespace predicates, maintaining consistency with the existing knowledge graph architecture.
2. **Maintains in-memory history** for efficient temporal queries without full KG scans.
3. **Integrates with CausalManager** for multi-step action simulation.
4. **Persists via bincode** through the tiered store, following the same pattern as CausalManager.
5. **Adds e-graph rules** for symbolic temporal reasoning (ec-persist, ec-terminate, cause-trans, enable-cause).

### What we chose NOT to do

- **Separate temporal database**: Would add complexity; the KG + in-memory history is sufficient.
- **Full EC with continuous time**: We use discrete timestamps (u64 seconds), which matches the KG's existing timestamp model.
- **Frame axioms**: Omitted for now — the EC persistence axiom covers the common case. If needed, Phase 15c can add explicit frame rules.

## Implementation

### Core types
- `EventCalculusPredicates` — 8 well-known KG relations (`ec:initiates`, `ec:terminates`, `ec:happens`, `ec:holds-at`, `ec:clipped`, `ec:is-fluent`, `ec:is-event`, `ec:at-time`)
- `Event` — symbol_id, name, timestamp, initiates/terminates fluent lists
- `Fluent` — symbol_id, label, current_value, last initiated/terminated by/at
- `FluentHistoryEntry` — timestamp, initiated (bool), by_event
- `StateProjection` — holding fluents, terminated fluents, events in interval
- `SimulationResult` — action sequence, state trajectory, final state, confidence

### Key methods
- `record_event()` — stores event + KG triples + updates fluent state + history
- `holds_at(fluent, time)` — core EC axiom evaluation
- `project_state(from, to)` — what holds at future time
- `simulate_actions(seq, causal_mgr, engine)` — multi-step prediction with 0.9^n confidence decay
- `what_changed_since(time)` — temporal diff
- `fluent_history(fluent)` — full lifecycle

### MCP tools (6 new)
- `record_event` — create temporal events
- `holds_at` — check fluent at time
- `project_state` — project future state
- `simulate_actions` — predict action sequence outcomes
- `what_changed_since` — temporal diff
- `fluent_history` — full fluent lifecycle

### E-graph rules (4 new)
- `cause-trans` — causal transitivity
- `enable-cause` — enabling chains
- `ec-persist` — EC persistence axiom
- `ec-terminate` — EC termination axiom

### Provenance
- `DerivationKind::EventCalculusProjection { fluent_count, event_count, interval_secs }` (tag 78)

## Consequences

- Enables Phase 15c (Counterfactual Reasoning) — "what if I had done Y instead?"
- Enables Phase 15d (Temporal Experience) — felt duration, urgency, boredom
- Enables Phase 16 (MCTS Planning) — uses event calculus for action sequence simulation
- The agent can now reason about temporal state evolution and predict multi-step outcomes
- Total MCP tool count: 22 → 38 → **44** tools
- 19 new tests (13 EC engine + 4 e-graph rules + 2 existing test expansions)

## Verification

- `cargo check` / `cargo check --features mcp` — clean
- 13 event calculus unit tests pass
- 6 reason module tests pass (including 4 new EC rule tests)
- E-graph rules derive expected temporal conclusions (ec-persist, cause-trans verified)
