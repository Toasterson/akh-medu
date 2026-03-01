# ADR 019 — PIM as Overlay on Goals

**Date**: 2026-02-21
**Status**: Accepted
**Phase**: 13e

## Context

Phase 13e adds personal task management (GTD + Eisenhower + PARA). A key design
question is whether PIM tasks should be a new entity type alongside Goals, or an
overlay on existing Goals. The agent already has a mature Goal system with status
tracking, priority scoring, success criteria, decomposition, and OODA-loop
integration.

## Decision

### 1. PIM tasks are Goal overlays, not separate entities

`PimMetadata` annotates existing `Goal` entities. A goal becomes a PIM task when
`PimManager::add_task()` is called, which adds GTD state, Eisenhower quadrant,
PARA category, and other PIM-specific fields to the goal's KG representation via
predicates in the `pim:` namespace.

**Rationale**: Creating a separate task type would split the agent's attention
between two collections of actionable items, duplicate status tracking logic,
and require complex synchronization. The overlay approach reuses all existing
goal infrastructure (OODA integration, decomposition, reflection, persistence)
while adding PIM-specific semantics.

### 2. KG predicates in `pim:` namespace

All PIM metadata is stored as KG triples using 14 well-known relations:
`pim:gtd-state`, `pim:context`, `pim:energy`, `pim:time-estimate`, `pim:urgency`,
`pim:importance`, `pim:para-category`, `pim:deadline`, `pim:quadrant`,
`pim:blocked-by`, `pim:blocks`, `pim:recurrence`, `pim:next-due`, `pim:last-done`.

**Rationale**: Consistent with the agent's pattern of using well-known relation
predicates (cf. `agent:` namespace for goals, `email:` for email metadata). Makes
PIM data queryable via SPARQL and visible in the KG.

### 3. Petgraph for dependency DAG with custom serde

Task dependencies use petgraph `DiGraph<SymbolId, DependencyEdge>` with a custom
`SerializableDag` wrapper for persistence, since petgraph doesn't implement serde
traits. Cycle detection uses DFS before insertion.

**Rationale**: Petgraph is already a dependency (used in `decomposition.rs`), provides
well-tested graph algorithms (topological sort, DFS), and handles the DAG semantics
correctly. The custom serde wrapper is minimal (nodes + edges vectors).

### 4. Validated GTD state transitions

`GtdState::can_transition_to()` enforces valid transitions at runtime. The machine
prevents backward transitions (e.g., Done → Inbox) and invalid paths (e.g.,
Reference → Next).

**Rationale**: GTD methodology has well-defined workflow rules. Enforcing them
prevents accidental state corruption and teaches the agent proper GTD hygiene.

### 5. E-graph rewrite rules for PIM

Two e-graph rules in `reason/mod.rs`:
- `pim-unblock`: when a blocker is done, the blocked task becomes next
- `pim-deadline-chain`: earliest-start constraint propagation from blocker deadlines

**Rationale**: Enables symbolic reasoning about task dependencies, complementing
the petgraph-based DAG analysis with e-graph-based equational reasoning.

## Consequences

- Goals and PIM tasks share the same lifecycle, reducing complexity
- PIM metadata is SPARQL-queryable via KG triples
- Weekly review surfaces stale/overdue/stalled items via `gtd_weekly_review()`
- VSA priority encoding enables similarity-based task grouping
- CLI commands (`akh-medu pim inbox/next/review/matrix/deps/overdue`) provide
  direct user interaction
- Phase 13f (calendar) and 13g (preferences) can build on PIM metadata
