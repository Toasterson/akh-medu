# Phase 11 — Autonomous Task System with Self-Goal Setting

- **Date**: 2026-02-17
- **Status**: Planned
- **Depends on**: Phase 9 (microtheories, TMS, argumentation for reasoning quality), Phase 10 (code generation for implementation tasks)

## Goal

Evolve the agent from "given goals, work on them" to "observe the world, identify what needs doing, set its own goals, and execute." The agent should be able to: discover gaps in its knowledge, identify opportunities for improvement, decompose large ambitions into actionable tasks, prioritize autonomously, and track progress across sessions — all grounded in the reasoning infrastructure from Phase 9.

## Existing Building Blocks

| Component | Location | What It Does |
|-----------|----------|-------------|
| `Goal` type | `agent/goal.rs` | Hierarchical goals with priority, status, criteria, stall detection, decomposition |
| OODA loop | `agent/ooda.rs` | Observe → Orient → Decide → Act with utility-based tool selection |
| Reflection | `agent/reflect.rs` | Tool effectiveness, goal progress, meta-reasoning adjustments |
| Planning | `agent/plan.rs` | Multi-step plans with backtracking, strategy alternation |
| Episodic memory | `agent/memory.rs` | Consolidation, recall, experience-informed decisions |
| Psyche | `compartment/psyche.rs` | Personality-driven tool bias, shadow constraints, growth tracking |
| Gap analysis tool | `agent/tools/` | Identifies missing knowledge in KG |
| Provenance ledger | `provenance.rs` | Full derivation history for audit |
| Working memory | `agent/memory.rs` | Ephemeral state with relevance scoring and reference counting |

## What's Missing

1. **Goal generation** — the agent can work on goals but can't create them from observation
2. **Task decomposition intelligence** — current decomposition splits on commas; no semantic understanding
3. **Priority reasoning** — priorities are numbers, not justified by argumentation
4. **Cross-session continuity** — goals persist but there's no "agenda" or "project" abstraction
5. **World monitoring** — no triggers for "when X changes, consider Y"
6. **Self-evaluation** — reflection adjusts priorities but doesn't question the goals themselves
7. **Resource awareness** — no model of time, effort, or opportunity cost

---

## Phase 11a — Goal Generation from Observation

**Problem**: The agent can only work on goals given to it. It cannot identify what needs doing by observing its own knowledge state.

**Design**:
- **Gap-driven goals**: After `gap_analysis`, auto-generate goals to fill identified knowledge gaps
  - KG gap (missing expected triple) → "Investigate and resolve: [gap description]"
  - Schema gap (incomplete pattern) → "Complete the [entity] record"
  - Coverage gap (thin area) → "Deepen knowledge of [domain]"
- **Anomaly-driven goals**: When contradiction detection (9l) or TMS retraction (9c) fires, generate investigation goals
  - "Resolve contradiction between [A] and [B]"
  - "Verify [X] after retraction of [Y]"
- **Opportunity-driven goals**: When new knowledge arrives (ingest, inference), check if it enables previously impossible tasks
  - "Now that [X] is known, we can [Y]"
- **Reflection-driven goals**: Meta-reasoning (existing) expanded to propose new top-level goals
  - "Tool [X] has been ineffective — investigate alternative approaches"
  - "Goal [Y] has been stagnant — consider whether it's still relevant"
- Goal generation is **proposal-based**: generated goals start as `Proposed` status, require promotion to `Active` (by user confirmation or agent deliberation via argumentation from 9e)

**Key types**:
```rust
GoalProposal { description: String, rationale: String, source: ProposalSource, priority_suggestion: u8 }
ProposalSource: GapAnalysis | ContradictionDetected | OpportunityIdentified | ReflectionInsight | WorldChange
```

**Changes**:
- [ ] New file: `src/agent/goal_generation.rs`
- [ ] `src/agent/goal.rs` — add `Proposed` status variant
- [ ] `src/agent/ooda.rs` — goal generation phase after Act (optional, configurable frequency)
- [ ] `src/agent/reflect.rs` — reflection produces `GoalProposal`s, not just `Adjustment`s
- [ ] Integration with gap_analysis, contradiction detection (9l), TMS (9c)

**Estimated scope**: ~600–900 lines

---

## Phase 11b — Intelligent Task Decomposition

**Problem**: Goal decomposition splits on commas/"and" — no semantic understanding of task structure.

**Design**:
- **Semantic decomposition**: Use KG structure to identify natural sub-tasks
  - Query related entities and relations for the goal's subject
  - Identify prerequisite knowledge (what must be known before this can be done?)
  - Identify dependent capabilities (what tools/skills are needed?)
  - Identify output expectations (what should exist when done?)
- **Decomposition strategies** (selected by goal type):
  - **Research**: observe → gather → synthesize → verify
  - **Construction**: design → implement → test → document
  - **Investigation**: hypothesize → gather evidence → evaluate → conclude
  - **Maintenance**: audit → identify issues → fix → verify
- **Dependency tracking**: Sub-goals can declare `blocked_by` relationships
  - "Implement X" blocked by "Design X"
  - "Test X" blocked by "Implement X"
- **Effort estimation**: Each sub-goal gets an estimated cycle count based on similar past goals (from episodic memory)
- **Recursive decomposition**: Sub-goals that are still too large get decomposed further (depth limit: 3)

**Key types**:
```rust
DecompositionStrategy: Research | Construction | Investigation | Maintenance | Custom(String)
TaskNode { goal: Goal, children: Vec<TaskNode>, blocked_by: Vec<SymbolId>, estimated_cycles: u32 }
TaskTree { root: TaskNode, strategy: DecompositionStrategy }
```

**Changes**:
- [ ] `src/agent/goal.rs` — `blocked_by` field on Goal, `TaskTree` type
- [ ] `src/agent/plan.rs` — decomposition strategies replace comma-splitting
- [ ] `src/agent/ooda.rs` — respect `blocked_by` when selecting next goal
- [ ] `src/agent/memory.rs` — effort estimation from episodic similarity

**Estimated scope**: ~600–800 lines

---

## Phase 11c — Priority Reasoning via Argumentation

**Problem**: Priorities are arbitrary numbers. No justification for why one goal is more important than another.

**Design**:
- **Argumentation-backed priorities**: Use pro/con argumentation (9e) to justify priority assignments
  - Arguments for higher priority: urgency (blocking other goals), opportunity (time-sensitive), importance (core mission), momentum (making progress)
  - Arguments for lower priority: risk (uncertain outcome), cost (high effort), dependency (waiting on blockers), staleness (no progress)
- **Priority re-evaluation**: After each reflection cycle, re-argue priorities for all active goals
- **Comparative ranking**: Instead of absolute priority numbers, use pairwise argumentation: "Is goal A more important than goal B right now? Why?"
- **User alignment**: When the agent's priority reasoning diverges from user-expressed preferences, flag it and explain

**Changes**:
- [ ] `src/agent/goal.rs` — `priority_rationale: Vec<Argument>` field
- [ ] `src/agent/reflect.rs` — argumentation-based re-prioritization in reflection
- [ ] `src/argumentation/` (from 9e) — goal-comparison argument templates
- [ ] `src/agent/ooda.rs` — priority selection explains its reasoning

**Depends on**: 9e (argumentation framework)

**Estimated scope**: ~400–600 lines

---

## Phase 11d — Project Abstraction and Cross-Session Continuity

**Problem**: Goals are flat lists. No "project" concept to group related goals across sessions. No long-term agenda.

**Design**:
- **Project**: A named microtheory (from 9a) that groups related goals with a shared context
  - Projects have: name, description, success criteria, deadline (optional), status
  - Projects own goals: `project:contains_goal` triples
  - Projects own context: domain assumptions, relevant compartments
  - Projects track progress: percentage of sub-goals completed
- **Agenda**: Ordered list of projects with priority reasoning (from 11c)
  - Persisted across sessions
  - Agent selects active project → selects active goal within project
  - Context switching: when switching projects, load/unload relevant compartments
- **Session summary**: At session end, auto-generate a summary of what was accomplished, what's pending, and what changed
  - Stored as episodic memory
  - Next session starts by reviewing the summary

**Key types**:
```rust
Project { id: SymbolId, name: String, goals: Vec<SymbolId>, context: SymbolId, status: ProjectStatus }
Agenda { projects: Vec<(SymbolId, u8)>, active_project: Option<SymbolId> }
SessionSummary { accomplished: Vec<String>, pending: Vec<String>, changed: Vec<String>, next_steps: Vec<String> }
```

**Changes**:
- [ ] New file: `src/agent/project.rs`
- [ ] `src/agent/agent.rs` — `Agenda` field, project selection in OODA observe phase
- [ ] `src/agent/memory.rs` — `SessionSummary` generation at persist time
- [ ] `src/agent/ooda.rs` — project-scoped goal selection, context switching
- [ ] `src/compartment/` — project-as-microtheory integration (9a)

**Depends on**: 9a (microtheories for project contexts)

**Estimated scope**: ~700–1000 lines

---

## Phase 11e — World Monitoring and Reactive Goals

**Problem**: The agent only acts when told to run a cycle. No triggers for responding to changes.

**Design**:
- **Watch expressions**: Declarative rules that monitor KG state and fire when conditions are met
  - `watch("new triple matching (?, is-a, Vulnerability)")` → generate security review goal
  - `watch("confidence of [X] drops below 0.5")` → generate investigation goal
  - `watch("file modified: src/*.rs")` → generate re-ingest goal
- **Event sources**:
  - KG triple additions/removals (already have provenance hooks)
  - TMS retraction cascades (from 9c)
  - File system changes (new watcher)
  - Timer-based (periodic check)
- **Reactive goal generation**: When a watch fires, create a `GoalProposal` with `ProposalSource::WorldChange`
- **Debouncing**: Watches have a cooldown period to avoid goal spam

**Key types**:
```rust
Watch { id: SymbolId, condition: WatchCondition, cooldown: Duration, last_fired: Option<Instant> }
WatchCondition: TripleMatch { pattern: TriplePattern } | ConfidenceThreshold { symbol: SymbolId, below: f64 } | FileChanged { glob: String } | Periodic { interval: Duration }
```

**Changes**:
- [ ] New file: `src/agent/watch.rs`
- [ ] `src/engine.rs` — hook into `add_triple()` / `retract()` for watch evaluation
- [ ] `src/agent/goal_generation.rs` — watches produce `GoalProposal`s
- [ ] `src/agent/agent.rs` — watch registry, evaluation in observe phase

**Estimated scope**: ~500–700 lines

---

## Phase 11f — Self-Evaluation and Goal Questioning

**Problem**: Reflection adjusts priorities but never questions whether a goal is still worth pursuing.

**Design**:
- **Goal relevance check**: Periodically re-evaluate whether each active goal is still relevant
  - Has the context changed since the goal was created?
  - Has the goal been superseded by a completed sibling?
  - Is the goal's success criteria still achievable given current knowledge?
  - Has the user's expressed priorities shifted?
- **Sunk cost avoidance**: If a goal has consumed many cycles with no progress and decomposition has failed, explicitly recommend abandonment with reasoning
- **Goal reformulation**: Instead of just failing, propose a modified version of the goal that might be achievable
  - "I couldn't implement X in full, but I could implement a simpler version Y"
- **Competence model**: Track which types of goals the agent succeeds/fails at (from episodic memory), and factor this into goal acceptance
  - "I've successfully completed 8/10 research goals but only 2/10 implementation goals"

**Changes**:
- [ ] `src/agent/reflect.rs` — `evaluate_goal_relevance()` in reflection cycle
- [ ] `src/agent/goal.rs` — `Reformulated` status, `reformulated_from` link
- [ ] `src/agent/goal_generation.rs` — goal reformulation proposals
- [ ] `src/agent/memory.rs` — competence tracking from episodic patterns

**Estimated scope**: ~400–600 lines

---

## Phase 11g — Resource Awareness

**Problem**: The agent has no model of effort, time, or opportunity cost. It can't reason about whether a goal is worth the cycles.

**Design**:
- **Effort model**: Track cycles-per-goal-type from episodic memory, build statistical model
  - "Research goals take ~8 cycles on average, implementation goals take ~20"
- **Opportunity cost**: When selecting which goal to work on, consider what else could be done with those cycles
  - Argumentation (9e): "Working on A costs ~10 cycles; in that time I could complete B and C"
- **Diminishing returns detection**: If each cycle on a goal produces less progress than the last, signal diminishing returns
- **Budget allocation**: Configurable per-project cycle budget; agent respects limits
  - When approaching budget, escalate to user or compress remaining plan

**Changes**:
- [ ] `src/agent/goal.rs` — `estimated_effort` field, historical averages
- [ ] `src/agent/reflect.rs` — diminishing returns detection, resource reporting
- [ ] `src/agent/project.rs` — cycle budgets per project
- [ ] `src/agent/ooda.rs` — opportunity cost in goal selection scoring

**Estimated scope**: ~400–600 lines

---

## Implementation Order

```
11a (Goal Generation) ──→ 11b (Intelligent Decomposition) ──→ 11d (Projects)
                                                                    │
11c (Priority Reasoning) ──→ 11d (Projects)                       │
                                                                    │
11e (World Monitoring) ──→ feeds into 11a                          │
                                                                    │
11f (Self-Evaluation) ←── depends on 11a, 11c                     │
                                                                    │
11g (Resource Awareness) ←── depends on 11d, 11f                  │
```

**Wave 1**: 11a (goal generation) — the core capability
**Wave 2**: 11b (decomposition) + 11c (priority reasoning) — making goals smart
**Wave 3**: 11d (projects) — cross-session structure
**Wave 4**: 11e (monitoring) + 11f (self-evaluation) — autonomous awareness
**Wave 5**: 11g (resource awareness) — economic reasoning

## Total Estimated Scope

~3,600–5,200 lines across 7 sub-phases.

## Relationship to Prior Phases

Phase 11 is where **all prior phases converge**:
- **Phase 9a** (microtheories) → projects are microtheories with scoped contexts
- **Phase 9c** (TMS) → retraction cascades trigger reactive goals
- **Phase 9e** (argumentation) → priority reasoning and goal evaluation
- **Phase 9l** (contradiction detection) → anomaly-driven goal generation
- **Phase 10** (code generation) → implementation goals can be executed end-to-end
- **Phase 8 agent infrastructure** → OODA, memory, reflection, planning are the execution substrate

This phase transforms akh-medu from a tool that does what it's told into an agent that decides what to do.
