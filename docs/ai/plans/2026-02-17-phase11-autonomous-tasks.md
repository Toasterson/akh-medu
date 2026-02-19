# Phase 11 — Autonomous Task System with Self-Goal Setting

- **Date**: 2026-02-17
- **Updated**: 2026-02-19 (Phase 11b complete)
- **Status**: In Progress (11a–11b complete)
- **Depends on**: Phase 9 (microtheories, TMS, argumentation for reasoning quality), Phase 10 (code generation for implementation tasks)

## Goal

Evolve the agent from "given goals, work on them" to "observe the world, identify what needs doing, set its own goals, and execute." The agent should be able to: discover gaps in its knowledge, identify opportunities for improvement, decompose large ambitions into actionable tasks, prioritize autonomously, and track progress across sessions — all grounded in the reasoning infrastructure from Phase 9 and informed by cognitive architecture research (Soar, ACT-R, CLARION, BDI, GDA).

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
| Triggers | `agent/trigger.rs` | ECA-style reactive rules (Interval, GoalStalled, MemoryPressure, NewTriples) |
| Provenance ledger | `provenance.rs` | Full derivation history for audit |
| Working memory | `agent/memory.rs` | Ephemeral state with relevance scoring and reference counting |
| Schema discovery | `autonomous/schema.rs` | Discovers type clusters from KG structure |
| Tool history | `GoalToolHistory` in OODA | Per-goal (tool, count, recency) tracking from WM Decision entries |

## What's Missing

1. **Goal generation** — the agent can work on goals but can't create them from observation
2. **Task decomposition intelligence** — current decomposition splits on commas; no semantic understanding
3. **Priority reasoning** — priorities are numbers, not justified by argumentation
4. **Cross-session continuity** — goals persist but there's no "agenda" or "project" abstraction
5. **World monitoring** — triggers exist but don't monitor KG state changes or expectations
6. **Self-evaluation** — reflection adjusts priorities but doesn't question the goals themselves
7. **Resource awareness** — no model of effort, time, or opportunity cost
8. **Procedural learning** — no way to compile successful goal strategies into reusable methods

---

## Phase 11a — Goal Generation from Observation

**Problem**: The agent can only work on goals given to it. It cannot identify what needs doing by observing its own knowledge state.

**Research basis**: Schmidhuber compression progress, IMGEP learning-progress goal sampling, Soar impasse-driven subgoaling, CLARION drive system, GDA discrepancy detection, BDI event-triggered goals.

**Design**:

### Drive System (CLARION-inspired)
Four measurable drives that generate `GoalProposal`s when their strength exceeds a configurable threshold:
- **Curiosity drive**: `strength = 1.0 - learning_progress_rate`. When KG growth stagnates, generate exploration goals. Track triple-addition rate per type cluster (from `schema::discover_schema`).
- **Coherence drive**: `strength = contradiction_count / total_triples`. When contradictions accumulate, generate investigation goals.
- **Completeness drive**: `strength = 1.0 - coverage_score` from `gap_analysis`. When gaps are detected, generate completion goals.
- **Efficiency drive**: `strength = tool_failure_rate`. When tools fail frequently, generate strategy-improvement goals (trigger reflection).

### Signal Sources
- **Gap-driven**: After `gap_analysis`, auto-convert each `KnowledgeGap` to a `GoalProposal`:
  - `GapKind::DeadEnd` → "Discover relationships for {entity}" with criteria "entity has >= N connections"
  - `GapKind::MissingPredicate` → "Find {predicate} for {entity}" with criteria "triple exists"
  - `GapKind::IncompleteType` → "Complete {entity} as {type}" with criteria "entity has all typical predicates"
  - Priority from `gap.severity` (scaled 0.0–1.0 to 0–255)
- **Anomaly-driven**: When contradiction detection (9l) or TMS retraction (9c) fires:
  - "Resolve contradiction between [A] and [B]"
  - "Verify [X] after retraction of [Y]"
- **Opportunity-driven**: When new knowledge arrives, check dormant goals (previously failed):
  - Parse failure reason, check if missing prerequisite now exists in KG
  - If resolved, reactivate with boosted priority
- **Impasse-driven** (Soar-inspired): When `decide()` produces a tie or all tool scores are below threshold:
  - Generate meta-goal: "Resolve decision impasse for goal {X}"
  - Process meta-goal through normal OODA (recursive metacognition)
- **Reflection-driven**: Meta-reasoning expanded to produce `GoalProposal`s (not just `Adjustment`s)
- **SPARQL-based**: Periodic queries for missing expected properties (via `FILTER NOT EXISTS`)

### Three-Phase Pipeline (GDA-informed)
1. **Signal Collection** (every N cycles or on trigger): Drives + gap analysis + anomaly checks + opportunity checks all produce raw `GoalProposal`s
2. **Deliberation** (filters and ranks):
   - Deduplicate via VSA Hamming distance (< 2000 from existing goal → duplicate)
   - Conflict check via e-graph (contradictory criteria in the same e-class → conflict)
   - Feasibility check (at least one tool's keyword match fires against goal description)
   - Priority = `drive_strength × gap_severity × novelty`
3. **Activation**: Top N proposals (configurable, default 3) become Goal entities. Remainder stored as dormant for opportunity detection. Provenance: `DerivationKind::AutonomousGoalGeneration`

**Key types**:
```rust
GoalProposal { description: String, rationale: String, source: GoalSource,
               priority_suggestion: u8, conflicts_with: Vec<SymbolId>, feasibility: f32 }

GoalSource: GapDetection { gap: KnowledgeGap } | ContradictionDetected { triples: (Triple, Triple) }
          | OpportunityDetected { newly_satisfied: Triple } | DriveExceeded { drive: DriveKind, strength: f32 }
          | ImpasseDetected { impasse: DecisionImpasse } | ReflectionInsight | WorldChange

DriveKind: Curiosity | Coherence | Completeness | Efficiency

DriveSystem { drives: [Drive; 4], thresholds: [f32; 4] }
Drive { kind: DriveKind, strength: f32, last_computed: u64 }
```

**Changes**:
- [x] New file: `src/agent/goal_generation.rs` — GoalProposal, GoalSource, three-phase pipeline, 12 unit tests
- [x] New file: `src/agent/drives.rs` — DriveSystem, Drive, DriveKind, strength computation, 8 unit tests
- [x] `src/agent/goal.rs` — add `Proposed` and `Dormant` status variants, `GoalSource` enum, `source` field on Goal
- [x] `src/agent/ooda.rs` — `DecisionImpasse`/`ImpasseKind` types, `detect_impasse()` function, integrated into `select_tool()`, 4 unit tests
- [x] `src/agent/reflect.rs` — `goal_proposals` field on `ReflectionResult`, `generate_reflection_proposals()` for stagnant goals/ineffective tools/high memory pressure, 3 unit tests
- [x] `src/agent/agent.rs` — `GoalGenerationConfig` in `AgentConfig`, `DriveSystem`/`last_impasse` fields, `generate_goals()` method, integrated into `run_cycle()`, drive persistence in `persist_session()`/`resume()`
- [x] `src/provenance.rs` — `AutonomousGoalGeneration { drive, strength }` variant (tag 40)
- [x] `src/main.rs` — `format_derivation_kind()` arm for `AutonomousGoalGeneration`
- [x] Integration with gap_analysis, contradiction detection (9l)

**Actual scope**: ~800 lines, 27 new tests (total: 856)

---

## Phase 11b — Intelligent Task Decomposition

**Problem**: Goal decomposition splits on commas/"and" — no semantic understanding of task structure.

**Research basis**: HTN planning (SHOP2, PANDA), HTN-MAKER method learning, Goal-Task Networks (GTN), resource-rational decomposition, Koopman's decomposition taxonomy.

**Design**:

### HTN Method Registry
Store decomposition methods as first-class KG entities with:
- **Preconditions**: SPARQL queries that must return results for the method to apply
- **Subtask list**: Ordered sequence of sub-goals or concrete plan steps
- **Ordering constraints**: Which subtasks must precede which (partial order)
- **Strategy type**: Research | Construction | Investigation | Maintenance | Custom

Method triples in KG:
```
method:investigate  has_precondition  "ASK { ?goal agent:involves_unknown ?x }"
method:investigate  has_step_1        "hypothesize"
method:investigate  has_step_2        "gather-evidence"
method:investigate  has_step_3        "evaluate"
method:investigate  has_step_4        "conclude"
method:investigate  has_ordering      "step_1 < step_2 < step_3 < step_4"
```

### VSA-Based Method Selection
Encode each method template as a hypervector (bundle of concept vectors from its description and preconditions). Compare the goal vector against each method vector via Hamming distance. Select the method with highest similarity — upgrades from keyword/threshold approach to distributed semantic matching.

### Goal-Task Network (GTN) Unification
A single decomposition can produce a mix of sub-goals (recursive) and concrete plan steps (terminal):
- Sub-goals stay in `goal.rs` and get their own decomposition
- Plan steps stay in `plan.rs` and execute directly
- The `TaskTree` type unifies both

### Dependency DAG
- Upgrade from flat `Vec<PlanStep>` to petgraph `DiGraph<TaskNode, DependencyKind>`
- **Topological sort** (Kahn's algorithm via petgraph) produces valid execution order
- **Critical path**: SPARQL query for the longest dependency chain — steps on it get priority
- **VSA dependency inference**: Encode each step's preconditions and effects as hypervectors. If step B's precondition vector has high similarity to step A's effect vector, infer a likely `blocked_by` dependency automatically

### Resource-Rational Decomposition
When multiple decomposition strategies apply, prefer the one that reduces planning complexity the most (cognitive science finding). Measure by: fewer subtasks, shallower depth, more concrete steps vs abstract sub-goals.

**Key types**:
```rust
DecompositionMethod { id: SymbolId, precondition_sparql: String, subtasks: Vec<TaskNode>,
                      ordering: Vec<(usize, usize)>, strategy: DecompositionStrategy }
DecompositionStrategy: Research | Construction | Investigation | Maintenance | Custom(String)
TaskNode { kind: TaskNodeKind, blocked_by: Vec<SymbolId>, estimated_cycles: u32 }
TaskNodeKind: SubGoal(Goal) | PlanStep(PlanStep)
TaskTree { root: TaskNode, children: DiGraph<TaskNode, DependencyKind>, strategy: DecompositionStrategy }
DependencyKind: FinishToStart | StartToStart | FinishToFinish
```

**Changes**:
- [x] New file: `src/agent/decomposition.rs` — DecompositionStrategy, MethodRegistry (6 built-in methods), TaskTree DAG (petgraph), decompose_goal_htn orchestrator, VSA+keyword method selection, 12 unit tests
- [x] `src/agent/goal.rs` — `blocked_by: Vec<SymbolId>` field on Goal, `is_blocked()` method, KG roundtrip via `agent:blocked_by` triples, 3 unit tests
- [x] `src/agent/agent.rs` — `blocked_by` predicate in AgentPredicates, `method_registry_htn` field, HTN-based `decompose_stalled_goal()`, method stats persistence
- [x] `src/agent/ooda.rs` — `decide()` skips blocked goals
- [x] `src/agent/plan.rs` — `Plan::from_task_tree()` flattens TaskTree into linear Plan
- [x] `src/provenance.rs` — `HtnDecomposition { method_name, strategy, subtask_count }` variant (tag 41)
- [x] `src/main.rs` — `format_derivation_kind()` arm for HtnDecomposition
- [ ] `src/agent/memory.rs` — effort estimation from episodic case retrieval (see 11g)

**Actual scope**: ~600 lines new code, ~18 new tests (total: ~874)

---

## Phase 11c — Priority Reasoning via Argumentation

**Problem**: Priorities are arbitrary numbers. No justification for why one goal is more important than another.

**Research basis**: Dung's abstract argumentation, Bench-Capon's Value-Based Argumentation Frameworks (VAF), Ferretti et al. argumentation-based goal reasoning, ACT-R utility formula.

**Design**:

### Value-Based Argumentation
Each argument promotes a **value** (first-class KG entity):
- `value:timeliness` — goal has urgency or blocks other goals
- `value:thoroughness` — goal improves knowledge quality or covers gaps
- `value:efficiency` — goal has low effort relative to impact
- `value:accuracy` — goal resolves contradictions or improves confidence

Value orderings define **audiences** (operational modes):
- Exploration mode: thoroughness > accuracy > efficiency > timeliness
- Deadline mode: timeliness > efficiency > accuracy > thoroughness
- Quality mode: accuracy > thoroughness > timeliness > efficiency

An attack succeeds only if the attacked argument's value is not preferred to the attacker's value in the current audience.

### Argument Structure
Arguments as KG triples:
```
arg:goal-A-urgent    attacks      arg:goal-B-important
arg:goal-A-urgent    promotes     value:timeliness
arg:goal-A-urgent    supports     arg:goal-A-has-deadline
```

### Extension Computation
Compute the **preferred extension** (maximal admissible set) under the current value ordering. Goals whose pro-arguments survive in the extension are justified. Goals whose pro-arguments are all defeated are deprioritized.

### Practical Implementation Path
Start with weighted pro/con: each goal gets `Vec<Argument>`. Priority = `Σ(pro_weight) - Σ(con_weight)`, where weights derive from the current value ordering. This is already more principled than `u8` and can evolve toward full extension computation. VSA similarity bootstraps arguments for new goals from past similar goals. E-graph discovers argument equivalences.

### ACT-R Utility Integration
Extend tool scoring with: `U(tool) = P(success) × G(goal_value) - C(estimated_cost)`. The competence model (11f) provides `P(success)`. The effort model (11g) provides `C`.

**Key types**:
```rust
Argument { id: SymbolId, conclusion: ArgumentConclusion, promotes: Value,
           attacks: Vec<SymbolId>, supports: Vec<SymbolId> }
ArgumentConclusion: PrioritizeGoal(SymbolId) | DeprioritizeGoal(SymbolId) | PreferTool(String)
Value { id: SymbolId, name: String }
ValueOrdering { name: String, ordering: Vec<SymbolId> }  // most preferred first
```

**Changes**:
- [ ] `src/agent/goal.rs` — `priority_rationale: Vec<Argument>` field, `Value` type
- [ ] `src/agent/reflect.rs` — argumentation-based re-prioritization in reflection
- [ ] `src/argumentation/` (from 9e) — goal-comparison argument templates, VAF extension computation
- [ ] `src/agent/ooda.rs` — priority selection explains its reasoning, ACT-R utility in tool scoring

**Depends on**: 9e (argumentation framework)

**Estimated scope**: ~500–700 lines

---

## Phase 11d — Project Abstraction and Cross-Session Continuity

**Problem**: Goals are flat lists. No "project" concept to group related goals across sessions. No long-term agenda.

**Research basis**: Cyc microtheories, Soar's 4-memory architecture, ACT-R activation-based retrieval, HGN (Hierarchical Goal Networks).

**Design**:

### Projects as Microtheories
- A **Project** is a named microtheory (from 9a) that groups related goals with a shared context
- Projects have: name, description, success criteria, deadline (optional), status
- Projects own goals: `project:contains_goal` triples
- Projects own context: domain scope (list of relevant concepts), assumptions, compartments
- Projects track progress: percentage of sub-goals completed

### VSA-Based Project Scoping
Bundle all concept symbols in a project's scope into a single project vector. When a new goal is created, compare its semantic vector against each project's vector. Automatically assign the goal to the project with highest similarity. If no project exceeds a threshold (Hamming distance > 4000 from all projects), create a new project.

### E-Graph for Project Relationships
Use rewrite rules to discover significant scope overlap: if two projects share > 60% of scope concepts (measured by VSA similarity of bundled scope vectors), propose a merge or parent-project creation.

### Soar/ACT-R Memory Integration
- **ACT-R activation-based retrieval**: Replace static `WorkingMemoryEntry.relevance` with `activation = ln(Σ t_i^{-d})` where `t_i` are times since each access and `d ≈ 0.5`. The existing `reference_count` tracks frequency; add timestamps of each reference. This naturally implements forgetting and recency bias.
- **Session summary**: At session end, consolidate WM into a `SessionSummary` stored as episodic memory. On resume, load summary into WM as context. This gives the agent a "what was I doing?" sense.

### Agenda
Ordered list of projects with priority reasoning (from 11c):
- Persisted across sessions via durable store
- Agent selects active project → selects active goal within project
- Context switching: when switching projects, load/unload relevant compartments
- Contextual reasoning: restrict KG queries to the active project's scope concepts

**Key types**:
```rust
Project { id: SymbolId, name: String, goals: Vec<SymbolId>, scope_vector: HyperVec,
          context: SymbolId, status: ProjectStatus, cycle_budget: Option<u32> }
Agenda { projects: Vec<(SymbolId, u8)>, active_project: Option<SymbolId> }
SessionSummary { accomplished: Vec<String>, pending: Vec<String>,
                 changed: Vec<String>, next_steps: Vec<String> }
```

**Changes**:
- [ ] New file: `src/agent/project.rs` — Project, Agenda, VSA scoping, project auto-assignment
- [ ] `src/agent/agent.rs` — Agenda field, project selection in OODA observe phase
- [ ] `src/agent/memory.rs` — SessionSummary generation at persist time, ACT-R activation formula
- [ ] `src/agent/ooda.rs` — project-scoped goal selection, context switching
- [ ] `src/compartment/` — project-as-microtheory integration (9a)

**Depends on**: 9a (microtheories for project contexts)

**Estimated scope**: ~800–1100 lines

---

## Phase 11e — World Monitoring and Reactive Goals

**Problem**: The agent only acts when told to run a cycle. Triggers exist but don't monitor KG state changes or expectations.

**Research basis**: GDA expectation monitoring, BDI event-triggered goals, ECA rules, situation calculus / fluent tracking.

**Design**:

### GDA Expectation Monitoring
After each Act phase, compare predicted KG state change (from plan step) against actual change:
- Plan step predicts: "this tool should add triples about X"
- After execution: compute actual triple delta
- Discrepancy = `missing_expected ∪ unexpected_present`
- Each discrepancy generates an explanation → a `GoalProposal` with `source: WorldChange`

### Pattern-Based KG Watches
Extend `TriggerCondition` with content-matching conditions:
```rust
TriggerCondition::TriplePattern {
    subject_pattern: Option<String>,   // e.g., "concept:*"
    predicate: Option<String>,         // e.g., "has_status"
    object_pattern: Option<String>,    // e.g., "status:failed*"
}
TriggerCondition::ConfidenceThreshold {
    symbol: SymbolId,
    below: f64,
}
```
When a triple matching the pattern appears in the KG, fire the trigger.

### VSA Semantic Trigger Matching
Encode trigger conditions as hypervectors. When new triples are added, encode them and compare against registered trigger vectors via HNSW. If similarity exceeds threshold, fire. This generalizes exact pattern matching to semantic matching.

### Fluent-Style State Tracking
Maintain a "world state snapshot" as a set of active fluents (KG triple count + key metrics at cycle start). After each action, compute the delta (added/removed triples). Plan preconditions are evaluated against the current fluent state. This enables GDA-style "what changed?" analysis.

### BDI Event-Triggered Goals
Generalize triggers to pattern-match on triple content:
- When `engine.add_triple()` fires, emit a "belief added" event
- Plans stored as KG triples: `(trigger_pattern, guard_condition, goal_to_generate)`
- Matching plans whose guards are satisfied generate new goals

### Debouncing
All watches have a cooldown period to avoid goal spam. Cooldown is configurable per-watch.

**Key types**:
```rust
Watch { id: SymbolId, condition: WatchCondition, cooldown: Duration, last_fired: Option<Instant> }
WatchCondition: TripleMatch { pattern: TriplePattern } | ConfidenceThreshold { symbol: SymbolId, below: f64 }
              | FileChanged { glob: String } | Periodic { interval: Duration }
Expectation { expected_triples_added: Vec<TriplePattern>, expected_triples_present: Vec<TriplePattern> }
Discrepancy { missing_expected: Vec<TriplePattern>, unexpected_present: Vec<Triple>, explanation: String }
```

**Changes**:
- [ ] New file: `src/agent/watch.rs` — Watch, WatchCondition, evaluation
- [ ] `src/agent/trigger.rs` — extend TriggerCondition with TriplePattern and ConfidenceThreshold
- [ ] `src/engine.rs` — hook into `add_triple()` / `retract()` for watch evaluation
- [ ] `src/agent/goal_generation.rs` — watches produce `GoalProposal`s
- [ ] `src/agent/ooda.rs` — expectation tracking after Act, fluent state snapshots
- [ ] `src/agent/agent.rs` — watch registry, evaluation in observe phase

**Estimated scope**: ~600–800 lines

---

## Phase 11f — Self-Evaluation and Goal Questioning

**Problem**: Reflection adjusts priorities but never questions whether a goal is still worth pursuing.

**Research basis**: Cox's MCL metacognitive loop, Nelson-Narens monitoring/control model, AGM belief revision, autoepistemic reasoning, oversubscription planning, Zone of Proximal Development.

**Design**:

### Metacognitive Monitoring/Control Layer
Nelson-Narens dual-process: **monitoring** produces judgments, **control** acts on them.

**Monitoring functions** (added to reflection cycle):
- **Progress rate**: Sliding window (last 5 cycles) tracking improvement rate per goal. When rate drops below threshold, signal diminishing returns.
- **Competence assessment**: Per-tool and per-task-category success rates (extend existing `ToolInsight`). Store in KG as `(tool:X, agent:success_rate, "0.73")`.
- **Failure pattern matching**: Encode failure contexts as VSA hypervectors. Search HNSW for similar past failures with known resolutions (Cox's IMXP pattern).
- **Confidence calibration**: Compare agent's predicted success probability against actual outcomes. Track calibration error over time.

**Control functions** (new `Adjustment` variants):
- `Adjustment::ReformulateGoal { goal, relaxed_criteria }` — when full achievement fails, propose simpler version via e-graph rewrite rules
- `Adjustment::SuspendGoal { goal, reason }` — sunk cost avoidance: if diminishing returns persist after decomposition
- `Adjustment::ReviseBeliefs { retract, assert }` — AGM-style belief revision when contradictions affect goals

### Zone of Proximal Development (ZPD)
Prioritize goals slightly beyond current competence (not trivially easy, not impossibly hard):
- For each goal, estimate difficulty from past similar goals (VSA similarity search)
- Compute agent's competence for that goal type from the competence model
- ZPD sweet spot: difficulty between `competence * 0.8` and `competence * 1.3`
- Goals in ZPD get a priority boost; goals far outside get a penalty

### Autoepistemic Goal Questioning
Implement Moore-style self-reflection during the reflection cycle:
- "If I were going to make progress on goal X, I would have found evidence by now (given N cycles and improvement rate R). I haven't. Therefore, I should not continue believing X is achievable."
- Triggers when `cycles_worked > 2 × estimated_effort` and improvement rate ≤ 0

### AGM Belief Revision for Goals
Treat goals as defeasible beliefs with justifications:
```rust
GoalJustification: UserRequested           // highest entrenchment
                 | DecomposedFrom(SymbolId) // depends on parent
                 | InferredFromKG(Vec<SymbolId>)  // depends on supporting triples
                 | DefaultAssumption(String)       // lowest entrenchment, defeasible
```
When a justification is undermined (parent abandoned, supporting triples retracted), cascade to the goal. Epistemic entrenchment determines retraction order: `UserRequested` is most entrenched.

### Goal Reformulation via E-Graph
Use equality saturation to find equivalent but simpler goals:
- Rewrite rule: `(achieve X fully)` → `(achieve X partially)` when `X.difficulty > agent.competence`
- Rewrite rule: `(achieve X and Y)` → `(achieve X)` when Y has been proven infeasible
- Extract the simplest equivalent goal from the e-graph via `AstSize` cost function

**Changes**:
- [ ] New file: `src/agent/metacognition.rs` — monitoring/control layer, progress tracking, competence model
- [ ] `src/agent/reflect.rs` — `evaluate_goal_relevance()`, autoepistemic check, ZPD scoring
- [ ] `src/agent/goal.rs` — `GoalJustification` enum, `Reformulated` status, `reformulated_from` link
- [ ] `src/agent/goal_generation.rs` — goal reformulation proposals
- [ ] `src/agent/memory.rs` — competence tracking from episodic patterns, failure pattern encoding

**Estimated scope**: ~600–800 lines

---

## Phase 11g — Resource Awareness

**Problem**: The agent has no model of effort, time, or opportunity cost. It can't reason about whether a goal is worth the cycles.

**Research basis**: Russell & Wefald's Value of Computation (VOC), Zilberstein's anytime algorithms, CBR effort estimation (Shepperd & Schofield), opportunity cost via marginal value analysis.

**Design**:

### Value of Computation (VOC)
Before each OODA cycle, compute VOC for the active goal:
```
VOC(goal) = P(improvement) × magnitude(improvement) - cycle_cost
```
- `P(improvement)`: From competence model (11f) — probability that one more cycle advances the goal
- `magnitude(improvement)`: From progress tracking — average progress per cycle (sliding window)
- `cycle_cost`: Opportunity cost of not working on any other goal (see below)

If `VOC ≤ 0`, switch to a different goal or stop deliberating. This is the **optimal stopping criterion** from metareasoning theory.

### CBR Effort Estimation
Each completed goal becomes a case stored via `put_meta`:
```rust
EffortCase { goal_description: String, goal_vector: Vec<u32>, cycles_consumed: u32,
             succeeded: bool, tool_usage: HashMap<String, u32>, initial_coverage: f32 }
```
For new goals, retrieve 3 most similar past cases via HNSW Hamming distance. Use median `cycles_consumed` as estimate. Adapt estimate based on current KG richness: if more triples exist about the goal's subject now than in the analogous case, reduce estimate; if sparser, increase.

**Dynamic stall threshold**: Replace fixed `DEFAULT_STALL_THRESHOLD` with `1.5 × estimated_effort`. Goals with higher estimated effort get more patience before stall detection fires.

### Opportunity Cost Reasoning
When selecting which goal to work on, compare marginal values:
- For each active goal, compute `marginal_value = VOC(goal) × priority / 255`
- The goal with highest marginal value wins
- Record the opportunity cost: `value of best alternative not chosen`
- Store as KG triples for auditability:
  ```
  (cycle:142, meta:selectedGoal, goal:X)
  (cycle:142, meta:opportunityCost, "0.35")
  (cycle:142, meta:alternativeGoal, goal:Y)
  ```

### Diminishing Returns Detection
Sliding window (last 5 cycles) tracking improvement per cycle:
- Compute `improvement_rate = progress_delta / cycles_delta`
- If `improvement_rate(current_window) < 0.5 × improvement_rate(previous_window)`, signal diminishing returns
- Feed into VOC computation: `P(improvement)` decreases when diminishing returns detected
- Also triggers meta-level control: suggest goal decomposition, strategy rotation, or goal suspension

### Budget Allocation
Per-project cycle budgets (from 11d):
- When approaching budget (> 80% consumed), escalate to user or compress remaining plan
- Anytime algorithm pattern: always maintain a "best so far" result that can be returned if budget runs out
- Agent can negotiate budget increases via argumentation (11c): "Goal X needs 10 more cycles because Y"

**Key types**:
```rust
EffortCase { goal_description: String, goal_vector: Vec<u32>, cycles_consumed: u32,
             succeeded: bool, tool_usage: HashMap<String, u32>, initial_coverage: f32 }
EffortEstimate { estimated_cycles: u32, confidence: f32, basis: Vec<SymbolId> }
ResourceReport { goal: SymbolId, cycles_consumed: u32, budget_remaining: Option<u32>,
                 voc: f32, improvement_rate: f32, diminishing_returns: bool }
```

**Changes**:
- [ ] New file: `src/agent/resource.rs` — VOC computation, effort estimation, opportunity cost
- [ ] `src/agent/goal.rs` — `estimated_effort` field, `EffortEstimate` type, dynamic stall threshold
- [ ] `src/agent/reflect.rs` — diminishing returns detection, resource reporting
- [ ] `src/agent/project.rs` — cycle budgets per project, budget escalation
- [ ] `src/agent/ooda.rs` — VOC-based goal selection, opportunity cost in scoring

**Estimated scope**: ~500–700 lines

---

## Phase 11h — Procedural Learning (Chunking)

**Problem**: The agent solves the same types of problems from scratch every time. Successful strategies are not compiled into reusable methods.

**Research basis**: Soar's chunking mechanism, ACT-R production compilation, HTN-MAKER method learning from traces.

**Design**:

### Chunking: Goal Success → Learned Method
When a goal succeeds after multiple OODA cycles, analyze the decision sequence and compile it into a **learned decomposition method**:

1. **Extract the trace**: Collect the sequence of (tool_selected, tool_args_pattern, outcome) from working memory Decision entries for the completed goal
2. **Generalize**: Replace specific SymbolIds with type-based patterns (e.g., "any Entity of type Person" instead of a specific person). Use the KG type hierarchy to determine the right level of generality.
3. **Store as HTN method**: Create a new `DecompositionMethod` in the KG with:
   - Preconditions: The goal type pattern (SPARQL query matching the generalized goal)
   - Subtasks: The generalized tool sequence as concrete plan steps
   - Ordering: Sequential (from the trace)
   - Provenance: `DerivationKind::ProceduralLearning { source_goal }`
4. **Index for retrieval**: Encode the method's precondition pattern as a VSA hypervector and insert into HNSW. Future similar goals will match the method via VSA similarity before falling back to generic decomposition.

### Method Refinement
When a learned method is applied and succeeds, increase its confidence. When it fails, either:
- Refine: Narrow the preconditions to exclude the failing case
- Retract: If failures exceed successes, retract the method (using provenance for cascade)

### ACT-R Production Compilation Analog
Soar chunking happens at impasse resolution. For akh-medu, compilation also triggers when:
- A reflection cycle identifies a successful pattern across multiple goals
- Two consecutive goals with similar vectors used the same tool sequence
- A user explicitly approves a strategy ("yes, always do it that way")

### Method Library Growth
Over time, the agent builds a library of domain-specific decomposition methods:
- New agents start with the built-in strategies (Research/Construction/Investigation/Maintenance)
- Learned methods are preferred over generic ones (VSA similarity match, not just keyword)
- The library is persisted in the durable store and restored on session resume
- Pruning: Methods not used in the last N sessions are demoted to dormant

**Key types**:
```rust
LearnedMethod { id: SymbolId, precondition_vector: HyperVec, precondition_sparql: String,
                steps: Vec<GeneralizedStep>, confidence: f32, usage_count: u32,
                success_count: u32, failure_count: u32 }
GeneralizedStep { tool: String, arg_pattern: String, expected_outcome: String }
```

**Changes**:
- [ ] New file: `src/agent/chunking.rs` — trace extraction, generalization, method compilation
- [ ] `src/agent/plan.rs` — method library integration, learned method preference in selection
- [ ] `src/agent/ooda.rs` — trigger chunking on goal completion
- [ ] `src/agent/memory.rs` — pattern detection across multiple goal completions

**Depends on**: 11b (HTN method registry), 11d (session persistence)

**Estimated scope**: ~500–700 lines

---

## Implementation Order

```
11a (Goal Generation + Drives) ──→ 11b (HTN Decomposition) ──→ 11d (Projects)
                                                                     │
11c (Value-Based Argumentation) ──→ 11d (Projects)                  │
                                                                     │
11e (World Monitoring + GDA) ──→ feeds into 11a                     │
                                                                     │
11f (Metacognitive Layer) ←── depends on 11a, 11c                   │
                                                                     │
11g (VOC + Resource Reasoning) ←── depends on 11d, 11f             │
                                                                     │
11h (Procedural Learning) ←── depends on 11b, 11d                  │
```

**Wave 1**: 11a (goal generation + drive system) — the core capability
**Wave 2**: 11b (HTN decomposition) + 11c (value-based argumentation) — making goals smart
**Wave 3**: 11d (projects + Soar/ACT-R memory integration) — cross-session structure
**Wave 4**: 11e (GDA monitoring) + 11f (metacognitive layer) — autonomous awareness
**Wave 5**: 11g (VOC + resource reasoning) + 11h (procedural learning) — economic reasoning + learning

## Total Estimated Scope

~4,500–6,900 lines across 8 sub-phases.

## Relationship to Prior Phases

Phase 11 is where **all prior phases converge**:
- **Phase 9a** (microtheories) → projects are microtheories with scoped contexts
- **Phase 9c** (TMS) → retraction cascades trigger reactive goals and belief revision
- **Phase 9e** (argumentation) → value-based priority reasoning and goal evaluation
- **Phase 9l** (contradiction detection) → anomaly-driven goal generation, coherence drive
- **Phase 10** (code generation) → implementation goals can be executed end-to-end
- **Phase 8 agent infrastructure** → OODA, memory, reflection, planning are the execution substrate

## Research Grounding

Every design decision in Phase 11 traces back to established cognitive architecture or AI research:

| Sub-phase | Primary Research Basis |
|-----------|----------------------|
| 11a Goal Generation | Schmidhuber compression progress, CLARION drives, GDA, Soar impasses, IMGEP |
| 11b Decomposition | SHOP2/PANDA HTN, HTN-MAKER, Goal-Task Networks, Koopman taxonomy |
| 11c Priority | Dung argumentation, Bench-Capon VAF, Ferretti goal reasoning, ACT-R utility |
| 11d Projects | Cyc microtheories, Soar 4-memory, ACT-R activation, HGN |
| 11e Monitoring | GDA expectation monitoring, BDI events, ECA rules, situation calculus |
| 11f Self-Evaluation | Cox MCL, Nelson-Narens, AGM belief revision, autoepistemic reasoning |
| 11g Resources | Russell-Wefald VOC, Zilberstein anytime, Shepperd-Schofield EBA, CBR |
| 11h Chunking | Soar chunking, ACT-R production compilation, HTN-MAKER method learning |

See `docs/ai/decisions/003-autonomous-tasks-research.md` for the full research analysis.

This phase transforms akh-medu from a tool that does what it's told into an agent that decides what to do.
