# ADR-003: Autonomous Tasks Research — Goal Generation, Metacognition, and Resource-Bounded Reasoning

- **Date**: 2026-02-17
- **Status**: Accepted
- **Context**: Deep research for Phase 11 optimization, specifically how to ground the autonomous task system in established cognitive architecture principles, intrinsic motivation theory, and metareasoning

## Research Question

How should a neuro-symbolic agent (VSA + KG + e-graph) autonomously generate, decompose, prioritize, evaluate, and manage its own goals — without LLM dependency?

## Key Findings

### 1. Intrinsic Motivation and Curiosity-Driven Goal Generation (Most Foundational)

**Schmidhuber's Compression Progress** (2009) formalizes curiosity as the first derivative of compression: an agent is "curious" about data that, once learned, would most improve its world model compression. The curiosity reward is the *rate of learning*, not the prediction error itself. This avoids both boredom (already-learned regions) and noise fixation (incompressible regions).

**Pathak's Intrinsic Curiosity Module (ICM)** (2017) operationalizes this: train a forward model that predicts the next state given an action, then use the prediction error as intrinsic reward. Crucially, the prediction operates in a learned *feature space* rather than raw observations, filtering out unpredictable noise.

**IMGEP** (Forestier, Portelas, Mollard & Oudeyer, 2022) formalizes intrinsically motivated exploration with four components: (1) a goal space, (2) learning-progress-based goal sampling, (3) a goal-parameterized policy, and (4) systematic reuse. The agent tracks which regions of goal space are improving fastest and samples from those — an automatic curriculum.

**Mapping to akh-medu**: The VSA provides a natural goal space — each goal is a hypervector bundle of involved concepts. The KG triple-addition rate per domain cluster serves as learning progress. HNSW enables fast search over the goal space. The key mechanism: regions of the KG where triple-addition is *accelerating* get more exploration goals; plateaued regions are deprioritized. This is compression progress without any neural network.

### 2. Cognitive Architecture Goal Generation (Soar, CLARION, BDI)

**Soar's impasse-driven subgoaling** (Laird, 2022) generates goals automatically from failures: when the agent can't select an operator (tie, conflict, no-change), a substate is created with a goal to resolve the impasse. After resolution, *chunking* compiles the solution into production rules for future instant use.

**CLARION's drive system** (Sun) uses explicit drives (curiosity, coherence, competence, self-preservation) that generate goals when drive strength exceeds a threshold. The motivational subsystem is separate from the action subsystem, monitoring its performance.

**BDI (AgentSpeak/Jason)** triggers plan selection from belief changes: when a belief is added or deleted, matching plans fire, and plan bodies can generate new achievement goals.

**Mapping to akh-medu**: Soar impasses map to the OODA `decide()` function — when tool selection ties or scores are all low, that's an impasse triggering a meta-goal. CLARION drives map to measurable KG metrics (gap coverage, contradiction count, learning rate, tool failure rate). BDI event-triggered goals map to the existing trigger system (`agent/trigger.rs`), generalized to pattern-match on triple content.

### 3. Goal-Driven Autonomy (GDA)

**Aha, Cox & Munoz-Avila (2018)** defined GDA as a four-step process: (1) discrepancy detection — compare expected vs observed state, (2) explanation generation — hypothesize why, (3) goal formulation — generate corrective goals, (4) goal management — select among formulated goals.

**Mapping to akh-medu**: After each Act phase, compare the plan step's predicted KG changes against actual changes. Discrepancies become explanations, which become `GoalProposal`s. This closes the gap between "plan what should happen" and "react to what actually happened."

### 4. HTN Planning for Task Decomposition

**SHOP2** (Nau et al., 2003) is the canonical HTN planner: decomposition methods specify how to break abstract tasks into subtasks with preconditions. **PANDA** adds plan-space search for partially ordered decomposition. **HTN-MAKER** (Hogg, Muñoz-Avila & Kuter, 2008) learns decomposition methods from plan traces — given examples of successful task completions, it induces the hierarchical structure.

**Goal-Task Networks** (Shivashankar et al., IJCAI 2016) unify goal decomposition and task decomposition: a single method can decompose goals into subgoals OR tasks into subtasks.

**Resource-rational decomposition** (cognitive science) shows humans decompose tasks by trading off solution utility against planning cost — sub-goals that reduce planning complexity the most are preferred.

**Mapping to akh-medu**: Store decomposition methods as KG triples with preconditions (SPARQL queries) and subtask lists. Use VSA similarity between the goal vector and method-template vectors for method selection. Learn new methods from successful episodes (HTN-MAKER pattern). Goal-Task Networks unify the existing `goal.rs` hierarchy and `plan.rs` steps.

### 5. Dependency Tracking and Partial-Order Planning

Task dependencies form a DAG with four canonical types: Finish-to-Start (most common), Start-to-Start (parallel execution), Finish-to-Finish (co-termination), Start-to-Finish (deadline-triggered). **Topological sorting** (Kahn's algorithm) produces valid execution orders. **Critical path analysis** identifies the minimum-time chain.

**Partial Order Causal Link (POCL)** planning maintains causal links between steps rather than committing to a total order — steps with no ordering constraint can execute concurrently.

**Mapping to akh-medu**: Upgrade `Plan.steps` from flat `Vec` to a petgraph `DiGraph<PlanStep, DependencyKind>`. Add `blocked_by` as an `AgentPredicates` member. VSA encodes step preconditions/effects — if step B's precondition vector is similar to step A's effect vector, infer a likely dependency automatically.

### 6. Value-Based Argumentation for Priority Reasoning

**Dung's abstract argumentation** (1995) defines frameworks as `(arguments, attacks)` with formal semantics (conflict-free, admissible, preferred, grounded extensions). **Bench-Capon's VAF** (2003) extends this with *values* — an attack succeeds only if the attacked argument's value isn't preferred. Different audiences (value orderings) produce different acceptable sets.

**Ferretti et al. (2022)** applied argumentation to goal reasoning: pro/con arguments cite expected benefits/costs, and goal selection uses Assumption-Based Argumentation (ABA) with explanations for every decision.

**Mapping to akh-medu**: Replace `priority: u8` with `Vec<Argument>` per goal. Each argument promotes a value (timeliness, thoroughness, efficiency, accuracy). Priority emerges from the preferred extension under the current value ordering. Store arguments as KG triples. VSA similarity bootstraps arguments for new goals from past similar goals. E-graph discovers argument equivalences.

### 7. Metacognition and Self-Evaluation

**Cox's Metacognitive Loop** (2005, 2007) defines a monitoring/control architecture: the meta-level monitors the object-level for failures, diagnoses them using Introspective Meta-Explanation Patterns (IMXPs), and applies control strategies. **Nelson & Narens** (1990) formalized the monitoring → control flow: monitoring produces judgments (ease-of-learning, feeling-of-knowing, confidence), control acts on them (allocation of study time, selection of strategies, termination).

**Mapping to akh-medu**: The existing `reflect.rs` is a simplified metacognitive controller. Extend it with: (1) per-tool and per-task-category success rate tracking (competence model), (2) Zone of Proximal Development — prioritize goals slightly beyond current competence, (3) failure pattern matching via VSA — encode failure contexts as hypervectors and search HNSW for similar past failures with known resolutions.

### 8. Resource-Bounded Reasoning and Metareasoning

**Russell & Wefald (1991)** introduced Value of Computation (VOC): before spending a cycle on deliberation, estimate whether the expected improvement in decision quality exceeds the cost of computing. If VOC ≤ 0, stop deliberating and act on current best.

**Zilberstein's anytime algorithms** (1996, 2011) produce progressively better results with more computation. The meta-level decides when to stop based on the performance profile (quality vs time curve). The optimal stopping point is where marginal improvement equals marginal cost.

**Mapping to akh-medu**: Before each OODA cycle, compute VOC for the active goal: `VOC = P(improvement) × magnitude(improvement) - cycle_cost`. If VOC ≤ 0, switch to a different goal or stop. Track progress via a sliding window — when improvement rate drops below a threshold (diminishing returns), signal the meta-level.

### 9. Effort Estimation via Case-Based Reasoning

**Estimation by Analogy (EBA)** (Shepperd & Schofield, 1997) estimates effort by retrieving similar past tasks and reusing their effort values. The process: feature extraction → similarity computation → k-nearest retrieval → adaptation → retention.

**Mapping to akh-medu**: Each completed goal becomes a case: `(goal_semantic_vector, cycles_worked, success, tool_sequence)`. Store in durable store via `put_meta`. For new goals, retrieve the 3 most similar past cases via HNSW (already available) and use the median `cycles_worked` as the estimate. Dynamic stall threshold: replace fixed `DEFAULT_STALL_THRESHOLD` with `1.5 × estimated_effort`.

### 10. Cross-Session Continuity (Soar + ACT-R Memory Models)

**Soar's four memory systems**: Working Memory (ephemeral situational awareness), Procedural Memory (production rules learned via chunking), Semantic Memory (long-term facts), Episodic Memory (temporal WM snapshots with delta encoding).

**ACT-R's activation-based retrieval**: chunks have base-level activation `= ln(Σ t_i^{-d})` where `t_i` are times since each access and `d ≈ 0.5`. This naturally implements forgetting and recency bias.

**Mapping to akh-medu**:

| Soar Component | akh-medu Equivalent | Status |
|---|---|---|
| Working Memory | `WorkingMemory` | Implemented |
| Procedural Memory | Tool registry + plan templates | Static — needs chunking |
| Semantic Memory | KG (Oxigraph) | Implemented |
| Episodic Memory | `EpisodicEntry` + consolidation | Implemented |
| Chunking | Not yet | **New for 11h** |

The chunking analog: when a goal succeeds after multiple OODA cycles, compile the decision sequence into a learned HTN method stored in the KG. Future similar goals fire the method directly, bypassing multi-cycle exploration. ACT-R activation can replace the static `WorkingMemoryEntry.relevance` with a decay function using the existing `reference_count` and timestamps.

### 11. Belief Revision and Goal Questioning

**AGM framework** (1985) defines three operations: expansion (add), contraction (remove + cascade), revision (add while maintaining consistency). **Epistemic entrenchment** orders beliefs by retractability — axioms are most entrenched, agent-derived beliefs least.

**Autoepistemic reasoning** (Moore): "If I were going to make progress on goal X, I would have found evidence by now. Since I haven't, I should not believe X is achievable." This applies directly to goal questioning.

**Mapping to akh-medu**: Treat goals as defeasible beliefs with justifications. When a justification is undermined (parent abandoned, supporting triples retracted), cascade to the goal. E-graph rewrite rules enable goal reformulation: `(achieve X fully)` → `(achieve simpler-X)` when full achievement fails. The provenance system already supports dependency tracking for cascade retraction.

### 12. World Monitoring (GDA + BDI + ECA)

**GDA expectation monitoring**: After each action, compare predicted state changes against actual. Discrepancies trigger explanation → goal formulation.

**BDI event-triggered goals**: When beliefs change, matching plans fire and can generate achievement goals. This generalizes the trigger system to pattern-match on triple content.

**ECA rules** and the existing trigger system (`agent/trigger.rs`) already implement basic reactive behavior. Extension: add `TriplePattern` conditions that match on triple content, plus **fluent-style state tracking** that maintains a KG snapshot and computes deltas per cycle.

**Mapping to akh-medu**: Extend `TriggerCondition` with `TriplePattern { subject_pattern, predicate, object_pattern }`. Add an expectation-monitoring step after Act. VSA encodes trigger conditions for semantic matching — a new triple is compared against all registered trigger vectors via HNSW.

## Decision

Enhance Phase 11 with research-informed additions:

### Enhanced 11a — Add drive system and three-phase pipeline
Four measurable drives (curiosity, coherence, completeness, efficiency) plus the Signal Collection → Deliberation → Activation pipeline from cognitive architecture research.

### Enhanced 11b — HTN method registry with learned decomposition
Store methods as KG triples with SPARQL preconditions. VSA-based method selection. HTN-MAKER-style method learning from successful episodes.

### Enhanced 11c — Value-Based Argumentation (VAF)
Arguments promote values with audience-specific orderings. Priority emerges from preferred extensions, not arbitrary numbers.

### Enhanced 11d — Soar/ACT-R memory model integration
ACT-R activation-based retrieval for session continuity. Session summaries as episodic memory.

### Enhanced 11e — GDA expectation monitoring
Predicted vs actual state comparison after each Act. Discrepancy → explanation → corrective goal pipeline.

### Enhanced 11f — Metacognitive monitoring/control layer
Nelson-Narens monitoring → control flow. Competence model with ZPD. Autoepistemic goal questioning. AGM belief revision with epistemic entrenchment.

### Enhanced 11g — VOC-based resource reasoning
Russell & Wefald VOC for cycle allocation. Anytime algorithm patterns. CBR effort estimation with HNSW case retrieval. Opportunity cost via marginal value comparison.

### 11h — Procedural Learning (New)
Soar-inspired chunking: compile successful goal-resolution sequences into HTN methods for instant reuse. Bridges 11b (decomposition) and 11d (cross-session). The agent gets faster at recurring task patterns.

## The Autonomous Agent Pipeline

The research converges on a unified architecture:

```
                        ┌─────────────────────────────────────────┐
                        │  META-LEVEL (11f + 11g)                 │
                        │                                         │
                        │  Monitoring:        Control:            │
                        │  - Progress rate    - VOC stopping      │
                        │  - Competence       - Goal reformulation│
                        │  - Failure patterns - Strategy rotation  │
                        │  - Diminishing ret  - Belief revision    │
                        │  - Opportunity cost - Effort budgeting   │
                        │                                         │
                        └───────────┬─────────────┬───────────────┘
                          monitoring│(up)  control│(down)
                        ┌───────────▼─────────────▼───────────────┐
                        │  OBJECT-LEVEL (existing OODA loop)      │
                        │                                         │
                        │  Observe → Orient → Decide → Act        │
                        │  + Goal Generation (11a)                │
                        │  + HTN Decomposition (11b)              │
                        │  + Argumentation Priority (11c)         │
                        │  + World Monitoring (11e)               │
                        │  + Procedural Learning (11h)            │
                        │                                         │
                        └───────────┬─────────────┬───────────────┘
                                    │             │
                        ┌───────────▼─────────────▼───────────────┐
                        │  SUBSTRATE                              │
                        │                                         │
                        │  KG (petgraph+oxigraph)  VSA (HNSW)    │
                        │  E-graph (egg)  Provenance  Store       │
                        │  Projects/Microtheories (11d)           │
                        └─────────────────────────────────────────┘
```

## Sources

### Intrinsic Motivation and Curiosity
- [Schmidhuber (2009) — Driven by Compression Progress](https://arxiv.org/abs/0812.4360)
- [Schmidhuber (2010) — Formal Theory of Creativity, Fun, and Intrinsic Motivation](https://people.idsia.ch/~juergen/ieeecreative.pdf)
- [Pathak et al. (2017) — Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)
- [Forestier, Portelas, Mollard & Oudeyer (2022) — IMGEP with Automatic Curriculum Learning](https://www.jmlr.org/papers/v23/21-0808.html)
- [Colas et al. (2022) — Autotelic Agents with Intrinsically Motivated Goal-Conditioned RL](https://www.jair.org/index.php/jair/article/download/13554/26824/31188)
- [Bellemare et al. (2016) — Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)

### Open-Ended Learning
- [Ecoffet et al. (2021) — Go-Explore: First Return Then Explore](https://arxiv.org/abs/1901.10995)
- [Wang et al. (2019) — POET: Paired Open-Ended Trailblazer](https://arxiv.org/abs/1901.01753)
- [Wang et al. (2020) — Enhanced POET](https://arxiv.org/abs/2003.08536)
- [Brant & Stanley (2017) — Minimal Criterion Coevolution](https://dl.acm.org/doi/10.1145/3071178.3071186)

### Cognitive Architectures
- [Laird (2022) — Introduction to the Soar Cognitive Architecture](https://arxiv.org/pdf/2205.03854)
- [Anderson — ACT-R](https://en.wikipedia.org/wiki/ACT-R); [ACT-R vs Soar Analysis](https://arxiv.org/abs/2201.09305)
- [Sun — CLARION Tutorial](https://escholarship.org/content/qt149589jb/qt149589jb.pdf)
- [Franklin — LIDA and Global Workspace Theory](https://www.worldscientific.com/doi/10.1142/S1793843009000050)
- [40 Years of Cognitive Architectures (Springer 2018)](https://link.springer.com/article/10.1007/s10462-018-9646-y)

### Goal-Driven Autonomy and BDI
- [Aha, Cox & Munoz-Avila (2018) — Goal Reasoning: Foundations](https://onlinelibrary.wiley.com/doi/abs/10.1609/aimag.v39i2.2800)
- [Munoz-Avila et al. — GDA Project](https://www.cse.lehigh.edu/~munoz/projects/GDA/)
- [Choi & Langley (2011) — Reactive Goal Management in ICARUS](https://www.sciencedirect.com/science/article/abs/pii/S1389041711000039)
- [BDI Architecture Survey (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0684.pdf)
- [Thangarajah et al. (2013) — Runtime Goal Conflict Resolution](https://ieeexplore.ieee.org/document/6511591/)

### HTN Planning and Decomposition
- [Nau et al. — SHOP2 HTN Planning System](https://www.cs.umd.edu/~nau/papers/nau2003shop2.pdf)
- [Bercher, Alford & Höller (2019) — PANDA Framework](https://link.springer.com/article/10.1007/s13218-020-00699-y)
- [Hogg, Muñoz-Avila & Kuter (2008) — HTN-MAKER](https://www.cse.lehigh.edu/~munoz/Publications/AAAI08.pdf)
- [Shivashankar et al. (2013) — Hierarchical Goal Networks](https://www.cs.umd.edu/~nau/papers/shivashankar2013hierarchical.pdf)
- [Koopman — Taxonomy of Decomposition Strategies](https://users.ece.cmu.edu/~koopman/decomp/decomp.html)
- [Resource-rational Task Decomposition (CogSci 2020)](https://cognitivesciencesociety.org/cogsci20/papers/0747/0747.pdf)

### Argumentation
- [Dung (1995) — Abstract Argumentation Framework](https://en.wikipedia.org/wiki/Argumentation_framework)
- [Bench-Capon (2003) — Value-Based Argumentation Frameworks](https://arxiv.org/pdf/cs/0207059)
- [Ferretti et al. (2022) — Argumentation-Based Goal Reasoning](https://academic.oup.com/logcom/article-abstract/33/5/984/6661105)
- [Argument-Driven Planning & Autonomous Explanation (AGI 2024)](https://alumni.media.mit.edu/~kris/ftp/argument-driven-planning-autonomus-explanation-agi2024.pdf)

### Metacognition and Metareasoning
- [Cox (2005) — Metacognition in Computation](https://www.sciencedirect.com/science/article/pii/S0004370205001530)
- [Cox (2007) — Perpetual Self-Aware Cognitive Agents](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2027/1920)
- [Nelson & Narens (1990) — Metamemory Framework](https://sites.socsci.uci.edu/~lnarens/1990/Nelson&Narens_Book_Chapter_1990.pdf)
- [Russell & Wefald (1991) — Principles of Metareasoning](https://www.sciencedirect.com/science/article/abs/pii/000437029190015C)
- [Zilberstein — Metareasoning and Bounded Rationality](http://rbr.cs.umass.edu/shlomo/papers/ZCh3-2011.pdf)
- [Metacognitive AI: Framework for Neurosymbolic Approach (Springer 2024)](https://link.springer.com/chapter/10.1007/978-3-031-71170-1_7)

### Belief Revision
- [AGM Theory — Logic of Belief Revision (Stanford Encyclopedia)](https://plato.stanford.edu/entries/logic-belief-revision/)
- [Epistemic Entrenchment (PhilArchive)](https://philarchive.org/archive/HUBBRI)
- [Non-Monotonic Logic (Stanford Encyclopedia)](https://plato.stanford.edu/entries/logic-nonmonotonic/)
- [Provenance-Aware Knowledge Representation (Springer 2020)](https://link.springer.com/article/10.1007/s41019-020-00118-0)

### Effort Estimation
- [Shepperd & Schofield (1997) — Estimation by Analogy](https://dl.acm.org/doi/10.1109/32.637387)
- [CBR for Task Execution Prediction (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-24586-7_2)
- [Enhancing Intelligent Agents with Episodic Memory (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1389041711000428)

### Truth Maintenance and Contradiction
- [de Kleer (1986) — ATMS](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/download/866/784)
- [Dis/Equality Graphs (2025)](https://programming-group.com/assets/pdf/papers/2025_Dis-Equality-Graphs.pdf)

### VSA Applications
- [VSA Survey Part I (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3538531)
- [VSA Survey Part II (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3558000)

## Consequences

- Phase 11 expands from 7 to 8 sub-phases (11a–11h)
- New drive system requires measurable KG metrics (gap coverage, contradiction count, learning rate, tool failure rate)
- HTN method registry requires new KG predicate set (`method:precondition`, `method:subtask`, `method:ordering`)
- Argumentation infrastructure reused from Phase 9e, extended with value orderings
- Metacognitive monitoring/control layer is a cross-cutting concern touching `reflect.rs`, `ooda.rs`, and new `metacognition.rs`
- Procedural learning (chunking) creates a feedback loop: goal success → learned method → faster future goals
- CBR effort estimation leverages existing HNSW + episodic memory infrastructure
- VOC computation requires per-goal progress tracking with sliding windows
- Estimated additional scope for 11h: ~500–700 lines
- Total Phase 11 scope revised: ~4,100–5,900 lines across 8 sub-phases
