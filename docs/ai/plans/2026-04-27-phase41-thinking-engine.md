# Phase 41 — The Thinking Engine: Unified Reasoning Loop with Test-Time-Compute Scaling

> Date: 2026-04-27
> Status: Proposed
> Phase: 41
> Depends on: Phase 38 (Relocatable Akh — for ThoughtState persistence), Phase 17 (evidence intervals), Phase 11g (VOC), existing reasoning primitives (Phases 9, 15, 16, 18, 19, 20)
> Enhances: every existing reasoning surface
> ADR: [052-unified-thinking-loop](../decisions/052-unified-thinking-loop.md)
> Research: [docs/ai/research/2026-04-27-thinking-in-llms-vs-vsa.md](../research/2026-04-27-thinking-in-llms-vs-vsa.md)
> Anima ticket: `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9`

## Executive summary

Phase 41 unifies akh-medu's existing reasoning primitives (spreading activation, e-graph saturation, MCTS, argumentation, counterfactual fork, Dempster-Shafer evidence combination, superposition merging) into a single `ThinkingLoop` with a measurable budget and a faithful reasoning trace. The loop holds a `ThoughtState` (HyperVec + superposed hypotheses + open goals + trace + budget), iterates a rotation of primitive steps, scores intermediate states with a structural verifier, and halts on convergence, budget exhaustion, or verifier sufficiency.

The empirical bet: this loop demonstrates a logarithmic accuracy-vs-compute scaling law on at least narrow reasoning benchmarks, matching the property that defines frontier LLM reasoning architectures — but on a substrate where every step is faithful, auditable, and persistent across hosts via Phase 38's WAL.

The phase ships in seven sub-phases (41a–41g), ordered so that an end-to-end loop exists at the end of 41a and each subsequent sub-phase strengthens it.

## Motivation

See ADR 052 and the underlying research synthesis for the full argument. In short:

1. **LLMs implement an approximate VSA.** Recent mech-interp work ([Attention as Binding](https://arxiv.org/html/2512.14709), [Hyperdimensional Probe](https://arxiv.org/html/2509.25045v1)) shows transformer attention is a soft, lossy VSA emerging from gradient descent. Akh-medu has the exact, explicit version.
2. **LLM "thinking" is autoregressive token sampling shaped by RL.** It works because more tokens = more forward passes = more compute. Latent-reasoning research ([COCONUT](https://arxiv.org/abs/2412.06769)) suggests tokens are a 15-bit straw against a 40,960-bit hidden-state channel.
3. **Akh-medu has every reasoning primitive frontier LLMs amortise into weights** — but they're invoked ad hoc, not as a unified loop with a budget.
4. **The missing piece is integration.** A single dial labelled "think harder" with a faithful trace and economic budget control is what would put akh-medu at the frontier.

## Scope and non-goals

### In scope

- A `ThinkingLoop` that drives the existing reasoning primitives in a rotation.
- A `ThoughtState` data type carrying HyperVec focus, superposed hypotheses with belief intervals, open goals, trace, and budget.
- A structural verifier composed of Phase 9l contradiction detection + Phase 17 Dempster-Shafer evidence combination + Phase 18 source reliability.
- Persistence of `ThoughtState` snapshots and `ThoughtStep` traces via the Phase 38 WAL → `akhomed.thinking_traces`.
- A budget controller that consults Phase 11g VOC for adaptive halting.
- A benchmark harness that measures accuracy vs. step budget on representative reasoning tasks.

### Explicitly out of scope

- **New reasoning primitives.** This phase composes existing ones. New primitives (e.g. probabilistic programming, SAT solving) belong to separate phases.
- **Replacing Phase 9f Reasoner Dispatch.** Dispatch remains valid for cases where a single competitive reasoner is the right answer. The Thinking Loop is an additional surface.
- **Replacing Phase 16b MCTS.** MCTS for action planning continues to work standalone. The Loop calls MCTS as one of its primitive steps.
- **Self-modifying primitive rotation.** 41b ships a static rotation; learned rotation policies are deferred to a possible Phase 41h or later phase.
- **Distributed thinking.** A `ThoughtState` lives on one Akh at a time. Multi-Akh thinking handoffs go through Phase 36 delegation (separate phase).
- **LLM training.** The Loop never trains an LLM; if it calls one, it's read-only via Phase 26.

## Architecture

### Data types

```rust
/// A unit of cognitive focus during deliberation.
///
/// Held in memory during a thinking session; flushed to akhomed.thinking_traces
/// on every ThoughtStep emit; recoverable across host relocation via Phase 38 WAL.
pub struct ThoughtState {
    /// HyperVec representation of current cognitive focus.
    /// Analogous to the residual stream in a transformer: accumulates as the
    /// loop proceeds via VSA bundle (majority-vote superposition).
    pub focus_vec: HyperVec,

    /// Multiple hypotheses held simultaneously in VSA superposition.
    /// Each (vec, interval) pair: vec is a candidate answer encoding,
    /// interval is the Dempster-Shafer belief bound on its truth.
    pub hypotheses: Vec<(HyperVec, BeliefInterval)>,

    /// Goals that opened during deliberation; closed when satisfied or abandoned.
    pub open_goals: Vec<SymbolId>,

    /// The reasoning trace, monotonically grown.
    pub trace: Vec<ThoughtStep>,

    /// Remaining budget. Decremented on every primitive step.
    pub budget: ThinkingBudget,

    /// Provenance bookkeeping: parent thought ids, originating goal, etc.
    pub origin: ThoughtOrigin,
}

pub struct ThinkingBudget {
    pub max_steps: u32,
    pub max_wall_ms: u64,
    pub voc_credits: f32,   // Phase 11g VOC currency
    pub used_steps: u32,
    pub used_wall_ms: u64,
    pub used_voc: f32,
}

pub enum ThoughtStep {
    SpreadingActivation { frontier_size: u32, new_concepts: Vec<SymbolId>, novelty: f32 },
    EgraphSaturate { iterations: u32, eclasses_added: u32, best_cost_delta: f32 },
    McTsRollout { trajectory_len: u32, return_value: f32, action: SymbolId },
    Argumentation { topic: SymbolId, verdict: VerdictKind, score: f32 },
    Counterfactual { intervention: TripleRef, divergence: f32 },
    EvidenceCombine { conflict: f32, narrowed_interval: BeliefInterval },
    SuperpositionMerge { hypotheses_in: u32, hypotheses_out: u32, mean_similarity: f32 },
    VerifierScore { score: f32, reasons: Vec<VerifierReason> },
}

pub struct Conclusion {
    pub answer: HyperVec,
    pub answer_symbol: Option<SymbolId>,    // resolved via item_memory
    pub trace: Vec<ThoughtStep>,
    pub confidence: BeliefInterval,
    pub steps_used: u32,
    pub halt_reason: HaltReason,
}

pub enum HaltReason {
    Converged { delta: f32 },
    BudgetExhausted { resource: BudgetResource },
    VerifierSufficient { score: f32 },
    Aborted { reason: String },
}
```

### Loop skeleton

```rust
impl ThinkingLoop {
    pub fn deliberate(
        &mut self,
        goal: SymbolId,
        budget: ThinkingBudget,
    ) -> ThinkingResult<Conclusion> {
        let mut state = self.initial_state(goal, budget)?;
        let scheduler = self.scheduler.clone();   // 41b chooses primitive rotation

        loop {
            // Halting checks
            if let Some(reason) = self.should_halt(&state)? {
                return Ok(self.conclude(state, reason)?);
            }

            // Pick next primitive
            let primitive = scheduler.next(&state)?;

            // Run one step
            let step = self.run_primitive(primitive, &mut state)?;

            // Update budget + trace
            state.trace.push(step);
            self.charge_budget(&mut state, primitive)?;

            // Persist via Phase 38 WAL (41d)
            self.persistence.flush_step(&state)?;
        }
    }
}
```

### The verifier (the structural process-reward equivalent)

```rust
pub trait ThoughtVerifier {
    fn score(&self, state: &ThoughtState) -> ThinkingResult<VerifierVerdict>;
}

pub struct VerifierVerdict {
    pub score: f32,                       // [0, 1]
    pub sufficient: bool,                 // halt-now signal
    pub reasons: Vec<VerifierReason>,
}

pub enum VerifierReason {
    /// Phase 9l: a contradiction surfaced — score drops, may abort.
    ContradictionDetected { kind: ContradictionKind, locus: TripleRef },
    /// Phase 17: belief interval narrowed — score rises.
    BeliefIntervalNarrowed { from: BeliefInterval, to: BeliefInterval },
    /// Phase 18: an unreliable source dominates evidence — score drops.
    UnreliableSourceDominant { source: SymbolId, weight: f32 },
    /// Convergence: focus_vec stable across N steps.
    FocusStable { delta: f32, window: u32 },
    /// VOC: marginal value of next step below threshold.
    DiminishingReturns { last_voc: f32 },
}
```

The verifier is a **composition of existing systems**. It does not introduce a new learned model — it cannot lie about the reasoning because the reasoning *is* the trace.

## Sub-phases

### 41a — `ThinkingLoop` skeleton + `ThoughtState` + budget primitive (~900 lines)

**Priority**: High (critical path)
**Depends on**: nothing additional — uses existing primitives in synchronous calls

Files:
- New: `src/thinking/mod.rs`, `src/thinking/state.rs`, `src/thinking/loop_runner.rs`, `src/thinking/budget.rs`, `src/thinking/error.rs`
- Modify: `src/lib.rs` (export module), `src/error.rs` (add `AkhError::Thinking` transparent variant)

Deliverables:
- `ThoughtState`, `ThinkingBudget`, `ThoughtStep`, `Conclusion`, `HaltReason` types.
- A round-robin `ThinkingLoop` that calls each primitive in order, in-memory only.
- ~12 unit tests covering: budget exhaustion, halt-on-converge, halt-on-verifier, primitive registration.

End-of-41a state: `engine.deliberate(goal, budget)` returns a `Conclusion` with a populated trace. No persistence, no adaptive scheduler, no real verifier.

### 41b — Primitive-rotation scheduler (~500 lines)

**Priority**: High
**Depends on**: 41a

Files:
- New: `src/thinking/scheduler.rs`

Deliverables:
- `Scheduler` trait with `RoundRobin`, `Weighted`, and `GoalShape` implementations.
- `GoalShape` heuristics: factual queries get heavier `spreading_activation`; planning queries get heavier `mcts_rollout`; explanation queries get heavier `argumentation`. Mapping is hard-coded and small.
- Tests: scheduler picks expected primitives for canonical goal kinds; weighted scheduler honours configured weights.

End-of-41b state: the loop biases its rotation toward primitives that match the goal's shape.

### 41c — Verifier integration (~700 lines)

**Priority**: High
**Depends on**: 41a, Phase 17 (already shipped)

Files:
- New: `src/thinking/verifier.rs`, `src/thinking/verifier/{contradiction,evidence,source,convergence,voc}.rs`

Deliverables:
- `ThoughtVerifier` trait + `CompositeVerifier` that runs the four sub-verifiers and aggregates scores.
- `ContradictionVerifier`: wraps Phase 9l contradiction detection over the current `focus_vec` materialisation.
- `EvidenceVerifier`: wraps Phase 17 Dempster-Shafer combination on `state.hypotheses`.
- `SourceReliabilityVerifier`: wraps Phase 18a — penalises hypotheses dominated by low-reliability sources.
- `ConvergenceVerifier`: tracks `focus_vec` Hamming delta over a sliding window; signals convergence below `delta_threshold`.
- Tests: verifier correctly flags introduced contradictions; correctly narrows belief intervals; correctly halts on convergence.

End-of-41c state: the loop halts intelligently rather than only on step count.

### 41d — Persistence: `ThoughtStep` + `akhomed.thinking_traces` (~600 lines)

**Priority**: Medium-high
**Depends on**: 41a, Phase 38a–c (WAL + snapshot)

Files:
- New: `src/thinking/persistence.rs`
- New migration: `crates/akhomed-db/migrations/00NN_thinking_traces.sql`
- Modify: `src/provenance.rs` (add `DerivationKind::ThoughtStep` — next free tag)

Schema:

```sql
CREATE SCHEMA IF NOT EXISTS akhomed;

CREATE TABLE akhomed.thinking_traces (
    thought_id     UUID PRIMARY KEY,
    akh_id         UUID NOT NULL REFERENCES akhomed.akhs(akh_id),
    workspace_id   UUID NOT NULL REFERENCES workspaces.workspaces(workspace_id),
    origin_goal    BIGINT NOT NULL,                    -- SymbolId
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at       TIMESTAMPTZ,
    halt_reason    TEXT,
    final_answer   BIGINT,                             -- SymbolId, nullable
    confidence_lo  REAL,
    confidence_hi  REAL,
    steps_used     INT NOT NULL DEFAULT 0,
    voc_used       REAL NOT NULL DEFAULT 0
);

CREATE TABLE akhomed.thinking_steps (
    thought_id     UUID NOT NULL REFERENCES akhomed.thinking_traces(thought_id) ON DELETE CASCADE,
    step_idx       INT NOT NULL,
    kind           TEXT NOT NULL,                      -- e.g. "SpreadingActivation"
    payload_json   JSONB NOT NULL,
    emitted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thought_id, step_idx)
);

CREATE INDEX thinking_traces_akh_started_idx
    ON akhomed.thinking_traces (akh_id, started_at DESC);

CREATE INDEX thinking_steps_thought_emitted_idx
    ON akhomed.thinking_steps (thought_id, emitted_at);
```

Deliverables:
- WAL-flushed step emission (every step is a Postgres write; throughput is bounded by the same WAL infrastructure as Phase 38b mutation log).
- Resume-from-snapshot: on Akh wake, restore in-flight `ThoughtState`s from the most recent snapshot + replay tail of `thinking_steps`.
- New `DerivationKind::ThoughtStep { thought_id, step_idx }` for cross-referencing facts derived during deliberation.
- Tests: kill-9 mid-loop, restart, verify the trace continues from the last emitted step.

End-of-41d state: thinking is durable and relocatable.

### 41e — VOC-driven adaptive budget (~400 lines)

**Priority**: Medium
**Depends on**: 41a, 41c, Phase 11g (already shipped)

Files:
- New: `src/thinking/voc_budget.rs`
- Modify: `src/agent/voc.rs` (expose marginal-value query for thinking workloads)

Deliverables:
- `VocBudgetController` consults Phase 11g `compute_voc()` after every N steps.
- Halts when marginal value < threshold (Apple-illusion-of-thinking fix: stop when more thinking won't help).
- Extends budget when marginal value remains high and step budget is near exhaustion (within configured global cap).
- Tests: synthetic VOC curves trigger expected halt/extend decisions.

End-of-41e state: budget is economic, not arbitrary.

### 41f — Benchmark harness + scaling-law measurement (~800 lines)

**Priority**: High (this is the empirical case for the phase)
**Depends on**: 41a–e

Files:
- New: `benches/thinking_scaling.rs`
- New: `crates/akh-bench-thinking/` (separate crate to keep heavy deps out of the main build)
- Datasets in `data/bench/thinking/`

Benchmarks:
1. **Multi-hop KG queries** (synthetic, generated from the corpus) — depths 2, 4, 8, 16.
2. **Analogical retrieval (A:B :: C:?)** — using the existing VSA encoding from ADR 002.
3. **Raven's Progressive Matrices subset** — to compare against the [NVSA 87.7% baseline](https://research.ibm.com/projects/neuro-vector-symbolic-architecture).
4. **Bootstrap-domain Q&A** — questions generated from each Akh's own knowledge.

For each benchmark, run with `max_steps ∈ {100, 300, 1000, 3000, 10000}` and plot:
- Accuracy vs. step budget (the scaling law).
- Accuracy vs. wall-clock.
- Accuracy vs. VOC spent.
- Halt-reason distribution.

Acceptance criteria:
- Loop runs end-to-end on all four benchmarks.
- Plots are emitted as PNG + raw CSV in `target/bench-results/thinking/`.
- A scaling curve exists (positive slope) on at least two of the four benchmarks.
- A scaling curve does NOT exist on at least one benchmark — informative negative result, drives Phase 41g.

End-of-41f state: an empirical, falsifiable artefact for the "frontier architecture" thesis.

### 41g — Findings ADR + plan revision (~documentation only)

**Priority**: High (closes the loop)
**Depends on**: 41f

Deliverables:
- Write ADR 053 — *Thinking Loop Findings* — recording the actual scaling curves observed.
- If the curves are good: extension plan for Phase 42 (deeper integration of LLM-as-oracle via Phase 26d, multi-Akh thought sharing via Phase 36).
- If the curves are bad: post-mortem analysis identifying which primitives don't compose, which need step-wise APIs, and what an alternative rotation looks like.
- Update `docs/ai/architecture.md` with the Thinking Engine module.

## Estimated effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 41a — Loop skeleton | ~900 | Medium | — |
| 41b — Scheduler | ~500 | Low | 41a |
| 41c — Verifier | ~700 | Medium | 41a, Phase 17 |
| 41d — Persistence | ~600 | Medium-high | 41a, Phase 38a–c |
| 41e — VOC budget | ~400 | Low | 41a, 41c, Phase 11g |
| 41f — Benchmark harness | ~800 | Medium | 41a–e |
| 41g — Findings ADR | ~docs | Low | 41f |
| **Total** | **~3,900** | | |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scaling curve is flat — primitives don't compose well in a unified loop | Phase 41g is designed to record this honestly. The plan is structured so even a negative result is informative. Iterate on rotation in a follow-up phase. |
| Persistence overhead dominates per-step cost | 41d uses the same WAL infrastructure as Phase 38b, which is already designed for high-frequency mutations. If overhead is real, batch step flushes (configurable). |
| Verifier is expensive at every step | Make the verifier hierarchical: cheap checks (convergence) every step, expensive checks (full Dempster combination) every K steps. |
| The Thinking Loop becomes the only path and ad-hoc primitives atrophy | The Loop is *additive*. Existing call sites for `infer()`, `simplify_expression()`, `mcts.plan()` continue to work. The Loop is one more API, not a replacement. |
| Dependency on Phase 38 delays Phase 41 indefinitely | 41a–c can ship without persistence (in-memory `ThoughtState`). 41d is the only sub-phase that hard-depends on Phase 38. Sequence accordingly. |

## Open questions

1. **HyperVec evolution policy.** How should `focus_vec` evolve during the loop? Bundle every emitted concept? Bind by step type? The first iteration uses majority-vote bundle of last-N concepts; revisit after 41f data.
2. **Hypothesis pruning.** Superposition has finite capacity (~7 fillers in 10K bits before similarity collapse). When and how do we prune low-confidence hypotheses? Initial heuristic: prune below `Bel < 0.05`.
3. **Cross-thought dependencies.** Can two `ThoughtState`s reference each other's intermediate steps? Intuition: yes, via `DerivationKind::ThoughtStep` cross-references. Defer concrete API to 41g if needed.
4. **Anytime semantics.** The user might want to ask "give me your current best answer" while the loop is still running. Initial design: read-only access to `state.hypotheses` is safe; expose via the engine API in 41a.
5. **Reasoner Dispatch (Phase 9f) interaction.** Should the Thinking Loop *call* Reasoner Dispatch as one of its primitives, or operate at the same level? Tentative: at the same level. Dispatch is an alternative top-level entry point optimised for "one shot answer", the Loop is the deliberation surface.

## References

- ADR: [052-unified-thinking-loop](../decisions/052-unified-thinking-loop.md)
- Research synthesis: [docs/ai/research/2026-04-27-thinking-in-llms-vs-vsa.md](../research/2026-04-27-thinking-in-llms-vs-vsa.md)
- Persistence backbone: [Phase 38 plan](2026-04-25-phase38-relocatable-akh.md)
- LLM-VSA bridge (peripheral oracle path): [Phase 26 plan](2026-03-26-phase26-llm-vsa-deep-integration.md), [ADR 037](../decisions/037-llm-vsa-deep-integration.md)
- Existing primitives: [Phase 16 plan (MCTS)](2026-02-22-phase16-predictive-planning.md), [Phase 17 plan (evidence)](2026-02-22-phase17-evidence-theory.md), [Phase 18 plan (source reliability)](2026-02-22-phase18-source-reliability.md), [Phase 11 plan (VOC, metacognition)](2026-02-17-phase11-autonomous-tasks.md), [Phase 9 plan (Cyc-inspired HOL)](2026-02-17-phase9-cyc-inspired.md)
- External: [Attention as Binding (Dec 2025)](https://arxiv.org/html/2512.14709), [Hyperdimensional Probe (Sept 2025)](https://arxiv.org/html/2509.25045v1), [COCONUT (Meta 2024)](https://arxiv.org/abs/2412.06769), [Apple Illusion of Thinking (NeurIPS 2025)](https://arxiv.org/abs/2506.06941), [Snell et al. test-time compute (DeepMind 2024)](https://arxiv.org/html/2408.03314v1), [DeepSeek R1 (Nature 2025)](https://arxiv.org/abs/2501.12948), [NVSA RPM 87.7%](https://research.ibm.com/projects/neuro-vector-symbolic-architecture).
