# ADR 052 — Unified Thinking Loop (Phase 41)

> Date: 2026-04-27
> Status: Proposed
> Phase: 41 (The Thinking Engine)
> Depends on: Phase 38 (Relocatable Akh — WAL + snapshot), Phase 17 (evidence theory), Phase 11g (VOC), existing reasoning primitives (Phases 9, 15, 16, 18, 19, 20)
> Related research: [docs/ai/research/2026-04-27-thinking-in-llms-vs-vsa.md](../research/2026-04-27-thinking-in-llms-vs-vsa.md)
> Anima ticket: `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9`

## Context

Frontier reasoning LLMs (o1/o3, Claude with extended thinking, DeepSeek R1, Gemini 2.5 Deep Think) demonstrate a *test-time compute scaling law*: accuracy increases logarithmically with thinking budget. The mechanism is autoregressive token generation shaped by RL on verifiable outcomes; the resulting "thinking" tokens perform implicit search, backtracking, and self-verification.

Recent mechanistic-interpretability work clarifies what this is *actually doing*:

- The [*Attention as Binding* paper (Dec 2025)](https://arxiv.org/html/2512.14709) argues transformer attention is a **soft, approximate VSA**: keys/queries as roles, values as fillers, QK-OV circuit as binding, residual-stream accumulation as superposition. Failure modes match VSA failure modes.
- The [Hyperdimensional Probe (Sept 2025)](https://arxiv.org/html/2509.25045v1) recovers concept-driven patterns from LLM residual streams (355M to 109B) using VSA hypervector algebra.
- Latent-reasoning research ([COCONUT](https://arxiv.org/abs/2412.06769), [Quiet-STaR](https://arxiv.org/abs/2403.09629), recurrent-depth transformers) shows that token-level CoT is squeezing reasoning through a 15-bit straw against a ~40,960-bit hidden-state channel.
- [Filler-token results](https://arxiv.org/abs/2404.15758) show CoT content is often irrelevant — what matters is the additional compute, not the words.
- [Apple's *Illusion of Thinking*](https://arxiv.org/abs/2506.06941) (NeurIPS 2025) shows reasoning models *give up* on hard problems before exhausting their token budget, revealing the absence of an internal value-of-computation control loop.
- [*Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410) shows the verbal CoT diverges from the actual circuit — reasoning traces are not faithful by construction.

The implication for akh-medu: **frontier LLMs are noisy, expensive, language-tied implementations of an approximate VSA.** Akh-medu has an *exact* VSA (`src/vsa/`) and every reasoning primitive an LLM amortises into weights — argumentation, MCTS, evidence theory, counterfactual, event calculus, epistemic logic, active inference. What is missing is **integration**: a single loop with a budget that drives all primitives and emits a faithful trace.

## Decision

Add **Phase 41 — The Thinking Engine**: a unified `ThinkingLoop` that holds a `ThoughtState` (HyperVec + superposed hypotheses + open goals + trace + budget), iterates a rotation of existing reasoning primitives, scores intermediate states with a structural verifier, halts on convergence / budget exhaustion / verifier sufficiency, and emits a `Vec<ThoughtStep>` reasoning trace as first-class persisted state.

Architecture sketch:

```
┌─ ThinkingLoop::deliberate(goal, budget) ─────────────────────────┐
│                                                                   │
│  ThoughtState {                                                   │
│    focus_vec:    HyperVec   // current cognitive "residual stream"│
│    hypotheses:   Vec<(HyperVec, BeliefInterval)>   // superposed  │
│    open_goals:   Vec<SymbolId>                                    │
│    trace:        Vec<ThoughtStep>                                 │
│    budget:       ThinkingBudget { steps, wall_ms, voc_credits }   │
│  }                                                                │
│                                                                   │
│  loop until converged | budget exhausted | verifier_sufficient: │
│    spreading_activation_step()    // 1 frontier ring              │
│    egraph_saturate_step()         // N rewrites                   │
│    mcts_rollout()                 // 1 trajectory                 │
│    argumentation_round()          // pro/con on top hyp           │
│    counterfactual_probe()         // optional fork                │
│    evidence_combine()             // DS update                    │
│    superposition_merge()          // bundle results               │
│    verifier_score()               // PRM-equivalent (structural)  │
│    thoughtstep_emit()             // trace record + WAL flush     │
│                                                                   │
│  return Conclusion { answer, trace, confidence_interval }         │
└──────────────────────────────────────────────────────────────────┘
```

### Key design choices

1. **Thought State is a HyperVec, not a token sequence.** The bandwidth advantage of latent reasoning, but exact rather than gradient-discovered.
2. **Hypotheses live in superposition.** Bundle-encoded; `O(1)` cost where Tree-of-Thoughts pays `O(N)` token cost.
3. **Verifier is structural, not learned.** Dempster-Shafer belief-interval narrowing + Phase 9l contradiction detection + Phase 18 source reliability. Cannot lie about the reasoning because the reasoning *is* the trace.
4. **Budget is economic, not token-count.** Wired to Phase 11g VOC: marginal-value scoring with diminishing-returns detection. Adaptive halting — keeps thinking when value justifies, stops when it doesn't.
5. **Provenance + Phase 38 → persistent thinking.** `ThoughtState` snapshots through the Postgres WAL. A thinking session can pause, persist, relocate to another host, and resume. No LLM has this capability.
6. **LLM as peripheral.** When surface-form fluency or massive prior is needed, call Candle T5 (Phase 26b) and project hidden states through the neural→VSA bridge (Phase 26d). LLM becomes a noisy oracle attached to the symbolic core, never the core itself.

### Why this is not duplicating Phase 9f or Phase 16b

- **Phase 9f Reasoner Dispatch** is bid-based competitive selection: at each call, *one* reasoner wins and runs. The Thinking Loop runs *all* relevant primitives interleaved with superposition merging.
- **Phase 16b MCTS** explores *action sequences* for planning. The Thinking Loop explores *knowledge sequences* for inference. MCTS becomes one of the primitives the loop drives, not a substitute.

### Demonstrating frontier status: a scaling-law benchmark

To call akh-medu a frontier architecture, demonstrate the same logarithmic accuracy-vs-compute curve LLMs show. Concretely:

- Pick a benchmark: ARC-AGI, Raven's Progressive Matrices (where IBM's [NVSA hit 87.7%](https://research.ibm.com/projects/neuro-vector-symbolic-architecture)), or a custom KG-reasoning benchmark drawn from the corpus.
- Run with `ThinkingBudget.steps ∈ {100, 300, 1000, 3000, 10000}`.
- Plot accuracy vs. wall-clock and accuracy vs. steps.

A scaling curve — even sub-logarithmic — is a publishable result. A flat curve is informative: it tells us the primitive rotation is wrong and needs revision.

## Consequences

### Positive

- **A single dial labelled "think harder"** that scales accuracy with compute, on a substrate where every step is auditable.
- **Faithful reasoning by construction**: the trace *is* the computation, not a verbal post-hoc rationalisation. Solves the [LLM faithfulness problem](https://arxiv.org/abs/2505.05410) by architecture rather than training.
- **Persistent, relocatable, multi-Akh-shareable thinking**: hard problems can be worked on for days across hosts and Akhs.
- **Compositional cognition**: thoughts bind and unbind. Analogical reasoning by VSA arithmetic on `ThoughtState` HyperVecs.
- **An empirical thesis to validate**: the scaling-law benchmark is a falsifiable bet. We learn either way.

### Negative

- **Implementation surface is large.** Six primitives, a budget controller, a verifier, persistence, and a benchmark harness. Estimated 4,000–6,000 lines (see Phase 41 plan).
- **Depends on Phase 38** for the persistence story to hold. Phase 41 implementation should follow Phase 38a–c at minimum.
- **The benchmark may show poor scaling** — primitive rotation might be wrong, or the existing primitives may not compose well. We're betting on integration; integration may need iteration.
- **Risk of premature unification.** Some primitives may need refactoring to expose step-wise APIs (`spreading_activation_step()` rather than `infer()`-to-completion). This could ripple back into existing modules.

### Neutral

- New `DerivationKind::ThoughtStep` (next free tag, ~77 if Phase 25 used 72-76).
- New top-level module: `src/agent/thinking/` (or `src/thinking/` — TBD in plan).
- New Postgres table: `akhomed.thinking_traces` (per-Akh, indexed by `thought_id`, `started_at`).
- The existing ad-hoc invocation paths for spreading activation / e-graph / MCTS remain valid; the Thinking Loop is additive, not a replacement.

## Rejected Alternatives

### "Just train a CoT model"
A small T5 with CoT supervision is what Phases 32–34 already explore for surface generation. It does not solve the architectural question — it builds another approximate VSA on top of a substrate that already has an exact one. Wrong layer of the stack.

### "Use Phase 9f Reasoner Dispatch as the loop"
Reasoner Dispatch is *competitive*: pick one reasoner, run, return. The frontier observation from latent-reasoning research is that thinking benefits from *parallel* hypotheses in superposition with structural reconciliation. Reasoner Dispatch is a complement to the Thinking Loop, not a substitute — the Loop can use it as one of its sub-steps.

### "Make MCTS the universal loop"
MCTS searches action trees. Inference often has no action structure — the question is *what is true*, not *what to do*. A purely MCTS-shaped loop forces every inference into a planning frame, which is the wrong abstraction.

### "Defer until Phase 38 ships"
We could delay this ADR until 38 is in place. Rejected because (a) Phase 41a–b can be implemented without persistence, with persistence layered in at 41d once 38 is ready; (b) writing the ADR now anchors the design and lets the empirical bet (scaling-law benchmark) be evaluated as soon as the loop skeleton exists.

## References

- [docs/ai/research/2026-04-27-thinking-in-llms-vs-vsa.md](../research/2026-04-27-thinking-in-llms-vs-vsa.md) — full synthesis underpinning this ADR.
- [docs/ai/plans/2026-04-25-phase38-relocatable-akh.md](../plans/2026-04-25-phase38-relocatable-akh.md) — persistence backbone.
- [ADR 037 — LLM-VSA Deep Integration](037-llm-vsa-deep-integration.md) — Phase 26 neural↔symbolic bridge.
- [ADR 035 — Evidence Theory](035-evidence-theory.md) — verifier substrate (Phase 17).
- [ADR 034 — MCTS Planning](034-mcts-planning.md) — one of the primitives the loop drives.
- [ADR 002 — Code Generation Research](002-code-generation-research.md) — VSA analogical-arithmetic precedent.
- External: [Attention as Binding (Dec 2025)](https://arxiv.org/html/2512.14709), [Hyperdimensional Probe (Sept 2025)](https://arxiv.org/html/2509.25045v1), [COCONUT (Meta 2024)](https://arxiv.org/abs/2412.06769), [Apple Illusion of Thinking (NeurIPS 2025)](https://arxiv.org/abs/2506.06941), [Snell et al. test-time compute (DeepMind 2024)](https://arxiv.org/html/2408.03314v1).
