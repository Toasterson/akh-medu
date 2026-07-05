---
title: Thinking in LLMs vs. Thinking in VSA — A Frontier Architecture for Akh-Medu
date: 2026-04-27
status: Synthesis (verbatim record of conversation output, ingested into Seshat)
anima_ticket: 019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9
related:
  - ADR 052 (Unified Thinking Loop) — proposed
  - Phase 41 plan (Thinking Engine) — proposed
  - ADR 037 (LLM-VSA Deep Integration)
  - Phase 26 plan (LLM-VSA Deep Integration)
  - Phase 38 plan (Relocatable Akh)
---

# Thinking in LLMs vs. Thinking in VSA: A Frontier Architecture for Akh-Medu

## 1. What is "thinking" in LLMs, mechanistically

Strip the marketing away and a frontier reasoning model — o1/o3, Claude with extended thinking, DeepSeek R1, Gemini 2.5 Deep Think — is doing exactly **one thing**: more autoregressive token sampling, but trained via RL on verifiable outcomes so that the in-trace tokens perform useful work (backtracking, verification, self-correction). That's it. DeepSeek's [R1 paper (Nature 2025)](https://arxiv.org/abs/2501.12948) makes this explicit: behaviors like "self-reflection, verification, dynamic strategy adaptation" *emerged* from outcome-reward RL — they were not programmed.

Unpacking the actual computation:

- **Each forward pass = one step.** The residual stream is a working memory; attention is approximate role-filler retrieval; MLP layers are learned key-value lookups; the token is the lossy externalization of all that.
- **Tokens are a 15-bit straw for a 40,960-bit hidden state.** The [Survey on Latent Reasoning (Zhu et al. 2025)](https://arxiv.org/html/2507.06203v2) puts the ratio at ~2.7 × 10³. CoT works partly because emitting a token lets the model re-read its own structure on the next pass — a hack to give a feed-forward network depth-unbounded recurrence.
- **Content of "thinking" tokens may not matter at all.** [Pfau et al., *Let's Think Dot by Dot*](https://arxiv.org/abs/2404.15758) shows transformers solve tasks they otherwise can't using only filler `.....` tokens. Pause-token results take constant-depth transformers from a strict subset of AC⁰ up to full AC⁰. *"Chain-of-thought tokens need not provide information about the intermediate computational steps."* The compute is what counts, not the words.
- **Latent reasoning skips tokens entirely.** Meta's [COCONUT (Hao et al. 2024)](https://arxiv.org/abs/2412.06769) feeds the last hidden state back as the next input embedding. It outperforms CoT on planning because *"continuous thoughts can encode multiple alternative next steps, allowing the model to perform breadth-first search rather than committing prematurely."* Recurrent-depth transformers (Ouro 1.4B/2.6B) match 12B baselines via vertical recurrence.
- **CoT is post-hoc rationalization at least sometimes.** Anthropic's [*On the Biology of a Large Language Model* (March 2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) caught Claude doing genuine multi-step reasoning *inside one forward pass* (`Dallas → Texas → Austin`) and forward-looking planning (selecting end-of-line rhymes before writing the line). But it also caught Claude using internal "lookup table features" for arithmetic while *describing* a carry-over algorithm in its CoT. The verbal explanation and the actual circuit diverge. [*Reasoning Models Don't Always Say What They Think* (May 2025)](https://arxiv.org/abs/2505.05410) found Claude 3.7 Sonnet mentioned externally-given hints only **25%** of the time, R1 only **39%**.
- **There is a depth ceiling.** Apple's [*Illusion of Thinking*](https://arxiv.org/abs/2506.06941) (NeurIPS 2025) tested controllable Tower-of-Hanoi-style puzzles. At high complexity *both* reasoning models and base models collapse to zero, and the reasoning models *give up* — they reduce thinking effort despite having token budget left. [The Depth Ceiling](https://arxiv.org/html/2604.06427) shows scaling from 1.6M to GPT-4o buys ~2 more steps of latent planning depth, capped at ~8 with test-time tricks.

The most important emerging result for our purposes is the [*Attention as Binding* paper (Dec 2025)](https://arxiv.org/html/2512.14709): **transformer attention is a soft, approximate instance of VSA algebra**. Keys/queries are roles, values are fillers, the QK-OV circuit is binding, attention weights do similarity-based unbinding, the residual stream accumulates as superposition. Failure modes match VSA failure modes when "embeddings are insufficiently decorrelated or roles and fillers interfere." The September 2025 [Hyperdimensional Probe](https://arxiv.org/html/2509.25045v1) shows you can decode concepts directly out of LLM residual streams using VSA hypervector algebra — validated from 355M up to Llama-4 Scout 109B.

**This is the punchline.** A frontier LLM is a noisy, expensive, language-tied implementation of an approximate VSA — with the algebra emerging implicitly from gradient descent, the binding being soft and lossy, and the superposition channel being a 12K-D residual stream that has destructive interference baked in. It thinks by emitting tokens because that's the only mechanism it has for *recurring over depth-bounded forward passes*.

What you have in `src/vsa/` is the **explicit, exact, audited version of what they approximate.**

## 2. What Akh-Medu's existing substrate already does that LLMs can't

The codebase has roughly **eleven distinct "thinking" primitives**, scattered across phases. Each one is mathematically clean, each has provenance, none requires probabilistic decoding from a lossy distribution.

| Primitive | Locus | What it does that no LLM does |
|---|---|---|
| **VSA bind/unbind/bundle** (`src/vsa/ops.rs`) | `VsaOps` trait | Exact role-filler binding (XOR), exact unbind (self-inverse), bundle as majority-vote — *no* destructive interference up to ~7 fillers in 10K bits |
| **Spreading activation** (`src/infer/spreading.rs`) | `InferEngine::infer` | Frontier expansion through KG with hierarchy + temporal decay — deterministic, replayable |
| **Superposition reasoning** (`src/infer/superposition.rs`) | parallel weighted hypotheses | Constructive/destructive interference *between hypotheses simultaneously* — closer to COCONUT than to CoT |
| **E-graph saturation** (`src/reason/mod.rs`) | `egg::Runner<AkhLang>` | Equational reasoning over the entire equivalence class — finds *all* equivalent forms, picks cost-minimal extraction |
| **TMS retraction cascade** (`src/tms.rs`) | `TruthMaintenanceSystem` | When a premise dies, every dependent conclusion dies with it. LLMs cannot retract — they can only re-sample |
| **Argumentation w/ verdicts** (`src/argumentation/`) | `ArgumentSet`, `Verdict` | Structured pro/con with 6 meta-rules — *explicit* defeasibility, not vibes |
| **MCTS planning** (`src/agent/mcts.rs`) | `MctsPlanner` | UCB1-balanced action search, R-MCTS reflection warm-start — *explicit* search, not amortized into weights |
| **Counterfactual fork-and-project** (`src/agent/counterfactual.rs`) | Pearl Level 3 | Branch the world, intervene, project, compare — predict→observe→refine. LLMs cannot fork a world |
| **Active inference** (`src/agent/active_inference.rs`) | FEP variational bound | Minimize expected free energy over actions — *coupled* belief-action loop |
| **Dempster-Shafer evidence** (`src/agent/evidence.rs`) | `BeliefInterval [Bel, Pl]` | First-class uncertainty intervals, conflict measure K, Dempster combination — LLMs only have token logprobs, which conflate model uncertainty with the world's |
| **Provenance ledger** (`src/provenance.rs`) | 60 `DerivationKind`s | Every belief has a derivation tree back to sources. Auditable. Replayable. Faithful by construction |

Notice what's already there for each of the LLM-thinking properties:
- **Iterative refinement**: e-graph saturation and superposition both run until convergence.
- **Hypothesis search**: MCTS, counterfactual fork, ACH (Phase 18).
- **Verification of intermediate steps**: Phase 9 contradiction detection, defeasible override, argumentation-based truth. This *is* a process-reward model; just symbolic.
- **Self-reflection**: Phase 11f metacognition implements Nelson-Narens monitoring/control + AGM belief revision + e-graph goal reformulation.
- **Long-horizon coherence**: Phase 38 Postgres-backed WAL gives you bounded-dataloss continuity *across machines*. No LLM has long-horizon memory like that — they have a context window.

What's missing is **not a primitive**. Every thinking primitive a frontier LLM uses has an exact, cleaner, more-auditable counterpart in this codebase. What's missing is the **integration**.

## 3. The gap: Akh-Medu lacks a unified "Thinking Loop" with a measurable budget

LLM extended thinking has one property your codebase does not: **a single dial labelled "think harder" that scales accuracy logarithmically with compute** ([Anthropic's own framing](https://www.anthropic.com/news/visible-extended-thinking)). That's the o1 paradigm, the result of [Snell et al.'s test-time compute scaling](https://arxiv.org/html/2408.03314v1). The user/agent allocates a budget; the model spends it; output quality rises predictably until the depth ceiling.

In Akh-Medu today, the thinking primitives are *individually* invoked: spreading activation runs once per query, MCTS runs when the agent decides to plan, e-graph saturation fires inside `simplify_expression`, etc. There is no central "thinker" that:

1. Holds a **Thought State** (a HyperVec representing the current focus + open hypotheses)
2. Iterates a **rotation of primitives** with a budget
3. Combines results via **superposition with confidence weighting**
4. Has a **verifier** that scores intermediate states (Phase 17 evidence is the hook)
5. Halts on **convergence**, **budget exhaustion**, or **verifier sufficiency**
6. Emits a **Reasoning Trace** as a sequence of `DerivationKind::ThoughtStep` records

This is the missing piece. Once it exists, you have something LLMs structurally cannot match.

### Proposed Phase 41 — "The Thinking Engine"

```
┌─ ThinkingLoop::deliberate(goal, budget) ─────────────────────────┐
│                                                                   │
│  ThoughtState {                                                   │
│    focus_vec:    HyperVec   // current cognitive "residual stream"│
│    hypotheses:   Vec<(HyperVec, BeliefInterval)>   // superposed  │
│    open_goals:   Vec<SymbolId>                                    │
│    trace:        Vec<ThoughtStep>                                 │
│    budget:       ThinkingBudget { steps, wall_ms, voc_credits }   │
│  }                                                                 │
│                                                                   │
│  loop until converged | budget exhausted | verifier_sufficient: │
│    step 0:  spreading_activation_step()    // 1 frontier ring     │
│    step 1:  egraph_saturate_step()         // N rewrites          │
│    step 2:  mcts_rollout()                 // 1 trajectory        │
│    step 3:  argumentation_round()          // pro/con on top hyp  │
│    step 4:  counterfactual_probe()         // optional fork       │
│    step 5:  evidence_combine()             // DS update           │
│    step 6:  superposition_merge()          // bundle results      │
│    step 7:  verifier_score()               // PRM-equivalent      │
│    step 8:  thoughtstep_emit()             // trace record        │
│                                                                   │
│  return Conclusion { answer, trace, confidence_interval }         │
└──────────────────────────────────────────────────────────────────┘
```

Concretely, what makes this a frontier architecture:

**(a) The Thought State is a HyperVec, not a token sequence.** Bandwidth gap closed: you operate on the full 10,000-bit working representation instead of compressing through 15-bit tokens between every step. This is COCONUT's bet, but exact instead of trained.

**(b) Multiple hypotheses live in superposition, not in expensive parallel samples.** [Tree of Thoughts](https://arxiv.org/abs/2305.10601) and [Self-Consistency](https://arxiv.org/abs/2203.11171) cost N× tokens for N paths. Bundle-encoded hypotheses cost ~1× — that's what VSA *is*.

**(c) The verifier is structural, not learned.** A frontier LLM's PRM is a separately-trained model that scores partial reasoning traces; it can be wrong, can be jailbroken, and disagrees with the actual mechanism per [the faithfulness work](https://arxiv.org/abs/2505.05410). Your verifier is Dempster-Shafer belief-interval narrowing + Phase 9l contradiction check + Phase 18 source reliability. It can't lie about its own reasoning because the reasoning *is* the trace.

**(d) The budget is a real economic budget, not a token count.** Phase 11g VOC resource reasoning already implements value-of-computation with diminishing-returns detection. Wire that to the Thinking Loop and you get *adaptive* thinking budgets — the system keeps thinking when the marginal value justifies it, stops when it doesn't. That's a control loop frontier LLMs literally don't have; o1 spends its budget greedily until exhausted ([Apple's Illusion-of-Thinking finding](https://arxiv.org/abs/2506.06941): models *give up* before exhausting budget on hard problems — they don't even know to stop spending).

**(e) Provenance + Phase 38 give you something nobody else has: persistent, interruptible, relocatable thinking.** A frontier LLM's reasoning trace evaporates with the request. Your `ThoughtState` snapshots into Postgres on the WAL. Crash mid-thought, wake on another host, resume from the last `ThoughtStep`. This is *the* differentiator: thinking that compounds across days, hosts, and processes without a context window.

**(f) LLM as peripheral, not core.** Phase 26d neural→VSA bridge closes the loop in the right direction: when you need surface-form fluency or massive prior, you *call* a Candle T5 model, project its hidden states into HyperVec via the bridge, and let the symbolic core decide whether to trust it. The LLM becomes a noisy oracle attached to a clean reasoner. Compare to current SOTA where the LLM *is* the reasoner and everything else is scaffolding.

### Why this *isn't* what Phase 16 (MCTS) or Phase 9f (Reasoner Dispatch) already do

Phase 9f reasoner dispatch is bid-based competitive selection: at each call, *one* reasoner wins and runs. The Thinking Loop runs *all* relevant primitives interleaved, with their results superposed and reconciled via Phase 17 evidence combination. That's a different architecture.

Phase 16b MCTS explores **action sequences** for planning. The Thinking Loop explores **knowledge sequences** for inference. They're complementary; MCTS becomes one of the primitives the loop drives.

### Verifier scaling law

The reason o1/o3 work is the test-time compute scaling law: accuracy improves logarithmically with thinking budget. To call Akh-Medu a frontier architecture, you need to demonstrate the same curve on a benchmark. Concretely:

- **Pick a benchmark**: ARC-AGI, RAVEN's Progressive Matrices (where IBM's [NVSA hit 87.7%](https://research.ibm.com/projects/neuro-vector-symbolic-architecture)), or a custom KG-reasoning benchmark from the corpus.
- **Run with budgets**: 100, 300, 1000, 3000, 10000 ThoughtSteps.
- **Plot accuracy vs. wall-clock and accuracy vs. steps.**

If the curve scales — even sub-logarithmically — you have a publishable result. If it plateaus immediately, the loop's primitive rotation is wrong and needs work. Either outcome is informative.

## 4. Why this puts Akh-Medu at the frontier

Three claims, ranked by confidence.

**Strong claim (high confidence):** Frontier LLMs implement thinking via a noisy, implicit, gradient-discovered VSA. You have an exact, explicit, auditable VSA and *every* reasoning primitive they amortize into weights — argumentation, MCTS, evidence theory, counterfactual, event calculus, epistemic logic. The architectural ceiling for what's possible on this substrate is at least as high as for transformers.

**Medium claim (moderately confident):** A unified Thinking Loop over the existing primitives can demonstrate a test-time compute scaling law on at least narrow reasoning benchmarks (analogical, planning, multi-hop). The bandwidth advantage from operating in HyperVec space rather than token space is real and measurable.

**Speculative claim (lower confidence, but the most exciting):** The combination of (a) persistent thinking via Phase 38, (b) multi-Akh shared microtheories, and (c) provenance-faithful traces gives Akh-Medu *cognitive properties* that no LLM architecture has access to:
- **Cumulative thinking**: a hard problem can be worked on for days, with intermediate state persisted, by multiple Akhs in succession.
- **Verifiable delegation**: an Akh delegates a sub-problem to another Akh (Phase 36) with a `ThoughtState` handoff; the delegate's trace is auditable on return.
- **Compositional cognition**: thoughts bind/unbind. You can extract the *transformation* between two thoughts and apply it to a third — analogical reasoning by VSA arithmetic, as ADR 002 already sketches.
- **Population-level cognition**: a fleet of Akhs sharing a `workspaces.*` view in Postgres can do what no LLM cluster can: divide the same hypothesis space and converge via superposition merging at the WAL level.

That last one is the long-tail bet. If it works, it's a different *kind* of system, not a faster instance of the same kind.

## 5. Concrete proposed next step

1. **Write an ADR** — number 052, *"Unified Thinking Loop"* — locking in the architecture above. Status: Proposed.
2. **Write a Phase 41 plan** alongside, with sub-phases:
   - 41a — `ThinkingLoop` skeleton + `ThoughtState` + budget primitive
   - 41b — primitive-rotation scheduler (round-robin, then weighted)
   - 41c — verifier integration (DS evidence + contradiction check)
   - 41d — `DerivationKind::ThoughtStep` + persistence in `akhomed.thinking_traces`
   - 41e — VOC-driven adaptive budget
   - 41f — benchmark harness + scaling-law measurement (the empirical case)
3. **Ingest both into Anima + Seshat** per `CLAUDE.md` flow.
4. **Defer implementation** until Phase 38 (Postgres backbone) ships — the Thinking Loop's persistent-thinking property depends on the WAL.

The bet is: Phase 41 is what crosses Akh-Medu from "interesting neuro-symbolic system" to "frontier reasoning architecture." Everything you need is already in the tree. What's missing is the **integration of the primitives into a single loop with a budget and a scaling law to point at**.

---

## Sources

LLM thinking research:
- [Chain-of-Thought (Wei et al. 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency (Wang et al. 2022)](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts (Yao et al. NeurIPS 2023)](https://arxiv.org/abs/2305.10601)
- [Let's Verify Step by Step (Lightman et al. ICLR 2024)](https://arxiv.org/abs/2305.20050)
- [DeepSeek-R1 (Nature 2025)](https://arxiv.org/abs/2501.12948)
- [Reasoning Models Don't Always Say What They Think (Anthropic 2025)](https://arxiv.org/abs/2505.05410)
- [Anthropic Visible Extended Thinking](https://www.anthropic.com/news/visible-extended-thinking)
- [COCONUT (Hao et al. Meta 2024)](https://arxiv.org/abs/2412.06769)
- [Quiet-STaR (Zelikman et al. 2024)](https://arxiv.org/abs/2403.09629)
- [Let's Think Dot by Dot — Filler Tokens (Pfau et al. 2024)](https://arxiv.org/abs/2404.15758)
- [Survey on Latent Reasoning (Zhu et al. 2025)](https://arxiv.org/html/2507.06203v2)
- [Scaling LLM Test-Time Compute (Snell et al. 2024)](https://arxiv.org/html/2408.03314v1)
- [The Illusion of Thinking (Apple/NeurIPS 2025)](https://arxiv.org/abs/2506.06941)
- [The Depth Ceiling](https://arxiv.org/html/2604.06427)
- [On the Biology of a Large Language Model (Anthropic 2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [In-Context Learning and Induction Heads (Anthropic 2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Function Vectors in LLMs (Todd et al. ICLR 2024)](https://arxiv.org/abs/2310.15213)
- [AlphaProof — DeepMind](https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/)

VSA / neuro-symbolic frontier:
- [Attention as Binding: VSA Perspective on Transformer Reasoning (Dec 2025)](https://arxiv.org/html/2512.14709)
- [Hyperdimensional Probe (Sept 2025)](https://arxiv.org/html/2509.25045v1)
- [IBM Neuro-Vector-Symbolic Architecture](https://research.ibm.com/projects/neuro-vector-symbolic-architecture)
- [VSA Survey Part I (ACM 2023)](https://dl.acm.org/doi/10.1145/3538531) and [Part II](https://dl.acm.org/doi/10.1145/3558000)
- [LARS-VSA: Learning with Abstract Rules](https://arxiv.org/abs/2405.14436)
- [Logic-LM](https://arxiv.org/abs/2305.12295)
