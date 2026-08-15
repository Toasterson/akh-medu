# Phase 51 P3-RL — Step-Level Rewards, VSA Geometric Criteria, and the Axis-A Depth Wall

- **Status:** Plan (not built). Numbering: ADR **096** provisional — `docs/ai/decisions/` already carries two `088-*` files from the worktree/develop split noted in ADR 087 §Status; reconcile on merge.
- **Date:** 2026-08-15
- **Decision record:** [ADR 096](../decisions/096-step-level-rewards-and-recurrent-depth.md)
- **Parent:** [Phase 51 plan](2026-07-05-phase51-grug-brain-baby-llm.md) (P3 → RL stage, ADR 087 §4 curriculum); [P3-R corpus plan](2026-07-11-phase51-p3r-reasoning-corpus.md)
- **Measured basis:** [P3-R results](../2026-07-11-phase51-p3r-results.md), [GWT emergence probes](../2026-07-11-phase51-gwt-emergence-probes.md)
- **Language / scorer:** [ADR 092](../decisions/092-high-grug-v1-reasoning-register.md) (v1 register, D6 "engine replay is the scorer"), [ADR 093](../decisions/093-dual-process-trace-handoff.md)
- **Anima work item:** [`01a004e0-8d5a-7e50-8e82-42b5062d1337`](https://anima.wegmueller.it/) (project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, "Akh-Medu") — filed `draft`. **Do not dispatch as one slice**: the seven sub-items below have independent gates and must be split first.

---

## 0. Where this starts

P3-R measured, at step 16000, two identical 11.7M runs differing only in target rendering:

| axis | trace arm | answer-only arm |
|---|---|---|
| axis-B answer accuracy | **0.840** | 0.516 |
| axis-B executability | 0.800 | — |
| axis-B trace validity | 0.770 | — |
| axis-A answer accuracy (depths 5–6) | 0.039 | **0.105** |
| axis-A trace validity | **0.000** | — |

Two separable problems, and the P3-R results doc already named both:

- **Gate 1 near-miss + the error ordering `acc 0.840 > exe 0.800 > valid 0.770`** — correct answers riding on imperfect traces. This is a *credit-assignment* problem and SFT cannot fix it, because SFT has no way to say "this step was the wrong one."
- **Gate 3 negative, axis-A trace validity 0.000** — a *depth/length* problem. No reward shaping touches it.

Workstream A addresses the first. Workstream B addresses the second. They are independent and can run in parallel.

---

## 1. Workstream A — step-level rewards from the verifier we already have

### 1.1 The framing (why no LLM judge)

The rubric-RL literature (RaR, [arXiv:2507.17746](https://arxiv.org/abs/2507.17746); Rubric Anchors, [2508.12790](https://arxiv.org/abs/2508.12790)) exists because open-ended domains have *no verifier*, so they hire an LLM judge and pay for it in reward hacking. Our reasoning corpus is a synthetic KG: it has a verifier. `replay_trace` (`crates/grug-data/src/reason/replay.rs:193`) is ADR 092 D6 and already returns a per-statement marking scheme:

| `ReplayReport` field | rubric-RL equivalent |
|---|---|
| `statements_valid / statements_total` (`executability`) | RaR explicit aggregation `Σwⱼcⱼ/Σwⱼ`, uniform weights |
| `broken_by_refs: Vec<usize>` | SRaR step attribution `kⱼ` — **already computed** |
| `malformed_lines` | format criterion |
| `answer_correct` | outcome reward |
| `gap_confirmed` / `abstained` | abstention criteria |

SRaR ([2605.17291](https://arxiv.org/abs/2605.17291)) needs an LLM judge to produce `(sⱼ, kⱼ)` — "criterion satisfied, and which step it judges." We get it deterministically. **Introducing an LLM judge here would be a strict downgrade** and would import the verifier-failure taxonomy (partial-compound / implicit-as-explicit / imprecise-verification, 53,447 cases in [2605.12474](https://arxiv.org/abs/2605.12474)) that we are currently immune to.

### 1.2 A1 — VSA geometric criteria (no GPU; runs on existing checkpoints)

Three criteria `replay_trace` structurally cannot check, all expressible in the existing kernel (`bind`/`bundle`/`permute`/`similarity`, `src/vsa/ops.rs:81–157`):

1. **Redundancy / self-correction looping.** Bind each step to `relation ⊗ subject ⊗ object`; Hamming-compare against prior steps. Re-deriving a fact you already hold is *individually valid every time* — replay scores it 1.0. This is the pathology SRaR measured at **48.1% of RaR rollouts**, cut to 26.5%. Build this first: it is the highest-value criterion and the cheapest.
2. **Goal drift.** Bundle the trace's step vectors, compare to the goal vector. Locally-valid wandering scores lower. Reuses `ConvergenceVerifier`'s focus-Hamming pattern (`src/thinking/verifier/convergence.rs`), pointed at a trace instead of a `ThoughtState`.
3. **Coverage.** Bundle-similarity between emitted chain and required relation path — continuous "did it do the work," rather than counting lines.

**Scope limit (hard):** VSA judges *geometry*, never semantics. The plan already records binary-VSA analogy as degenerate — that is the founding reason the neural organ exists. Do not ask it whether an explanation is good.

**Deliverable:** a `reason::geometric` scorer alongside `score_reason_outputs`, plus a retro-scoring pass over `runs/p3r-trace/` step-16000 outputs. Measurable with **no GPU run** — this is the cheapest information in the plan and should land first.

### 1.3 A2/A3 — the RL stage

`rg` over `crates/grug-train/` and `crates/grug-data/` for reward/GRPO/rollout returns nothing. ADR 087 §4 puts RL after SFT with executable rewards; that stage is unbuilt. Build it in the SRaR shape:

```
Â(t) = (answer_correct advantage, group-normalised)      ← outcome term
     + (per-step replay offset, normalised per step      ← rubric term
        across sibling rollouts, broadcast to that
        step's token span)
```

Two details, both load-bearing in the source paper:

- **Per-step cross-rollout normalisation.** Steps every rollout gets right contribute zero signal; only steps whose validity *varies* teach. Removing this collapsed SRaR's training outright.
- **Decoupled advantage.** The rubric term stays out of the group baseline, so no rollout can inflate its advantage by padding the trace. This is what removed the self-correction-looping incentive.

We have the step spans (ADR-092 `### Step N:` grammar + tokenizer offset mapping, the same mechanism `hidden_from` already relies on) and `broken_by_refs` gives failures by step number.

### 1.4 A4 — anti-farming guards

We are immune to *verifier failure* (no judge to fool). We are **not** immune to *rubric-design limitation* — the failure that survived even the strong verifier in [2605.12474](https://arxiv.org/abs/2605.12474), where gains concentrated in presence/completeness criteria while real quality declined.

Our version is specific and predictable: **executability rewards emitting valid statements**, so a policy can farm it with trivially-valid `recall` lines it does not need. Guards:

- Normalise per **required hop** (`example.depth`), not per emitted statement.
- Register A1's redundancy criterion as an explicit **negative** criterion.
- Keep the decoupled advantage (A3).
- Optional, only if farming is observed: **Rubric Dropout** ([2608.11669](https://arxiv.org/abs/2608.11669)) — drop a random 30–50% of criteria per step, **one mask per rollout group** (per-rollout masks make GRPO advantages incomparable and corrupt the gradient). Their sweet spot is 30–50%, sign flips at 60%. Note their negative result: POW3R-style criterion *reweighting* scored below no intervention at all.

One asymmetry in our favour worth recording: RaR's ablation found synthetic *pitfall* criteria useless, because anticipating a model's failure modes needs an examiner's intuition. Ours are **measured, not guessed** — `broken_by_refs`, `malformed_lines`, the axis-A `trace_valid = 0.000`.

---

## 2. Workstream B — the axis-A depth wall

### 2.1 The decomposition (this is the main new claim)

P3-R's axis-A failure conflates two mechanisms, and next-lever (a) in the results doc was proposed precisely to separate them. The literature gives a distinct fix for each branch, and they are **not** interchangeable:

| branch | failure | literature |
|---|---|---|
| **B-render** | cannot *emit* a 5-step trace — the sequence is longer than anything trained | positional-encoding / length-generalization |
| **B-reason** | cannot *compose* 5 hops | recurrent-depth / looped transformers |

`trace_valid = 0.000` on axis-A is the signature of B-render dominating: not "wrong answer" but "no well-formed output at all." Run B0 before concluding anything about B-reason.

### 2.2 B0 — the positional-encoding ablation (cheapest, highest information)

**Finding from source inspection:** `TinyTransformer` uses a **learned absolute position embedding** — `position_embedding: Embedding<B>` over a `max_seq_len` table (`crates/grug-train/src/transformer.rs:290, 310`; the doc comment at :302 notes learned-not-sinusoidal was chosen so both embeddings share a type).

A depth-5/6 trace is strictly longer than any depth-2/3/4 trace. Its later position rows were therefore **never trained** — they sit at or near initialisation. That is a *mechanically sufficient* cause of `trace_valid = 0.000`, independent of any reasoning limitation, and it has not been on the Phase 51 risk list.

The literature is unusually decisive here:

- **Kazemnejad et al., [2305.19466](https://arxiv.org/abs/2305.19466)** — systematic comparison of APE / T5-rel / ALiBi / RoPE / NoPE on reasoning and math. **NoPE outperforms every explicit scheme** on downstream length generalization, at zero extra compute. Learned APE is the worst family for our case.
- **Ruoss et al., [2305.16843](https://arxiv.org/abs/2305.16843)** — 6000 models across 15 algorithmic tasks: failure is *caused by* position encodings being OOD at longer lengths. Randomized positional encodings simulate longer sequences and sample an ordered subset; **+12.0% test accuracy on average**.
- **Cho et al., [2405.20671](https://arxiv.org/abs/2405.20671)** — position *coupling*: assign the same position ID to structurally-equivalent tokens. Trained on 1–30-digit addition, generalizes to **200 digits (6.67×)**. Our ADR-092 grammar is explicitly step-indexed (`step n`, `by i j`), so coupling by hop index is a natural fit rather than a stretch.

Notably, Kohli et al. (below) used APE for their 2-hop experiment but switched to **NoPE for the depth-extrapolation experiment**, "which shows better generalization in pilot studies" — the same split we are hitting.

**B0 protocol:** re-run the P3-R trace arm, identical recipe (seed 42, 40k examples, 20k steps, 13M), varying *only* the position scheme: learned-APE (control) → NoPE → randomized-PE. One GPU run each. Read axis-A `trace_valid` first, answer accuracy second. If NoPE moves `trace_valid` off 0.000, a large fraction of the gate-3 null was never a reasoning result.

### 2.3 B1 — mixed-depth curriculum, with the threshold protocol

Already next-lever (a). Adopt Kohli et al.'s easy-to-hard discipline rather than plain depth-mixing: train on depth 2 until ≥95% on a held-out depth-2 split, then admit depth 3, and so on, **jointly training on all previously introduced depths** to prevent forgetting. Their term for the largest reachable `k` is the model's *learnable recursion depth* — a cleaner metric than our current per-depth table, and one we should adopt.

### 2.4 B2 — recurrent depth (the real B-reason fix)

**Kohli et al., "Loop, Think, & Generalize," [2604.07822](https://arxiv.org/abs/2604.07822)** is close to a mirror of our setup: synthetic KG, from-scratch controlled training, multi-hop composition, logit-lens analysis, and **depth extrapolation as an explicit named challenge** (train ≤5-hop, evaluate 10-hop). Their depth-extrapolation config is `L=4` layers, `|E|=200`, `|R|=10`, out-degree 10 — the same order as ours.

Findings that transfer:

- **Vanilla transformers fail both compositional challenges; recurrent-depth succeeds.** At 2-hop, `R=1` (vanilla) fails completely while even `R=2` generalizes.
- **Depth extrapolation is unlocked by scaling inference-time recurrence** — more iterations at test time, no retraining, no architecture change.
- **Dynamic training recurrence extrapolates best**: sample `R ~ clip(Poisson(λ), R_min, R_max)` per batch, `R_min=2, R_max=8`. Fixed recurrence is weaker.
- **Zero-init the `c_proj` output matrices** of both attention and FFN, so each recurrent block is exact identity at init. Weight-tied stacks are otherwise unstable when unrolled.
- **Systematic generalization arrives via three-stage grokking** — memorization → in-distribution → systematic — and only *after* near-perfect in-distribution accuracy, at ~10⁴ epochs vs ~10² for overfitting. Anyone reading our loss curves needs to know not to kill the run early.

Supporting: **Saunshi et al. [2502.17416](https://arxiv.org/abs/2502.17416)** — a `k`-layer transformer looped `L` times nearly matches a `kL`-layer model and is far better than `k` layers; "many reasoning problems require large depth but not necessarily many parameters." That is our exact constraint (13M params, 6 layers, need depth). **Geiping et al. [2502.05171](https://arxiv.org/abs/2502.05171)** scales the same idea to 3.5B/800B tokens.

**Why this specifically fits our measured probe result.** The GWT probes found no revisitable middle register — "feed-forward pipeline," crossover pinned at layer 5–6 of 6, axis-A pinned at 6, source span dead by mid-stack. Recurrent depth *manufactures* the revisitable middle by construction: the same layers are re-entered, and Kohli's logit lens shows the bridge entity becoming decodable at deeper **effective** depths. Our harness already has the instruments — `hidden_layers`, `lens_logits`, `hidden_from`, `probe_lens.rs`, `probe_swap.rs` — so the emergence probe reruns unchanged over effective depth `L × R`.

**Seam:** the existing `Attention<B>` trait (`transformer.rs:56`) is the wrong seam — recurrence wraps *blocks*, not attention. The change is local: `blocks: Vec<DecoderBlock<B>>` (`transformer.rs:311`) with its `for block in &self.blocks` loop (`:355`) becomes a weight-tied block applied `R` times, with `R` a forward-pass argument. `hidden_layers` / `hidden_from` need an effective-depth index rather than a block index.

### 2.5 B3 — halting, and the known failure mode

Kohli et al. name **overthinking**: excessive recurrence degrades predictions and caps very-deep generalization. Corroborated by **STARS ([2605.26733](https://arxiv.org/abs/2605.26733))** — looped LMs peak at some depth then collapse with further recurrence; their fix is Jacobian spectral-radius regularization toward stable fixed points.

We have already solved the analogous problem once. Phase 41e's VOC controller encodes the **two-regime rule** — diminishing-returns halting only in refinement, never during frontier search — adopted precisely because the first sweep reproduced the illusion-of-thinking failure. That rule is the right prior for recurrence halting, and it is ours, measured (ADR 085).

Cheap mechanism to borrow: **[2607.14427](https://arxiv.org/abs/2607.14427)** measures per-token fixed-point convergence in a 135M depth-recurrent model — successive-output KL falls 3.9e-1 → 8.5e-6 by loop 16, median token converges by loop 6, ~10% still updating at depth 8. A **training-free** halt-when-stable rule matched uniform depth-8 quality at 4.94 average loops (38% fewer). Reading convergence beat learning to predict it.

### 2.6 Honest limits of the B literature

- Kohli et al.'s task is **implicit** reasoning — predict a single tail entity, no emitted chain. Ours must render a whole grammar-valid trace. Their result is evidence for B-reason and says **nothing** about B-render. Do not let it substitute for B0.
- **LOTUS ([2606.31779](https://arxiv.org/abs/2606.31779))** is the closest bridge — a looped padded transformer with cross-entropy on *each latent position's gold CoT-step token*, i.e. our intermediate-trace supervision moved into latent space; first latent-CoT method to close the gap to explicit CoT at 3B, 2.5–6.9× lower thought-phase latency. But it is a 3B result; nothing here says it survives at 13M.
- Recurrent depth costs wall-clock linearly in `R`. Kohli et al. report `R=4` converging in fewer epochs *and* less absolute wall-clock than `R=2`, so this may be cheaper than it looks — but that is their task, not ours.

---

## 3. Sequencing

Cheapest-information-first. A1 and B0 are independent and should run concurrently.

| # | item | cost | gate |
|---|---|---|---|
| **A1** | VSA geometric scorers + retro-score `runs/p3r-trace` | **no GPU** | redundancy rate measurable; is looping actually present in our traces? |
| **B0** | position-scheme ablation (APE / NoPE / randomized) | 3 GPU runs | axis-A `trace_valid` off 0.000 |
| **A2** | replay → per-step reward vector | no GPU | unit-tested against known-bad traces |
| **B1** | threshold curriculum, depths 2→6 | 1 GPU run | learnable recursion depth > 4 |
| **A3** | GRPO + decoupled advantage in `grug-train` | build + 1 GPU run | axis-B exe ≥ 0.9 (gate 1 cleared); acc−exe gap < 0.05 |
| **B2** | weight-tied recurrent block, dynamic `R`, zero-init | build + 2 GPU runs | axis-A answer acc > answer-only arm's 0.105 |
| **B3** | convergence-based halting, VOC two-regime rule | build | no overthinking collapse at `R > R_max` |

**Reportable nulls.** House rule stands. If B0 leaves `trace_valid` at 0.000, B-render is not a positional artefact and the depth wall is wholly B-reason — itself a clean result that promotes B2. If A1 finds no redundancy in our traces, the SRaR looping pathology does not reproduce at 13M, and A4's guard list shrinks.

## 4. What this does not address

- **Systematic generalization** (composing atomic facts never composed in training) — Kohli et al.'s other axis. Our corpus does not currently split `C_ID`/`C_OOD` that way. Worth adding to `grug-data` as a third axis, but out of scope here.
- **The Goyal/Bengio bandwidth-limited slot** stays in the back pocket. Its trigger (gate 4 flat AND gate 3 negative) did not fire — gate 4 moved 5→4. B2 is now the better-evidenced architectural candidate and should be tried first.
- **P1b/Apertus comprehension** — unchanged, still the separate lever.
- **The founding thesis is not touched.** Phase 41's VSA loop remains the owner of unbounded depth; `handoff()` (`crates/grug-data/src/reason/handoff.rs:86`) already ships the System-2 continuation. Recurrent depth buys the organ *more* depth, not *all* depth.
