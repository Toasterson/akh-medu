# Global Workspace in Language Models — GWT for Grug-Brain Training — akh-medu

- **Date:** 2026-07-09
- **Status:** Imported research (external, fetched). Primary source is a peer-reviewed-adjacent Anthropic interpretability paper; the two supporting papers are ICLR/arXiv. Findings are the papers' own — treat mechanisms as validated *in frontier-scale models*, and the akh mapping as design hypothesis.
- **Fetched for:** Phase 51 (Grug-Brain / BabyLLM) model training — [plan](../plans/2026-07-05-phase51-grug-brain-baby-llm.md), [ADR 087](../decisions/087-grug-brain-baby-llm-training.md).
- **Companion to:** [High Grug reasoning-language](2026-07-04-grug-reasoning-language.md), [Project Grug-Brain](2026-07-05-project-grug-brain.md), [Thinking in LLMs vs VSA](2026-04-27-thinking-in-llms-vs-vsa.md).

## Sources

1. **[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/)** — Anthropic / Transformer Circuits (2026). Blog framing: [A global workspace in language models](https://www.anthropic.com/research/global-workspace); community writeup: [LessWrong](https://www.lesswrong.com/posts/3PaLrzxagpbnNtPLT/a-global-workspace-in-language-models). **The primary paper.**
2. **[Coordination Among Neural Modules Through a Shared Global Workspace](https://arxiv.org/abs/2103.01197)** — Goyal, Bengio et al. (ICLR 2022). The foundational *architectural* GWT-for-deep-learning mechanism.
3. **[Learning to Chain Operations by Routing Information Through a Global Workspace](https://arxiv.org/abs/2503.01906)** — (2025). GWT applied to *compositional multi-step reasoning* training. PDF cached under `tool-results/`.

Adjacent (not extracted here, worth a later pass): [Semi-supervised Multimodal Representation Learning through a Global Workspace](https://arxiv.org/pdf/2306.15711), [Zero-shot cross-modal RL transfer through a Global Workspace](https://arxiv.org/pdf/2403.04588), [Functional Advantages of the Selection-Broadcast Cycle](https://arxiv.org/pdf/2505.13969).

---

## 0. Why this matters to Phase 51 (the one-paragraph version)

The Anthropic paper is **empirical validation of the founding Grug-Brain bet**. It shows that a **low-capacity, verbalizable, broadcast "workspace"** *emerges on its own* in a trained LLM's intermediate layers — **not because the architecture enforces it, but because it is functionally useful for flexible planning**. Ablate it and the model keeps fluent language but loses multi-step reasoning. That is *exactly* the split ADR 087 designs for ("weights hold reasoning + tool-use + syntax; facts injected at runtime"). **High Grug is the deliberate engineering of what Anthropic found emerges by accident** — a designed verbalizable reasoning register whose tokens *are* SymbolIds. The de-risking is significant: we are not betting on an untested idea, we are *specifying* a structure that large models converge to unsupervised.

---

## 1. Paper 1 — The J-space / verbalizable global workspace (Anthropic 2026)

### 1.1 The claim
LLMs maintain "a privileged set of internal representations, available for report, modulation, and flexible internal reasoning" on top of automatic processing — functionally analogous to *conscious access* (Baars' Global Workspace Theory). They name it the **J-space**.

### 1.2 What J-space *is*, mechanically
- **Capacity-limited:** only ~10% of activation variance; typically **~25 active "J-lens" vectors at once**. The scarcity is the feature, not a bug.
- **Layer-localized:** lives in **intermediate layers only** (~L38–L92 of a ~100-layer model), not in early "sensory" or late "motor" layers.
- **Token-indexed:** each workspace direction corresponds to a **single vocabulary token** (acknowledged limitation for multi-token concepts). → *The workspace vocabulary is discrete and symbol-like.*
- **Mechanistically privileged:** J-lens vectors compose more broadly with downstream weight matrices than random directions (Gram-matrix evidence) — they are *read by many circuits*, i.e. broadcast.

### 1.3 The Jacobian lens (how they find it)
A causal lens: for layer ℓ, average the Jacobian of the final-layer activation w.r.t. layer-ℓ activation over positions and 1,000 prompts:

```
J_ℓ = E[ ∂h_final,t' / ∂h_ℓ,t ]        lens(h_ℓ) = softmax(W_U · norm(J_ℓ · h_ℓ))
```

Improves on **logit lens** (which assumes `J_ℓ = I`) and **tuned lens** (correlational, "skips ahead") by being *causal* and by isolating a "general disposition to verbalize" through context-averaging. Supports **write ops**: steering (`h ← h + α·v_t`) and lens-coordinate patching (swap one workspace vector, preserve the orthogonal remainder).

### 1.4 The five functional properties (why they call it a workspace)
1. **Reportable** — the model can say what's in J-space when asked.
2. **Modulable** — it can deliberately change J-space contents on request.
3. **Causal for reasoning** — J-space drives multi-step reasoning/planning.
4. **Flexible / broadcast** — one representation feeds many downstream tasks (the key property).
5. **Bypassed by routine** — most automatic processing never touches J-space.

### 1.5 The three experiments that matter for us
- **France→China broadcast:** one J-lens swap (France→China) applied *once*, then read correctly by *four different downstream computations* — capital ("Beijing"), language ("Chinese"), continent ("Asia"), currency ("Yuan"). **Write once, read by many.** This is the broadcast property made concrete.
- **Task-selective routing:** language-identity of a passage sits in J-space in *all* conditions, but is only *causally required* (swaps change the output) when the task demands **flexible report/inference**; for automatic continuation the same info is present but inert. → *The workspace is engaged by task demand, not by information availability.*
- **Ablation = surgical:** zeroing top-k J-lens directions selectively **impairs flexible reasoning** (analogy, translation, multi-hop, sonnet-writing) while **preserving** shallow classification/extraction/sentiment. Chain-of-thought externalization *reduces* dependence on the internal workspace (the model offloads to tokens).

### 1.6 The training result we should copy
**Counterfactual Reflection Training:** train the model to *articulate* ethical principles if interrupted and asked to reflect (targeting the J-space population in those contexts). Result: **behavior improved in the original, uninterrupted contexts with no direct training on them** — and *ablating the implanted J-space concepts reverted the improvement.* Conclusion: **the representations used for verbal report are the same ones governing silent reasoning.** Training the *verbalization* trains the *thinking*.

---

## 2. Paper 2 — Shared workspace architecture (Goyal, Bengio; ICLR'22)

The *architectural* recipe, if we ever want to build a workspace in rather than let it emerge:

- **Mechanism:** specialist modules (transformer blocks / object slots) communicate **only through a shared, bandwidth-limited workspace** — no direct module-to-module wiring.
- **Competitive write:** modules **compete via attention** for the scarce write slots each step; winners write, losers don't.
- **Broadcast read:** workspace contents are broadcast back to *all* modules each cycle.
- **Why the bottleneck helps:** the bandwidth limit **forces specialization and compositionality** and **synchronizes otherwise-independent specialists** — scarcity is rationalized as a *learning-inductive-bias*, not a cost. Contrast with Transformers' all-pairs attention, which gives no global coherent state.

## 3. Paper 3 — Chaining operations through a workspace (2025)

The *training-objective* evidence for multi-step reasoning:

- **Setup:** modules read/write a shared workspace across **iterative recurrent broadcast cycles**; trained on **chaining tasks** ("compute A+B, then ×C").
- **Objective:** supervise both per-operation accuracy **and** correct sequencing; **ground-truth intermediate states guide what the workspace stores/retrieves at each step** (not just final-answer supervision).
- **Result:** better **compositional generalization to longer/unseen operation chains**, **reduced error accumulation** vs. sequential pipelines, and interpretable intermediate contents.

---

## 4. Direct implications for Phase 51 training (actionable)

Ordered by leverage on the current plan.

### 4.1 Reframe High Grug as "the engineered global workspace vocabulary" — and it de-risks the bet
The emergent-J-space finding says a **discrete, token-indexed, low-capacity, broadcast reasoning register** is *the general solution* large models converge to. High Grug (tokens = SymbolIds, [High Grug doc](2026-07-04-grug-reasoning-language.md)) is that structure specified up front. **Update ADR 087 §5 / the High Grug language-design work item** to state this explicitly: High Grug is not just a "dense reasoning language," it is *the model's global workspace vocabulary*, and its SymbolId grounding is what makes it broadcastable across the neural↔VSA boundary. This is the strongest external support the project has for the whole design.

### 4.2 The capacity bottleneck is a *design lever*, keep the register sparse
Both Paper 1 (J-space ≈ 10% variance, ~25 active vectors) and Paper 2 (bandwidth limit *forces* specialization/compositionality) say **a small workspace is why it generalizes**. For a <500M model this is doubly good news — the constraint we're forced into is the one that helps. Design High Grug to be **token-dense and few-active-symbols-at-a-time**; resist the urge to let the reasoning register sprawl. Consider an explicit sparsity/low-rank prior on whatever internal register we designate.

### 4.3 Supervise intermediate High Grug states, not just final answers (P3 curriculum)
Paper 3's core result: **guiding what the workspace stores at each step** (ground-truth intermediates) is what buys length-generalization and kills error accumulation. This argues for the **High Grug trace being a supervised intermediate**, chain-of-operations style, during SFT cold-start and shaped by RL — *not* an unsupervised latent we only read the endpoints of. Fold this into the P3 reasoning curriculum and the RL reward shaping (reward correct intermediate High Grug steps, not only executable final tool calls).

### 4.4 Train the verbalization to train the silent reasoning (pretrain + SFT)
Paper 1's Counterfactual Reflection result: teaching the model to *articulate* improved *silent* behavior, because report and silent reasoning share representations. This **directly justifies ADR 087 §5/§7's placement of High Grug in pretrain** (not deferring it entirely to SFT): making the model fluent in *emitting* High Grug should improve the reasoning it does *without* emitting. Cheap, high-leverage, and it's a testable prediction — measure silent-reasoning eval with vs. without High-Grug-fluency pretraining.

### 4.5 The round-trip (P6) IS a global-workspace broadcast — use the coordination protocol
ADR 087 §8 (shared-SymbolId dual-process peer) is, unnamed, a **two-module shared-workspace system**: neural System-1 and VSA System-2 as Goyal/Bengio "specialist modules" coordinating through the SymbolId register as the workspace. Adopt the paper's protocol explicitly for P6 dual-process scheduling: **competitive write** (which substrate populates a reasoning step) + **broadcast read** (both substrates + the CompositeVerifier read the same symbols). The France→China "write once, read by many" is the exact semantics the round-trip needs. Each substrate covers the other's failure mode (degenerate binary-VSA analogy ↔ neural fluency) — the kludge-brain *is* a global workspace across substrates.

### 4.6 Add a "does a workspace emerge?" interpretability metric to the eval harness
The eval harness (ADR 087 §9) currently measures the boundary (valid JSON, schema, arg-correctness, anti-collapse). Add a **Jacobian-lens-style probe** as a *reasoning-quality* metric for P3+: does the trained BabyLLM develop **verbalizable broadcast directions** in its intermediate layers, and are they **token-indexed to our SymbolIds**? A cheap version: apply the France→China-style *single-swap-read-by-many* test on our own tool/reasoning tasks. If our engineered workspace is working, an intervention on one High Grug symbol should propagate correctly across multiple downstream computations. This turns "is High Grug actually the workspace?" from a hope into a measurement. **Burn exposes activations**, so a logit/Jacobian lens is buildable in-crate.

### 4.7 Routine-vs-flexible split validates "facts at runtime, reasoning in weights"
Paper 1's ablation (kill workspace → lose reasoning, keep fluency) and task-selectivity (routine bypasses the workspace) are direct evidence for the **memorization-vs-reasoning separation** the whole project rests on. Fluent surface generation and factual recall are the "automatic" pathway; multi-step reasoning is the workspace pathway. Injecting facts from the KG at runtime targets the automatic pathway; High Grug reasoning targets the workspace pathway. The architecture the project chose is the architecture the data says large models discover.

---

## 5. One naming-collision caveat

ADR 027 "**Shared Workspace Agent**" is a **different, unrelated** concept — it's the *software* pattern of one `Agent` per workspace (`Arc<Mutex<Agent>>` in `ServerState`), nothing to do with GWT. When wiring the GWT ideas in, keep the vocabulary distinct: call the GWT concept the **reasoning workspace** / **global workspace** / **J-space analogue**, never bare "shared workspace," to avoid conflating it with the daemon's agent-sharing pattern.

## 6. Open questions this raises

- **Does an emergent workspace even appear at <500M / our data scale?** Anthropic measured a ~100-layer frontier model. The 6–8-layer Grug models may be too shallow for a *spontaneous* intermediate-layer workspace — which is precisely the argument for *engineering* it via High Grug rather than hoping it emerges. §4.6's probe is how we'd find out.
- **Multi-token concepts.** J-space is single-token-indexed; High Grug SymbolIds can name multi-token concepts. Does that make High Grug a *stronger* workspace vocabulary than the emergent one, or does it break the "one direction per token" property that makes the lens work? Design question for the High Grug spec.
- **Recurrent broadcast cycles (Paper 3) vs. feed-forward transformer.** Paper 3's gains come partly from *iterating* the workspace. Our vanilla-attention transformer is feed-forward. Is the Phase 41 VSA loop's iteration the akh equivalent of the recurrent broadcast cycle (System-2 doing the iterating that the feed-forward organ can't)? Likely yes — see §8.

---

## 7. Concrete eval spec — the workspace-emergence probe for `grug-train`

The single highest-value follow-up: **measure whether a verbalizable workspace actually forms in our tiny model, and whether High Grug is it.** This turns the entire GWT hypothesis into a number on the eval dashboard. It's cheap because the P1a-4 refactor already gave us the hooks: `TinyTransformer::hidden()` (intermediate hidden states) + `vocab_logits()` (unembed), `PointerTransformer::forward_parts` (exposes internal parts), and `Vocab::token_pieces`. Three staged probes, in ascending cost:

### 7.1 Stage L — logit lens per layer (cheapest, build first)
For each layer ℓ, take the intermediate hidden `h_ℓ` and apply the model's own unembedding: `topk(vocab_logits(norm(h_ℓ)))`. **Question:** does an *intermediate* layer already "name" the target tool / the High Grug reasoning symbols *before* the final layer emits them? Plot the per-layer rank of the eventual output token across a batch of tool-call and (later) reasoning prompts.
- **Positive signal:** a "verbalization crossover" layer where the right token surfaces mid-stack → the tiny analogue of J-space's intermediate-layer localization.
- **Cost:** one forward pass, no autodiff. Pure inference. `examples/probe_lens.rs`.

### 7.2 Stage S — single-swap-read-by-many (the France→China test on our tools)
The decisive broadcast diagnostic. Build a **prompt family** sharing one variable but asking different downstream things — e.g. one target entity/region fixed, four requests (`get_capital`, `get_language`, `get_currency`, `get_continent`) over it. At the Stage-L crossover layer, **patch a single lens direction** (swap the entity's direction A→B, preserve the orthogonal remainder — exactly the paper's op) and measure whether the swap propagates *correctly across all four* downstream computations.
- **If High Grug is a real broadcast workspace:** one symbol-swap redirects every reader (write once, read by many). **If it isn't:** the swap only affects the one task that lexically mentions the entity.
- This is the experiment that distinguishes "High Grug is a genuine workspace vocabulary" from "High Grug is decorative CoT." **Run it once High Grug tokens exist (P3+); a JSON-era proxy over shared tool arguments is runnable now** on the p2/p1a2 checkpoints.
- **Cost:** patching = forward passes with one steered activation; no training. `examples/probe_swap.rs`.

### 7.3 Stage A — ablation (routine-vs-flexible split)
Zero the top-k Stage-L "verbalizable" directions at each position and measure the differential:
- **Expect to survive:** single-tool valid-format rate, schema adherence, sentiment/extraction-shaped tasks (the *automatic* pathway).
- **Expect to degrade:** multi-step tool chains, analogy, translation, any High-Grug-mediated reasoning (the *workspace* pathway).
- A clean dissociation here is our-scale replication of Anthropic's surgical ablation and **direct evidence for the memorization-vs-reasoning architecture the whole project rests on**. `examples/probe_ablate.rs`, reuses the two-axis eval harness.

### 7.4 Emergence-over-training metric (add to the JSONL time series)
Log, per checkpoint: (a) the Stage-L crossover-layer index and its confidence, (b) the count of active "verbalizable" directions (the tiny analogue of J-space's ~25) and their share of activation variance, (c) once High Grug lands, **what fraction of those directions are token-indexed to SymbolIds.** **The question this answers:** does a workspace *sharpen* over training the way it does in frontier models — or does it never form at 6–8 layers (see §8)? Either answer is decision-grade.

**Jacobian vs logit lens:** the paper's causal J-lens needs the layer-to-layer Jacobian (Burn `Autodiff` can produce it via vector-Jacobian products, but a full Jacobian over 1000 prompts is expensive). **Start with the logit lens** (Stage L, free) and a cheap **tuned-lens linear probe** (one trained affine map per layer) before paying for the true Jacobian — the paper itself notes the logit lens is the `J_ℓ = I` special case, so it's a valid first cut.

---

## 8. Strategic synthesis — why the kludge-brain *is* the workspace at our scale

The most important structural conclusion, stated plainly:

**Anthropic's J-space lives in the middle ~40% of a *100-layer* model (L38–L92).** A Grug organ is **6–8 layers**. It has almost no "middle" — layers 3–5 of 8 — so the *sensory → workspace → motor* tripartite division that emergence needs **may have no room to form**. This is not a footnote; it is the central risk to any hope that a workspace appears spontaneously in a tiny model.

Three consequences reorganize the plan:

1. **Engineering beats hoping.** If depth blocks spontaneous emergence, we cannot *rely* on it — we must **impose** the workspace via the training objective. That is precisely what High Grug + supervised intermediate states (§4.1, §4.3) do: an *externally specified* verbalizable register, not an emergent one. GWT stops being a nice analogy and becomes the *reason the design is shaped the way it is*.

2. **Goyal/Bengio (Paper 2) is promoted from "theory" to "fallback architecture."** If the §7.4 emergence probe comes back negative (no internal workspace at 8 layers), the explicit architectural remedy is a **shared bandwidth-limited slot the layers compete to write and all read** — buildable behind the existing `Attention<B>` seam as a new concrete type, exactly where the plan already reserves room for RoPE/EM-LLM/DuoAttention. Keep it in the back pocket; the probe decides whether we need it.

3. **★ The VSA loop is the workspace the tiny transformer architecturally lacks.** Paper 3's compositional gains come from *iterating* the workspace across recurrent broadcast cycles. A feed-forward 8-layer transformer **cannot iterate internally.** The Phase 41 VSA Thinking Engine **is an external, iterative, broadcast workspace** operating over the shared SymbolId register. So the kludge-brain (ADR 087 §8 / ADR 089) is not merely a division of labor between neural System-1 and symbolic System-2 — **it is supplying the iterative-workspace substrate the neural organ is too shallow to contain.** The tiny model proposes into the workspace; the VSA loop *is* the recurrent broadcast cycle that refines it; both read the same symbols. This is the sharpest reframing the papers give us, and it should anchor the P6 integration design: **the workspace is not inside the organ, it is the SymbolId register shared with the VSA loop, and iteration lives in System-2.**

Corollary for ADR 089: the three organ mechanisms (retrieval-scoping, KG-repair, gap-token) are all **workspace operations** — they populate, correct, and signal-absence-in the shared register the neural organ writes to. The "zero-shot belongs to the organ, not the weights" thesis is the *same claim* GWT makes at the architecture level: flexible cross-task use is a **workspace (System-2) property**, not a weights (System-1) property. Anthropic's ablation (kill workspace → lose flexible reasoning, keep fluency) is the frontier-scale echo of our four-levers-fail / one-symbolic-pass-wins result. **Two independent lines of evidence, one conclusion: reasoning flexibility is a workspace property. That convergence is the finding.**
