# A capable 8–12B Akh on hardware we can actually buy

Date: 2026-09-19. Status: proposal for decision. No purchase, training run or pod start is authorised by
this document.

Supersedes nothing. Extends [the R1 release plan](2026-09-11-akh-350m-graph-grounded-release-plan.md)
(candidate range 350M–1.1B) upward to the 8–12B range the operator asked about, and re-opens the size
question its [research program](2026-09-12-akh-capability-and-model-size-research-program.md) left
unresolved. Governed by [ADR 105](../decisions/105-autonomous-run-governance.md) (one work item per run,
machine-checkable exit condition) and [ADR 106](../decisions/106-local-burn-metal-lane-and-engine-in-the-loop-training.md)
(cloud is the measured exception, requiring a recorded local miss, operator approval and a dollar cap).

**Naming (from [ADR 111](../decisions/111-kaine-architecture-ladder.md), on `main`).** *ADAM* is the sovereign
BabyLLM lineage: `small_8k` (33.4M) → 350M → a planned ≈1.1B rung. *Kaine* is the next lineage above ADAM's top rung,
whose size ADR 111 leaves as an output of the 350M and 1.1B measurements. In this document "full-Adam" always means the
Adam *optimizer* state (16 bytes/param); the ADAM *lineage* is written in capitals. "A real ADAM training run" in the
operator's framing is a run on that ladder, not an optimizer choice. *(2026-09-27: the inherited 8–12B workhorse is
now named **Lilith**, not Kaine; see the naming note under "Parallel track" and [Project Lilith](2026-09-21-project-lilith.md).)*

## The question

Train or obtain an 8B–12B model capable of the Akh function, deployable to a 16 GB Mac mini, using
hardware we own or can plausibly buy: an RX 7900 XTX (24 GB, ROCm 6.3.4), an M2 Mac mini (16 GB), a
planned RTX 3090 (24 GB), or a newly-shipping M5 Mac Studio (up to 512 GB unified).

## The honest answer

**Every route to a *sovereignly trained* 8–12B model is out of reach on any single machine we could buy,
by two to four orders of magnitude — and the reason is FLOPs, not memory.** A high-memory Mac Studio
solves the one constraint that is not binding.

But the goal is reachable, because it was mis-stated. We do not need to *train* an 8–12B model. We need an
8–12B model that *performs the Akh function*, and the cheapest test of whether we already have one has
never been run.

## 1. What the hardware can do

Full-parameter Adam costs **16 bytes/parameter** (ZeRO, 1910.02054: 2 B bf16 param + 2 B grad + 4 B fp32
master + 4 B momentum + 4 B variance), so 8B = **128 GB** and 12B = **192 GB** as floors before
activations. Apple's GPU ceiling defaults to ≈75 % of unified memory (`sysctl iogpu.wired_limit_mb`
raises it; unsupported, resets on reboot).

| machine | full-Adam ceiling | effective TFLOPs | price |
|---|---|---:|---|
| RX 7900 XTX 24 GB (owned) | ~1.3B | ~30 est. | owned |
| M2 Mac mini 16 GB (owned) | none meaningful | measured 0.3 on our stack | owned |
| RTX 3090 24 GB (planned) | ~1.3B | 71 peak, ~28 at 40 % MFU | ~€700 |
| M5 Max Studio 128 GB | ~5–7B | unpublished | €5,199 |
| **M5 Ultra Studio 256 GB** | **~12B** (192 GB usable) | unpublished | **€10,999** |
| M5 Ultra Studio 512 GB | ~24B | unpublished | unpriced, ships late Oct |

Apple publishes **no training TFLOPS for M5**. Their own measured figures are 3.19–4.06× over M4 for
compute-bound work and 1.19–1.27× for generation; no third-party M5 GEMM benchmark exists yet. Any M5
TFLOPS number in circulation is an extrapolation from the "4× vs M4" claim.

**The decisive arithmetic.** A 50B-token recovery run on an 8B model is `6·N·D = 2.4e21` FLOPs:

| machine | 50B-token 8B recovery run |
|---|---|
| RTX 3090 | **~2.7 years** |
| M5 Ultra 256 GB (estimated) | **~3–5 years** |
| 8× H100 rented | **~4.4 days, ≈ $2,400** |

**€11,000 of Mac buys memory we cannot feed.** If the route needs a real recovery run, the only machine
that finishes it is rented, at a fifth of the Studio's price. This is the single most important number in
this document.

For completeness, from-scratch at Chinchilla 20 tok/param on one 3090: 350M ≈ **7 days**, 1.1B ≈ **8
weeks**, 8B ≈ **10 years**, 12B ≈ **15 years**. The existing R1 range of 350M–1.1B is exactly the band a
single consumer card can pretrain, which is presumably why it was chosen.

Data is not a constraint: the foundation corpus is FineWeb-Edu / FineWeb2-HQ (ADR 102), effectively
unlimited at our scale.

## 2. What the compression literature actually costs

Read in full text. The recovery phase, not the pruning phase, is the expense — and the recipes split
sharply into two tiers by how they recover.

**Tier 1 — datacenter recovery (full-parameter continued pretraining):**

| recipe | reduction | recovery tokens | authors' hardware |
|---|---|---:|---|
| Sheared LLaMA (2310.06694) | 7B→1.3B/2.7B | 0.4B + **50B** | multi-GPU cloud (count not stated) |
| Minitron (2407.14679) | 15B→8B/4B | **94B** | **128× A100-80GB** |
| Minitron in practice (2408.11796) | 12B→8B | **380B** + ~100B teacher correction | **256× H100** |
| SlimQwen (2605.08738) | MoE 80A3B→23A2B | **400B** | undisclosed |
| DarwinLM (2502.07780) | various | search ~10B/candidate | search: **4 consumer GPUs, 8 h**; final: **40× H100, 13 h** |
| ShortGPT recovery (2403.03853) | 25 % | 50B | 8× A100-80GB |
| LaCo recovery (2402.11187) | 25–30 % | 1–1.25B | 4× A100-80GB, 28–35 h |

Minitron's headline "40× fewer tokens" is real but measured against a same-size sibling trained from
scratch on 3.8T tokens — it still means 94B tokens on 128 A100s.

**Tier 2 — single-GPU recovery (QLoRA healing). This tier is the plan.**

| recipe | pruning step | recovery | hardware |
|---|---|---|---|
| **Deeper Layers (2403.17887)** | angular-distance layer selection, forward passes only | **QLoRA heal, 164M–328M tokens**, 5000 steps, rank 64 | **"each experiment... can be performed on a single 40GB A100 GPU"** |
| ShortGPT (2403.03853) | Block-Influence cosine ranking | **training-free** | forward passes; trivially single-GPU |
| LaCo (2402.11187) | RDSC layer merge | **training-free**, "within minutes on a single GPU" | ~19.4 GB bf16 for a pruned 13B |
| SliceGPT (2401.15024) | one-shot PCA slice, gradient-free | **LoRA**, ~1–5 k sequences | **1× H100, 1–5 h total** |

The pruning *decision* is essentially free in all four — a cosine or angular-distance metric over a few
hundred calibration samples. **2403.17887 removes "up to roughly half of the layers before performance
collapses" and heals with QLoRA on 164–328M tokens on one card** — roughly three orders of magnitude
cheaper than Minitron's 94B tokens, and it is the only pipeline in the literature measured end-to-end on
a single GPU. That is a real route, and my earlier reading that no single-GPU recovery existed was wrong.

**Three findings still constrain it:**

- **Pruning damages precisely what an agentic Akh needs.** UniComp (2602.09130) finds a "knowledge bias":
  factual knowledge survives while **multi-step reasoning, instruction-following and multilingual ability
  degrade sharply**, plus "performance-reliability decoupling" — retained accuracy does not imply retained
  safety or truthfulness. ShortGPT independently reports **generative tasks degrading far faster than
  multiple-choice** ones, and 2403.17887 finds C4 loss staying smooth up to ~80 % pruning *after* QA
  accuracy has already collapsed. Loss is not a safe gate here; the capability contract is.
- **At equal total budget, from-scratch matches coarse pruning.** 2606.14150 shows pruning wins at equal
  *retraining* budget, but once the parent's pretraining is counted, from-scratch **matches or surpasses**
  coarse structured pruning. Pruning's durable edge survives only at fine/unstructured granularity, which
  has the weakest hardware support and would gain nothing on Metal or RDNA3.
- **Our served parent is the architecture the literature does not cover.** Qwen3.8-27B is a hybrid
  gated-delta + full-attention decoder (`src/nlu/candle_backend.rs:29`). SlimQwen states plainly that
  "Minitron is applicable only to non-MoE models"; whether SliceGPT's RMSNorm rotational-invariance trick
  or BI-style layer ranking is even valid across gated-delta blocks is unresolved. ShortGPT does validate
  its BI metric on Mamba-2.8B and RWKV-7B, which is weak evidence that layer-similarity pruning survives
  non-vanilla attention — but that is not the same as validating it on a gated-delta hybrid.

## 2b. For tool-calling, the evidenced baseline is offline SFT on verified traces, not logit distillation

This is the most directly actionable finding, and it changes what Stage 1 should be.

- **MiniLLM (2306.08543)** and **GKD (2306.13649)** — the on-policy KD literature — were evaluated only
  on instruction-following, summarization, translation and arithmetic. **Neither tests tool use.** Their
  gains over plain SFT are real but single-digit (LLaMA-7B Dolly 73.0 → 76.4), and MiniLLM's ablation
  shows the stabilizers, not on-policy-ness, are load-bearing. MiniLLM cost: 16× V100, <10 h for 7B-from-13B.
- **The dominant recipe for tool-calling is offline SFT on execution-verified synthetic traces.**
  **xLAM (2409.03215)** reached #1 on the Berkeley Function-Calling Leaderboard — beating GPT-4 and
  Claude-3, *including at 1B and 7B scale* — using APIGen to synthesise **60,000 verified examples from
  3,673 APIs**, plain offline SFT, no RL. **ToolACE (2409.00920)** got an **8B model to SOTA on the same
  leaderboard** by the same route. ToolACE-DEV (2505.07512) explicitly warns that distilling tool use from
  a stronger teacher "incurs significant costs" and causes knowledge-scope mismatch.
- **The bottleneck is trace verification, not model scale or training algorithm.**
- **2025 nuance (abstract-level only):** MENTOR (2510.18383) and agent-trajectory distillation (2505.17612) report that
  reward-shaped distillation from an agentic teacher beats plain SFT *out of domain*; magnitude unverified. Our episode
  generator already emits causal step rewards, so the lane can test both without new infrastructure.

**We already own the expensive half of that recipe.** `src/bin/akh_episode_generator.rs` generates
engine-in-the-loop episodes with symbolic completion targets, event-calculus validity checking and causal
step rewards — an execution-verified trace generator of exactly the APIGen/ToolACE shape, built for P6.3.
The published evidence says 60k such traces plus offline SFT is enough to put an 8B model at the top of a
tool-calling leaderboard.

## 3. Licensing: the gate several candidates fail

Decision-grade for a project with a FLOSS-sovereignty principle. Verified against actual LICENSE blobs,
not model-card badges.

| family | licence | redistribute derivatives | MAU trigger | anti-distillation |
|---|---|---|---|---|
| **Apertus v1.5** (8B / 70B; Mini 0.5B–4B) | **Apache-2.0**, training-data recipe public | yes | none | none |
| **Qwen3 / 3.5 / 3.6 / 3.8** (8B–32B dense + MoE) | **Apache-2.0** | yes | none | none |
| **Gemma 4** (12B/31B dense, 26B-A4B MoE) | **Apache-2.0** | yes | none | none |
| **Mistral Small / NeMo / Ministral** (Apache tier) | Apache-2.0 | yes | none | none |
| **DeepSeek-R1-Distill-Qwen** (1.5B–32B) | **MIT** | yes | none | none |
| GLM-4.5-Air | MIT (README) / Apache-2.0 (LICENSE file) — pin the revision | yes | none | none |
| DeepSeek-V3/V2 raw | custom, Apache-derived + Attachment A | yes, with pass-through | none | none (distillation contemplated) |
| **Meta Llama 3.x / 4** | Llama Community License | yes, with branding | **700M MAU — persists through Llama 4** | none (naming only) |

Two results matter. **Llama fails the gate** — the MAU trigger is exactly the pattern the project rules
out. And **Gemma 4 switched to Apache-2.0** in April 2026, the first Gemma generation to do so; Gemma 2
and 3 remain under the older custom terms with a prohibited-use pass-through. Since `gemma4-12b.gguf` is
already on disk, we hold an **Apache-2.0, ~12B, 256K-context, natively tool-using model** that clears the
sovereignty gate outright. Qwen3.5-9B (262K native context, tool-calling in the chat template) is the
other clean candidate in the target band.

**Apertus v1.5-8B is the strongest sovereign candidate and was scoped in this repo as a corpus, not yet as a base**
([P1b memo](../2026-07-11-phase51-p1b-apertus-scoping.md)). Released 2026-07-24 by EPFL/ETH/CSCS as continued
pretraining of Apertus-8B-2509 (+4T tokens): 262,144-token context, a documented tool-call format
(`swiss-ai/apertus-format`, vLLM `--tool-call-parser apertus`), optional thinking mode, and native multimodal towers
that a community MLX "text-only" trim strips for a 16 GB Mac. Architecture is a plain dense GQA decoder with QK-norm and
the xIELU activation — so QLoRA, PEFT and every pruning recipe in section 2 apply unchanged, with xIELU as a separate
pip dependency (axolotl documents a QLoRA config at ~8.7 GB VRAM). llama.cpp support merged 2025-10-02, so Q4_K_M
exports exist. It is the only candidate whose *training data recipe* is public, which is a stronger sovereignty claim
than Apache-2.0 weights alone, and it sits under Swiss jurisdiction. Two caveats, both measurable in Stage 0: the 2509
tech report's post-training section does not mention tool-use data (tool integration is a v1.5-era claim without a
published v1.5 report yet), and **no BFCL number exists for either version** — Portuguese gains (Global-MMLU-pt
39.5→63.2, Belebele-pt 57.2→78.7) are community-reported, not first-party. Apertus should therefore enter Stage 0 as a
candidate next to Gemma 4, not replace it by assumption.
**Measured locally 2026-09-20** ([probe](../research/2026-09-20-apertus-v1.5-8b-local-probe.md)): the Q8_0 and
Q4_K_M text-only trims load on the RX 7900 XTX via Ollama at 70–78 and 91–97 tok/s decode, answer EN/DE/PT/FR/Swiss
German on-topic, return exact strict JSON, and emit the documented `<|tools_prefix|>…<|tools_suffix|>` framing (proven
by token id) — then fabricate a tool output unless the harness stops at the suffix.

**Decision 2026-09-20 (operator): Gemma 4 is the trunk; Apertus is a routed component, not the base.** Prompted by
guidance relayed from the Public AI Matrix channel by Janik von Rotz (maintainer of apertus-ai.org, the official
site — beware of clones): instruction following has improved in newer releases, Apertus "can definitely connect in
your harness as a generator or decision step", but is "not recommended as primary agent right now". Recorded as a
chat statement, not a published benchmark. Consequences: Stage 0/1 run on **Gemma 4 12B** as the trunk; Apertus
v1.5-8B is wired as a per-akh `ChatBacking` for Swiss-language turns (German, Swiss German, Romansh, Portuguese
basics) and as a generator/decision step where the captain routes to it; it stays in Stage 0 only for the
Portuguese-basics rows. Revisit if a later Apertus release publishes an agentic (BFCL-class) number.

## 4. Why the obvious plan fails, and what replaces it

The obvious plan — buy a big Mac and prune the served 27B down to 12B — fails three ways: the deep
reduction needed (27B→12B is ~55 %) sits past where 2403.17887 reports collapse; pruning degrades
reasoning and instruction-following, which is what an Akh is; and the parent is a gated-delta hybrid that
no pruning paper covers.

What has never been measured is the cheap thing. **We already own Apache-2.0 12B-class models that clear
the licence gate** — `gemma4-12b.gguf`, plus `qwen2.5-14b` and `qwen2.5-32b` on disk and the served
Qwen3.8-27B. The Ptah runtime already executes this family on HIP and Vulkan (20.2 steps/s decode, 23.9 s
load, 10 W resident idle at `9347805b`). **Nobody has run the R1 twelve-family capability contract against
any of them.** Until that number exists, every size, purchase and training decision is blind.

Measure first, buy last.

## 5. The plan

### Stage 0 — Baseline what we already own (no purchase, no training)

Run the R1 capability contract's **100-world development pack** against **`gemma4-12b`** (Apache-2.0, the
trunk), `qwen2.5-14b` and the served Qwen3.8-27B; run **Apertus v1.5-8B** on the Portuguese-basics and Swiss-language
rows only (its role is a routed language component, per the 2026-09-20 decision) through the existing Akh agentic loop, scored per the frozen
contract: final world state, correct target, authorization, evidence-backed graph delta, reuse after
restart, fresh verification, truthful completion, interaction repair — with engine-only successes counted
separately from neural generalization, as the contract requires.

*Exit condition:* a per-family score table for four models, with Portuguese-basics scored separately. *Cost:* GPU time on owned hardware.
*Blocking dependency:* **the 100-world development pack does not exist yet.** Building it is the real
first task. It is required by every branch below, so it is not wasted under any outcome.

### Stage 1 — Offline SFT on engine-verified traces, via QLoRA

Not distillation. Follow the xLAM/ToolACE recipe, which is what actually tops tool-calling leaderboards at
this scale: generate ~60k execution-verified episodes with `akh_episode_generator`, then QLoRA-finetune
the Gemma 4 trunk (or the strongest Stage-0 model, if Stage 0 overturns that choice) on them plus the Akh curriculum (tool contracts, High Grug register, graph
operations, Dojo kata). A 12B at NF4 is ~6.5 GB frozen base plus adapters — comfortable in 24 GB.
Re-run the contract.

*Exit condition:* per-family deltas vs Stage 0. *Cost:* days on an owned card.

### Gate A — decide on evidence

- **Contract met** → quantize to 4-bit, deploy to the Mac mini, done. No purchase, no Adam run. 8B at
  4-bit ≈ 4.5 GB, 12B ≈ 6.5 GB; both fit 16 GB alongside KV cache, runtime and graph.
- **Gap is knowledge or format** → more verified traces; stay in Stage 1. This is the cheap axis and the
  literature says it is usually the right one.
- **Gap is reasoning capacity** → a better base is the answer, not a training run. Re-baseline.
- **Gap is genuinely un-adaptable by LoRA** → Stage 2.

### Stage 2 — only on a measured Stage-1 miss

Before either Stage-2 option: **within-lineage breeding of the Gemma 4 trunk** — separate LoRA children per trait
(Portuguese, High Grug register, tool use, Dojo kata) recombined by TIES/DARE or an evolutionary recipe search with
the 100-world pack as fitness. Fits one 24 GB card; cross-family weight merging is impossible and logit-level
cross-family distillation is a datacenter job (output-level FuseChat-3.0-style fusion may not be — cost unpublished),
so this is the cheap version of "traits from other models". **Tooling gap:** upstream `mergekit` has no Gemma 4
architecture entry yet; LoRA Soups' CAT (2410.13025) is the preferred operator over TIES/DARE. Design and sources:
[model breeding with Gemma 4 as the trunk](../research/2026-09-20-model-breeding-gemma4-trunk.md).

Two options, in cost order:

1. **Layer-prune a larger Apache-2.0 parent + QLoRA heal**, per 2403.17887: angular-distance layer
   selection (free), then heal on 164–328M tokens on one card. Reaches ~half the layers before collapse,
   so 27B→14B is in range and 27B→12B is marginal. Gate on the capability contract, never on loss —
   both 2403.17887 and ShortGPT show loss staying smooth after downstream accuracy has collapsed.
2. **A real full-Adam run**, which is a rented multi-GPU job (≈$2,400 for ~4.4 days on 8× H100 at 50B
   tokens), requiring an ADR-105 work item with a dollar cap and explicit operator approval per ADR 106.
   It is not a local job on any machine in section 1.

### Parallel track — the ADAM ladder and where Kaine forks

Nothing above produces a sovereign from-scratch foundation. That is the ADAM ladder's job, and the band a single
consumer card can pretrain is **350M–1.1B**, unchanged from the R1 plan and the [P7 plan](2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md):
350M in ~1 week, 1.1B in ~8 weeks on a 3090 (P7.8 already scopes the 1.1B fit on one 80 GB A100). ADAM is a separate,
legitimate objective and stays tracked as one.

**Kaine is where this plan and ADR 111 have to be reconciled.** ADR 111 defines Kaine as a *larger from-scratch rung*
above 1.1B on our own architecture (SWA + spaced global layers, optional Gated DeltaNet parallel heads, product-key memory
seam per P7.9). Section 1's arithmetic says a from-scratch rung above ~1.1B does not finish on any machine we own or
would buy — it is a rented multi-GPU job by construction. So there are exactly two consistent readings of "Kaine":

1. **Kaine = the ADR 111 architecture, trained from scratch, on rented compute**, sized by P7.8. Sovereign, expensive,
   and not before ADAM 1.1B has been measured.
2. **Kaine = the harness-adapted inherited workhorse** (Stage 1 of this plan: an Apache-2.0 8–12B base, QLoRA on
   engine-verified traces, engine memory reached through tools rather than a trained memory layer). Cheap, on owned
   hardware, not sovereign in lineage.

These are different artifacts with different provenance. Whichever is chosen, the name should mean one thing; this plan
recommends reading 2 as the *first* Kaine trial and keeping 1 as the sovereign target it is measured against, and asks
that the operator record that choice in ADR 111 rather than letting the name drift.

> **Naming resolved (note added 2026-09-27).** Per [Project Lilith](2026-09-21-project-lilith.md), **"Lilith"** names the
> inherited Gemma track (reading 2 above), and **"Kaine"** keeps its sovereign [ADR 111](../decisions/111-kaine-architecture-ladder.md)
> meaning (reading 1). Where this plan says "Kaine" for the inherited workhorse, read "Lilith". The recommendation above is
> kept as the historical proposal.

## 6. Hardware recommendation

**Buy the €700 RTX 3090 before the €10,999 Mac Studio, and buy neither until Stage 0 reports.**

The 3090's value is not FLOPs — it is **ecosystem access**. QLoRA, which every affordable route above
depends on, needs bitsandbytes, PEFT and FlashAttention. Those are CUDA-native; on ROCm they are partial;
on Apple silicon bitsandbytes does not exist at all, and neither do GaLore, LOMO, Muon or any other
optimizer-state compression. Every paper in Tier 2 used CUDA. MLX supports `--fine-tune-type full`, but
its full-finetune throughput has never been credibly measured — the "1–3 tok/s" figure in circulation is
unsourced and must not be budgeted against.

The M5 Ultra Studio is the right purchase for a **different goal**: serving and deployment. At 256 GB it
would run the 27B unquantized or host many akhs concurrently, and would remove the 16 GB ceiling that
motivates the 8–12B target in the first place. That is a real justification — it is simply not a training
one, and should be decided on serving requirements, not on this plan.

**A cheaper lever than either purchase exists.** Our Burn/Metal stack measures ≈1.5k tokens/s on the 33.4M
model (`training/results/p6-1-metal-smoke-20260913-RESULT.md`) = `6·N·D` ≈ **0.30 TFLOPs against ~7.1
TFLOPS fp16 peak, roughly 4 % MFU**, where tuned CUDA training reaches 40–55 %. Gradient checkpointing and
recompute-backward attention deliberately trade FLOPs for memory (raw MFU perhaps 7 %), and a
`d_model=512` model has poor arithmetic intensity by nature, so this does not straightforwardly
extrapolate. But **any purchase multiplies whatever MFU the stack achieves**, so measure it at a realistic
width before spending.

## 7. Open items that could change this plan

- **The 100-world development pack does not exist.** Stage 0 cannot run without it. Critical path; file it
  first.
- **Whether layer-similarity pruning is valid for hybrid gated-delta architectures** is unresolved.
  ShortGPT validated its BI metric on Mamba and RWKV, which is suggestive but not equivalent. If it does
  not hold, the served 27B cannot be a pruning parent — but the Apache-2.0 Gemma-4 and Qwen3.5 dense
  models can.
- **MLX full-finetune throughput on M5 silicon is unmeasured by anyone.** Measure before any Mac training
  purchase.
- **Our 4 % MFU figure is one width on one model**; it needs a second point at ≥1B before it justifies or
  rejects a purchase.
- **Apertus v1.5 tool-calling accuracy is unmeasured.** BFCL is named in its vLLM docs but no score is published; the
  v1.5 tech report is still unreleased. Stage 0 measures it on our contract instead of trusting the claim.
- **GLM-4.5-Air's licence is inconsistent** between README (MIT) and LICENSE file (Apache-2.0); pin an
  exact revision if it is ever a candidate. Both readings clear the gate.
- Whether ~60k engine-generated episodes reproduce xLAM's result **for our contract** rather than for BFCL
  is untested — the leaderboard measures function-calling, the R1 contract measures graph acquisition,
  restart recall and repair as well.

## What this plan does not claim

It does not claim any model passes any gate, that the 100-world packs exist, that a 12B QLoRA run will
close the capability gap, or that a specific throughput will be achieved on unmeasured silicon. Every
number is labelled measured, published or estimated; the estimates must not be promoted to measurements by
later citation.
