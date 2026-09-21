# Model breeding with Gemma 4 as the trunk: what weight-space merging can and cannot do, and what has to be distillation

Answers the operator's question of 2026-09-20: *"is there some sort of model breeding — distilling traits of parent
models into child ones? We could use Gemma 4 as the architectural basis and distil other knowledge onto it. Is there
weight-combining research?"* Twelve primary sources were read in full text and every arXiv ID below was checked
against its title. Context: [8–12B hardware plan](../plans/2026-09-19-capable-8-12b-akh-hardware-plan.md) (Gemma 4
12B is the trunk as of 2026-09-20; Apertus is a routed language component; Qwen3.8-27B is the served workhorse).

## Short answer

Yes, on both counts, with one hard rule that decides everything: **weight-space methods need a common ancestor;
cross-lineage transfer must be distillation.** "Breeding" in the literal sense — searching over how to recombine
parents — exists and works (Sakana), but only among descendants of one base. So there are two different programmes:

1. **Within the Gemma 4 lineage: breed.** Train LoRA children on the trunk for separate traits, recombine them by
   merging, and let a search pick the recipe with the capability contract as fitness. Runs on one 24 GB card.
2. **From other families (Apertus, Qwen3.8): distil.** Their weights cannot be combined with Gemma 4's under any
   method. Their *behaviour* can be transferred by cross-tokenizer logit distillation or, for tool use, by plain
   SFT on execution-verified traces — which is what actually tops tool-calling leaderboards.

## 1. Weight-space merging: the methods and their precondition

| method | arXiv | what it does | precondition | strongest result | stated limit |
|---|---|---|---|---|---|
| Model Soups | 2203.05482 | average weights of many fine-tunes of one checkpoint | same base, same init, same basin | ViT-G soup 90.94 % ImageNet top-1 | fails once fine-tunes leave the shared basin |
| Task Arithmetic | 2212.04089 | task vector = fine-tuned − base; add/negate/combine | same base | negation removes toxicity with little fluency loss; addition multi-tasks | interference when vectors conflict |
| TIES-Merging | 2306.01708 | trim small deltas → elect sign → merge agreeing params | same base | beats averaging/Task Arithmetic across modalities and sizes | mitigates, does not remove, interference |
| DARE | 2311.03099 | drop 90–99 % of deltas, rescale the rest, then merge | **"homologous models"** (abstract) | #1 7B on Open LLM Leaderboard | degrades on heavily continued-pretrained models (§4.6) |
| Model Stock | 2403.19522 | Soup's centre from two fine-tunes via layer-wise proximity | same as Soups | matches Soups with 24× fewer runs | same basin assumption |
| Git Re-Basin | 2209.04836 | permutation-align *independently trained* nets, then average | same architecture only | zero-barrier LMC between independent ResNets on CIFAR-10 | **never tested on LLMs**; paper reports a counterexample to the single-basin hypothesis |
| LoraHub | 2307.13269 | gradient-free composition of many LoRAs for a new task | all LoRAs on the same base | approaches ICL on BBH | does not beat ICL outright |

Every method except Git Re-Basin needs elementwise tensor correspondence: same shapes, same layer semantics, and
**the same tokenizer indexing the embedding and unembedding matrices**. Git Re-Basin relaxes "same lineage" to
"same architecture" and has no LLM evidence at all.

## 2. Breeding proper

- **Evolutionary Model Merge — Sakana AI, 2403.13187.** CMA-ES over merge recipes in parameter space (blend
  weights) and data-flow space (stack layers from different models). Parents: shisa-gamma-7b-v1, WizardMath-7B-v1.1,
  Abel-7B-002 — **all fine-tunes of Mistral-7B-v0.1**. The bred model scores 52.0 on MGSM-JA when no parent exceeds
  30.0, and beats a 70B Japanese model. The paper states the constraint outright: parameter-space merging
  "confine[s] the space of models to be merged to fine-tuned descendants of the same parent base model". Data-flow
  merging has no such constraint but was only shown across two models and was scoped to fit one GPU. **Its Table 3 is
  the most important number for us: naive TIES on the same three parents scores 4.4, naive layer-stacking 0.0.** The
  search does the work; the merge operator alone is not safe to assume.
- **Branch-Train-Merge, 2208.03306, and Branch-Train-MiX, 2403.07816.** Branch experts from one seed, train each on a
  domain independently with no cross-node sync, recombine. A 22.4B BTM model (64 domains, 192B tokens) matched a
  dense model trained with 2.5× the compute. BTX folds the independently trained FFN experts into MoE layers,
  averages the rest, and adds a short routing fine-tune — and frames itself as generalising both BTM and Sparse
  Upcycling (2212.05055). This is the **sovereign-lineage** form of breeding: one seed, diverged, recombined into one
  deployable model.
- **Depth up-scaling — SOLAR 10.7B, 2312.15166** — duplicates layers of a smaller model and continues pretraining;
  single-parent growth, adjacent rather than breeding.

## 3. Cross-lineage transfer is distillation

- **FuseLLM, 2401.10491, and FuseChat ("Knowledge Fusion of Chat LLMs"), 2402.16107.** Exactly the "trunk + other
  families" idea. Trunk: Llama-2 7B. Teachers: Llama-2, OpenLLaMA, MPT — three architectures, three tokenizers.
  Tokenizers aligned by minimum-edit-distance dynamic programming over token sequences; the trunk is continually
  trained toward the fused distribution (MinCE or cross-entropy-weighted AvgCE). Gain: +5.16 % relative on BBH
  (27 tasks), +1.25 % on commonsense — real, modest, and **non-uniform**: Geometric Shapes −33.7 %, Ruin Names
  −4.7 %. Cost: **8× A100-40GB, ≈33 hours, 1.8B tokens** (MiniPile). FuseChat extends it to teachers of different
  sizes via pairwise fusion then a parameter-space merge of the same-size targets.
- **Universal Logit Distillation, 2402.12030.** Optimal transport over logit distributions; needs no vocabulary
  overlap heuristic at all.
- FuseLLM's own related-work section states why weight blending across families is not an option: it "typically
  assumes uniform network architectures… which is often unattainable in the context of LLMs".

## 4. What this means with Gemma 4 as the trunk

**Apertus ↔ Gemma 4 cannot be weight-merged.** Different tokenizers (Mistral-Nemo-derived vs Gemma's), different
activations (xIELU vs Gemma's), different norms. Coincidentally equal shapes do not make weights play the same
functional role. The same holds for Qwen3.8-27B, which is additionally a gated-delta hybrid.

The breeding programme that fits owned hardware:

| track | mechanism | fits one 24 GB card? | evidence | gate |
|---|---|---|---|---|
| **A. LoRA children on Gemma 4** — Portuguese, High Grug register, tool use, Dojo kata as separate adapters | **LoRA Soups "CAT" (2410.13025)** — layer-wise learned concatenation weights from a handful of examples — over TIES/DARE grid search; LoraHub-style composition as fallback | yes | CAT beats TIES/DARE/SLERP/linear by avg +43 % and data-mix SFT by +12 % on skill composition; interference is the known failure (2601.18350, 2510.03262) | capability contract per family, never loss |
| **B. Evolve the recipe** | Sakana-style search over A's children with the 100-world pack as fitness | yes — cost is wall-clock (~10² – 10³ evaluations), not FLOPs | 2403.13187, incl. its 4.4 → 52.0 naive-vs-evolved gap | same |
| **C. Distil tool use from Qwen3.8** | baseline: offline SFT on execution-verified traces (xLAM 2409.03215, ToolACE 2409.00920), the recipe that tops BFCL at 1B–8B; 2025 evidence that **teacher-reward RL distillation** (MENTOR 2510.18383) and **agent-trajectory distillation** (2505.17612) beat plain SFT out of domain — magnitude unverified | yes | `akh_episode_generator` already produces verified traces *with causal step rewards*, i.e. the MENTOR-shaped signal | same |
| **D. Distil Swiss-language competence from Apertus** | logit-level: FuseLLM/ULD (8× A100-40GB, 33 h — not a single-card job); **output-level: FuseChat-3.0 / WRPO (2503.04222, 2412.03187)** — SFT then DPO on teacher *outputs*, no vocabulary alignment, +6.8 avg over 14 benchmarks on an 8B student; mechanically QLoRA-shaped, so plausibly single-card, **cost unpublished** | logit-level no; output-level unverified | 2401.10491, 2402.12030, 2503.04222, 2412.03187, 2503.20083 | — |
| **E. Sovereign breeding on the ADAM ladder** | Branch-Train-MiX from one ADAM seed: domain branches, recombined into MoE | per branch, yes | 2403.07816 | ADAM gates |

Track D is the expensive one and is also the one the 2026-09-20 routing decision makes unnecessary: Apertus is
wired as a per-akh backing for Swiss-language turns, so its competence is *used*, not transferred. Revisit D only if
routing proves insufficient.

Gemma 4's own family shape matters for track E's dense counterpart: the 26B-A4B MoE sibling is what Sparse
Upcycling / BTX would produce from a dense trunk, so an upcycled-Gemma-4 experiment has a first-party reference
point to compare against.

## 5. Where merging fails, as the papers report it

- Sign and magnitude interference in naive averaging and Task Arithmetic (TIES's premise).
- **Untuned merges collapse**: Sakana's baselines — TIES 4.4, layer-stacking 0.0 — on parents that evolved to 52.0.
- Heavily shifted fine-tunes resist DARE's sparsification (a 100B-token continued pretrain degraded; only evolved
  reweighting recovered it).
- Selective task regression under an improving aggregate (FuseLLM's −33.7 % on one BBH task).
- Basin-assumption failure: Soups/Model Stock need a shared basin; Git Re-Basin's own paper shows a counterexample.

Two failure modes widely reported in community write-ups — instruction-following loss after merging and safety
regressions — did **not** appear as measured findings in any of these twelve papers. Treat them as plausible, not
established; the capability contract is the only gate that would catch them here.

## Second pass, 2026-09-20: tooling and successors (verified first-party unless marked)

- **Gemma 4 facts** (tech report 2607.02770, HF cards): 12B dense (11.95B), 31B dense, 26B-A4B MoE (3.8B active, 8 of
  128 experts + 1 shared); SentencePiece **vocab 262,144**, identical between base and instruct; **256K context** on
  12B; hybrid **5:1 local-sliding-window : global** attention (window 1024, p-RoPE on global layers), QKNorm, keys
  reused as values in global layers. **Separate base (`google/gemma-4-12B`) and instruct (`-it`) checkpoints exist**,
  so Task-Arithmetic deltas are available. FFN activation not stated in the retrievable report text (GeGLU in Gemma
  1–3; unconfirmed for 4).
- **`mergekit` does not support Gemma 4 upstream.** `mergekit/_data/architectures/` holds `gemma`, `gemma2`, `gemma3`,
  `gemma3vl` only. A community fork (`EldritchLabs/mergekit-exp`) carries an experimental `gemma4.json`; a few HF
  Gemma-4 `dare_ties` merges exist with **no reported benchmarks**. `mergekit-evolve` and `mergekit-extract-lora` exist
  upstream; native LoRA-level TIES/DARE is unverified. **Tracks A/B need either the fork, an upstream `gemma4.json`,
  or our own delta arithmetic** — a real gap, not a documentation lag.
- **LoRA Soups, 2410.13025** (Prabhakar et al.): independently trained skill LoRAs composed by "CAT", layer-wise learned
  concatenation weights fitted from a few examples; avg **+43 % over TIES/DARE/SLERP/linear** and **+12 % over
  data-mix SFT** on math-word-problems-with-code. This displaces grid-searched TIES/DARE as track A's operator.
- **Interference is documented, and worse than "some loss":** 2601.18350 shows merging a domain LoRA with an
  instruction LoRA can *reactivate suppressed behaviour* through final-layer representation conflicts; 2510.03262
  shows inter-LoRA orthogonality does not imply semantic disentanglement. Model Breadcrumbs (2312.06795) sparsifies
  task vectors at both tails to reduce it without per-task tuning.
- **FuseChat-3.0, 2503.04222, and WRPO, 2412.03187** (same group): "implicit model fusion" — SFT then DPO on the
  *outputs* of heterogeneous teachers (Gemma-2-27B-it, Mistral-Large, Qwen-2.5-72B, Llama-3.1-70B), no vocabulary
  alignment; **+6.8 avg over 14 benchmarks** on Llama-3.1-8B-Instruct, +37.1 AlpacaEval-2, +30.1 Arena-Hard. Compute
  cost not stated in the abstract or landing page. **Cross-tokenizer distillation via approximate likelihood
  matching, 2503.20083** (Minixhofer et al.), handles subword→byte transfer. **HeteroFusion, 2604.01674**, fuses across
  Llama/Qwen/Mistral backbones; whether it covers a gated-delta hybrid teacher is unverified.
- **Agentic distillation beyond SFT exists (direction confirmed, magnitude not):** MENTOR, 2510.18383, frames plain
  SFT as off-policy distillation with poor out-of-domain generalisation and reports RL with teacher-optimised rewards
  beating both SFT and strict RL out of domain; "Distilling LLM Agent into Small Models with Retrieval and Code
  Tools", 2505.17612, transfers full task-solving behaviour to 0.5B–3B students. Both read at abstract level only.
  This corrects the earlier note's "the field does not use distillation at all": plain SFT on verified traces is the
  2024 baseline that topped BFCL; 2025 work claims reward-shaped distillation beats it out of domain.
- **Language LoRA × tool-use LoRA interference: no primary source exists.** Closest analog is "Multi-linguistic LoRA
  Merging" (NeurIPS 2025, OpenReview `VAnFWVbYxG`, no arXiv ID found — community-tier provenance), merging a task
  adapter with a language adapter plus a post-merge alignment step. **Track A would generate novel evidence here,
  not confirm existing evidence.** That is fine, but it must be measured, not assumed.

## What we do not know

- Gemma 4's FFN activation (not in the retrievable report text).
- Whether `mergekit-exp`'s `gemma4.json` is correct; whether LoRA-level TIES/DARE exists natively anywhere.
- GPU-hours for FuseChat-3.0 / WRPO / ULD / 2503.20083 — none published at abstract level; the FuseAI repo is the
  place to check. Until then "output-level fusion fits a 24 GB card" is a plausibility, not a fact.
- The numeric gap between reward-shaped agentic distillation (2510.18383, 2505.17612) and plain SFT.
- Whether any heterogeneous-fusion paper has used a hybrid gated-delta teacher.
- Whether a Portuguese adapter and a tool-use adapter interfere on Gemma 4. Nobody has measured it.
