# Phi ("Textbooks Are All You Need") as Curriculum Blueprint for a 350M Sovereign Reasoning LLM

**Date:** 2026-07-11
**Status:** Research scoping (sourced) — feeds the [Phase 51 P4 plan](plans/2026-07-11-phase51-p4-350m-foundation.md)
**Purpose:** Ground the 350M pretrain design in what Phi's papers measured; separate transferable principle from Phi-specific knowledge-packing; flag every unverified number.

## 1. The Phi thesis, precisely

- **phi-1** (1.3B): <7B total tokens — ~6B *filtered* web code (educational-value classifier over The Stack + StackOverflow) + <1B GPT-3.5 synthetic textbooks + ~180M-token synthetic "CodeExercises". 4 days × 8 A100. **HumanEval 50.6%, MBPP 55.5%**, beating models 10×+ larger. Ablation: filtering raised HumanEval 12.2→17.7% at fixed steps; +synthetic → 20.1% — **data curation ≈ 5–8 HumanEval points at fixed size/compute.** [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)
- **phi-1.5** (1.3B): phi-1 code data + ~20B new synthetic "textbook-like" NLP tokens, trained 150B tokens (≈5 epochs), 1.5K A100-h. **Beats Llama2-7B on WinoGrande/ARC-C/SIQA (reasoning) but loses on HellaSwag/MMLU (knowledge).** [arXiv:2309.05463](https://arxiv.org/abs/2309.05463)

**The fault line that is our whole thesis: the textbook recipe buys *reasoning*, not *world knowledge*.** We offload knowledge to the KG, so we sit exactly on the side of the line Phi's data wins.

## 2. Synthetic generation

phi-1 teacher = **GPT-3.5**; diversity via topic+audience constraints (textbooks) and constrained function names (exercises); decontamination = 13-gram sweep + embedding/AST near-dup pruning (thresholds 0.95→0.8). phi-1.5's 20B synthetic seeded from **20,000 curated topics** + web-snippet injection for lexical diversity (teacher not explicitly named — **unconfirmed**). [2306.11644, 2309.05463]

## 3. Classifier filter (our FineWeb-Edu inheritance)

phi-1's "educational value" classifier = random forest over codegen embeddings, trained on ~100K GPT-4-annotated samples (survival fraction **unconfirmed**). **FineWeb-Edu operationalized the identical idea with an open-weight annotator:** Llama3-70B scored ~450–500K samples 0–5, a Snowflake-Arctic-Embed-M + linear head classifier was trained and applied corpus-wide; threshold **≥3 keeps ~8%** (~1.3T of ~15T tokens), F1≈82%. [fineweb-edu-classifier card, [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)] **⇒ our P1b FineWeb-Edu choice already inherits Phi's filtering trick sovereignly (open-weight teacher, not GPT-4).**

## 4. phi-2/3/4 — transferable principles

- **phi-3-mini** (3.8B, 3.3T tokens): explicitly **"data optimal" not compute-optimal** (~870 tok/param, not Chinchilla's 20); **two-phase curriculum** (Phase 1 general web breadth → Phase 2 heavily-filtered web + synthetic reasoning). [arXiv:2404.14219]
- **phi-4** (14B, ~10T): mixture ≈ synthetic 40% / web-rewrite 15% / organic web 15% / code 20% / academic 10%. Two findings: **(a) more epochs over synthetic beat fresh web tokens for reasoning**; **(b) synthetic-only underperforms on knowledge + hallucinates → they keep organic web *for knowledge grounding*.** [arXiv:2412.08905]

**⇒ Finding (b) is Phi's reason to keep organic web — a failure mode we architecturally deleted (KG owns facts). We keep web *only for linguistic breadth*, a far smaller requirement.**

## 5. Scaling-down reality check

- **phi-1-small (350M, same recipe): HumanEval 45%** (vs 1.3B's 50.6%, ~11% relative drop). **The textbook advantage substantially survives to 350M** — our only apples-to-apples primary datapoint at target scale. [2306.11644] **Gap: no phi-1.5-at-350M exists** — the *common-sense* (non-code) textbook advantage at 350M is unmeasured by anyone.
- **TinyStories** (<10M params): narrowing the output distribution buys in-distribution fluency at tiny scale — theoretical backing for our own 11.7M near-perfect in-distribution result. [arXiv:2305.07759]
- (**1.5-Pints** [arXiv:2408.03506] on-topic by title but numbers unverified — PDF parse failed; follow-up if decision-relevant.)

## 6. Applicable recipe for us

**Token budget:** Chinchilla floor 350M×20 = **7B**; phi-1.5-ratio-scaled ceiling **~35–50B**. **Recommend 7–10B as a checkpoint-gate, extend only if still climbing.**

**Copy:** two-phase curriculum; classifier filter (inherited); diversity-injection prompting; aggressive decontamination; "epochs-over-synthetic beat fresh web" for reasoning.
**Do NOT copy:** knowledge-packing (deleted by design); English-only (P1b already diverges); **closed-API teacher dependence** (phi used GPT-3.5/4 — conflicts with sovereignty).
**Our lever Phi never had:** the thinking engine emits **perfectly correct** reasoning traces + glosses, zero teacher, zero hallucination — a *stronger* sovereignty position than Phi's own pipeline.

### Recommended 350M mixture + phasing (synthesis, not a Phi number)

| Phase | Share | Source | Rationale |
|---|---|---|---|
| **1 — comprehension** | ~40% | FineWeb-Edu sample-10BT + FineWeb2-HQ German + apertus-swiss | web text teaches fluency/register, not reasoning |
| **2a — synthetic reasoning prose** *(optional arm)* | ~15% | local-open-weight or Ollama-Cloud teacher (measured) | phi-1.5-style reasoning-language gains; sovereignty-flagged |
| **2b — engine High-Grug fluency + traces** | ~35% | our thinking engine (sovereign ground-truth) | correct reasoning, our differentiator |
| repeats over 2a+2b | to budget | — | phi-4: epochs-over-synthetic |

### Open decisions (resolved in the P4 plan / grilling)

Token budget point; teacher choice (→ optional arm, Ollama-Cloud for the experiment, local-reproduce if it wins); 2a:2b ratio; narrow-first phi-1-small-style run (→ yes); decontamination policy (→ 13+7-gram + embedding vs our holdouts); 1.5-Pints follow-up.
