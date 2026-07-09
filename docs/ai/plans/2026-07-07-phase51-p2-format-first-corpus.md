# Phase 51 P2 — Format-First Corpus + Real Model Building

- **Status:** A/B **DONE** (2026-07-07). Format **locked in** ([ADR 088](../decisions/088-high-grug-boundary-format.md)); zero-shot verdict → **the organ, not the weights** ([ADR 089](../decisions/089-organ-kg-zeroshot-path.md), Phase 51 P3). The format hypothesis was half-right: it removed the *tax* (half the context, trivially-valid syntax) but was **not** the zero-shot cure — a fourth negative on that axis.
- **Date:** 2026-07-07
- **Continues:** [P1 scaling results](../2026-07-05-phase51-p1-scaling-results.md) (§10 Step A, §11 Step B, §11.x Step B2), [Phase 51 plan](2026-07-05-phase51-grug-brain-baby-llm.md), [ADR 087](../decisions/087-grug-brain-baby-llm-training.md), ADR 088 (this pivot).

## 0. Why pivot (the evidence)

P1a-3→P1a-4 chased zero-shot tool-calling through capacity, data, constrained decoding, a copy/pointer mechanism, and copy-supervision. Every lever that added *model machinery* failed or barely moved zero-shot. The reframing question — **is JSON too complex a target for a tiny model?** — is backed by the literature and by our own numbers:

- **SLMs pay a "format tax."** *"The Constraint Tax"* and *"When Correct Isn't Usable"* (2026) find that for sub-3B models, format fidelity "consumes a material part of the generation problem": JSON's braces, quotes, per-field keys, arrays, and type constraints compete with the model's limited capacity for the actual task. Constrained decoding fixes *syntax* but yields "wrong answer with valid schema" — **precisely our Step A result** (constrained zero-shot schema 0.219 / arg 0.125).
- **The field already left JSON.** Llama 3.2+ emit **pythonic** calls (`get_weather(city='SF')`) explicitly to remove JSON-schema ambiguity; Gemma uses **special-token** delimiters (`<tool_call>`…`</tool_call>`) or ` ```tool_code `. Structure is carried by *single special tokens + a minimal payload*, not hand-balanced punctuation.
- **Phi sets the target.** phi-1-small = **350M**, 45% HumanEval, from curated *textbook-quality synthetic data + exercises* aimed at a **skill**, not world knowledge.
- **Our own data double-taxes.** We serialize **both** the offered schema (prompt) *and* the call (target) as JSON. Two structural taxes on a 13M model.

**Decision (ADR 088): retire JSON as the boundary format. Adopt a minimal, special-token-delimited "High-Grug v0" format for both the offered schema and the emitted call.** Then the tiny model spends capacity on schema-reading and value-grounding, not brace-balancing.

Sources: [Textbooks Are All You Need (2306.11644)](https://arxiv.org/abs/2306.11644), [The Constraint Tax (2605.26128)](https://arxiv.org/pdf/2605.26128), [When Correct Isn't Usable (2605.02363)](https://arxiv.org/pdf/2605.02363), [vLLM tool-calling (pythonic vs JSON)](https://docs.vllm.ai/en/latest/features/tool_calling/), [FunctionGemma formatting](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices).

## 1. High-Grug v0 — the boundary format

Design goals, in priority order: **(a)** structure decisions are "emit the next special token" (no character-level balancing/quoting/nesting); **(b)** deterministic, trivial parse (split on special tokens); **(c)** values may contain spaces without quoting; **(d)** aligns with the "tokens are structural symbols" High-Grug thesis (these special tokens are the first designed symbols).

Five new special/added tokens (registered in the BPE tokenizer next to `<bos>`/`<eos>`/`<pad>`/`<sep>`): `<tool>`, `<param>`, `<call>`, `<arg>`, `<end>`.

**Offered schema (prompt side)** — one segment per tool, params inline:
```
<tool>book_table<param>party_size int req<param>time str req<tool>get_weather<param>city str req
```
Within a `<param>` segment: `name TYPE REQ` (space-separated; TYPE ∈ {str,int,num,bool,enum}; REQ ∈ {req,opt}; for enum, variants follow: `mode enum req low medium high`).

**Emitted call (target side)** — the model's output:
```
<call>book_table<arg>party_size 4<arg>time 20:00<end>
```
Within an `<arg>` segment: the first space-delimited token is the arg **name**, the **rest of the segment** (up to the next special token) is the **value** — so spaces in values need no quoting. No `{}`, `""`, `:`, `,`. Structure is entirely special tokens.

Parsing a call for scoring: split on `<call>`/`<arg>`/`<end>`; the tool name is between `<call>` and the first `<arg>`; each `<arg>` splits once on the first space into (name, value). Deterministic and total.

**Why this should help zero-shot specifically:** the offered schema is now flat and special-token-delimited, so "which params does *this* tool have" is a trivial read (each `<param>` is one segment), not a JSON-array parse. The call mirrors the schema's structure, so emitting the right params is copy-adjacent-by-layout. Constrained decoding (Step A grammar) ports trivially to this format if we still want the syntax floor.

## 2. The first experiment — format A/B (the real-model-building step 1)

Hold **everything** constant from the `p1a2-ep8` recipe (seed 42, 1000 procedural tools, 13M base `TinyTransformer` — **no pointer**, back to the clean base model, 8 epochs, easy 2-distractors) and change **only the target/schema format**: JSON vs High-Grug v0. Fresh BPE tokenizer per format (Grug adds the 5 special tokens). Eval the same two axes; the scorer is format-aware.

- **Gate / hypothesis:** Grug format lifts in-distribution arg-correct above the JSON baseline (0.78) **and** moves zero-shot schema meaningfully off 0.03 — because the structural tax is gone and the schema is readable. If it clears zero-shot where every model lever failed, format was the real bottleneck.
- **Snapshotting:** checkpoint every 500 steps (frequent, per the new directive) so we can bisect *when* a capability appears and resume cheaply.
- **Runtime:** ~1 hr GPU, local (no API cost).

If confirmed, this retroactively explains the whole P1a-4 arc: capacity/data/copy all failed because they targeted the wrong bottleneck.

**Result (measured, `runs/p2-grug/`, base 13M, seed 42, only format varies):**
- **Tax removed:** encoded seq len **halved** (median 109 vs JSON's ~345); valid-format trivial from step 0.
- **In-distribution accuracy: a wash** — Grug arg tracks *near, marginally below* JSON (~0.63–0.66 vs ~0.78). Freed capacity did not visibly lift accuracy on this synthetic task.
- **Zero-shot: unchanged** — axis-A schema **mean 0.030, max 0.055**, identical to JSON's 0.031, flat the whole run.

**Verdict:** the format is the right boundary representation to **lock in** (efficiency + reliability + High-Grug alignment) but is **not** the zero-shot cure. Four levers now leave zero-shot at ~0.03. → [ADR 088](../decisions/088-high-grug-boundary-format.md) locks the format; [ADR 089](../decisions/089-organ-kg-zeroshot-path.md) moves zero-shot correctness to the organ/KG (P3).

### P3 organ results (as they land)

- **P3b — KG-backed repair (mechanism 2), 2026-07-07:** ✅ the first lever to move zero-shot — arg-correct **~0.03 → ~0.55–0.60 (≈20×)** on two independently-trained models, no retraining. Zero-shot was a *grounding* problem, not a weights problem. See [scaling-results §12](../2026-07-05-phase51-p1-scaling-results.md#12).
- **P3c — gap-token abstention (mechanism 3), 2026-07-08:** ✅ built + measured (`runs/p3c-gap/`, `examples/train_gap.rs`). Abstention *is* in-distribution learnable — recall off zero to a **~0.24 plateau at ~0.92 precision, ~0.02 false-abstention, no call regression** — categorically unlike zero-shot (which no lever moved). But a 13M model **under-abstains** (recall low) *under random distractors*: a precise, conservative safety valve. See [scaling-results §13](../2026-07-05-phase51-p1-scaling-results.md#13) and [ADR 089](../decisions/089-organ-kg-zeroshot-path.md).
- **P3a — retrieval-scoped tool universe (mechanism 1), 2026-07-08:** ✅ built + measured (`runs/p3a-registry/`, `examples/train_registry.rs`, retriever `crates/grug-data/src/retriever.rs`). **Two wins: (1) premise proven — retrieval recall@8 = 1.000 over the 850-tool registry → the target is essentially always offered, so zero-shot-to-unseen-tools is dissolved by scoping; (2) M1 rescues M3 — abstention recall jumps 0.24 → ~0.998 because retrieval-scoped offered sets make "right tool present vs only near-misses" a crisp signal.** Registry call reliability (schema 0.91 / arg 0.61) matches the easy baseline despite offering 8 confusables. **All three ADR 089 mechanisms now built + measured.** **Full stack (M2 wired into the retrieval eval, `eval_registry_repair.rs`, n=256): registry call arg 0.594 raw → 0.594 repaired (Δ +0.000).** Error-structure inspection (`inspect_registry.rs`, n=400) **refutes the guess that this gap is tool-selection error: 0/160 arg-wrong cases chose the wrong tool.** The residual is argument fidelity — **value-only 68%** (right tool+names, wrong value; partly float-format noise), **presence-only 22%** (optional-inclusion mismatch), **invalid-name 10%** (M2-repairable). So tool selection is *already solved* (M1 + model) → **a re-ranker is a no-op**; the last-mile residual is argument-value fidelity. **Value-grounding (`ground_values`, composed after M2): since the correct value is in the request 100% of the time and ~70% of errors are numeric, snap each mis-copied numeric value to the nearest unused request number + wrong enums to the request's variant → whole-call arg 0.594→0.738 (+0.145, ~24% rel), per-value 0.748→0.824, beats the easy-2 baseline.** The last-mile stack is now **four** stages: retrieve (presence) → propose (selection) → M2-repair (naming) → value-ground (values) → M3-abstain (novelty); the leftover residual (in-request misalignment, string/bool) is genuine model value fidelity. See [scaling-results §14–15](../2026-07-05-phase51-p1-scaling-results.md#14) and [ADR 089](../decisions/089-organ-kg-zeroshot-path.md).
- **P3a follow-up levers, 2026-07-09** (`grug_data::{ground,retriever,organ}`, `crates/grug-data/examples/gap_organ.rs`): (a) **string value-grounding** built + measured **null** on this corpus (grounded numbers byte-identical — the string values are a short closed bank the model copies correctly; residual is numeric/enum), booleans declined; (b) **semantic HDC retriever** probes the lexical caveat — clean recall@8 lexical 1.000 / HDC 0.980 (both key on the verbatim name), under name typos lexical → 0.000 / HDC → 0.588 (**robustness** is the HDC/VSA contribution; genuine paraphrase stays untestable on this name-only corpus); (c) **organ gap-response** (`GapResponse{Widen,Synthesize}`, `respond_to_gap`) built + measured — retrieval saturated (0 natural misses; widen proven synthetically), genuine novelty synthesize **1.000** / false-widen **0** (all-tokens `covers` test never fooled by a confusable); (d) **confirmatory seed-43 run** launched for n=2 on the headline (pending). See [scaling-results §15](../2026-07-05-phase51-p1-scaling-results.md#15).

## 3. Corpus design (delegated — reasoned here)

Principles, adapted from Phi + the SLM-format literature + our anti-memorisation finding:

1. **Format-minimal (§1).** The single highest-leverage change. Both schema and call in High-Grug v0.
2. **Skill-targeted, not knowledge-targeted (Phi).** The corpus teaches *tool-calling as a skill*: read an offered schema → emit a schema-consistent, value-grounded call. No world facts (those come from the KG at runtime — the project's whole thesis).
3. **Diversity over volume, to defeat memorisation (our P1a-2 finding + still true).** Keep the 1000+ procedural tool catalog (few examples/tool). Expand toward more roles/types/arities so no tool-shape is memorisable.
4. **Difficulty curriculum (Phi "exercises"):** single-required-arg → multi-arg → optional-arg mix → disambiguation among similar offered tools (the "hard distractor" axis, but now cheap because the format is readable) → multi-tool / no-applicable-tool (abstain).
5. **Value-grounding built in:** every arg value appears verbatim in the request (copying is *possible*); values drawn from a wide space so they aren't memorisable — this is where the real "grounding" skill lives, and the flat format lets the model actually exercise it.
6. **Reasoning slot reserved (P3):** leave room for an optional pre-call `<think>…</think>` trace (Gemma/Qwen style) — the eventual High-Grug *reasoning* language. Not in P2; P2 is tool-calling format only.

## 4. Real model building + frequent snapshotting (the standing shift)

Per the new directive, after the format A/B we stop one-off ablations and build the *pipeline*: curated corpus generator (§3) → base model → **checkpoint every 500 steps** → two-axis eval curve → keep the best. Snapshots are the unit of progress; every capability jump is bisectable and resumable. Scale (params, tokens, curriculum stages) increases only once the format+corpus foundation reads clean.

## 5. Status of the P1a-4 copy-mechanism thread (closed)

- **Step A (constrained decoding):** kept as a free inference-time syntax floor (7× zero-shot schema), format-agnostic.
- **Step B (pointer-generator):** null — gate collapsed (measured `p_gen` ≈ 0.96).
- **Step B2 (copy-supervision):** `p_gen` responds to a BCE (0.96→~0.77) but **grad-norm-clip caps λ's effect** (λ=1 and λ=4 gave identical `p_gen` at step 1000), and even with the gate open, in-distribution copying recovers slowly while **zero-shot plateaus ~0.05** — the copy *attention* doesn't learn accurate in-context pointing at this scale. Not the lever. Superseded by the format pivot.

The honest lesson: three model-machinery levers (capacity, copy mechanism, copy supervision) all failed on zero-shot because the bottleneck was upstream, in the **representation the model is asked to produce**.
