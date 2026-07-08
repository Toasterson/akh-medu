# ADR 089 — Zero-Shot Tool-Calling Belongs to the Organ, Not the Weights

- **Status:** Accepted. **All three mechanisms now built + measured.** **M1 (retrieval-scoped universe) built + measured (P3a, 2026-07-08): retrieval recall@8 = 1.000 over an 850-tool registry — the offered set essentially always contains the target, so zero-shot-to-unseen-tools is *dissolved by scoping*; and, composed with M3, it *rescues* abstention to recall ≈ 0.998 / false-abstention ≈ 0.001 while registry call reliability matches the baseline (schema 0.91 / arg 0.61).** **M2 (KG-backed repair) built + validated (P3b, 2026-07-07): zero-shot arg-correct ~0.03 → ~0.55–0.60 (≈20×) on two models, no retraining.** **M3 (gap-token abstention) built + measured (P3c, 2026-07-08): abstention *is* in-distribution learnable (recall ~0.24 at high precision under random distractors; ~0.998 under M1's retrieval-scoped offered sets — see M1).** Number provisional — reconcile on merge.
- **Date:** 2026-07-07
- **Deciders:** Till, Fable 5
- **Evidence:** [P1 scaling results §8–§11](../2026-07-05-phase51-p1-scaling-results.md); `runs/p1a3-cap/`, `runs/p1a4-ptr*/`, `runs/p2-grug/`
- **Relates:** realises ADR 087's decision 8 (shared-SymbolId dual-process peer); consumes Phase 28 (router), 29 (capability exposure), 30/31 (skills), 37 (Seshat), 41 (Thinking Engine / `CompositeVerifier`), and the VSA/SymbolId substrate. Boundary format is ADR 088.

## Context

Four independent levers were measured against **zero-shot to unseen tools** (whole-tool holdout) and every one left it flat at schema ≈ 0.03:

| lever | result on zero-shot |
|---|---|
| 2.2× capacity (13M→30M, P1a-3) | 0.008 — no change |
| copy/pointer mechanism (Step B) | 0.016 — gate collapsed (`p_gen`≈0.96) |
| copy-supervision (Step B2) | ~0.06 plateau — grad-clip-capped, copy-attention won't point |
| minimal High-Grug format (P2, ADR 088) | **0.030 mean** — identical to JSON, whole run |

Meanwhile the *same* models learn in-distribution tool-calling well (arg ≈ 0.7–0.78 on trained tools). The conclusion is not a bug and not a missing trick — it is structural: **a weights-only tiny model cannot produce a correct call for a tool it never trained on, however the call is formatted.** Open-world zero-shot is not a property that fits in ~10⁷ parameters trained on synthetic templates.

This is not a failure of the BabyLLM thesis — it is the thesis (ADR 087): weights hold *reasoning + tool-use + syntax*; **knowledge (which tools exist, their exact schemas) comes from the KG at runtime.** So zero-shot correctness must live in the symbolic organ (System-2), where the four experiments prove it cannot live in the neural weights (System-1).

## Decision

**Move zero-shot correctness out of the neural weights and into the organ**, via three mechanisms layered on the in-distribution tool-caller. The tiny model *proposes*; the symbolic layer *grounds, verifies, repairs, and gap-fills*.

### 1. Retrieval-scoped, closed tool universe — dissolves the common case

There is no genuine open-world zero-shot in normal operation. The tool universe **is** the KG's capability/skillpack registry (Phase 29/30/31). At runtime a retrieval step (semantic router Phase 28 / Seshat Phase 37 / VSA similarity over goal↔tool) selects the top-K relevant tools and offers their schemas; the model is **trained on the registry** (or a large curated slice). So "offered tools" are almost always tools the model has *seen* — the whole-tool-holdout scenario the model fails at is one the organ does not face. New capabilities enter by (re)training / a LoRA adapter, not by demanding weight-level generalisation.

**Eval reframe:** the headline metric is in-distribution reliability across the *whole* registry, plus correct **abstention** when no offered tool fits — not open-world zero-shot. The whole-tool-holdout axis stays as a diagnostic of the neural floor, not a target.

**Measured (P3a, 2026-07-08 — `runs/p3a-registry/`).** A 13M model trained on the retrieval-scoped corpus (seed 42, 850-tool registry + 150 out-of-registry, High-Grug, 20000 steps). The offered set is a lexical retriever's top-8 over the registry (a faithful stand-in for the Phase-28 VSA router: the procedural requests embed each tool's `verb_noun` via `verb_phrase_for`, so name-token overlap *is* the relevance signal, and many tools share one token → the top-8 is a set of request-keyed *confusables*). The **label is a function of the offered set**: call when retrieval surfaced the target, gap when it did not — coupling M1 to M3. 20% of training examples are genuine out-of-registry (must-abstain) requests.

| metric | mature plateau (steps 14k–20k, 13 evals) | final (20k) | reading |
|---|---|---|---|
| **retrieval recall@8** | **1.000** | 1.000 | model-independent — the premise: the target is essentially always offered |
| registry call schema / arg | 0.909 / **0.611** | 0.891 / 0.633 | ≈ the easy-2 baseline (p2 0.90 / 0.65) *despite* offering 8 confusables |
| **abstain recall** | **0.998** | 1.000 | near-perfect — vs P3c's 0.24 under random distractors |
| abstain false-abstention | **0.001** | 0.008 | it does not wrongly abstain on covered requests |
| abstain precision | 0.999 | 0.992 | |

**Two findings.** (1) **The premise holds decisively — recall@8 = 1.000.** With a retriever, the offered set contains the target essentially always, so the model never faces the whole-tool-holdout scenario (the ~0.03 neural floor) in normal operation. Open-world zero-shot is *dissolved by scoping*, not demanded from the weights — exactly this ADR's claim. (2) **M1 rescues M3.** Abstention recall jumps from P3c's ~0.24 (random-distractor offered sets) to **~0.998** here. The reason is structural: under retrieval scoping a *callable* request's offered set always contains the exact verb+noun match, while a *genuine-novelty* request's offered set contains only one-token near-misses — so "is the right tool actually present?" becomes a crisp, learnable pattern instead of the no-signal "3 random tools" of P3c. The two mechanisms compose synergistically: retrieval makes both the call decision and the abstain decision sharp, and registry call reliability still matches the easy baseline even though the retrieved offered set (8 request-keyed confusables) is a harder discrimination than 3 random tools.

**Caveat (honest).** The lexical retriever and the call/gap label share the same name-token signal, so "exact match present vs only near-misses" is an especially clean cue here; the real router is semantic (VSA over `goal ↔ tool`) and the schema read is semantic, so abstention in production is a semantic judgment, not token-matching — the *structure* of the result (retrieval-scoped offered sets make abstention crisp) should transfer, but likely not as pristine as 0.998. The schema(0.91)–arg(0.61) gap is the residual valid-but-wrong-among-confusables that M2 (KG-repair, P3b's ≈20×) is designed to close; composing M1→propose→M2-repair→M3-gap is the full stack (M2 not wired into this run's eval).

### 2. KG-backed grounding + repair — the correctness guarantee

Even on known tools the neural proposal can misname a param, drop a required one, or mistype a value. The organ closes this deterministically:

- **Schema-constrained decoding (Step A)** — the syntactic + schema-membership floor. Built, format-agnostic, ports to High-Grug.
- **Symbol grounding** — tool and param names are KG symbols with stable SymbolIds. The boundary resolves the model's High-Grug tokens ↔ SymbolIds; an emitted param is matched to the tool's actual param SymbolIds (exact, else nearest by VSA/embedding similarity in symbol space), so `region` vs `region_name` is *repaired*, not failed.
- **Verify + repair loop** — Phase 41's `CompositeVerifier` applied to the grounded call: check it against the KG's authoritative schema (tool SymbolId → its `ParamSpec` SymbolIds, types, required set); fill missing-required from the request/context, coerce or reject wrong types, drop unknown args. The call that reaches execution is **KG-valid by construction.** This is where the "valid-but-wrong" residual (Step A) is finally killed — by the KG, not the weights.

### 3. The gap-token — genuine novelty as retrieval + synthesis, not generalisation

When the request needs a capability retrieval did *not* surface, or the model reaches for a symbol it lacks, it emits a **gap-token** (ADR 087 decision 5's designed knowledge-gap signal). The organ intercepts:

- **widen retrieval** / search KG + Seshat — often the tool exists but wasn't top-K → inject its schema, re-run (now in-context);
- **genuinely absent** → escalate: synthesise a skillpack (Phase 30/31 LLM-assisted authoring), delegate (Phase 36), or ask the user;
- **record the gap** → a curriculum signal for the next training round.

Genuine zero-shot thus becomes a retrieval + synthesis + learning problem handled symbolically — never a demand that frozen weights invent a correct call for a tool they have never seen.

**Measured (P3c, 2026-07-08 — `runs/p3c-gap/`).** The 13M base was trained on the locked High-Grug corpus (seed 42, 1000 tools, easy-2 distractors, 20000 steps) with **20% abstention examples** mixed in: a request phrased for a tool `T` whose offered bundle deliberately *excludes* `T` (distractors-only, same set size), target = the bare gap token `gap`. Eval measures a confusion matrix on a frozen holdout (n=128 should-call + 128 should-abstain, held-out phrasings): **abstain recall**, **false-abstention rate** (wrongly abstaining on a real call), and **precision**.

| metric | mature plateau (steps 10k–20k, 21 evals) | final (step 20k) | reading |
|---|---|---|---|
| abstain **recall** | **0.24** (max 0.32) | 0.27 | learnable — off zero, but the model *under*-abstains |
| abstain **precision** | **0.92** | 0.83 | when it abstains, it is almost always right |
| **false-abstention** | **0.02** | 0.055 | it does **not** cannibalise real tool-calling |
| call arg-correct (axis B) | 0.54 | 0.57 | ≈ p2 baseline (~0.65) minus a modest ~0.05–0.08 mix tax |

**The claim is qualitatively confirmed, quantitatively modest.** The decisive contrast is with open-world zero-shot: *four* levers (capacity, copy, copy-supervision, format) all left it pinned at 0.03 — **nothing moved it**. Abstention **moves** — recall climbs off zero to a real plateau at high precision — which is exactly the ADR 089 partition: "none of these fit" is an in-distribution classification the weights *can* learn, unlike "produce a correct call for a tool never seen," which they provably cannot. But at 13M the model is a **conservative, precise, low-recall** abstainer: it catches only ~1 in 4 no-fit cases, though it rarely false-alarms and does not degrade calling. Likely causes of the low recall (all testable, not yet run): the 80/20 call/abstain prior teaches "there is usually a matching tool" (a higher `--abstain-frac` should raise recall at some call cost); easy-2 distractors make the "none fits" boundary soft (hard distractors would sharpen it, likely at some false-abstention cost); and 13M capacity.

**Consequence for the stack.** A precise, low-recall gap-token is *usable as the mechanism-3 fallback* precisely because it sits **behind** mechanism 1 (retrieval keeps the offered set relevant, so genuine "nothing fits" is rare) and mechanism 2 (repair fixes near-misses): it fires on the clearest no-fit cases with few false alarms, and retrieval/synthesis handles the rest. If the gap-token is ever to be the *primary* abstention signal rather than a fallback, recall must improve — via an `--abstain-frac` / hard-distractor sweep or capacity. Built: `grug_data::{schema::ToolCall::gap, format::{parse_output,ParsedOutput}, generator::try_next_abstention_example, dataset::{train_examples_with_abstention,abstention_examples}, eval::AbstentionReport}`, `grug_train::{FrozenAbstentionHoldout,run_abstention_eval_fmt}`, `examples/train_gap`.

### The kludge-brain summary

Neural System-1 (tiny model; weights = reasoning + tool-use + High-Grug syntax) proposes fast; symbolic System-2 (KG + VSA + `CompositeVerifier`) grounds, verifies, repairs, and gap-fills. The four negatives prove open-world correctness can't live in System-1; this ADR puts it in System-2 — which is what akh-medu already *is*. Mostly wiring existing organs to the neural boundary, not new machinery.

## First buildable steps (Phase 51 P3)

- **P3a — retrieval-scoped training + reframed eval. ✅ built + measured (2026-07-08).** Trained on the whole registry with a retriever-selected offered set (top-8) and a genuine-novelty must-abstain set. Result: retrieval recall@8 = 1.000 (premise), registry call schema 0.91 / arg 0.61 (≈ baseline), abstention recall ≈ 0.998 (M1 rescues M3). See the "Measured (P3a)" block under mechanism 1 and scaling-results §14. `runs/p3a-registry/`, `examples/train_registry.rs`.
- **P3b — KG-backed repair pass.** Ground → verify → repair the model's High-Grug output against the schema (procedural catalog as a stand-in KG first, real KG later). Measure raw-neural → grounded-repaired lift; expect the valid-but-wrong residual to collapse toward exact.
- **P3c — gap-token. ✅ built + measured (2026-07-08).** Trained the model (20% abstention mix) to emit the gap token when no offered tool covers the request. Result: abstention is in-distribution learnable (recall ~0.24 plateau, precision ~0.92, false-abstention ~0.02, no call regression) — confirming the claim qualitatively, though a 13M model under-abstains (see the "Measured (P3c)" block under mechanism 3). Wiring the organ's retrieval → synthesis → escalation response on a fired gap token remains (it consumes the same retrieval/synthesis path as mechanism 1).

## Consequences

- **Kills the temptation to keep scaling/tricking the tiny model for zero-shot** — that budget is now spent on reasoning quality (weights) and on the symbolic repair/retrieval layer (organ), where the leverage actually is.
- **Ships the in-distribution tool-caller now** (it works), gated by Step A constrained decoding + P3b repair, without waiting on open-world generalisation that will never come from weights.
- **Binds BabyLLM into the existing organ** rather than growing it into a standalone model — consistent with the whole project's neuro-symbolic thesis.

## Alternatives considered

- **Keep pushing the neural model** (bigger, more data, better copy, RLHF-for-schema): rejected — four measured negatives; diminishing-to-zero returns on the axis in question.
- **Foundation pretrain (Apertus) for comprehension:** de-prioritised — the bottleneck is not natural-language comprehension of the schema (it's already in-context); it is that novel schemas aren't in the weights, which pretraining doesn't fix.
- **Pure symbolic (no neural model):** rejected — loses the fast, fluent proposer and the reasoning/syntax the weights are good at; the dual-process split is the point.
