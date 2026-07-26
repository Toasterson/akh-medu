# ADR 092 — High Grug: The Reasoning Register (v1 trace format → v2 ratified language)

- **Status:** **v2 Accepted (user-grilled 2026-07-11).** v1 was self-grilled; the D1–D6 resolutions were reopened and ratified/extended under user grilling into the v2 design in the section immediately below. v1's text is retained intact beneath it (never delete). Number provisional — reconcile on merge.

---

## v2 — Ratified under user grilling (2026-07-11)

**Deciders:** Till (grilled) + Fable 5. **Inputs added since v1:** the measured [P3-R + mixed-depth results](../2026-07-11-phase51-p3r-results.md) (supervision +0.48, curriculum-bounded depth, render-straitjacket), [ADR 093](093-dual-process-trace-handoff.md) (handoff economics), the [Phi-curriculum memo](../2026-07-11-phi-curriculum-350m-memo.md), and the 350M scale-up target. **Consumers:** the [Phase 51 P4 350M-foundation plan](../plans/2026-07-11-phase51-p4-350m-foundation.md).

### What changed since v1
v1 designed a trace format for an 11.7M synthetic-KG lab. v2 designs the reasoning register the **350M organ thinks in** — a *typed logic over a closed operator set*, its reserved token space, and how it threads pretrain vs SFT. v2 **supersedes** v1's D3 (entity-words → reserved special tokens + slots) and D1 (op inventory → larger, with logical connectives) and **promotes to first-class** the bilingual/pretrain-fluency channel v1 deferred.

### The governing principle (Egyptian, made literal)
The early Middle Kingdom sign set was small and *designed*; the Ptolemaic explosion was institutional decay, not linguistic necessity. Two intelligences with a shared goal hold the line: **the operator core is closed by governance, and the governance rule is "a corpus must measurably exercise it," not "it would be nice."** The Egyptian structure maps onto the register almost exactly, and we take the *structure*, never the *sign inventory*:

| Egyptian device | High Grug v2 |
|---|---|
| **phonograms** (small closed sound-sign set, composed) | **operators** — the closed, learned-as-schema core (Q3/Q6) |
| **determinatives** (unpronounced category glyph) | **slot type-classifiers** `x7·PLACE` (Q2) |
| **phonetic complements** (redundant error-check signs) | **`got` / `by i j`** — self-verifying redundancy (the *complement principle*: every statement carries the redundancy to verify itself) |
| **logograms** + ideogram-stroke marker | reserved SymbolId tokens (Q1, **dark**) + a reserved promotion marker |
| **cartouche** (name-scope oval) | microtheory/goal scope (deferred) |

### The eight ratified decisions

- **Q1 — Entities are entity-agnostic positional slots** `x1..xN`, bound to SymbolIds at the boundary. The reserved SymbolId block is **carved but left dark** (logogram promotion deferred, forward-compatible). *Supersedes v1 D3.* Rationale: never couple the tokenizer to a KG snapshot (sovereignty/relocatability); the Phi memo confirms baking in symbols = knowledge-packing, the failure mode we deleted.
- **Q2 — Determinative/classifier tokens on slots** (`x7·PLACE`), adopted as a **measured ablation** (typed vs bare) at 11M, with the sharp hypothesis that typed slots reduce **off-path derailment** (relation choice is type-gated: `RelationKind::of_source(ty)`).
- **Q3 — Larger closed operator core:** trace-mechanics (`step recall got derive by check gap goal call-answer`) + **structural inference relations** (`is-a part-of causes before after`) + **logical connectives** (`and or not implies`). ~19 ops — larger but closed. *Supersedes v1 D1.* The operator set is chosen to be **exactly the engine's inference-licensing predicates**, so organ, System-2, and verifier share one vocabulary (seamless ADR-093 handoff).
- **Q4 — Full commit to rule-based (first-order) inference.** `implies` introduces universally-quantified KG rules applied by **modus ponens** — the capability jump over 11.7M's graph-walking, mapped onto the `egg` e-graph. Corpus in two arms — **propositional** (`and/or/not` over ground facts) and **rule-based** — each **gated at 11M** before entering the 350M corpus.
- **Q5 — Full measurement instrumentation.** Decompose rule-application into `select / bind / fire / chain` so partial capability is *visible* (a false-abandon guard: an 11M fail must distinguish *broken* from *capacity-gated*). Run an **11M→50M→100M capacity ladder on both arms** and read the *slope*; reuse per-step replay + `probe_lens` crossover as "is-it-forming" reads.
- **Q6 — Operators/classifiers/markers are reserved special tokens** (the designed phonogram set) in one governed region of the definitive tokenizer + the dark SymbolId block; NL BPE for the open lexicon; **grammar parses from token IDs**. *Supersedes v1 D3's "learned ASCII words."* Enables VSA/gloss embedding-init (a head-start prior). Cost: tokenizer↔renderer lockstep is now load-bearing.
- **Q7 — Level 4 threading: High Grug enters *pretrain* as fluency.** Engine-generated High-Grug fluency + `gloss↔trace` bilingual pairs in pretrain + VSA/gloss embedding-init — because the emergence probes say the workspace must be *imposed*, and GWT #4 (articulation improves silent reasoning) is the mechanism. **The fluency corpus is engine-generated (sovereign, correct, zero teacher)** — the sharpest edge over Phi. Gated by an 11M/50M pretrain-fluency-vs-cold-SFT ablation.
- **Q8 — Teacher-free sovereign base + optional measured teacher arm.** Base = FineWeb-Edu + DE/Swiss (comprehension) + engine fluency/traces (reasoning); no teacher. Optional Phase-2a teacher-prose arm generated via **Ollama Cloud** (quota cleared) as an A/B, **provenance-flagged**: the experiment may use cloud, but if the arm wins the shipped recipe must be reproduced with a **local open-weight** model first.

### What v2 still defers
NSM primes, microtheory/cartouche scope ops, epistemic/DS masses, TMS ops (`assume/contradict/retract`), inter-akh Grug, the Phase-26 VSA projector for logogram entities — each still waits for a corpus that measurably exercises it. The governance rule is unchanged; the operator core simply grew to the set the 350M reasoning corpus will exercise.

---

## v1 (self-grilled, 2026-07-11) — retained

- **Status (v1):** Accepted (self-grilled under the standing /goal to implement the training roadmap; **revisable on user grilling** — the D1–D6 resolutions below are the points to re-open). Number provisional — reconcile on merge.
- **Date:** 2026-07-11
- **Deciders:** Fable 5 (autonomous, per /goal); Till to ratify/amend
- **Inputs:** [grilling brief](../2026-07-11-phase51-high-grug-v1-grilling-brief.md); [emergence-probe results](../2026-07-11-phase51-gwt-emergence-probes.md) (NEGATIVE at 6 layers — imposition, not emergence); [ADR 088](088-high-grug-boundary-format.md) (v0 pipe format, locked); [research 2026-07-04](../research/2026-07-04-grug-reasoning-language.md) (six design laws); Phase 51 plan §2a (GWT amendments)
- **Relates:** unblocks P3-reasoning (plan item 3); feeds the P1b definitive tokenizer (reserved SymbolId block per the [Apertus memo](../2026-07-11-phase51-p1b-apertus-scoping.md))

## Context

The tool-calling arc (P0→P3d) is closed: weights do in-distribution calls, the ADR-089 organ does the rest. The next skill is reasoning, and the emergence probes settled the architecture question the same day: a 6-layer organ forms no internal workspace (Stage L crossover at layer 5–6 of 6; Stage S source spans dead by mid-stack). The workspace must therefore be **imposed in the token stream** — the model must externalize state by emitting it and re-reading it. High Grug v1 is that imposed register. ADR 088's v0 covers only the *call* boundary; v1 extends the same pipe-form philosophy to reasoning traces.

## Decision — the v1 language

### Syntax

One workspace state per line; structure by leading op word + ` | ` separators; free-form inside segments, hard grammar only at statement boundaries; plain decode-safe ASCII (no tokenizer specials). A trace over prompt-supplied facts:

```text
prompt:  fact x7 lives_in c04 . fact c04 part_of r02 . <distractor facts...>
         goal region_of x7
target:  step 1 | recall lives_in x7 | got c04
         step 2 | recall part_of c04 | got r02
         step 3 | derive region_of x7 r02 | by 1 2
         call answer | value r02
```

### D1 — trace-op inventory: 9 ops, nothing unexercised

`step recall got derive by check ok gap goal` plus the v0 structural carryovers (`call`, `tool`, ` | `). Resolution rationale:
- **`assume` / `contradict` / `retract` are v2** — the deterministic synthetic KG exercises no belief revision; an unexercised token is an unmeasurable token (brief's own rule). They enter when a corpus with contradictory facts exists (TMS integration).
- **Epistemic/DS-mass buckets are v2** for the same reason — deterministic facts carry no uncertainty.
- **`gap` is reused as-is** from ADR 089/P3c: a goal unanswerable from the prompt facts targets `call <gap>`-analogous abstention (`step 1 | check <missing> | gap` then `call answer | value gap`). One abstention mechanism across skills, not two.

### D2 — explicit step numbers with `by <i> <j>` back-references

Supervisability beats density at 13M: every `derive ... | by i j` is independently checkable against steps i and j, which is exactly what intermediate supervision (GWT #3) and the TraceReplayer need. The off-by-one hallucination risk is itself a *measurement* (replay catches it as a broken reference). Positional/implicit chaining reconsidered at P1b scale.

### D3 — entities as plain ASCII words in v1; token-ID minting deferred to P1b

No special-token machinery for the P3 gate run: ops and entity names are ordinary words; the synthetic BPE learns them as single merges organically (measured: the procedural corpus saturates BPE at word level, so frequent ops/entities *will* be single tokens). The reserved contiguous SymbolId block (1–4k IDs) is a **P1b definitive-tokenizer** commitment, not a P3 one; the Phase-26 VSA projector remains gated on its own phase. Consequence: v1 traces are byte-identical whether or not minting has happened — the format survives the tokenizer transition.

### D4 — bandwidth limits as initial values, swept not assumed

≤ 4 payload symbols per statement (grammar-enforced), ≤ 8 active entities per step (corpus-supervised). These implement the Goyal/Bengio bandwidth constraint **in the format**; the architectural slot behind `Attention<B>` stays in the back pocket unless format-level imposition fails (probe crossover does not move down during P3 training).

### D5 — mint into the throwaway synthetic BPE now

The P3 run is a gate, not the final model; P1b re-pretrains from scratch with the definitive tokenizer anyway (plan line 40). Practically, with D3, "minting" for P3 is just training the usual per-run BPE over the trace corpus.

### D6 — engine replay is the scorer

`TraceReplayer` (grug-data) replays every emitted statement against the prompt's own facts: `recall r x` must match a prompt fact, `got` must carry the looked-up object, `derive ... by i j` must be the valid join of steps i and j, `gap` must correspond to a genuinely absent fact. **Executability rate** joins answer accuracy as a first-class metric — this is the SynthLang-killer (design law 3): valid-but-wrong *reasoning* becomes measurable the way `parse_call` made valid-but-wrong *calls* measurable.

### Bilinguality — deferred to P1b

Fixed reductive glosses per op are documentation in v1 (`derive` = "from what I know, it follows that"). Gloss↔trace training data belongs to P1b pretrain (GWT #4: articulation and silent reasoning share representations), not to the P3 gate, which measures reasoning in isolation.

## Consequences

- P3-reasoning (plan item 3) is unblocked: procedural multi-hop corpus over a synthetic KG, targets = v1 traces, intermediate supervision by construction (LM loss over every trace token), answer-only arm as the A/B control.
- The five measurement instruments from the brief stand: executability rate, intermediate-supervision differential, probe_lens crossover migration per checkpoint, true intra-forward read-by-many (multi-question prompts now expressible), anti-collapse + two-axis holdout.
- Deferred set is explicit: NSM primes, classifiers, microtheory ops, epistemic masses, TMS ops, inter-akh Grug, VSA projector, token-ID minting — each waits for a corpus that exercises it.

## Alternatives considered

- **Lock the full v0-sketch vocabulary (~245–450 tokens) now:** rejected — unexercised tokens are unmeasurable; the tool-calling arc's lesson is to add mechanism only against a measuring corpus.
- **Wait for user grilling before any lock:** rejected under the standing /goal; this ADR records the strawman + resolutions so the grilling has a concrete artifact to amend rather than a blank page.
- **Latent/continuous reasoning (no discrete register):** rejected in the research doc (§2.2) and now doubly so — the probe shows the model cannot hold state internally; an externalized discrete register is the only medium it can revisit.
