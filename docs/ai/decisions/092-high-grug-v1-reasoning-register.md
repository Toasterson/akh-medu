# ADR 092 — High Grug v1: The Reasoning Register (Trace Format for P3)

- **Status:** Accepted (self-grilled under the standing /goal to implement the training roadmap; **revisable on user grilling** — the D1–D6 resolutions below are the points to re-open). Number provisional — reconcile on merge.
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
