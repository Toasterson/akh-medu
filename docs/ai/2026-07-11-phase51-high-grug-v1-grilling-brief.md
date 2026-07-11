# High Grug v1 — Design Brief for Grilling (the Reasoning Register)

**Date:** 2026-07-11
**Status:** DRAFT — grilling material, not a decision. The ADR comes out of the grilling session.
**Inputs:** [research 2026-07-04](research/2026-07-04-grug-reasoning-language.md) (six design laws, v0 sketch, risks), [ADR 088](decisions/088-high-grug-boundary-format.md) (v0 call format, locked), [GWT amendments](plans/2026-07-05-phase51-grug-brain-baby-llm.md) (§2a items 1–3), [emergence-probe results](2026-07-11-phase51-gwt-emergence-probes.md) (NEGATIVE at 6 layers), [Apertus scoping](2026-07-11-phase51-p1b-apertus-scoping.md) (reserved token block), [ADR 089](decisions/089-organ-kg-zeroshot-path.md) (organ = workspace ops).

## What changed since the v0 sketch — three facts that reshape v1

1. **The probe says imposition, not emergence.** The 6-layer organ has no internal workspace (crossover at layer 5–6; source spans dead by mid-stack). High Grug is therefore not a *description* of a workspace the model grows — it **is** the workspace, held in the token stream, revisited by emitting and re-reading. Every design choice should optimize for "state that survives outside the residual stream."
2. **We own the tokenizer.** The v0 sketch's token mechanics (ToolkenGPT PEFT, mean-of-gloss init, GGUF CONTROL flags, DSI++ forgetting) are frontier-model retrofit machinery. BabyLLM trains from scratch on its own BPE — minting tier-1 tokens is a *vocab file entry*, not a research problem. The hard tier-2 question (VSA projector) can be simplified at our scale (see D3).
3. **The v0 boundary format works and is locked** (ADR 088): `call <name> | <arg> <value>`, pipe-separated, quote-free, decode-safe, sequence length halved. v1 extends this shape to reasoning traces; it does not reopen it.

## Strawman v1 (the thing to grill)

### Syntax — one workspace state per line, pipe-form throughout

```text
step <n> | <op> <payload...>
```

Extending ADR 088's grammar: structure carried by leading keyword + ` | ` separators, one statement per line, statement = one workspace write. Free-form inside segments, hard grammar only at statement boundaries (the format-tax dodge, arXiv:2408.02442). Examples over a KG-hop task:

```text
step 1 | recall inhabits x7 | got x7 city_04
step 2 | recall part_of city_04 | got city_04 region_02
step 3 | derive inhabits_region x7 region_02 | by chain 1 2
step 4 | check region_02 | ok
call answer | region region_02
```

### Tier-1 inventory — start brutally small (grill the blocks)

| Block | v1 count | Contents | Engine check (executability law) |
|---|---|---|---|
| Trace ops | ~12 | `step recall derive assume check contradict retract gap goal by got ok` | each maps to a KG/TMS/DS operation the engine can replay |
| Epistemic | ~6 | confidence-mass buckets (DS-aligned), `likely certain refuted` | DS interval comparison |
| Structural | ~4 | ` | `, `call`, `tool`, end-marker (all v0 carryovers) | parser totality |
| **Total minted** | **~22** | | |

Deliberately NOT in v1 (v0 sketch had them; defer): NSM semantic primes (~65–100), classifier/determinative tokens (~50–100), microtheory push/pop. Rationale: P3-reasoning's synthetic corpus doesn't exercise them, and every unexercised token is an unmeasurable token. Add blocks when a corpus measures them.

### SymbolId space — discrete block now, projector later

Reserve a contiguous trailing block (Apertus memo: 1–4k IDs) in the P1b tokenizer. At 13M/synthetic scale, entities enter as **minted discrete tokens with identity mapping to KG SymbolIds** (`x7` ↔ SymbolId) — no xRAG projector needed while the entity inventory is corpus-bounded. The Phase-26 VSA→embedding projector remains the open-world answer and gets its own gate later. This is D3 below — the biggest divergence from the v0 sketch.

### Bandwidth limit — the GWT constraint made syntactic

Grammar caps **≤ 4 payload symbols per statement** and the corpus supervises **≤ N active entities per step** (strawman N=8, measured). This is Goyal/Bengio's bandwidth-limited slot implemented in the *format* rather than the architecture — cheaper than a new attention type, and the probe gives us the metric (does the crossover layer move down when the register is narrow?). The architectural slot stays in the back pocket if format-level imposition fails.

### Bilinguality — gloss channel from day one

Every tier-1 op carries a fixed reductive English gloss (`derive` → "from what I know, it follows that"). Corpus trains both directions (trace→gloss, gloss→trace); Phase 33's verbalizer is the decompiler; monitors read the gloss. P1b pretrain carries High-Grug fluency text (GWT #4: articulation and silent reasoning share representations).

## Decision points for the grilling (D1–D6)

- **D1 — trace-op inventory:** is the ~12-op set right? Specifically: does `assume/contradict/retract` belong in v1 (TMS integration) or v2? Is `gap` (ADR 089's abstention token) reused as-is?
- **D2 — numbering:** explicit `step <n>` + `by chain 1 2` back-references make intermediate supervision trivial (each step independently checkable) but cost tokens and invite off-by-one hallucination. Alternative: positional (no numbers, `by chain` refers to previous lines implicitly). Strawman keeps explicit numbers — supervisability beats density at 13M.
- **D3 — entities as discrete tokens vs VSA projector:** strawman = discrete block for P3 (corpus-bounded), projector deferred to a Phase-26 gate. Grill: does this fork High Grug into two dialects, and is that acceptable?
- **D4 — bandwidth numbers:** ≤4 payload symbols/statement, ≤8 active entities/step — arbitrary until measured. Accept as initial values, sweep in the P3 corpus?
- **D5 — tokenizer timing:** mint v1 tokens into the *current* throwaway synthetic BPE for the P3-reasoning run (fast, disposable), or wait for the definitive P1b tokenizer? Strawman: mint into the synthetic BPE now — the P3 format-first run is a gate, not the final model, and P1b re-pretrains anyway (plan line 40).
- **D6 — engine replay:** should the P3 eval *actually replay* each trace op against the KG (executability law, the SynthLang-killer), or score traces textually against gold? Strawman: replay — grug-data owns the synthetic KG, so a `TraceReplayer` is cheap and makes "valid-but-wrong reasoning" measurable the way `parse_call` made calls measurable.

## What v1 explicitly does not decide

Inter-akh Grug over ACP (v0 sketch P4), NSM prime inventory, classifier tokens, microtheory ops, the P6 competitive-write/broadcast-read protocol details, and the Goyal/Bengio architectural slot (back pocket; format-level bandwidth first).

## Measurement plan (instruments already exist)

1. **Executability rate** — fraction of emitted statements that replay cleanly against the KG (new `TraceReplayer`, D6).
2. **Intermediate-supervision differential** — P3 corpus trained with vs without per-step supervision (GWT #3), same harness discipline as P2's format A/B.
3. **Crossover migration** — `probe_lens` per checkpoint during P3 training; a downward-moving crossover layer = the workspace is taking hold in the weights (probe report, follow-on metric).
4. **True read-by-many** — multi-question prompts sharing one workspace state make the intra-forward Stage S test runnable at last (the current proxy only shows patch generality).
5. **Anti-collapse + two-axis holdout** — unchanged from P1/P2/P3 discipline.
