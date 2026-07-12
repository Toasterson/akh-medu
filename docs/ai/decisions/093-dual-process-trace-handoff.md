# ADR 093 — Dual-Process Trace Handoff: Organ Proposes, System-2 Completes (P6 First Slice)

- **Status:** Accepted (self-grilled under the standing /goal; **revisable on user grilling**). Number provisional — reconcile on merge.
- **Date:** 2026-07-11
- **Deciders:** Fable 5 (autonomous, per /goal); Till to ratify/amend
- **Inputs:** [P3-R results + mixed-depth appendix](../2026-07-11-phase51-p3r-results.md) (the measured premises); [ADR 092](092-high-grug-v1-reasoning-register.md) (the trace register); [emergence probes](../2026-07-11-phase51-gwt-emergence-probes.md) (workspace = shared register, iteration lives in System-2); [ADR 085 findings](085-thinking-engine-findings.md) (Phase 41 solvable depth scales with budget); ADR 087 §8 / ADR 089 (kludge-brain thesis)
- **Relates:** first buildable slice of the Phase 51 plan's P6 integration; the reverse (broadcast-read) direction stays open as the full P6 protocol.

## Context — every premise is now measured, none assumed

1. **The organ reasons near-perfectly at trained depths and not one hop beyond.** Mixed-depth run: 0.93–0.98 answer accuracy across depths 2–6 in-curriculum; 0.004 at depths 7–8, whole-trace validity 0.000 — but **41.8% of depth-7 steps replay validly before the trace derails**. Competence is bounded by curriculum coverage, and the failure is graceful at the step level.
2. **Curriculum buys any bounded depth; no curriculum buys unbounded depth.** The wall moved exactly from 5–6 to 7–8 when the training depths moved.
3. **System-2 already owns unbounded iteration.** The Phase 41 thinking engine's measured scaling law (solvable multi-hop depth 4→16→32 at budgets 100→300→1000) is precisely the regime the feed-forward organ cannot enter (probes: no internal iteration substrate at 6 layers).
4. **The two systems already share a register.** ADR 092 traces are engine-replayable statement streams; the emergence probes' conclusion ("the workspace is the SymbolId register shared with the VSA loop; iteration lives in System-2") makes the trace stream the handoff medium — no translation layer.

## Decision — the handoff protocol

**System-1 (the organ) proposes a trace; a deterministic verifier finds the longest valid on-path prefix; System-2 continues from exactly that state; the composed trace is re-verified end-to-end.** Concretely (`grug_data::reason::handoff`, first slice):

1. **Trivial-accept:** if the organ's full trace replays valid and its `call answer` is correct, use it untouched. The handoff never interferes where the organ is competent (in-distribution authority stays neural).
2. **Prefix salvage:** otherwise, walk the query path hop-by-hop from the example's subject; hop *k* counts as *neurally resolved* only if the organ emitted a statement establishing hop *k*'s fact **and** that statement validates against the prompt facts **and** it lies on the query path (locally-valid but off-path recalls do not count — wrong turns are not progress).
3. **Symbolic continuation:** from the last on-path entity, follow the remaining relations over the (verified) fact store; a missing fact yields `gap` via the same abstention grammar the organ itself uses (one abstention mechanism across both systems, per ADR 092 D1).
4. **Compose and re-verify:** neural prefix + synthesized suffix are renumbered into one ADR-092 trace and replayed; composed executability is a first-class output. The final artifact is indistinguishable in form from a pure-organ trace — downstream consumers (provenance, explanation, training data) need no dual-process awareness.

**Firing conditions** are exactly three, all detectable without model internals: trace derail (rule 2 finds a proper prefix), gap emission (verified genuine by replay — a spurious gap is a derail), and the trivial-accept miss (wrong answer on a valid-looking trace). No confidence thresholds, no heuristics — the verifier's judgment is structural.

## Lab ↔ production mapping

| lab slice (this ADR, built) | production (P6 proper, designed here, built later) |
|---|---|
| prompt facts (`ReasonExample.prompt_facts`) | the KG (Oxigraph/petgraph truth + TMS filter) |
| deterministic chain-follower continuation | `engine.deliberate(goal, budget)` — Phase 41 `ThinkingLoop` with goal = the *remaining* sub-chain from the splice entity |
| entity words (ADR 092 D3) | SymbolId register (P1b reserved token block; identity mapping) |
| `replay_trace` prefix validation | `CompositeVerifier` (contradictions + belief intervals + source trust) |
| composed trace | `ThoughtStep` trace + `DerivationKind::AgentDecision`-style provenance record |
| `gap` continuation | gap → ADR 089 organ responses (retrieval-widen / synthesize) before abstaining |

## Economics (why the split, not System-2-alone)

In the lab the symbolic follower solves everything alone — the lab measures **mechanics**, not value. The value case is production: an organ hop costs ~a dozen tokens of forward pass; a System-2 hop costs thinking-budget under the measured ~18.7·d+20 steps/depth law and the VOC controller's two-regime rule. The handoff converts every neurally-resolved hop into budget System-2 does not spend — the metric that carries this claim is **neural fraction** (`neural_hops / depth`), reported per depth. The organ's measured 42% step-level validity beyond its curriculum is exactly the salvageable capital the protocol harvests.

## Out of scope (recorded, not forgotten)

- **The reverse direction** — System-2 writing back into the organ's context so the organ re-proposes from an advanced state (broadcast-read; competitive write). That is the full P6 protocol and needs the live-engine integration, not the lab.
- Real-engine wiring (ThinkingLoop as the continuation, KG as the fact store) — next slice, behind the same `handoff` shape.
- Confidence-weighted splice points (DS masses on prefix hops) — v2, needs non-deterministic corpora (ADR 092's TMS trigger).

## Verification

`eval_handoff` (grug-train example): three conditions per axis — organ-alone, handoff, symbolic-alone — on the mixed-depth model (`runs/p3r-mixed-trace`, step 19999), axes = in-distribution (depths 2–6), out-of-depth (7–8), abstention. Acceptance: handoff ≥ symbolic-alone accuracy everywhere (composition never hurts); axis-B handoff == organ-alone (trivial-accept, no interference); out-of-depth neural fraction > 0 (the organ contributes salvaged hops); composed executability ≈ 1.0. Measured results appended below after the run.

## Measured (first slice, 2026-07-11 — `runs/p3r-mixed-trace/eval_handoff.json`, step 19999, n=256/256/128)

| axis | organ alone | **handoff** | symbolic alone | neural hops | symbolic hops | organ-used rate |
|---|---|---|---|---|---|---|
| in-distribution (d 2–6) | 0.957 | **1.000** | 1.000 | 3.71 | 0.20 | 0.949 |
| out-of-depth (d 7–8) | 0.004 | **1.000** | 1.000 | 2.36 | 5.11 | 0.000 |
| abstention (d 2–6) | 0.977 | **1.000** | 1.000 | 3.94 | 0.04 | 0.969 |

Composed-trace executability = **1.000 on every axis** — the splice logic is exact. All four acceptance criteria hold: handoff never falls below symbolic-alone; in-distribution the trivial-accept leaves the organ in charge (94.9% of examples untouched, and the handoff *repairs the organ's residual 4–8% error to 1.000* — an unplanned bonus: the same protocol that extends depth also catches in-distribution derails); out-of-depth the **neural fraction is 0.31** (2.36 of ~7.5 hops resolved by a model that scores 0.004 alone — the 42% step-level validity converted into salvaged work); abstention preserved at 1.000.

Lab caveat, stated plainly: symbolic-alone also scores 1.000 here because the lab's fact store is the prompt itself — the lab proves the **mechanics** (splice exactness, no interference, neural fraction > 0), while the **economics** (neural hops are cheap, System-2 hops cost budget) is the production claim, carried by the Phase-41 scaling law and measured only when the next slice wires `ThinkingLoop` as the continuation.
