# Phase 54 — Knowledge-work codification (the codifier)

- **Date:** 2026-09-26
- **Status:** Planned. Items are drafts in Anima; the operator promotes them.
- **ADR:** [114](../decisions/114-knowledge-work-codification.md)
- **Research:** [AgentRun harvest](../research/2026-09-26-agentrun-harvest/README.md)
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, **Plan `01a0e1f9-ca71-74a3-9baf-0552f96bf329`** (the phase umbrella; workflow Feature Development). Slice work items under the Plan, all draft, `source=architect-session`: 54a `01a0e1fa-173a-7e22-b6d1-40da7a8b1255`, 54b `01a0e1fa-320a-7a72-bde3-210ad96239f3`, 54c `01a0e1fa-80c3-7f40-a76e-38c6cc5a93af`, 54d `01a0e1fa-9ce6-7a02-97d3-8f8effbc74b2`, 54e `01a0e1fa-b8e3-7171-a81c-948a66a1110a`, 54f `01a0e1fa-cce7-7883-97ff-f9f26293c3e9`. (A first set created 2026-09-26 without `plan_id`, plus a mistaken umbrella WI `01a0df4f-5d03`, were cancelled 2026-09-27.)
- **Depends on:** Phase 11h chunking (live), Phase 30/31 skillpacks (live), Phase 53c/53d/53e (landing), typed-execution slices A1–A3 (proposed, [plan](2026-09-22-typed-execution-and-akhpack-v1.md))

## Goal

An akh on a capable model does a repetitive job by hand, leaves a note per
run, and after enough runs writes the job down as a checked procedure. The
procedure takes the ordinary cases through a reactor; the akh handles what the
procedure hands back. Cost per case and escalation share are measured per
procedure version. No Jev-class model is in the dispatch path in this phase.

## Slices

| Slice | Depends on | Delivers | Acceptance |
| --- | --- | --- | --- |
| 54a Job signature, notes and cost ledger | none | `JobSignature` (blake3 over the Phase 11h generalized trace + goal shape); a per-run ledger row (tokens, tool calls, wall time, outcome, escalation) from existing telemetry; an end-of-turn lesson note stored as episodic memory keyed by the signature; codifier candidate scoring over methods (usage, success rate, cost, variance) | Two runs of the same job with different wording share a signature; ledger rows reconcile with provider usage; notes are recalled by signature, never by phrase matching |
| 54b Decision records and `DecisionBackend` seam | 53d | `TypedQuestion`/`Answer` types, retained decision records (schema, options, answer, probability, backend identity, state revision, provenance); `Structural` backend over `RouteFilter`, event calculus and DS intervals; `Shadow` adapter ported from `laya_shadow.py` semantics (observe, never execute); `akh decide replay` re-runs retained records against changed reader code | Replay of N retained records completes without a model call; a shadow backend's disagreement can never change the structural answer; backend failure preserves the host path |
| 54c Codifier authoring turn | 54a, 54b, ADR 113 budgets | Background task in the consolidation cycle; when a candidate crosses thresholds, one capable-model turn authors the procedure as skillpack `plans.json` + `actions.json` + `procedures.md` (rung 3); the authoring prompt is fed the traces, notes and ledger of that job only; provenance links the artifact to its episodes and method | Authored procedure passes existing skillpack validation; spend is charged to the akh's budget and stops at the cap; a job below threshold is never authored |
| 54d Shadow, canary ramp and promotion | 54c | Shadow execution beside the by-hand turn with per-node comparison; canary fraction then default with fallback to the full agent; promotion requires beating the current handler on a development set and a sealed held-out set; rollback by lock revision; escalation share and cost per version reported | A procedure that loses on the held-out set is not promoted even if it wins on the development set; escalated cases reach the full agent with the procedure's partial record |
| 54e Package and reactor binding | A1–A3, 53e, 54d | Surviving procedures packed as `.akhpack` (WASM component or checked TS glue) and registered as tools; a reactor with route filters derived from the procedure's preconditions binds the package to events; runs under a per-delegation grant | Ordinary cases for a bound job produce no LLM turn; the package digest, grant and receipts appear in the case's provenance; unbinding restores the rung-3 handler |
| 54f Evidence on two fleet jobs (analysis) | 54d | Before/after on two real routine jobs of the Romans (candidates: the Anima lane triage the observer source now emits, Vesta's monitoring reactions) with cost, accuracy against operator labels, escalation share per version | A written report under `docs/ai/` with retained ledgers; numbers in tables, no claims beyond the two jobs |

54a and 54b can proceed in parallel. 54c is the first slice that spends LLM
budget. 54e waits for the package slices of the typed-execution plan and is the
only slice that needs them; rungs 0–3 and the reactor binding over existing
tools deliver the cost saving without it.

## Out of scope

- A typed-decision model with dispatch authority (53g D1 stands; a later ADR
  reopens it per decision family once the replay panel clears).
- A new workflow language.
- Any shell surface in authored procedures.
- Training work; nothing here touches Phase 51 lanes.

## Operator touch points

- Promote 54a/54b from draft when the 53d reactor admission PR is on main.
- Set the per-akh codifier budget before 54c is promoted; the default is zero.
- Choose the two jobs for 54f when 54d is in testing.
