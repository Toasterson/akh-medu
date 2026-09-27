# ADR 114 — Knowledge-work codification: an akh does the job, writes it down, packages it, and steps aside

> Date: 2026-09-26
> Status: Proposed
> Phase: 54
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, Plan `01a0e1f9-ca71-74a3-9baf-0552f96bf329`; slice WIs are listed in the plan doc
> Builds on: ADR 040 (procedural skills), ADR 041 (skill intelligence), ADR 091
> (transcript capture), ADR 105 (run governance), ADR 113 (full akh always),
> [ADR 115](115-typed-execution-and-akhpack-v1.md) (typed execution and akhpack v1), Phase 11h chunking
> Input: [AgentRun harvest](../research/2026-09-26-agentrun-harvest/README.md)

## Context

Grep.ai published a harness whose whole point is that the expensive model does
a repetitive job once, leaves notes, and after a few runs writes the job down
as a program of cheap, inspectable steps. The program takes the ordinary cases
and the frontier agent stays for the odd ones. They report the cost of a
compliance job falling by an order of magnitude with accuracy going up, because
the program stops reading pages that cannot change the answer.

We already own most of the parts. Phase 11h compiles repeated successful tool
traces into `LearnedMethod`s. Phase 31d detects latent skills during
consolidation. Phase 30 skillpacks carry procedures, action schemas and HTN
plans. Phase 53 puts every side effect behind akhexecd with receipts and gives
delegates scoped grants. The 2026-09-22 proposal adds a typed, shell-free
TypeScript cell environment and `.akhpack` packages. What we lack is the loop
that connects them: nothing today turns a frequently repeated agentic job into
a checked program, measures it against the by-hand run, and moves ordinary
cases off the LLM.

Two constraints frame the decision:

1. **Jev-class routing is out for now.** Phase 53g decision D1 retired the
   Laya model: on 52 complete inputs it answered `investigate` every time with
   near-uniform scores, and the 288-decision replay gave nothing controller
   grade. Routing today comes from structural rules that match machine codes.
2. **The REPL and package work is proposed, not landed.** Slices A0–A3 of the
   [typed-execution plan](../plans/2026-09-22-typed-execution-and-akhpack-v1.md)
   are now WI-502–WI-505 under ADR 115; they remain draft. The codification loop must be
   useful before they land and better after.

## Decision

### 1. Codification is a background process with a ladder, not a feature of a turn

An akh runs a **codifier** in its consolidation cycle, next to skill discovery,
under the per-akh LLM budget from ADR 113. It climbs a fixed ladder per job,
the same one a unix administrator climbs:

| Rung | Artifact | Who produces it | Exists today |
| --- | --- | --- | --- |
| 0. By hand | A full agentic turn on a capable model | The akh | Yes |
| 1. Note | One job-level lesson per run, stored as episodic memory keyed by the job signature | The akh, at the end of the turn | No |
| 2. Method | `LearnedMethod` compiled from repeated successful traces | Phase 11h, engine-side | Yes |
| 3. Procedure | Skillpack `plans.json` + `actions.json` + `procedures.md`; a typed TS cell once slice A2 lands | Capable model, one authoring turn | Authoring exists; the trigger does not |
| 4. Package | `.akhpack` with the procedure as a WASM component or checked TS glue, registered as a tool | Capable model + pack writer (slice A1/A3) | No |
| 5. Reactor | Route filters bind the package to events so the ordinary case never reaches the LLM | Engine, from the procedure's declared preconditions | Route filters yes; binding no |

Rung promotion is triggered by structural signals only: usage count and
success rate of the method, the measured cost of the by-hand runs (tokens,
tool calls, wall time from telemetry), and trace variance. No natural-language
matching decides what gets codified.

### 2. The job signature is structural

A job is identified by its generalized tool trace and goal shape, the same
generalization Phase 11h already performs (concrete symbols replaced by KG
types), hashed with blake3 like `TripleIdentity`. Notes, ledger rows, methods,
procedures and packages all key on it. Two runs with different wording and the
same shape are the same job.

### 3. Three step kinds, and code decides what follows

A codified procedure is composed from exactly three kinds of step, using the
surfaces we already planned rather than a new language:

- **Code** — a typed cell (no `any`, no shell) or, before A2 lands, an HTN plan
  step over existing tools. Arithmetic, counting, thresholds and ordering live
  here.
- **Decision** — a typed question over a structured state with named options,
  answered through the `DecisionBackend` seam in §4. Closed questions live here.
- **Agent** — a bounded delegated turn under a 53e grant, only where something
  must be looked up or done that no question can settle.

Shapes (chain, map/reduce, loop-until with a bound) are curated library
functions of the cell environment. **Escalate** is the reactor result that
requests a larger-model turn; it is how the program hands a case back.

### 4. The typed-question seam stays; the backend today is structural

```rust
/// Answers a typed question over a versioned state. Never has execution authority.
pub trait DecisionBackend {
    fn decide(&self, q: &TypedQuestion, state: &StateRevision) -> DecisionResult<Answer>;
    fn identity(&self) -> BackendIdentity; // model or rule-set + revision
}
```

- `Structural` — the engine's own rules: `RouteFilter` predicates, event
  calculus, Dempster–Shafer intervals, e-graph facts. Deterministic and the
  only backend allowed to influence dispatch in this phase.
- `Shadow(model)` — a Laya, typed-decisions or future Jev-class model, ported
  from `experiments/typed-decisions/akh_decisions/laya_shadow.py`. Records
  proposals and disagreements next to the structural answer. It gains
  dispatch authority per decision family only after the replay panel in
  `LAYA_DECISION_LAYER.md` clears, by a separate ADR.

Every answer is retained as a **decision record**: question schema, options,
answer, probability if any, backend identity, state revision, provenance. That
is what makes a judgment replayable: change the code that reads the answers
and re-run the cases without a model call.

### 5. Promotion is host-owned and measured

A procedure moves from shadow to live on a fixed ramp with the full agent as
fallback: shadow (run beside the by-hand turn, compare), then a canary fraction
of cases, then default. Each step requires beating the current handler on a
development set **and** on a sealed held-out set the authoring model never
sees. Per-node evaluation freezes the evidence and changes one node. Rollback
is a lock revision, not a redeploy. Escalation share and cost per case are
reported per procedure version.

### 6. Governance by construction

Each codified artifact carries: source job signature, contributing episode
ids, method id, authoring model and turn id, package digest, promotion
evidence, and the reviewer. Decisions name their node, backend identity,
inputs and questions. This reuses `DerivationKind` provenance, the WAL and
akhexecd receipts; nothing new is invented for audit.

## Alternatives considered

- **Adopt a workflow DSL like AgentRun's.** Rejected. The typed cell language,
  packages and route filters already give us every node kind; a second syntax
  would split evaluation and provenance.
- **Put a typed-decision model in the loop now.** Rejected by the 53g evidence
  and by ADR 112's own conclusion that the typed-decision head is not the
  chooser. The seam is kept so the replay property is not lost.
- **Codify from the first run.** Rejected. One run has no variance signal;
  frequency is what makes the note generalize.
- **Let the procedure run without escalation.** Rejected. The ordinary path is
  cheap because the odd case has somewhere to go.

## Consequences

- The capable model is spent on authoring and on exceptions, not on every case.
  Per-akh budgets from ADR 113 bound the authoring spend.
- Chunking, skill discovery, skillpacks and the REPL stop being separate
  islands: one ladder connects them, and each rung is useful without the next.
- Before slices A1–A3 land, codification stops at rung 3 with skillpack plans
  and reactor bindings over existing tools. That is still the cost saving on
  routine jobs; the package rung adds portability and isolation.
- Shadow model backends produce comparison data for free from live jobs, which
  is the corpus the Laya decision layer said it needed.
- Every akh becomes an author of packages for its own routine jobs, which is
  the feed the akhpack registry needs.
