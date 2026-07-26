# ADR 056 — Unified trace envelope (Phase 41i)

> Date: 2026-04-28
> Status: Proposed
> Phase: 41i (new sub-phase under Phase 41)
> Depends on: ADR 052 (Unified Thinking Loop), Phase 41a (Loop skeleton), existing OODA cycle records and provenance ledger
> Related: [Phase 41 plan](../plans/2026-04-27-phase41-thinking-engine.md), [synthesis plan](../plans/2026-04-28-prompt-to-plan-tool-workflow.md)
> Anima ticket: TBD (file under Phase 41 thread `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9` until promoted)

## Context

The synthesis plan recommends Option B: keep `run_ooda_cycle` ([`src/agent/ooda.rs`](../../src/agent/ooda.rs)) as the outer agent loop and have its Decide phase delegate to `ThinkingLoop::pick_tool` on impasse. This produces *two trace formats* the user must be able to read as one continuous reasoning trail:

1. **OODA cycle records** — `OodaCycleResult { cycle, observation, orientation, decision, action }` written today via `DerivationKind::AgentDecision { goal, cycle }`.
2. **`ThoughtStep` records** — emitted by Phase 41 `ThinkingLoop::deliberate`, persisted via `DerivationKind::ThoughtStep { thought_id, step_idx }` and (Phase 41d) the `akhomed.thinking_steps` table.

When the OODA Decide phase calls `ThinkingLoop::pick_tool`, the resulting deliberation is *part of the same agent action*. The user asking "show me what the agent was thinking when it decided to use `code_gen`" needs to see the OODA cycle marker, the impasse, the embedded ThoughtSteps that resolved it, and the final tool choice — as one ordered narrative, not two interleaved logs.

Without an explicit envelope, every consumer (Phase 29 capability exposure, Phase 41h.3 MCP `get_deliberation_trace`, Pattern D markdown plan rendering, Phase 41g benchmark harness) reinvents its own join logic against `provenance.rs` indices. Each consumer drifts; the trace surface fractures.

The need is for a single shared key — `trace_id` — that:
- Joins an OODA cycle's `AgentDecision` provenance record to the `ThoughtStep` records it spawned.
- Survives across surfaces (OODA→Loop→delegation→verbalizer all carry the same `trace_id`).
- Is queryable from any of the existing provenance traversal APIs.
- Is stable enough that an old trace can be loaded and replayed without schema drift.

## Decision

Add **Phase 41i — Trace envelope** (new sub-phase under Phase 41, lands between 41f and 41g, before 41h):

1. **A new `TraceEnvelope` type** in `src/provenance.rs`:
   ```rust
   pub struct TraceEnvelope {
       pub trace_id: TraceId,                      // UUID v7 for time-orderability
       pub root_goal: SymbolId,                    // The goal that opened this trace
       pub root_kind: TraceRootKind,               // Ooda | Thinking | Delegation
       pub started_at: DateTime<Utc>,
       pub ended_at: Option<DateTime<Utc>>,
       pub status: TraceStatus,                    // Open | Closed | Abandoned
       pub conclusion: Option<TraceConclusion>,
   }

   pub enum TraceRootKind {
       Ooda { cycle: u64 },
       Thinking { thought_id: ThoughtId },
       Delegation { delegation_id: DelegationId },
   }
   ```

2. **All existing trace-emitting `DerivationKind` variants gain an optional `trace_id` field** (additive, backwards-compatible):
   - `DerivationKind::AgentDecision { goal, cycle, trace_id: Option<TraceId> }`
   - `DerivationKind::ThoughtStep { thought_id, step_idx, trace_id: Option<TraceId> }`
   - `DerivationKind::Delegation { target, task_id, cost_usd, trace_id: Option<TraceId> }`
   - Future LLM-call patterns inherit the field at addition time.

3. **A new join key in the OODA Decide phase.** When `ThinkingLoop::pick_tool` is invoked from impasse (Phase 41h.2), the OODA Decide phase generates a `trace_id`, threads it into the loop's `ThoughtState.origin`, and uses the same id when storing the resulting `DerivationKind::AgentDecision`. The OODA `Decision` enum gets a new variant:
   ```rust
   pub enum Decision {
       /* ... existing variants ... */
       Deliberated {
           trace_id: TraceId,
           tool: ToolId,
           confidence: BeliefInterval,
       },
   }
   ```

4. **Postgres schema** (lands with Phase 41d persistence):
   ```sql
   CREATE TABLE akhomed.trace_envelopes (
       trace_id     UUID PRIMARY KEY,
       akh_id       UUID NOT NULL REFERENCES akhomed.akhs(akh_id),
       workspace_id UUID NOT NULL REFERENCES workspaces.workspaces(workspace_id),
       root_goal    BIGINT NOT NULL,
       root_kind    TEXT NOT NULL,                      -- 'ooda' | 'thinking' | 'delegation'
       started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
       ended_at     TIMESTAMPTZ,
       status       TEXT NOT NULL DEFAULT 'open',
       conclusion   JSONB
   );

   CREATE INDEX trace_envelopes_akh_started_idx
       ON akhomed.trace_envelopes (akh_id, started_at DESC);
   ```
   The Phase 41d `akhomed.thinking_traces` table gains a `trace_id` foreign key into `trace_envelopes`. Existing per-cycle provenance records similarly carry the foreign key.

5. **Traversal API** in `src/provenance.rs`:
   ```rust
   impl ProvenanceLedger {
       pub fn trace_records(&self, trace_id: TraceId) -> Vec<ProvenanceRecord>;
       pub fn trace_envelope(&self, trace_id: TraceId) -> Option<TraceEnvelope>;
   }
   ```
   Records are returned in `(emitted_at, step_idx)` order so the rendering layer can iterate without further joining.

6. **Migration path:** existing in-tree provenance records (with `trace_id: None`) continue to work. The Phase 41h adapter is the first call site that *requires* a `trace_id`; until then, trace envelopes are created lazily by the OODA loop only when its Decide phase is going to call into the ThinkingLoop.

## Consequences

### Positive

- Phase 29 capability exposure (`reasoning_trace`, `explain_decision`) and Phase 41h.3 MCP `get_deliberation_trace` get a single canonical join key. No more per-consumer ad-hoc reconstruction.
- Pattern D markdown plan rendering trivially walks one envelope to produce a complete plan document.
- Phase 41g benchmark harness can compare runs by `trace_id` without filtering across multiple kind-specific tables.
- Multi-Akh delegation (Phase 36 AkhMeduPeer) inherits the envelope: a remote Akh receives a delegation under a `trace_id`, runs its own deliberation under that same id, returns `ThoughtStep` records that the originating Akh stitches into its envelope. The audit story crosses machines.
- Cumulative thinking (ADR 052 §4 speculative claim): a long-running problem produces one envelope that grows over days, across hosts.

### Negative

- Schema migration touches `provenance.rs` and the `akhomed.*` tables. The additive `Option<TraceId>` softens the blow but the Postgres migration is real.
- Every existing call site that creates a `DerivationKind::AgentDecision` or similar is a candidate for an audit pass to ensure `trace_id` propagation is correct where applicable. ~15 call sites today; not all need updates, but all need review.
- A `TraceId` UUID-v7 per envelope adds 16 bytes per record; over millions of records this is non-trivial (mitigated by index pruning policies established for Phase 38 anyway).

### Neutral

- The `trace_id` is *optional* on individual `DerivationKind` records — legacy or non-deliberated records carry `None`, modern deliberated records carry a UUID. Consumers handle both.
- Phase 41i is a small phase (~150 LOC + schema migration) but it is a hard dependency of Phase 41h.2's `Decision::Deliberated` variant. Sequence: 41a → 41i → 41h → 41d.

## Rejected alternatives

### "Use the existing `cycle: u64` as the join key"

Cycle counters are per-Akh-session. They reset on session restart, conflict across multiple parallel goals in one cycle, and cannot identify a multi-step deliberation that spans cycles. Insufficient.

### "Use `goal_id` as the join key"

A single goal can be deliberated on across many cycles (planning failures, backtracking, MCTS rollouts). Goals are too coarse — they cluster all reasoning about a goal into one bucket, losing the per-deliberation grain.

### "Use `thought_id` as the join key for everything"

Backwards: `thought_id` lives on the ThinkingLoop side. OODA cycles that don't trigger a ThinkingLoop call still produce traces worth surfacing; they would have no `thought_id`. The envelope abstracts above both.

### "Defer until Phase 41h actually needs it"

Tempting but produces a circular dependency: Phase 41h.2's `Decision::Deliberated { trace_id, ... }` variant requires the envelope to exist. If 41h ships first, it commits to a trace key shape that 41i then has to honour. Better to land the envelope first and let the adapter use it cleanly.

## References

- [ADR 052 — Unified Thinking Loop](052-unified-thinking-loop.md)
- [ADR 054 — Hidden-state fidelity](054-hidden-state-fidelity.md)
- [ADR 055 — API trace re-encoding](055-api-trace-reencoding.md)
- [Phase 41 plan](../plans/2026-04-27-phase41-thinking-engine.md) — add 41i between 41f and 41g
- [Synthesis plan](../plans/2026-04-28-prompt-to-plan-tool-workflow.md) §6, §7 commitment 3
- [`src/provenance.rs`](../../src/provenance.rs)
- [`src/agent/ooda.rs`](../../src/agent/ooda.rs)
