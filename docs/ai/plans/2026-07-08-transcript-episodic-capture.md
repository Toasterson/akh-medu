# Plan: Transcript → Episodic Capture (ADR-081 write-side, first slice)

- **Date:** 2026-07-08
- **ADR:** [091](../decisions/091-transcript-episodic-capture.md) (extends [081](../decisions/081-akh-memory-schema-typing.md))
- **Status:** Design locked (grilling session 2026-07-08); implementation pending
- **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e`

## Why

The memory loop is broken at the first hop: conversations are recorded only into an
ephemeral, never-serialized 10-turn ring buffer, so nothing durable is ever captured for
the (already-wired) recall / sleep-consolidation / introspection downstream. Measured:
`akhomed_episodes = 0` after six conversations; LongMemEval A/B = 0/6 harness-off vs 4/6
harness-on, but the 4/6 was ephemeral within-session working memory, not durable episodic.
This slice adds the capture hop ADR 081 assumed but never sliced. See ADR 091 for the full
rationale and the eight locked decisions.

## The pipeline (target)

```
run_agentic_turn (turn end; agentic path; daemon/TUI/MCP via the shared Agent method)
  └─ emit_turn_wal → mutation_log row  kind="turn"
        payload = { v, ts, session_id, workspace, model,
                    user_prompt, agent_answer,
                    tools: [ {tool, ok} ... ]  (compact trace, not full args) }
        (durable in Postgres immediately; hypertable; wake-replay SKIPS kind="turn")
        │
  daily-log  ≡  SELECT payload FROM akhomed_mutation_log WHERE kind='turn' ORDER BY seq
        │
  run_sleep_cycle → harvest_daily_log phase (per-akh watermark in akh_state)
        └─ for each turn since watermark:
             keywords = extract_keywords(user_prompt + agent_answer)   (reuse nlu::kg_context)
             syms     = resolve_or_create_entity(keyword) for each
             mint episode:turn:{seq}
                --learned--> [syms]   --has_tag--> tag:conversation
                body_ref = seq        (provenance: SleepConsolidation / source=turn seq)
             advance watermark
        │
  recall_for_prompt → recall_episodes (wired) finds episode via symbols
        └─ if episode has body_ref: page turn payload from WAL by seq
           → CompactingManager budgets it into the RecalledEpisodes section
        │
  [LATER, ADR 081 slice 4] gpt-oss Librarian replaces crude keyword-distillation
                            with typed triple extraction over the same raw turns
```

## Build increments (small, individually testable commits)

1. **WAL `kind="turn"` + capture hook.**
   - `crates/akhomed-store`: allow/append a `MutationRow { kind: "turn", payload }`;
     add `kind="turn"` to the wake-replay match as a **skip** (do NOT call `add_triple`).
   - `src/agent/agentic_chat.rs`: `emit_turn_wal(...)` at turn end (next to
     `record_agent_turn`), gated by the capture flag (decision 7). Turn-record schema as
     above; keep the tool trace compact (tool name + ok/fail, not raw args).
   - Test: a turn writes one `kind='turn'` row; wake replays cleanly (no `add_triple` on it).

2. **Harvest phase in the sleep cycle.**
   - `src/agent/sleep.rs`: `harvest_daily_log(store, engine, predicates)` — SELECT turns
     since the per-akh watermark, mint the episode-index per turn, advance watermark.
   - Watermark: a row in `akhomed_akh_state` (travels on relocation).
   - Wire into `run_sleep_cycle` (`src/agent/sleep.rs`) so it rides the existing
     `sleep_tick` (`daemon.rs:381`) — covers TUI IdleScheduler too (same entry point).
   - Test: after a turn + a forced sleep, an `episode:turn:*` exists, `learned`-linked to
     the turn's symbols, watermark advanced; second harvest is a no-op (idempotent).

3. **Recall body-paging.**
   - `src/agent/memory.rs`: when `reconstruct_episode` sees a `body_ref`, fetch the WAL
     `turn` payload by seq and surface its text (capped) as the episode content.
   - Test: `recall_for_prompt` on a keyword from a past turn returns the turn's content.

4. **Opt-out control.**
   - Workspace toml `capture_transcripts` (default true) + `AKH_TRANSCRIPT_CAPTURE` env.
     Honored at the `emit_turn_wal` gate.
   - Test: with capture off, zero `kind='turn'` rows written.

5. **Acceptance test (the loop).**
   - Harness: extend the `run_memory_bench` A/B (`scratchpad/memory_ab.sh` pattern) with an
     inter-session **harvest + daemon restart**:
     `session A inject → close → sleep/harvest → RESTART daemon → session B ask`.
   - Assert: turn rows written → survive restart → episodes minted + watermark advanced →
     B answers a fact stated only in A. `passthrough` control stays 0.

## Implementation notes (not design forks)

- **Wake-replay skip** is load-bearing: any code assuming "every `mutation_log` row
  replays as a KG mutation" must special-case `turn` (snapshot/relocation paths in
  `crates/akhomed-store/src/snapshots.rs`).
- **Watermark travels**: store harvest progress in `akh_state`, not local disk, so a
  relocated akh (ADR 049) resumes without re-harvesting.
- **Keyword→symbol** reuse: `nlu::kg_context::extract_keywords` + `Engine::resolve_or_create_entity`
  (same path KG grounding uses) — no new NLP.
- **Episode label**: keep the existing `episode:*` convention but do NOT bake the transcript
  into the label (avoid the 80-char-excerpt lossiness of `persist_episode`); the body lives
  in the WAL via `body_ref`.
- **Provenance**: episodes carry `DerivationKind::AgentConsolidation` (or a new
  `TranscriptHarvest`) with the source turn seq, so the Librarian can later find and
  re-project the same turns.

## Open / follow-up (explicitly out of this slice)

- **gpt-oss Librarian** typed extraction over the raw turns (ADR 081 slice 4) — supersedes
  the crude keyword distillation.
- **Retention/compression policy** on the `turn` hypertable (window, compression cadence) —
  deferred until the Librarian is consuming the log; keep-all for now.
- **Symbolic `ChatProcessor` capture** (non-LLM grounded path) via the shared helper.
- **Privacy**: redaction / per-turn sensitivity marking beyond the coarse opt-out.
- **Introspection loop**: an explicit "reflect on my past conversations" pass (today's
  `reflect` tunes goals from tool outcomes, not transcripts).

## Acceptance criteria (this slice)

- [x] A completed agentic turn writes exactly one `kind='turn'` WAL row (capture flag on).
      Covered by `src/agent/agentic_chat.rs::tests::transcript_capture_gate_and_emit_turn_wal`
      (payload shape + gating) and `tests/wal_live.rs::turn_mutation_flushes_with_turn_kind`
      (the row lands in Postgres with `kind='turn'`, live).
- [x] Wake/relocation replays with `turn` rows present and does not error (they are skipped).
      Covered by the `"turn"` skip arm in `crates/akhomed-store/src/snapshots.rs::apply_mutation`
      and its unit test; exercised end-to-end by `tests/harvest_live.rs` and
      `tests/adr091_acceptance.rs`, both of which write `kind='turn'` rows and then run
      migrations/harvest against the same akh without error.
- [x] A sleep cycle harvests un-harvested turns into `episode:turn:*` indices and advances
      the per-akh watermark; re-running harvests nothing new.
      Covered by `tests/harvest_live.rs::harvest_mints_episodes_and_is_idempotent` and,
      at the acceptance level (Phase B of the full loop), `tests/adr091_acceptance.rs::
      transcript_to_episodic_recall_survives_restart_across_pg`.
- [x] A fresh session, **after a daemon restart**, recalls a fact stated only in a prior
      session (cross-session durable recall); the `passthrough` control returns 0.
      Covered by `tests/adr091_acceptance.rs::transcript_to_episodic_recall_survives_restart_across_pg`
      (Phases A–D): session A's turn is written by one `Engine` instance, flushed to
      Postgres, and that engine dropped; a second, completely fresh `Engine` instance
      (the restart discriminator) harvests it into an episode index; a third instance's
      question — which never saw session A — recalls the fact via WAL body-paging; a
      fourth, un-harvested control engine recalls nothing (`passthrough` stays 0). The
      restart boundary this test exercises is the WAL/Postgres hop (fresh `Engine`
      instances over the same database, per ADR 091 decision 8's "restart is the
      discriminator" framing) — full KG durability across a literal daemon-process
      restart (snapshot/wake) is already covered by the Phase 38 / ADR 049 live
      snapshot/wake tests, so this test does not re-prove that separately. A full
      live-daemon restart drill (start `akhomed`, converse, kill -9, restart, converse
      again) remains an ops runbook step, not a unit/integration test.
- [x] `capture_transcripts=false` produces zero turn rows (opt-out honored).
      Covered by `src/agent/agentic_chat.rs::tests::transcript_capture_gate_and_emit_turn_wal`
      step 3 (`AKH_TRANSCRIPT_CAPTURE=0` forces the gate off even with the config field
      on; `emit_turn_wal` records nothing beyond the prior enabled-path assertion) — a
      `RecordingSink` stand-in for the WAL, so no live-Postgres row is written to count.
