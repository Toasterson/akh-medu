# ADR 091: Transcript → Episodic Capture — the ADR-081 write-side first slice

- **Status:** Accepted (design) — implementation in progress
- **Date:** 2026-07-08
- **Plan:** [docs/ai/plans/2026-07-08-transcript-episodic-capture.md](../plans/2026-07-08-transcript-episodic-capture.md)
- **Extends:** [ADR 081](081-akh-memory-schema-typing.md) (Akh Memory & Schema-Typing)
- **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e`

## Context

A live-hardware audit this session exposed that the akh's memory loop is **broken at
its first hop**. After six real conversations against a GLM-5.2-backed akh, on a fresh
store:

- `akhomed_episodes` = **0 rows**; zero conversation content anywhere in the KG (the
  474 symbols / 441 facts were seed packs + agent scaffolding + tool schemas only).
- `ConversationState` is an ephemeral 10-turn ring buffer (`conversation.rs`,
  `agent.rs:217`) that is **never serialized** — a finished conversation leaves no
  durable trace except a goal-oriented `SessionSummary`.

Meanwhile the downstream is **built and wired but starved**: `recall_for_prompt`
(`agent.rs:1711`) reads durable KG episodes cross-session; `consolidate()` + the sleep
cycle run automatically on daemon timers (`daemon.rs:331,381`); introspection tools are
registered. But episodes only ever enter the store from OODA cycles, the *voluntary*
`memory_write` tool, or file ingestion — **never from a conversation.**

An A/B on LongMemEval oracle made the gap concrete: harness-off (a true
`PassThroughManager`, added this session) scored **0/6** — the akh literally answers "I
have no episodic memory of that conversation"; harness-on (`CompactingManager`) scored
**4/6**. But that 4/6 was **within-session working memory** (the manager rendering the
live ring buffer), not durable episodic memory. Close the session and it is gone.

ADR 081 *assumed* conversational capture — decision 3 ("capture is permissive: notes,
transcripts, episodes; daily logs are staging"), decision 10 ("capture is structural —
no `take_note` tool"), slice 4 ("the Librarian harvests … transcripts (the daily log)")
— but **no slice ever named the capture mechanism itself.** That hole is this ADR.

## Decision

Add the missing write-side capture hop as ADR 081's first buildable slice. Eight
decisions, grilling-locked 2026-07-08:

1. **Daily-log is the single permissive source of truth.** The existing sleep cycle
   harvests it into KG episodes now (crude, non-LLM); the gpt-oss Librarian (ADR 081
   slice 4) replaces that distillation later. One source, no forked episodic store.
2. **Per-turn, raw capture.** On each completed turn, a structured JSONL-style turn
   record (à la Claude Code's session transcript) is appended — undistilled. Distillation
   is downstream; staging stays raw.
3. **The daily-log is a WAL `kind`, not a Document.** Turn records are rows in the
   existing `akhomed_mutation_log` (a TimescaleDB hypertable) with `kind = "turn"` and a
   JSONB payload. The daily-log is the projection `WHERE kind='turn' ORDER BY seq`. This
   **refines ADR 081 P1**, which implied "daily log = library `Document`": the WAL gives
   immediate durability (the turn is in Postgres the moment it ends), reuses the seq
   watermark, and avoids a second store to keep in sync. **Wake-replay must skip
   `kind="turn"`** — these are raw staging, not KG mutations; the KG facts derived from
   them get their own `assert` rows.
4. **Harvest is a new phase in `run_sleep_cycle`.** It reads `turn` rows directly from
   the WAL since a per-akh watermark (stored in `akh_state`, so it travels on relocation)
   and mints episodes. Not bridged through working memory — the transcript is a distinct
   durable source and must not compete for (or be evicted from) the small OODA buffer.
5. **A minted episode is a lightweight index; the body stays in the WAL.** Harvest
   resolves the turn's content keywords → symbols and mints an `episode:turn:{seq}` entity
   `learned`-linked to them + `has_tag tag:conversation`, with a body pointer = the WAL
   seq. Recall finds the episode via its symbol links and **pages the body from the WAL on
   demand** (the context manager already budgets the recalled section). Keeps the KG free
   of transcript text (ADR 081 decision 11, "bodies live only in archival") and maps the
   episode onto the existing `memory.rs` `EpisodicEntry`, reconciling the two
   episodic concepts into one.
6. **Capture hook at turn-end in `run_agentic_turn`.** Co-located with the existing
   `record_agent_turn` (`agentic_chat.rs:370/483`) — one hook in the shared `Agent` method
   covers every surface (daemon, TUI, MCP). Capture is **structural and automatic** (every
   turn, no model deciding "is this noteworthy"). Agentic (LLM) path this slice; the
   symbolic `ChatProcessor` can share the helper later.
7. **Keep raw indefinitely, default-on, opt-out.** Raw turns are kept durably (hypertable
   compression handles size); no auto-prune this slice — the Librarian will need the raw
   text. Capture is default-on for all akhs, with a per-workspace opt-out
   (`capture_transcripts = false` / `AKH_TRANSCRIPT_CAPTURE=0`): capture is the akh's own
   memory, so on-by-default is right, but a sovereign akh must be able to refuse.
8. **Acceptance = cross-session recall across a daemon restart.** Inject a fact in session
   A → close it → trigger sleep/harvest → **restart the daemon** → ask in a fresh session
   B; B must answer a fact stated only in A. Plus DB assertions (turn rows written →
   survive restart → episodes minted + watermark advanced), with `passthrough` as the
   still-0 control. The restart is the discriminator that separates the real durable loop
   from the ephemeral within-session 4/6.

## Alternatives considered

- **Daily-log as a library `Document`** (literal ADR 081 P1 reading). Rejected for the
  first slice: a Document is flushed/vector-indexed on its own cadence, whereas the WAL
  makes the turn durable immediately and reuses the seq watermark + hypertable. The Note
  carrier (P1) remains valid for *authored* notes; the transcript daily-log is WAL-native.
- **A dedicated `akhomed_transcript_log` sibling table.** Rejected: keeps the KG-truth WAL
  "pure" but duplicates the append/watermark/snapshot/retention machinery. A skipped
  `kind` is cheaper and the hypertable already fits.
- **Feed the existing working memory** (chat turn → `WorkingMemoryEntry` → existing
  `consolidate()`). Rejected: WM is a small evicting buffer for OODA residue; turns would
  race eviction and conflate transcript with operational state. It is also the throwaway
  hook the design explicitly avoids.
- **Distilled capture** (summarise each turn at write time). Rejected: bakes a lossy
  "is this noteworthy" judgment into capture — exactly what ADR 081 decision 10 forbids.
  Staging stays raw; distillation is the harvest/Librarian's job.
- **Full transcript text inlined into the episode.** Rejected: bloats the KG with
  transcript prose and duplicates the WAL body; violates bodies-in-archival.
- **Crude heuristic fact-extraction at harvest** (emit `assert` triples now). Rejected for
  this slice: pre-Librarian keyword extraction is noisy and would pollute the KG with the
  junk triples the Librarian was meant to produce properly. Harvest indexes; the Librarian
  extracts.

## Consequences

**Positive**
- Lights up the entire already-wired-but-starved downstream (recall, sleep/consolidation,
  introspection) by fixing the one missing hop.
- Testable end-to-end **today**, without waiting for the gpt-oss Librarian.
- Squarely on the ADR 081 spine — a real slice, not a throwaway; the Librarian later
  supersedes only the crude distillation, not the capture or storage.
- Immediate durability + relocation-safe (WAL + `akh_state` watermark travel with the akh,
  ADR 049).

**Negative / risks**
- The WAL now carries a non-replayed `kind`; the wake path and any tooling that assumes
  "every `mutation_log` row is a replayable KG mutation" must special-case `turn`.
- Crude keyword→symbol distillation gives low-precision recall until the Librarian lands
  (acceptable: recall is re-ranked and budget-bounded; the loop still works).
- Volume: one WAL row per turn on chatty akhs (mitigated by hypertable compression; no
  prune this slice is a deliberate, revisitable choice).
- Recall gains a body-paging read (WAL fetch by seq) — a small addition to the recall path.
- Transcript capture persists user content to Postgres by default; the per-workspace
  opt-out is the mitigation, and privacy/retention windows are deferred to a follow-up.

## References

- Extends: [ADR 081](081-akh-memory-schema-typing.md) (permissive truth + typed
  projection; this is its write-side first slice).
- Related: [ADR 065](065-context-assembly-runtime.md) (context managers consume the
  recalled section; `PassThroughManager` = the harness-off control), [ADR 049](049-relocatable-akh.md)
  (WAL + snapshot/wake + relocation the watermark must survive), [ADR 072](072-normalized-akhomed-store.md)
  (normalized store).
- Code touchpoints: `src/agent/agentic_chat.rs` (capture hook), `crates/akhomed-store/`
  (WAL `kind`, wake-replay skip, watermark), `src/agent/sleep.rs` + `src/agent/daemon.rs`
  (harvest phase), `src/agent/memory.rs` (episode mint + `recall_episodes` body-paging).
- Empirical basis: this session's LongMemEval A/B (0/6 passthrough vs 4/6 within-session
  compacting) and the `akhomed_episodes = 0` audit.
