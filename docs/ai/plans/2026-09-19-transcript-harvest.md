# Plan — Agent Session Transcript Harvest

- **Date:** 2026-09-19
- **ADR:** [108](../decisions/108-agent-session-transcript-harvest.md)
- **Map:** [docs/wayfinder/transcript-harvest/](../../wayfinder/transcript-harvest/map.md) — every decision below links to the ticket carrying its reasoning and rejected alternatives
- **Status:** ready to build; all design decisions closed

## Components

| Component | Runs | Job |
|---|---|---|
| `akh-harvest` | every machine producing sessions | scan → push verbatim rows → advance local watermark |
| `akh-derive` | in-cluster, singleton | normalize, render, flag secrets, embed |
| `akh-transcripts-mcp` | in-cluster | exact + semantic reads |
| `akh-embed` (new crate) | library | `SentenceEmbedder` extracted from `crates/seshat` |

## Step 0 — stand up `harvest-pg`

**No cluster in the fleet has TimescaleDB** — `akh-pg`, `djehuty-board` and
`postgres-ha` all run PG 18 with `vector` available and `timescaledb` absent. So
the store gets its own cluster rather than an image swap on the database the akhs
depend on.

Artifacts are ready in
`docs/wayfinder/transcript-harvest/assets/deploy/`: a `Dockerfile` (CNPG 18 base
+ TimescaleDB 2.25.1, multi-arch, build-time extension assertion) and
`harvest-pg-cluster.yaml` (validated with `kubectl apply --dry-run=server`).
Build + push to `code.aopc.cloud`, then add the manifest to the Flux repo —
`kubectl apply` directly will be reverted by Flux.

*Done when:* `select extname from pg_extension` in `transcripts` returns both
`timescaledb` and `vector`.

**If this step is skipped**, drop `observed_at` from the raw primary key — it
exists solely to enable compression, and unpaid-for it only adds the
carry-forward determinism requirement.

## Increments

Each is independently testable; none requires the next.

### 1. Schema + migrations
SeaORM entities and `sea-orm-migration`, advisory-locked like
`akhomed_store::run_migrations_locked`. Raw table, normalized table,
`derivation_version`, drift counters, `CREATE EXTENSION vector`, HNSW index.
Guarded so TimescaleDB absence is a warn-and-continue and pgvector absence is a
**loud miette failure**.
*Done when:* migrations apply to a scratch database from `scripts/pg-standalone.sh`.

### 2. Source adapters (read-only, no database)
Three adapters producing verbatim records plus identity. Claude Code's recursive
layout (`<slug>/<sessionId>.jsonl`, `subagents/agent-<id>.jsonl`,
`subagents/workflows/wf_<id>/journal.jsonl` — path-derived identity, no
`sessionId` in content); Codex rollouts; OpenCode's `event` table (**not**
`message`/`part` — 24 migrations touch those, 1 touches `event`).
*Done when:* a local run reports **46 Claude streams and 33 Codex streams**. A
naive `<slug>/*.jsonl` glob finds 6 and fails silently — assert the count.

### 3. `akh-harvest`
Watermark `(inode, byte offset)` per `(machine, source, stream)`, invalidating to
a full re-read on inode change or shrink. Complete lines only. Batched
`INSERT … ON CONFLICT DO NOTHING`. Forward-only `observed_at` fill if it survives
the decision above.
*Done when:* two consecutive runs insert 66,084 then 0 rows, and a deleted
watermark re-runs to 0 net inserts.

### 4. `akh-embed` extraction
Move `SentenceEmbedder` out of `crates/seshat` into a small crate both depend on.
`ort = "=2.0.0-rc.11"` and `tokenizers = "0.22"` are already declared
byte-identically in two manifests.
*Done when:* Seshat's own tests are green and its embedding path is unchanged.

### 5. `akh-derive`
Anti-join discovery at current `derivation_version`, **`LIMIT` pushed into SQL**
(do not copy `embed_pending_batch_impl`'s `.all().take()`). Ten event kinds.
Per-record failure writes a `degraded` row — **a poison record must not re-pend
forever**. Render. Secret flags. Turn-boundary chunks (~1500 chars) embedded for
`prompt`/`reply` **only**.
*Done when:* the 66k backfill completes, degraded count is explainable, and a
version bump re-derives cleanly.

### 6. `akh-transcripts-mcp`
`list_sessions` · `get_session` · `get_lines` · `search_transcripts` ·
`grep_transcripts`. Byte budget with kind-aware tiers — prose untruncated, tool
output head-and-tail, telemetry/compaction/opaque-`thought` excluded by default.
Truncation always visible. Spliced reading order by default (subagent streams at
their spawn `tool_call`, via `lifecycle` parent/child ids). Bearer auth per
`crates/seshat/src/auth.rs`, read-only role.
*Done when:* an agent retrieves a specific line of a past session in each source,
and no response exceeds its budget.

### 7. Rollout
Per-machine client certificate (`CN = harvest-<machine>`) plus insert-only role;
systemd user timer at 15 minutes (`khepri-sweep.timer` is the local precedent);
launchd and SMF for the others. Report watermark age — **a harvester that
harvests nothing looks exactly like one that is broken.**

## Traps, all measured

- Claude Code: **46 files, 6 sessions**; 11.1% of records have no timestamp,
  including every `started`/`result`.
- Codex: `ordinal` absent in 22/33 files; `session_meta` re-emits on fork/resume
  in 18/33, carrying the **parent's** id; `event_msg` is *not* a duplicate of
  `response_item`.
- **Reasoning is not readable** — Codex 100% encrypted, 91.8% of Claude thinking
  blocks empty. Do not budget for it.
- `compacted`: 129 records, **12 MB**. A count-ordered census hides it; rank by
  bytes too.
- `journal.jsonl` is not a transcript.
- Secret detection: named patterns only. Entropy flags 15% of the corpus; "token"
  appears in 40,810 records meaning *LLM tokens*.

## Verification corpus

The measurements above come from 66,084 records / 182 MB on this workstation, and
the scripts that produced them are reproducible:
`docs/wayfinder/transcript-harvest/assets/scripts/` (`render_prototype.py`,
`redaction_prototype.py`) plus the survey scripts on branch
`research/source-format-stability`.

## Implementation decisions ratified during the build (2026-09-20)

Surfaced by the increment agents where the tickets were silent; ratified by the
integrator rather than left implicit in code.

- **`th_events` gained `subseq` and `disposition`.** One raw record can yield
  several events (a Claude `user` record with three `tool_result` blocks), so the
  unique key is `(derivation_version, source, session_id, stream_id, seq, subseq)`.
  Records that yield no event still get a row with `disposition ∈ noise |
  undetermined | failed` and `kind NULL`, so the anti-join never re-pends them
  (derivation-worker §2). The ten-kind set stays closed — no pseudo-kinds.
- **Codex dispatch keys on the record's own JSON `type`, not `origin.record_type`** —
  the adapter flattens `response_item` to its nested `payload.type` for drift
  counting, which would have sent every model-I/O record to Undetermined.
- **OpenCode payload is the whole `event` row**; the normalizer unwraps `data`.
  OpenCode records carry no timestamp; the harvester derives `observed_at` from
  `data.time.created|updated|timestamp|created_at` (ms epoch or RFC3339) before
  forward-fill, else quarantines.
- Claude `system` subtypes other than `model_refusal_fallback` → Undetermined.
  Claude `fallback` blocks and Codex `session_meta`/`token_usage_record`/
  `compacted`/`turn_context` serialise the whole block as `text` (nothing lost).
  Codex tool outputs take `call_id`/`id` as `tool_call_id`, mirroring the call.
  Codex `inter_agent_communication_metadata` → Undetermined (catch-all).
- `render()` returns a `Render` newtype; `Outcome::Degraded` is chosen from the
  built event's `degraded` flag, not by the caller.
- Chunk turn boundary = `EventKind` change (Prompt ↔ Reply), since kind encodes
  role for prose across all three sources.
- **OpenCode `message.part.updated` tool-part shape is a best-effort guess**
  (18-event local corpus): verify when real OpenCode volume exists.
- **Client certificates must be X.509 v3** and rustls verifies the hostname even
  under `verify-ca`; external harvesters use `harvest-pg.lab.home` + `verify-full`
  (runbook). Codex `cli_version` is persisted in the watermark so tail reads keep
  their schema marker.
