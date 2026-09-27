# ADR 119 — Agent Session Transcript Harvest

> Renumbered from ADR 108 on 2026-09-27 (duplicate number). ADR 108 is the Sokar artifact daemon.

- **Status:** Accepted (design user-grilled 2026-09-19). **Built 2026-09-20**: harvester live on hephaistos (systemd timer), `harvest-pg` populated (75k raw rows, all derived, 5k chunks embedded), worker + MCP read surface verified against the live store; image `code.aopc.cloud/toasterson/akh-transcripts:d72262cacdc2` pushed; in-cluster worker + MCP read surface rolled out via chart 0.1.18 and reachable at `http://transcripts.lab.home/mcp`.
- **Date:** 2026-09-19
- **Deciders:** Till (grilled), Opus 5
- **Plan:** [docs/ai/plans/2026-09-19-transcript-harvest.md](../plans/2026-09-19-transcript-harvest.md)
- **Map:** [docs/wayfinder/transcript-harvest/](../../wayfinder/transcript-harvest/map.md) — 23 decision tickets with full reasoning; [CONTEXT.md](../../wayfinder/transcript-harvest/CONTEXT.md) is the glossary
- **Relates:** extends the capture pattern of [ADR 091](091-transcript-episodic-capture.md); reuses the boundary-format idea of [ADR 088](088-high-grug-boundary-format.md); consumes but does not extend [ADR 047](047-seshat-archive.md)

## Context

Sessions from Claude Code, Codex and OpenCode accumulate on disk and are
unsearchable. ADR 091 solved the equivalent problem for an *akh's own* turns; this
is the same shape for *foreign* agent CLIs, deliberately independent of `akhomed`.

The corpus was measured rather than assumed — 66,084 records / 182 MB logical over
77 days on one machine, ≈0.85 GB/year/machine — and the measurements repeatedly
overturned the design's own premises. Those corrections are the substance of this
ADR.

## Decision

**A thin per-machine harvester pushes verbatim records into a dedicated Postgres
store; a central worker derives everything else; one MCP server serves exact and
semantic reads.**

1. **Own store, not the akh's — and its own cluster.** `akhomed_mutation_log` is
   an akh's *personal* memory and would drag foreign sessions into its recall and
   relocation payload. Placement was revised on 2026-09-19 from "a database in
   akh-pg" to a dedicated `harvest-pg` CNPG cluster, because **no cluster in the
   fleet has TimescaleDB** and adding it to akh-pg would mean a rolling image swap
   on the database the akhs depend on.
2. **Identity is three-level:** `(source, session_id, stream_id, seq)`, harvesting
   machine an attribute. **A file is not a session** — 46 Claude Code files hold 6
   sessions; subagents and resumed rollouts are *streams* of one session.
3. **Verbatim rows are the system of record**, with a derived normalized layer.
   All three formats are unstable by vendor admission, so re-derivability is the
   load-bearing property.
4. **Ten closed event kinds** — `prompt` `reply` `thought` `tool_call`
   `tool_result` `context` `session_meta` `lifecycle` `error` `telemetry`.
5. **A compact High-Grug-style render**, stored as a derived column.
6. **Insert-only per-machine credentials**, direct to Postgres, no spool — the
   source files are the spool.
7. **Central derivation worker**: anti-join discovery, `derivation_version` bump
   for re-derivation, single advisory-locked poller.
8. **One MCP server**, byte-budgeted with **kind-aware truncation**; Seshat is out
   of the design.
9. **Verbatim in, redact outward**, with a named-pattern detector.

## What measurement changed

Each of these reversed or corrected a decision already taken:

- **Reasoning content does not exist.** Codex `reasoning` is **100% encrypted**
  (4,642 records, `summary: []` + `encrypted_content`); **91.8% of Claude
  `thinking` blocks are empty signatures**. Total readable reasoning: **0.11 MB,
  0.06% of corpus.** `thought` was made first-class on a record count; it is a
  marker, not a body, and is excluded from embedding and from tier-1 budget.
- **Codex `event_msg` and `response_item` are not duplicates** — 22% text overlap,
  and `event_msg/agent_message` is absent from 11 of 33 files. Choosing it as
  canonical would have left a third of the corpus with no assistant messages.
- **Codex `ordinal` is unusable** as `seq` — absent in 22 of 33 files.
- **Prose is 14.2% of bytes**; tool output is 29.3% and `event_msg` alone 37.7%.
  This produced the kind-aware read budget.
- **The pure format tax is 2.6x**, not the 17.5x a naive corpus-wide ratio
  suggests — the rest is dropped noise and lossy truncation. Measured in **bytes**;
  no tokenizer was available, so the token claim is **unverified** and 2.6x should
  be read as an upper bound.
- **Named secret patterns flag 0.042%** of records and found **two real
  credentials** (a GitHub OAuth token, an SSH private key) ≈ one per 34,000
  records. A high-entropy fallback would flag **15% of the corpus, 68.4% of it
  model signatures and ciphertext**, and cue-word matching flags 61.4% because
  **"token" appears in 40,810 records as *LLM* tokens**.
- **TimescaleDB is not available anywhere in the fleet.** All three CNPG
  clusters (`akh-pg`, `djehuty-board`, `postgres-ha`) run PG 18 with `vector`
  available and `timescaledb` absent. Resolved by building our own image and
  cluster — a travelled path, since `postgres-ha` already runs a custom
  `cnpg-age` image from the user's registry.
- **ADR 091's hypertable does not exist.** `akhomed_mutation_log` holds 769,001
  rows with TimescaleDB uninstalled — it is a plain table. The guarded migration
  warned and no-opped, and nothing verified afterwards. Outside this ADR's scope,
  but it is why the new image asserts its extensions at build time.

## Build findings (2026-09-20)

- **The anti-join needed an index the schema ticket did not foresee.** On 75k
  rows the planner merged on three key columns and post-filtered `seq`
  (210,751,599 rows removed, 43.6 s per pass — the worker looked hung).
  `th_raw_records (source, session_id, stream_id, seq)` brings it to 80 ms.
- **`th_events` needed `subseq` and `disposition`**: one raw record can yield
  several events, and noise/undetermined/failed records must still get a row or
  the anti-join re-pends them forever.
- **Reading a record larger than the byte budget livelocked** the read surface
  (zero lines, cursor unchanged); oversized first items are now emitted
  head-and-tail truncated with a visible marker.
- **Client certs must be X.509 v3** for rustls, and rustls verifies the hostname
  even under `verify-ca`; **transcripts contain NUL bytes** Postgres rejects.
- Claude `attachment` records land as `context` but degraded (~3.8k): the
  normalizer does not find their text field. Tracked by the drift counters.

## Consequences

**Positive.** Re-derivation is free and central — a normalizer fix is one deploy,
not a four-machine three-OS rollout. Insert-only grants make writes
`ON CONFLICT DO NOTHING`, which makes watermark loss self-healing. Cutting Seshat
means a leaked secret is **retractable** (Seshat exposes neither delete nor
upsert). Embedding lives on one host because derivation is central.

**Negative / accepted.** No cross-corpus search alongside ADRs and plans. A machine
offline beyond the source-retention window loses that history (floor measured at
≥76 days). Pattern-based redaction misses novel formats — the store is
credential-bearing and its bearer token is the real protection.

**Open.** The `harvest-pg` cluster and its image must actually be built; until
then the design degrades to a plain table with an identical key. With the cluster
built, `observed_at` in the primary key is paid for by compression as intended.
Carry-forward must stay **forward-only** — backward fill would mutate the PK
between runs and insert duplicates.

## Alternatives considered

Reusing `akhomed_mutation_log` (ADR 091) — maximum reuse, rejected on memory
hygiene. Keeping Seshat for semantic search — chosen initially, reversed once
research showed no delete and no upsert. Deriving at the harvester — rejected
because formats break on upstream releases. Marker-column discovery (Seshat's
`embedded_at`) — rejected because raw must stay immutable and compressible.
