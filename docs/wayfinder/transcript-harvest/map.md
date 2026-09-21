---
label: wayfinder:map
created: 2026-09-19
status: complete — handed off to ADR 108 + plan
tracker: local-markdown (Anima unreachable — see tickets/tracker-restoration.md)
---

# Map: session-transcript harvester for Claude Code, Codex and OpenCode

## Destination

A **working harvester**: session transcripts from Claude Code, Codex and OpenCode
land in a dedicated TimescaleDB hypertable, addressable line by line, with a
normalized cross-tool projection and a compact render; one MCP server over that
store answers both exact and semantic queries. Done when a machine can ask for a
specific line of a specific past session in any of the three tools, and get it.

## Notes

- **This map carries execution.** Wayfinder's plan-don't-do default is
  deliberately overridden here (user decision, 2026-09-19): the destination is a
  running harvester, not a spec. Decision tickets still resolve as decisions;
  implementation tickets graduate from the fog once their decisions close.
- **Tracker is local markdown, not Anima.** Anima is the repo's primary tracker
  (see `docs/wayfinder/agent-loops/tracker.md`), but from this machine
  `anima-pat get-auth-header` reports no PAT stored and both
  `anima.wegmueller.it/mcp` and `seshat.lab.home/mcp` answer 401. This map lives
  in markdown until [Tracker restoration](tickets/tracker-restoration.md) closes.
- **Independent of `akhomed`.** User constraint: the harvester is its own
  process. It must not require, extend, or be scheduled by the akh daemon.
- **This is not ADR 091.** [ADR 091](../../ai/decisions/091-transcript-episodic-capture.md)
  captures *the akh's own* turns into `akhomed_mutation_log` (itself a
  TimescaleDB hypertable) for episodic recall. This effort captures *foreign*
  agent-CLI sessions into a *separate* store. ADR 091 is the proven pattern to
  copy — raw staging plus a derived index with body-paging — not the store to
  reuse. Whether the two stores ever converge is fog, not scope.
- Relevant existing work to read alongside: ADR 091 (capture pattern),
  [ADR 088](../../ai/decisions/088-high-grug-boundary-format.md) and
  [ADR 092](../../ai/decisions/092-high-grug-v1-reasoning-register.md) (the
  compact render's ancestry), [ADR 047](../../ai/decisions/047-seshat-archive.md)
  (Seshat), `crates/akhomed-store/src/migration/` (hypertable + watermark
  migrations to imitate).
- The effort's vocabulary lives in [CONTEXT.md](CONTEXT.md) — scoped to this
  context, glossary only, no implementation detail.
- Apply grilling and domain-modeling on every HITL ticket. Research tickets run
  as independent subagents on isolated `research/*` branches.
- **Measured corpus as of 2026-09-19** (the substrate every ticket reasons about):
  Claude Code 44 JSONL files / 22 MB; Codex 33 rollout files / 46 MB; OpenCode
  one SQLite DB, near-empty (2 messages, 5 parts, 18 events, 38 migrations).
  Codex's line budget is dominated by noise: 17,311 `token_count` events, and
  `event_msg` duplicates `response_item` for agent messages.

## Decisions so far

<!-- Resolved live during the destination-naming grill, 2026-09-19. Each has a
     ticket carrying the full reasoning and the alternatives rejected. -->

- [Execution mode](tickets/execution-mode.md) — the destination is a working
  harvester for all three CLIs, not a spec handoff; the map carries execution.
- [System of record](tickets/substrate.md) — a dedicated TimescaleDB hypertable
  in its own schema owns the truth; **not** the akh's `akhomed_mutation_log`,
  which would drag foreign sessions into one akh's personal memory and its
  relocation payload. Seshat is fed a derived projection.
- [Harvest unit](tickets/harvest-unit.md) — the row is the source record
  **byte-exact**, keyed `(source, session_id, seq)`; a normalized cross-tool
  event projection is derived from it and can be re-derived.
- [Compact render](tickets/compact-render.md) — the "custom encoding" is a
  High-Grug-flavored one-line render per normalized event, stored as a derived
  column, so machines scan sessions without paying JSON's punctuation tax.
- [Process model](tickets/process-model.md) — an idempotent one-shot CLI resumed
  from a watermark, driven by a platform timer (systemd user / launchd / SMF).
  No supervised daemon, no inotify portability burden.
- [Secrets posture](tickets/secrets-posture.md) — verbatim in, redact outward.
  The private store holds source records unmodified; everything leaving it is
  redacted, and OpenCode's `account`/`credential` tables are never read.
- [Session identity](tickets/session-identity.md) — identity is three-level,
  `(source, session_id, stream_id, seq)`, machine an attribute not a key. A file
  is **not** a session: 46 Claude Code files hold 6 sessions, subagents are
  streams of their parent; Codex `ordinal` is absent in 22/33 files so `seq` is
  the line index; OpenCode's `aggregate_id`/`seq` already fit. File order is a
  valid topological order (0 violations in 15,861 parent links).
- [Event normal form](tickets/event-normal-form.md) — ten closed event kinds
  (`prompt` `reply` `thought` `tool_call` `tool_result` `context` `session_meta`
  `lifecycle` `error` `telemetry`); Codex `response_item` canonical and
  `event_msg` mined only for lifecycle (they are **not** duplicates — 22% text
  overlap, absent in 11/33 files); duplicates and `atis-latch` are noise,
  `world_state` stays undetermined; reading order splices subagents at their
  spawn point; repo and workspace are separate fields; `thought` first-class but
  off by default. Vocabulary in [CONTEXT.md](CONTEXT.md).
- [Store schema](tickets/store-schema.md) — own database in the existing akh-pg
  cluster; raw verbatim rows are the hypertable partitioned on `observed_at`
  (forward-fill only — backward fill would mutate the PK and duplicate rows),
  PK `(observed_at, source, session_id, stream_id, seq)`; keep raw indefinitely,
  compress on age. TimescaleDB absent degrades to a plain table; **pgvector
  absent must fail loudly**, since the read surface advertises semantic search.
  Measured volume ≈ 3.4 GB/year across four machines.
- [Ingest transport](tickets/ingest-transport.md) — direct Postgres with a
  **per-machine insert-only role**; the harvester pushes **verbatim rows only**
  and a central derivation worker does normalization, rendering, redaction and
  embedding; **no local spool** — the source files are the spool. Insert-only
  forces `ON CONFLICT DO NOTHING`, puts the watermark on the machine, and makes
  watermark loss self-healing — which holds only because `observed_at` is
  deterministic.
- [Derivation worker](tickets/derivation-worker.md) — pending is an **absence**
  (anti-join against normalized rows at the current `derivation_version`), not a
  watermark; failures still write a `degraded` row so poison records cannot
  re-pend forever; re-derivation is a **version bump**; one advisory-locked
  polling worker in-cluster. Splicing stays a read concern. Residual: the
  `observed_at` carry-forward rule is a **frozen contract** the harvester must
  own, because it is in the raw primary key.
- [Embedding pipeline](tickets/embedding-pipeline.md) — embed **prose only**
  (`prompt`/`reply`/`thought`), not the lossy render and not tool output;
  **turn-boundary chunks** at ~1500 chars giving an exact seq range for
  pointer-back; **extract a shared embed crate** (ort/tokenizers are already
  declared byte-identically in two manifests); HNSW over 384-dim; re-embedding
  rides the existing `derivation_version` anti-join with no mechanism of its own.
  ~14,666 chunks, 22.5 MB of vectors.
- [Read MCP surface](tickets/read-mcp-surface.md) — five sharply-named tools
  (`list_sessions` `get_session` `get_lines` `search_transcripts`
  `grep_transcripts`); **byte budget with kind-aware truncation** — prose never
  truncated, process output head-and-tail, telemetry/compaction excluded by
  default — because prose is only **14.2%** of corpus bytes; truncation always
  visible; one central server, bearer token, read-only role, no per-caller
  scoping.
- [Read surface](tickets/read-surface.md) — **revised 2026-09-19**: one store,
  one MCP server, serving exact addressing *and* pgvector semantic search.
  Seshat is out; its embedder and chunker are reused as a **library** instead.
  Reverses the earlier decision to retain Seshat, after research showed it has
  neither deletion nor upsert.
- [Seshat ingest and search contract](tickets/seshat-ingest-contract.md) —
  `seshat_ingest` has **no upsert and no deletion**; document `metadata` JSONB
  round-trips verbatim on search hits, which is the pointer-back hook; a
  `agent_session_transcript` format already exists. Re-derivation and secret
  retraction both need answers the service does not provide — which reopened and
  then reversed the read-surface decision above.
- [Source format stability](tickets/source-format-stability.md) — all three
  formats are **unstable by vendor admission or by evidence**; Claude Code's docs
  say the JSONL "can break on any release". Dispatch on
  `(source, type_family, schema_marker)`, store-and-count unknowns, never guess.
  Read OpenCode's `event` table, not `message`/`part`. Codex `session_meta`
  re-emits on fork/resume in 18 of 33 local files.

## Not yet specified

- Whether a Seshat export ever returns, putting transcripts in the shared corpus
  beside ADRs, plans and papers. Ruled out of *this* effort (see Out of scope);
  it would come back as a fresh effort with its own destination, and would want
  the delete/upsert path Seshat still lacks.
- Whether the akh's own ADR-091 turns eventually become a fourth source, giving
  one store for every session the human and their akhs have ever had. Sharpen
  after the event normal form exists — it decides whether the two schemas can
  even meet.
- Whether this corpus feeds Phase-51 training (ADR 102/104). If it does, it
  drags in benchmark decontamination and a consent question about third-party
  content quoted inside sessions. Out of scope to *do*, but the store's shape
  may need to anticipate it.
- Reading sessions that are still in progress (the `--watch` mode the process
  model deliberately deferred). Blocked on whether any consumer actually wants
  live reads.
<!-- graduated 2026-09-19: source-file lifecycle (rotation, compaction growth,
     pruning, watermark unit) is now tickets/source-file-lifecycle.md -->
- Merge semantics across machines: clock skew between this box, the Mac mini,
  archibald and exec1 once several are harvesting into one hypertable. Sharpen
  after ingest transport picks direct-PG vs an endpoint.
- Expunging a session on request across verbatim rows, derived projections and
  Seshat embeddings. Depends on what the Seshat research finds about deletion.
- Whether T3 Code threads and Cursor become sources. Adapter shape first.

## Out of scope

- Modifying Claude Code, Codex or OpenCode themselves. The harvester reads what
  they already write; it does not ask upstream for a better format.
- Replacing or refactoring ADR 091's akh episodic capture. It keeps working
  exactly as it does today.
- Running any training on the harvested corpus.
- A human-facing transcript browser or UI. The consumers here are machines.
- **Pushing transcripts into Seshat** — ruled out 2026-09-19 when
  [the read surface](tickets/read-surface.md) collapsed to one store; the
  [Seshat projection](tickets/seshat-projection.md) ticket is closed unresolved.
  Accepted cost: no cross-corpus semantic search over transcripts alongside ADRs
  and plans. Seshat's *code* is still used, as a library dependency.
- Fixing Seshat's missing delete and upsert. It is ~40 lines in a crate in this
  workspace and a real gap for its other consumers, but it is no longer on the
  route to this destination.
