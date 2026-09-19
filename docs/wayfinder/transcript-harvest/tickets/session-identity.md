---
type: grilling
status: closed
resolved: 2026-09-19
blocked-by: []
---

## Question

What is the primary key of a harvested record, and what exactly is a "session"?

The addressing scheme `(source, session_id, seq)` was assumed by
[the harvest unit decision](harvest-unit.md). This ticket makes it real against
what the three sources actually do:

- **Claude Code** records form a **DAG**, not a list: every record carries `uuid`
  and `parentUuid`, plus an `isSidechain` flag for subagent branches. A subagent's
  records live in the same file as its parent. Is a sidechain the same session or
  a child session? What is `seq` when the ordering is a tree?
- **Compaction and resume**: a resumed or compacted Claude Code session — does it
  continue the same `session_id`, start a new file, or both? Whichever it does,
  the harvester must not double-count or orphan the tail.
- **Codex** already ships an `ordinal` per line. Is it dense, monotonic, and
  stable across a resumed rollout — i.e. can it be adopted as `seq` directly?
- **OpenCode** has no lines at all. `event(aggregate_id, seq, ...)` offers a seq
  per aggregate, not per session; `message`/`part` offer ids and timestamps.
  Which becomes the synthetic `seq`, and is it stable across the SQLite
  migrations the DB is clearly still accumulating?
- **Machine identity**: the same `session_id` could in principle be harvested from
  two machines, and the watermark is per-`(machine, source, session)`. Does
  machine belong in the key, or only in the watermark?
- **Worktrees**: `~/.claude/projects/` already contains both
  `-home-toasty-ws-agi-akh-medu` and a
  `-home-toasty-ws-agi-akh-medu--claude-worktrees-...` slug. Same project or not?
- **Idempotency**: re-running the harvester over an unchanged file must be a
  no-op, and over a file that grew must insert only the tail. State precisely
  what makes that true.

## Evidence from the format research (2026-09-19)

[Source format stability](source-format-stability.md) answered part of this
ticket's questions with measurements rather than assumptions:

- **Codex fork/resume is confirmed and common.** `session_meta` re-emits mid-file
  on forked or resumed sessions — **18 of 33 local rollout files** — embedding the
  parent's full header inline each time. So a Codex file is *not* one session, and
  `ordinal` cannot be assumed dense and monotonic across the whole file until this
  is modelled. This is the single sharpest input to the key design.
- **OpenCode's read path is settled**: read the `event` table, not
  `message`/`part`. One migration has ever touched `event` versus ~24 touching the
  others, and upstream is already moving toward `session_v2`/`session_message`.
  So the synthetic `seq` derives from `event(aggregate_id, seq)` — which makes
  "aggregate vs session" the question this ticket must still answer.
- **Claude Code version dispatch is unreliable**: 16.7% of lines carry no
  `version` at all, so identity and parsing must key on `type` first.

## Why it is first

Every other open ticket keys off this. The store schema indexes it, the MCP read
surface addresses by it, the Seshat projection must point back through it, and
the normalizer's output rows inherit it.

## Resolution (2026-09-19)

**Identity is three-level: `(source, session_id, stream_id, seq)`.** The machine
that harvested a record is an **attribute, not part of the key**.

### The model

| Concept | Claude Code | Codex | OpenCode |
|---|---|---|---|
| **Session** — one conversation | `sessionId` | `session_meta.payload.session_id` | `event.aggregate_id` |
| **Stream** — one append-only sequence in it | main file, or `agent-<agentId>` | the rollout file's own uuid | the session's single event stream |
| **Seq** — position in the stream | 0-based line index | 0-based line index | `event.seq` |

All three sources turned out to share this shape; nothing is bent to fit.

### What the measurements settled

- **A file is not a session.** 46 Claude Code files hold **6 sessions**. Subagent
  transcripts live at `<slug>/<sessionId>/subagents/agent-<agentId>.jsonl` (and
  deeper, under `subagents/workflows/wf_<id>/`), each carrying the **parent's
  `sessionId`** plus its own `agentId`, `isSidechain: true` and its own worktree
  `cwd`. So a sidechain **is** the same session — a different stream of it. That
  answers the ticket's opening question directly.
- **Codex `ordinal` cannot be `seq`.** It is **absent in 22 of 33 files** —
  clearly version-gated. Where present it is dense and monotonic, but the
  harvester cannot depend on it. Line index is used instead, uniformly.
- **Codex filename ≠ session.** The filename uuid differs from
  `session_meta.session_id` in **27 of 33 files**: the filename identifies a
  *rollout*, the payload a *conversation*, and several rollouts share one
  conversation (`01a02e64` spans four files). Resume/fork therefore produces a new
  **stream**, not a new session — which is exactly what the stream level is for.
- **The DAG does not need to be structure.** Across **15,861 parent links, zero
  children precede their parent**: file order is a valid topological order, so
  `seq` = line index is sound. `uuid`/`parentUuid` are retained as **attributes**
  for anyone who wants the tree. Branching is real but rare — 161 branch points,
  about 1%.
- **OpenCode needs no synthetic seq.** `aggregate_id` *is* the session id, `seq`
  is dense and monotonic per session, and `event_sequence` already stores the
  high-water mark — a ready-made watermark. This reinforces
  [the format research](source-format-stability.md): read `event`, not
  `message`/`part`.

### Machine as attribute

Session and stream ids are UUIDs/ULIDs, already globally unique, and every source
is append-only. So the same session harvested from two hosts — synced home
directory, restored backup, a worktree on a second machine — **dedupes naturally**
rather than duplicating, and no reader has to decide which copy is canonical.
Provenance is still recorded per row. The *watermark* remains per
`(machine, source, stream)`, since that tracks progress, not identity.

### Idempotency

The key does the work: records are append-only and keyed by
`(source, session_id, stream_id, seq)`, so re-inserting an already-seen record is
an upsert onto identical content — a re-run over an unchanged file is a no-op by
construction, not by bookkeeping. *Finding* the tail cheaply, and surviving
rotation and compaction, belong to
[source file lifecycle](source-file-lifecycle.md); this ticket only guarantees
that writing it twice is harmless.

### Addendum 2026-09-19 — path-derived stream identity

[The store schema](store-schema.md) found a stream kind this resolution did not
anticipate: `<slug>/<sessionId>/subagents/workflows/wf_<id>/journal.jsonl`
contains only `started`/`result` records, carries **no `sessionId` in its
content**, and has no timestamps at all. Its session and stream identity are
**derived from its path**, not read from the record. The three-level model still
holds — it is a stream of its parent session — but identity resolution is
per-stream-kind, not uniformly content-based.

### Surfaced, handed on

- **Streams within a session are not totally ordered.** Claude's main and subagent
  streams run *concurrently*; Codex's rollouts are *sequential* resumes. So
  "replay this session in reading order" is not a `seq` sort — it needs stream
  ordering or timestamp interleaving. Handed to
  [the event normal form](event-normal-form.md).
- **Project vs worktree identity** is unresolved and deliberately out of this
  ticket: `-home-toasty-ws-agi-akh-medu` and
  `-home-toasty-ws-agi-akh-medu--claude-worktrees-cool-carson-b63ef4` are separate
  slugs holding separate sessions of the *same repository*. It does not affect the
  key; it affects grouping and filtering. Handed to
  [the event normal form](event-normal-form.md).
- **Fork vs resume is indistinguishable** from the Codex data available. Both
  produce a new stream under the same session, which is why the model tolerates
  the ambiguity — but a reader wanting a single linear history cannot get one
  without more information.
