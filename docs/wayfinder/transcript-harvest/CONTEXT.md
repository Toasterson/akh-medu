# CONTEXT — transcript harvest

Glossary for the transcript-harvest effort. Scoped to this context: it defines
only the vocabulary this map introduces, and deliberately contains no
implementation detail. Decisions live in `tickets/`, not here.

## Identity

- **Source** — one agent CLI whose transcripts we harvest: `claude_code`,
  `codex`, `opencode`. Not "tool" (overloaded) and not "client".
- **Session** — one *conversation*. The unit a human means by "that session".
  Claude Code `sessionId`; Codex `session_meta.payload.session_id`; OpenCode
  `event.aggregate_id`. **A session is not a file** — 46 Claude Code files held
  6 sessions.
- **Stream** — one append-only sequence of records *within* a session. A session
  has one or more. Claude Code: the main file, plus one per subagent
  (`agent-<agentId>`). Codex: one rollout file; a resume or fork starts a new
  stream in the same session. OpenCode: the single event stream.
- **Seq** — a record's position in its stream. Line index for the JSONL sources,
  `event.seq` for OpenCode. Dense and monotonic within a stream.
- **Record** — one addressable item, `(source, session_id, stream_id, seq)`.
  The **verbatim record** is the source bytes; the **normalized event** is what
  we derive from it. Both are "the record" only when qualified.
- **Machine** — the host that harvested a record. Provenance, *not* identity;
  two machines harvesting the same session converge rather than duplicate.

## Content

- **Event kind** — one of ten closed values naming what a normalized event *is*:
  `prompt`, `reply`, `thought`, `tool_call`, `tool_result`, `context`,
  `session_meta`, `lifecycle`, `error`, `telemetry`.
- **Noise** — a verbatim record that is retained but yields no normalized event,
  because it duplicates another record or carries no transcript content.
  Noise is never deleted; "noise" is a statement about derivation, not value.
- **Undetermined** — a record type whose meaning could not be established from
  the data. Retained, counted, and never guessed at. Distinct from noise: we
  know noise is redundant, we merely don't know what undetermined records are.
- **Degraded** — a normalized event derived from a record whose shape was only
  partially recognised. Carries a visible marker; silence is the failure mode
  this exists to prevent.
- **Render** — the compact one-line form of a normalized event, lossy by design.

## Reading

- **Reading order** — the canonical linearization of a session: subagent streams
  spliced into the parent at the `tool_call` that spawned them, depth-first.
  Distinct from **stream order** (seq within one stream) and from wall-clock
  order.
- **Spawn link** — the relation from a parent's `tool_call` to the child stream
  it created. Claude Code provides it via `started`/`result` records bearing
  `agentId`.
- **Repo** — canonical project identity, resolved from the git remote or repo
  root. **Workspace** — the working directory a stream actually ran in, which
  for a worktree or a subagent differs from the repo root. Sessions group by
  repo; workspace is retained.
