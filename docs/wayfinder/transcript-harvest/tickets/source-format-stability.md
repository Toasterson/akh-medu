---
type: research
status: closed
resolved: 2026-09-19
branch: research/source-format-stability
asset: assets/source-format-stability.md (+ 4 survey scripts, commit 73c16e12)
---

## Question

How stable are the on-disk transcript formats of Claude Code, Codex and OpenCode
across versions, and what must the harvester do to survive drift without silently
losing or corrupting records?

## Resolution

Full findings and four reproducible survey scripts:
[assets/source-format-stability.md](../assets/source-format-stability.md) (also on the research branch)

**The vendors answer it for us.** Claude Code's own documentation states the JSONL
format is *internal*, *changes between versions*, and *can break on any release*.
Codex and OpenCode publish no guarantee at all, and each has currently-open issues
showing real breakage in the wild. Treat all three as unstable by default. This
retroactively vindicates [the harvest unit decision](harvest-unit.md): verbatim
capture is not belt-and-braces here, it is the only thing that survives.

**Version markers are weaker than assumed, and differ per source:**

- **Claude Code** — `version` appears on only 4 of 11 observed record types;
  **16.7% of lines (3,061/18,290) carry no version at all.** Dispatch must be
  `type`-first, version second.
- **Codex** — `cli_version` lives only in `session_meta.payload` and must be
  threaded forward. Critically, **`session_meta` re-emits mid-file on forked or
  resumed sessions — in 18 of 33 local files** — each time embedding the parent's
  full header inline.
- **OpenCode** — no per-record CLI version, but each event `type` carries an
  inline schema-version suffix (`message.updated.1`). The most self-describing of
  the three.

**Read OpenCode's `event` table, not `message`/`part`.** 38 migrations in ~146
days (one per 3.8 days); ~24 touch `session`/`message`/`part` shape directly,
while exactly one has ever touched `event`, which has needed no schema change
since. Corroborated by upstream evidence that OpenCode is already migrating away
from those very tables toward `session_v2`/`session_message`.

**Two record types could not be determined from data alone** and are flagged
rather than guessed: Claude Code's `atis-latch` (179 occurrences, a single opaque
hex token) and Codex's `world_state` (324 occurrences, opaque payload).

**Recommended strategy:** capture raw bytes/rows unconditionally; dispatch
normalization on `(source, type_family, schema_marker)`; unknown combinations are
stored and counted, **never guessed at**; degraded-known combinations carry a
visible `degraded=true` tag; drift alarms calibrated per source against its
measured baseline — any new Codex top-level `type`, any OpenCode event-type suffix
above the highest known `.N`, and any `no such column/table` error on OpenCode's
non-`event` tables treated as a hard page.

## Consequences

- [Session identity](session-identity.md) gains hard evidence on fork/resume and
  a settled answer for OpenCode's read path.
- [Event normal form](event-normal-form.md) inherits the census and the two
  undetermined types.
- New ticket: [source file lifecycle](source-file-lifecycle.md) — the survey
  surfaced `.jsonl` → `.jsonl.zst` silent rotation and unbounded compaction
  growth, which the scanner and watermark must handle and which no existing
  ticket owned.
