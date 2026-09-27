# Tracker operations — transcript-harvest map

**The tracker is this directory.** Anima is the repo's primary Wayfinder tracker
(see `../agent-loops/tracker.md`), but it is unreachable from this machine:
`anima-pat get-auth-header` reports no PAT stored, and `anima.wegmueller.it/mcp`
answers 401. Migrating this map to a native Anima Plan is itself a ticket —
[Tracker restoration](tickets/tracker-restoration.md).

## Conventions

- The map is [map.md](map.md); every ticket is a file in `tickets/`. The filename
  is the ticket id. **Refer to tickets by title, never by filename**, in anything
  a human reads.
- Ticket frontmatter carries `type` (`research` | `prototype` | `grilling` |
  `task`), `status` (`open` | `in-progress` | `closed`), and `blocked-by` as a
  list of ticket filenames — this stands in for a tracker's native dependency
  edges.
- A ticket is **claimed** by setting `status: in-progress` and a `claimed:` line
  before doing any work, so concurrent sessions skip it.
- The **frontier** is every open ticket whose blockers are all closed and which is
  unclaimed. Recompute it rather than trusting a stale list; the validation script
  in the charting session's history does this in ~30 lines of Python.
- Research findings land in `assets/`, produced on a `research/<name>` branch and
  linked from the ticket. Assets are never pasted into a ticket body.
- **Never resolve more than one ticket per session**, except research tickets.

## State after session 10 (autonomous), 2026-09-19

**23 tickets: 22 closed.** The only open ticket is
[tracker restoration](tickets/tracker-restoration.md), which requires a
human-supplied Anima PAT and is now a migration task rather than a blocker.

The map reached its destination as a design. Handed off as:

- **[ADR 119](../../ai/decisions/119-agent-session-transcript-harvest.md)** — the decisions and what measurement overturned
- **[Implementation plan](../../ai/plans/2026-09-19-transcript-harvest.md)** — seven increments, each independently testable
- **[CONTEXT.md](CONTEXT.md)** — the glossary
- Reproducible prototypes in `assets/scripts/`

**One decision is flagged for the user:**
[store schema](tickets/store-schema.md) keeps `observed_at` in the primary key to
enable TimescaleDB compression, and
[extension verification](tickets/extension-verification.md) then found TimescaleDB
is **not available** in akh-pg. Nothing is broken — the design degrades to a plain
table with an identical key — but the complexity is currently unpaid-for.
Recommendation: drop it from the key. Not reversed unilaterally.

Session log: 1 charting · 2 Seshat cut · 3 identity · 4 normal form · 5 store
schema · 6 ingest transport · 7 derivation worker · 8 read surface · 9 embedding
· 10 autonomous close-out (lifecycle, both prototypes, extension verification,
placement, ADR + plan).
