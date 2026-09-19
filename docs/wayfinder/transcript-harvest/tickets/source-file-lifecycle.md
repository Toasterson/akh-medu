---
type: grilling
status: closed
resolved: 2026-09-19
blocked-by: []
graduated-from: fog, 2026-09-19 (surfaced by source-format-stability.md)
---

## Question

What happens to a source file *after* the harvester has seen it, and how does the
scanner survive that?

[The format research](source-format-stability.md) surfaced lifecycle behaviour no
existing ticket owned, and it breaks the naive "stat the file, read from the
watermark offset" scanner that [the process model](process-model.md) implies:

1. **Silent rotation to `.jsonl.zst`.** Sessions get compressed in place. A
   scanner watching for `*.jsonl` simply stops seeing them — no error, just a
   session that quietly never grows again. Detection and decompression both need
   specifying, as does what the watermark means across the rotation boundary.
2. **Unbounded compaction growth.** A compacted session's file keeps growing.
   Does the harvester re-read, and does compaction rewrite history already
   harvested — which would make a byte-offset watermark lie?
3. **Deletion and pruning.** These tools clean up after themselves. Once a source
   file is gone, the verbatim rows in our store are the only copy. Is that
   intended (yes, by [the harvest unit](harvest-unit.md)) and does anything need
   to notice the disappearance?
4. **Watermark unit.** Byte offset, line count, or last-seen record id. Rotation
   and compaction argue against byte offsets; a record id needs
   [session identity](session-identity.md) to be meaningful.
5. **Discovery is recursive, and getting it wrong fails silently.**
   [Session identity](session-identity.md) found Claude Code's real layout is
   `<slug>/<sessionId>.jsonl` plus
   `<slug>/<sessionId>/subagents/agent-<agentId>.jsonl`, nesting further under
   `subagents/workflows/wf_<id>/`. **A naive `<slug>/*.jsonl` glob misses 40 of
   46 files** — and misses them quietly, which is the dangerous part. Specify
   discovery per source, and a check that proves the file count found matches
   what is on disk.
6. **Not every `.jsonl` is a transcript.** `journal.jsonl` under
   `subagents/workflows/wf_<id>/` holds only `started`/`result` lifecycle records
   — no timestamps, no `sessionId`. Discovery must classify stream kinds, not
   assume every matched file is a transcript stream, and must derive identity
   from the path for the kinds that lack it in-content. See
   [store schema](store-schema.md).
7. **Bound the offline window.** [Ingest transport](ingest-transport.md) chose
   no local spool — the source files *are* the spool — and explicitly handed
   this ticket the job of quantifying the accepted risk: how long can a machine
   be offline before rotation or pruning loses a window of history? Answer it in
   wall-clock terms per source, so placement can pick timer intervals against a
   real number.
8. **Partial writes.** A session being written *right now* has a trailing partial
   line. The scanner must not consume it and must not advance past it.

Sharp enough to ticket and unblocked: it is about the scanner's contract with the
filesystem, not about what a record means.

## Resolution (2026-09-19)

### Measured first

- **No rotation observed.** Zero `.zst`, `.gz` or `.bak` files exist under
  `~/.claude/projects` or `~/.codex/sessions`. The `.jsonl` → `.jsonl.zst`
  behaviour [the format research](source-format-stability.md) reported came from
  upstream issue reports, **not** from local evidence. Handle it defensively;
  do not claim to have seen it.
- **Files are append-only and never rewritten.** For every Codex file sampled,
  `mtime - (last record timestamp)` is **exactly 0 seconds**. History is not
  rewritten in place, which also means `compacted` records are *appended* rather
  than replacing what they summarise.
- **No pruning within 76 days.** Files dated 2026-07-05 are still present as of
  2026-09-19.

### 1. Discovery — recursive, per-source, and counted

Per source, with explicit stream-kind classification rather than "every `.jsonl`
is a transcript":

- Claude Code: `<slug>/<sessionId>.jsonl` (main), plus
  `<slug>/<sessionId>/subagents/agent-<agentId>.jsonl`, plus deeper nesting under
  `subagents/workflows/wf_<id>/`. **A naive `<slug>/*.jsonl` glob finds 6 of 46
  files** and fails silently.
- `journal.jsonl` is a **workflow journal**, not a transcript: `started`/`result`
  only, no timestamps, no `sessionId` in content — identity comes from its path
  ([session identity](session-identity.md)).
- Codex: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`.
- OpenCode: the SQLite `event` table, not files at all.

Every run reports **files discovered per source**. A count that drops without an
explanation is the alarm; silent under-discovery is the failure this ticket
exists to prevent.

### 2. Watermark — `(inode, byte offset)`, self-invalidating

Append-only files make a byte offset safe, and it is the only unit that avoids
re-reading a 44 MB rollout every tick.

Per `(machine, source, stream)` the harvester stores the file's **inode** and the
**byte offset** it consumed to. On each tick:

- inode unchanged and `size >= offset` → read from `offset`.
- inode changed, or `size < offset` → **re-read from zero**.

Re-reading is always safe, because [ingest transport](ingest-transport.md) made
writes `ON CONFLICT DO NOTHING` — a full re-read costs bandwidth and nothing else.
That is what lets this rule be crude and still correct: replacement, truncation,
restore-from-backup and a lost watermark all collapse into the same harmless path.

### 3. Rotation and compression — defensive, not speculative

If a `.jsonl` disappears and a same-named `.jsonl.zst` appears, it is **the same
stream**: decompress and re-read from zero. Records already stored dedupe away;
only genuinely new tail records land. Since rotation has never been observed
locally, this path must be **tested synthetically** rather than assumed to work.

### 4. Partial writes

Consume only **complete lines** — a trailing fragment without a newline is left
for the next tick, and the offset never advances past it. A session being written
right now is therefore harvested up to its last complete record, never into a
half-written one.

### 5. The offline window — measured lower bound, not a guarantee

[Ingest transport](ingest-transport.md) accepted that a machine offline long
enough for source files to rotate or be pruned loses that window, and asked this
ticket to bound it.

**Measured: at least 76 days** — the oldest local file is from 2026-07-05 and is
still present. That is a lower bound observed on one machine, **not** a vendor
guarantee; all three tools are explicitly unstable
([format research](source-format-stability.md)) and none documents a retention
policy.

Practical consequence: a timer interval anywhere from minutes to days is safe
against this bound, so [placement](placement-scheduling.md) should pick the
interval for freshness, not for data safety. A machine offline for **months**
is the only real exposure, and the honest mitigation is to notice — a stream whose
watermark has not advanced in N days should be reported, not silently tolerated.
