---
type: grilling
status: closed
resolved: 2026-09-19
blocked-by: []
graduated-from: ingest-transport.md, 2026-09-19
---

## Question

The derivation worker runs beside the database and turns verbatim rows into
normalized events, renders, redaction flags and embeddings. What is its
operational shape?

[Ingest transport](ingest-transport.md) put derivation centrally and made the
harvester thin. The *logic* is already specified —
[normal form](event-normal-form.md), [render format](render-format.md),
[redaction detector](redaction-detector.md),
[embedding pipeline](embedding-pipeline.md) — but nothing owns how the worker
actually runs.

1. **Trigger.** Polling a derivation watermark, `LISTEN/NOTIFY` on insert, or a
   timer? Note the akh daemon already uses `pg_notify` elsewhere in this project,
   so there is precedent either way.
2. **Its own watermark.** Distinct from the harvesters' local ones, and stored in
   the database since this component legitimately has `SELECT`. What does it key
   on — the hypertable's `observed_at`, or an insertion sequence? These differ,
   because a backfilled old session arrives *now* with an *old* `observed_at`,
   and a watermark on the wrong one silently skips it. This is the sharpest
   question in the ticket.
3. **Re-derivation.** The whole point of keeping raw. How is a full or partial
   re-derivation triggered, how does it avoid orphaning old normalized rows and
   embeddings, and can it run while harvesters are still inserting?
4. **Idempotency and ordering.** Derivation must be safe to run twice and safe to
   run concurrently with itself, or explicitly singleton via an advisory lock —
   the project already has `run_migrations_locked` as precedent.
5. **Partial failure.** A batch where one record's normalizer panics must not
   block the rest, and must be visible rather than silently skipped — the
   `degraded` marker and drift counters from the normal form are the mechanism.
6. **Where it runs.** In-cluster next to the CNPG database, given it needs the
   ONNX embedding model and low-latency database access. Same deployment story as
   `seshd`, or something smaller?
7. **Backfill of the existing corpus.** 66,084 records already on disk across two
   machines. The first run is a bulk backfill, not a steady-state tick, and
   should be exercised as such.

## Resolution (2026-09-19)

**Anti-join discovery at the current derivation version; re-derivation by version
bump; a single advisory-locked polling worker in-cluster.**

### 1. Discovery — pending is an absence, not a watermark

Pending = raw rows with **no normalized row at the current derivation version**
(`NOT EXISTS`, supporting index on the normalized key, `LIMIT` per batch).

This dissolves the question this ticket called its sharpest — whether the
watermark should key on `observed_at` or an insertion sequence. There is no
watermark. A session backfilled today with a three-month-old `observed_at` is
found because its normalized row is missing, not because its timestamp falls in
some window. Backfill and steady state are the same code path.

[Seshat's embed worker](../../../crates/seshat/src/embed_worker.rs) is the
precedent — it polls `WHERE embedded_at IS NULL` (`embed_worker.rs:171`) — but
its *marker column* form cannot be copied here: raw rows are immutable by grant
([ingest transport](ingest-transport.md) gave harvesters `INSERT` only) and live
on a compressed hypertable, so flagging one would mean decompressing a chunk to
write back. The anti-join is the same idea without mutating raw.

**Do not copy `embed_pending_batch_impl`'s batching.** It calls `.all(db)` and
then `.take(batch_size)` — every pending row is loaded into memory before being
truncated. Ours must push the limit into SQL.

### 2. Poison records must not re-pend forever

A direct consequence of discovery-by-absence, and the failure mode to design for:
a record whose normalizer always fails never gets a normalized row, so it is
found again on **every** tick, forever — a silent hot loop that also blocks
nothing and therefore looks healthy.

The fix is that failure still writes a row: a normalized row marked
**`degraded`** (or explicitly failed), carrying the reason, which satisfies the
anti-join and stops the retry. This reuses the `degraded` marker and the
per-`(source, type, schema_marker)` drift counters from
[the normal form](event-normal-form.md) rather than inventing a second mechanism.
A rising degraded count is the alarm; a hung record is not a possibility.

Partial failure is therefore per-record: one bad record degrades itself and the
rest of the batch commits.

### 3. Re-derivation — version bump

Normalized rows carry the `derivation_version` that produced them. Pending is
defined against the current version, so **bumping the version re-pends the
corpus** with no deletes, no window during which the corpus has a hole, and no
orphans — superseded rows are dropped lazily once their replacement exists.

It also makes partial re-derivation expressible (bump for one source, or one
stream kind), and makes it directly visible which normalizer produced which row —
which matters given [the format research](source-format-stability.md) expects the
normalizer to change repeatedly.

### 4. Runtime — one worker, advisory-locked

A single worker running in-cluster beside the CNPG database, holding a Postgres
advisory lock so a second instance refuses to start — the pattern already in
`run_migrations_locked` (`crates/akhomed-store/src/lib.rs:96`). Polls on a
configurable interval, as Seshat's does.

At 858 records/day this is the simplest correct thing, and it keeps the ONNX
embedding model loaded exactly once. Rejected: multiple workers with
`SKIP LOCKED` (real concurrency to get right, several resident copies of the
model, no volume to justify it); `LISTEN/NOTIFY` (seconds instead of a poll
interval, but notifications are not durable, so a sweep is needed anyway and both
paths must then be maintained).

**Backfill is the first run, not a special mode:** 66,084 records already exist
across two machines. Same code path, but it must be exercised as a bulk run —
specifically to confirm the anti-join is index-backed, since unindexed it
degrades badly at exactly that size.

### 5. Boundary — derivation records links, the read surface splices

[The normal form](event-normal-form.md)'s reading order splices subagent streams
into their parent at the spawning `tool_call`. That is a **cross-stream** join and
belongs to [the read surface](read-mcp-surface.md), not here. The worker's job is
to normalize each `started`/`result` record into a `lifecycle` event carrying the
parent and child identifiers; assembling them into a spliced session is a read
concern. Derivation stays per-record and order-free, which is what makes batching
and re-derivation safe.

## Residual risk — `observed_at` is derivation the worker cannot own

`observed_at` is part of the raw primary key ([store schema](store-schema.md)),
so the **harvester** must compute the forward-fill before inserting. That is a
piece of derivation logic living on the thin client, contradicting
[ingest transport](ingest-transport.md)'s "derive centrally" principle in exactly
one place — and the expensive kind, because changing the rule changes the primary
key, making it a re-*insert* rather than a re-derivation.

**Therefore: the carry-forward rule is a frozen contract**, shared by harvester
and store and versioned with the schema rather than with the normalizer. It is
deliberately trivial (own timestamp, else nearest preceding in stream, else
quarantine) so that it can stay frozen. If it ever must change, the correct move
is to reopen [the store schema](store-schema.md) and take `observed_at` out of
the key — not to patch the rule in place.
