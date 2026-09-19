---
type: grilling
status: closed
resolved: 2026-09-19
blocked-by: [store-schema.md]
---

## Question

How does a harvester on an arbitrary machine get records into the central store,
and what credential does it hold to do it?

The harvester runs wherever sessions are produced — this box, the Mac mini,
archibald, exec1 — while the store is central. Two shapes:

- **Direct Postgres.** Simplest code, transactional batch insert, no new service.
  Costs a database credential on every machine that has ever run a coding agent,
  and requires network reachability to the cluster from all of them.
- **An ingest endpoint.** One narrow write API, per-machine tokens that can be
  revoked individually, no database credential outside the cluster. Costs a
  service to build, deploy and keep available — and if it is down, the harvester
  must degrade without losing the watermark.

Also to settle:

- Where the credential lives per platform, and how it is provisioned. Note the
  cautionary precedent in `docs/wayfinder/agent-loops/tracker.md`: a dedicated
  credential per integration, read at use time, never written into config. Note
  also that the Anima PAT on *this* box is simply absent — the failure mode is
  real.
- Batch size and failure semantics: a partial batch must leave the watermark
  behind, never ahead.
- Whether the harvester ever runs on a machine that cannot reach the store at
  all, and whether that needs a local spool.

## Resolution (2026-09-19)

**Direct Postgres, per-machine insert-only role. The harvester pushes verbatim
rows only; all derivation happens centrally. No local spool.**

### 1. Transport — direct, least-privilege

Each machine connects with its **own PG role granted `INSERT` on the raw table
and nothing else**: no `SELECT` on other sessions, no `UPDATE`, no `DELETE`, no
DDL. This keeps the existing `akhomed` connection-URL precedent
(`src/config.rs:202`, `DATABASE_URL` override) while removing its weakness — a
credential sitting on a laptop that is full of other credentials. Revocation is
`DROP ROLE`, per machine, with no service to build and no new auth system to run.

Rejected: a full-rights credential per machine (the plain akhomed pattern — one
compromised host reads and destroys the whole corpus); and an ingest endpoint
(strongest isolation and a natural home for embedding, but a service to build,
deploy, monitor and degrade around, for four LAN machines).

### 2. Three consequences that make the design tighter

These fell out of the insert-only choice rather than being designed in, and they
are the reason it is worth more than the security argument alone:

- **Writes must be `INSERT … ON CONFLICT DO NOTHING`**, because `DO UPDATE` would
  need `UPDATE`. That is exactly right for this data: raw rows are immutable, so
  there is nothing a second write could legitimately change. The append-only
  model is now enforced by the grant, not by convention.
- **The watermark must live on the machine**, because an insert-only role cannot
  `SELECT` one back. That matches
  [the process model](process-model.md)'s per-`(machine, source, stream)`
  watermark and keeps the role genuinely insert-only.
- **Losing a watermark is therefore harmless.** The harvester re-scans from the
  beginning and every row collides into `DO NOTHING`. Wasteful, not corrupting —
  the system self-heals from local state loss.

This last property only holds because `observed_at` is deterministic. Had
[the store schema](store-schema.md) allowed backward-fill, a re-scan would have
computed different partition keys and inserted duplicates instead of colliding.
The two decisions depend on each other.

### 3. Derivation is central, the harvester is thin

The harvester reads files and pushes verbatim rows. A **derivation worker beside
the database** does normalization, rendering, redaction flagging and embedding.

The deciding argument is [the format research](source-format-stability.md): all
three formats break on upstream releases, so the normalizer *will* change
repeatedly. With derivation at the harvester, every fix is a four-machine,
three-OS redeploy, and re-derivation needs the original files to still exist
locally — which [source file lifecycle](source-file-lifecycle.md) says they will
not. Centrally, a normalizer fix is one deploy and re-derivation is one job.

Two further effects:

- The harvester becomes small enough to port to macOS and illumos without
  carrying an ONNX runtime, a tokenizer or a chunker.
- The embedding model lives on **one** host instead of four. This answers the
  sharpest open question in [the embedding pipeline](embedding-pipeline.md) —
  its item 3 — from the outside.

New ticket: [derivation worker](derivation-worker.md). The *logic* is specified
across the normal form, render, redaction and embedding tickets, but nothing
owned its operational shape until now.

### 4. Failure — the source files are the spool

The watermark advances only after a batch commits, so a failed or partial push
retries on the next timer tick and re-reads from the source files, which are
already durable on disk. The harvester holds no queue of its own; a second
durable store would only duplicate what the filesystem already guarantees.

**Accepted risk, explicitly:** a machine offline long enough for its source files
to rotate to `.jsonl.zst` or be pruned loses that window. That window is bounded
by upstream's retention behaviour, not ours — handed to
[source file lifecycle](source-file-lifecycle.md) to quantify, since it already
owns rotation and pruning.

### 5. Credential placement

One role per machine, its URL delivered by the platform's own secret mechanism
rather than written into a config file — following the caution recorded in
`docs/wayfinder/agent-loops/tracker.md` (a dedicated credential per integration,
read at use time, never committed). Provisioning each machine's role is part of
[placement and scheduling](placement-scheduling.md), which this resolution
unblocks.
