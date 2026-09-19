---
type: grilling
status: closed
resolved: 2026-09-19
---

## Question

Where do harvested transcripts live — what is the system of record?

## Resolution

**A dedicated TimescaleDB hypertable in its own schema is the system of record**,
one row per source record, exactly addressable by `(source, session_id, seq)`.
**Amended 2026-09-19:** the hypertable is now the *only* store — semantic search
runs on pgvector inside it. Seshat was originally to receive a derived, chunked
projection; see [the revised read surface](read-surface.md).

The two halves of the original ask are not alternatives: embedded chunks answer
meaning-shaped queries, and a hypertable answers "line N of session X". The
hypertable is truth either way; what changed on revision is that the embeddings
live in it rather than in a second service.

Rejected — **reusing `akhomed_mutation_log`** (ADR 091): maximum reuse, since
harvest, watermark, body-paging and compression all exist there, but that table is
an *akh's own* WAL. Foreign human-agent sessions would enter one akh's personal
episodic memory and travel with it on relocation, and wake-replay already has to
special-case `kind="turn"`. The user's "independently of the current daemon"
constraint points the same way. ADR 091 is the pattern to copy, not the table to
share.

Also rejected: Seshat-only (loses line addressing entirely) and hypertable-only
(later revisited and reaffirmed against — see [read surface](read-surface.md)).
