---
type: grilling
status: closed
resolved: 2026-09-19
revised: 2026-09-19 — flag cleared by reversing the Seshat half of this decision
---

## Question

What is the read surface, who is the first consumer, and does Seshat stay in the
design at all?

## Resolution (revised 2026-09-19 — supersedes the original below)

**One store. A single MCP server over the harvester's own Postgres answers both
exact and semantic queries. Seshat is out of the design.**

- Exact addressing — session listing, whole-session fetch,
  `get_lines(session, from, to)`, grep — served from the hypertable.
- Semantic search served by **pgvector in the same database**, embedded with the
  same `all-MiniLM-L6-v2` model Seshat uses, by depending on the `seshat` crate
  as a **library** (`pub mod embedder`, `pub mod ingest::chunk_paragraph_aware`).
  Nothing is reimplemented and `seshd` is not in the picture.
- First consumer remains a coding agent reading its own or a sibling's session.

### What changed the decision

The original resolution kept Seshat for semantic search on the understanding that
its cost was sync and dedup work. [The ingest research](seshat-ingest-contract.md)
found two harder gaps — no upsert (re-derivation duplicates) and **no deletion**
(a redaction miss is unretractable through Seshat's interface) — which tripped the
reopening condition this ticket had written for itself.

Costing the options against the code then moved the ground again: fixing Seshat is
**small** (`delete_document_and_chunks` already exists and is transactional at
`crates/seshat/src/scripts.rs:240`, already used as an upsert primitive at `:195`;
publishing it plus an MCP tool is ~40 lines), and collapsing is **also small**,
because `seshat` is a library crate whose embedder and chunker can simply be
called. With both paths cheap, the decision stopped being about effort and became
architectural.

Collapsing wins on: one store, one trust boundary, full delete and upsert control
(which is what makes [the secrets posture](secrets-posture.md) honest), no
cross-service sync, no re-ingest dedup, no "which tool do I reach for" ambiguity
for the model, and no dependency on a service that currently answers 401.

**Accepted cost:** transcripts no longer sit in the shared corpus beside ADRs,
plans and papers, so cross-corpus semantic search is lost. If that is later
wanted, it returns as a fresh effort — an export — not as a resumption of this one.

### Consequences

- [Seshat projection](seshat-projection.md) is **closed, out of scope**.
- [Substrate](substrate.md) is amended: Seshat is no longer fed a derived
  projection.
- [Read MCP surface](read-mcp-surface.md) grows the semantic half.
- New ticket: [embedding pipeline](embedding-pipeline.md) — with Seshat gone, we
  own chunking, embedding and the model asset, and must say where they run.

---

## Original resolution (2026-09-19, superseded above)

A dedicated MCP server for exact addressing, **Seshat retained** for semantic
search — a user decision taken against the standing recommendation to collapse to
one store. Rejected at the time: Seshat-only (no exact line addressing) and
SQL-and-CLI-only (agents cannot use it without a human pasting results). Those two
remain rejected.
