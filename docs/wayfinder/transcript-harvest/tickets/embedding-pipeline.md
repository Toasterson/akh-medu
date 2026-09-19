---
type: grilling
status: closed
resolved: 2026-09-19
blocked-by: [event-normal-form.md, store-schema.md]
graduated-from: read-surface.md revision, 2026-09-19
---

## Question

With Seshat out of the design, we own chunking and embedding. Where does that run,
over what text, and how does the model asset get to the machines that need it?

Seshat previously answered all of this behind its ingest tool. Collapsing to one
store hands it back to us.

0. **Do not embed everything.** [The read surface](read-mcp-surface.md)'s byte
   census says prose is 14.2% of the corpus and process output plus noise is most
   of the rest. Embedding tool output and compaction snapshots would spend the
   majority of the embedding budget on the least searchable material and pollute
   semantic hits. Decide the embedded subset explicitly — prose-first is the
   obvious starting point.
1. **What gets embedded.** The verbatim row is wrong (it is JSON scaffolding).
   The candidates are the normalized event text and
   [the compact render](render-format.md). Note the render is *lossy by design* —
   embedding it means searching a summary. Decide, and say what a hit points at.
2. **Chunk granularity.** A single event is often far too small to embed
   meaningfully and a whole session far too large.
   `seshat::ingest::chunk_paragraph_aware` (paragraph-aware, non-overlapping,
   1500-char target) is reusable — but it was tuned for prose documents, not for
   turn-structured transcripts. Reuse, retune, or chunk on turn boundaries?
3. **Where the embedder runs — SETTLED 2026-09-19.**
   [Ingest transport](ingest-transport.md) put all derivation centrally, in the
   [derivation worker](derivation-worker.md) beside the database. The embedding
   model lives on **one** host, not on every machine that ever ran a coding
   agent. The options below are retained only to record what was rejected:
   - *At the harvester*, per machine — needs `model.onnx` + `tokenizer.json` and
     an ONNX runtime on **every** machine that has ever run a coding agent.
   - *At a central ingest endpoint* — one copy of the model, but requires the
     endpoint shape rather than direct-Postgres writes.
   - *As a worker beside the store* — harvester writes rows, a worker embeds
     asynchronously. Mirrors Seshat's own `embed_worker` and its ~30s visibility
     lag, which is fine for a timer-driven harvester.
4. **The read path also needs the model.** Semantic search must embed the *query*.
   So the MCP read server needs the embedder too, regardless of where ingest
   embeds. Confirm this does not force the model onto the read host unexpectedly.
5. **Dependency shape.** Depending on the whole `seshat` crate for its embedder
   pulls in rmcp, axum, sea-orm and the rest of a service. Decide between
   depending on it as-is, factoring the embedder into a small shared crate, or
   vendoring. Extraction is the cleaner answer but touches a crate with other
   consumers.
6. **Re-embedding.** [The harvest unit](harvest-unit.md) exists so derivation can
   be re-run. Re-deriving text means re-embedding it — which, unlike with Seshat,
   we can now simply do, because we own delete and upsert. State the trigger and
   make sure a re-derivation cannot leave orphaned vectors.
7. **Index and dimensionality.** 384-dim vectors; which pgvector index (HNSW vs
   IVFFlat), what recall target, and whether the vector table is a hypertable or
   a plain table keyed into the raw one — coordinate with
   [the store schema](store-schema.md).

## Resolution (2026-09-19)

### 1. Embed prose only

`prompt`, `reply`, `thought` — the decisions and plans. **Not** the compact
render: it is lossy by design, so embedding it would bound search quality by
whatever the render chose to keep, and re-rendering would silently change what is
findable. **Not** tool output, per [the read surface](read-mcp-surface.md)'s byte
census — process output is 29.3% of bytes and the least searchable material in
the corpus.

**Sized:** prose is ~22 MB → **~14,666 chunks**, **22.5 MB of vectors**, growing
roughly **80,000 chunks per machine per year**. Nothing here is large.

Rejected: embedding tool-call names and argument digests, which would make "when
did I run the migration that dropped the constraint" findable. It is a real use
case and a small fraction of tool bytes — but it needs its own extraction rule
and it noisies the semantic neighbourhood. Revisit once the prose index has a
user complaining.

### 2. Turn-boundary chunks

Group consecutive prose events within a turn, splitting at ~1500 characters —
comfortably inside MiniLM's 512-token limit (`embedder.rs:15`).

Transcripts are **turn-structured, not paragraph-structured**. Seshat's
`chunk_paragraph_aware` (`ingest.rs:215`) is markdown-heading-aware and tuned for
prose documents; turn boundaries are invisible to it, so a chunk could span a
prompt and its reply and blur whose words they are. Turn grouping also yields an
exact **`(seq` range`)`** per chunk, which is precisely the pointer the read
surface hands back for widening a semantic hit into `get_lines`.

Rejected: one chunk per event — perfect provenance, but many events are a single
short sentence, which embeds poorly and fragments the neighbourhood.

### 3. Extract a shared embed crate

`SentenceEmbedder` moves into a small crate that both Seshat and the derivation
worker depend on.

The justification is already in the tree: `ort = "=2.0.0-rc.11"` and
`tokenizers = "0.22"` are declared **byte-identically**, with identical features,
in both `crates/seshat/Cargo.toml:66-67` and the root `Cargo.toml:168-169`. They
are already duplicated; extraction removes a copy rather than adding a third.
Depending on `seshat` wholesale would instead drag sea-orm, rmcp, axum and
tokio-full — 47 dependencies of a service crate — into a worker that wants two
modules.

**This touches a crate with other consumers.** It is a refactor for code sharing,
*not* the Seshat delete/upsert work the map ruled
[out of scope](../map.md) — but it carries the same obligation: Seshat's own tests
must stay green, and the extraction should be behaviour-preserving with no change
to its embedding path.

### 4. The read path needs the model too

Semantic search must embed the **query**, so
[the read MCP surface](read-mcp-surface.md) loads the same embedder from the same
shared crate. Both it and the derivation worker run in-cluster, so the model asset
(`model.onnx` + `tokenizer.json`, resolved by `resolve_model_dir`) is **baked into
the image** rather than fetched at runtime — deterministic, no startup network
dependency, and it is the same asset for both.

This is only tractable because [ingest transport](ingest-transport.md) put
derivation centrally. Had embedding stayed at the harvester, this model would need
to reach every machine that ever ran a coding agent.

### 5. Index and storage

- **384-dimensional**, cosine, matching `EMBEDDING_DIM` (`embedder.rs:12`).
- **HNSW**, not IVFFlat: no training step, better recall and latency at this
  size, and 22.5 MB of vectors makes the index-build cost irrelevant.
- Vectors live in the same database, keyed back to
  `(source, session_id, stream_id, seq)` plus the chunk's seq range.

### 6. Re-embedding needs no mechanism of its own

Embeddings are derived rows like any other, so they carry the
`derivation_version` from [the derivation worker](derivation-worker.md) and are
found by the same anti-join. **A version bump re-pends embeddings along with
everything else** — there is no separate re-embed trigger, no orphan sweep, and no
second watermark. An embedding failure writes a `degraded` row exactly as a
normalization failure does, so a poison chunk cannot re-pend forever.

### 7. Degradation

`pgvector` absent means semantic search is unavailable while exact addressing
still works — and per [the store schema](store-schema.md) that must **fail loudly
at startup**, not warn, because the read surface advertises the capability.
[Extension verification](extension-verification.md) is the open task that settles
whether this is hypothetical.

## Correction 2026-09-19 — embed `prompt` and `reply` only

Decision 1 named `prompt`, `reply` and `thought`. [The render
prototype](render-format.md) then measured reasoning content at **0.11 MB total**
— Codex reasoning is 100% encrypted, 91.8% of Claude thinking blocks are empty.

Embedding `thought` would mean embedding ~180 real chunks and, worse, would
embed base64 ciphertext if the encrypted payload were passed through. **Dropped
from the embedded subset.**

Revised sizing: the embedded corpus is `prompt` + `reply` ≈ **7.4 MB**, roughly
**4,900 chunks** rather than the 14,666 estimated from a prose figure that had
counted 11.8 MB of ciphertext as prose.
