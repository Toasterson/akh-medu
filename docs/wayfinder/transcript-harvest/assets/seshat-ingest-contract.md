# Seshat ingest and search contract

Research note for the transcript-harvester project. Answers what `seshd`
(`crates/seshat/`) actually accepts on ingest and returns on search, read
directly from source (the live `http://seshat.lab.home/mcp` endpoint 401s from
this box, so nothing here was validated against a running server). Every claim
below cites a `file:line`; anything not answered by the code is marked as a
gap.

## 1. Ingest entry points

There is exactly one ingest path, exposed both as an MCP tool and reused by
one internal REST route. There is **no dedicated ingest REST endpoint**.

- **MCP tool `seshat_ingest`** — `crates/seshat/src/mcp/mod.rs:511-566`.
  Declared via `#[tool(name = "seshat_ingest", ...)]` at
  `crates/seshat/src/mcp/mod.rs:510-513`. Params (`IngestParams`,
  `crates/seshat/src/mcp/mod.rs:91-123`):
  - `title: String` (required)
  - `slug: Option<String>` (default `None`, auto-derived if omitted —
    `crates/seshat/src/mcp/mod.rs:96-97`)
  - `format: String` (default `"markdown"` via `default_format()`,
    `crates/seshat/src/mcp/mod.rs:102,134-136`); accepted values per the tool
    description are `markdown`, `text`, `html`, `pdf`, `epub`, `mobi`, or
    `agent_session_transcript` (`crates/seshat/src/mcp/mod.rs:98-101`) — only
    `markdown`/`md` and `agent_session_transcript` change chunking behavior
    (see §2); everything else is opaque prose (`crates/seshat/src/ingest.rs:172-186`).
  - `body: String` (required, full document text)
  - `source_path: Option<String>` (default `None`)
  - `tags: Vec<String>` (default empty)
  - `metadata: serde_json::Value` (default empty; `null` is coerced to `{}`
    at `crates/seshat/src/mcp/mod.rs:544-548`)
  - `ingested_by: Option<String>` (default `None`)
  - `target_chunk_chars: usize` (default `1500` =
    `ingest::DEFAULT_TARGET_CHUNK_CHARS`, `crates/seshat/src/mcp/mod.rs:121-122,137-139`)

  Returns `IngestResult` (`crates/seshat/src/mcp/mod.rs:162-168`) as pretty
  JSON text content: `document_id` (UUID string), `slug`, `chunk_count`,
  `word_count`.

- **Underlying function**: `ingest::ingest_document(db, IngestRequest)` —
  `crates/seshat/src/ingest.rs:92-170`. Chunks the body, inserts one
  `pa_documents` row and N `pa_chunks` rows in a single transaction
  (`crates/seshat/src/ingest.rs:125-162`). **Embedding is not performed here**
  — chunks are inserted with `embedded_at = NULL`
  (`crates/seshat/src/ingest.rs:154`) and picked up later by the async
  embed worker (§7).

- **Companion MCP tool `seshat_embed_now`** —
  `crates/seshat/src/mcp/mod.rs:568-626`. No params. Forces one embed batch
  (256 chunks) synchronously and returns `{chunks_processed, embeddings_created}`.
  Only compiled in when the `embedding` feature is enabled; otherwise returns
  an `McpError::internal_error` (`crates/seshat/src/mcp/mod.rs:591-598`).

- **Other MCP tools** (context, not ingest): `seshat_search` (§5),
  `seshat_wish` (`crates/seshat/src/mcp/mod.rs:319-399`, RAG + optional LLM
  synthesis — synthesis itself is unimplemented, returns
  `"status": "synthesis_not_yet_implemented"` at
  `crates/seshat/src/mcp/mod.rs:374-386`), `seshat_status`
  (`crates/seshat/src/mcp/mod.rs:401-449`), `seshat_documents`
  (`crates/seshat/src/mcp/mod.rs:451-508`, paginated list, no filter by
  tag/metadata).

- **REST routes** — mounted in `crates/seshat/src/main.rs:160-176`:
  - `GET /scripts` — `crates/seshat/src/scripts.rs:74-86,126-146`
  - `GET /scripts/{name}` — `crates/seshat/src/scripts.rs:88-101,148-178`
  - `POST /scripts/{name}` — `crates/seshat/src/scripts.rs:103-122,180-218`

  These are **ELIZA KDL script storage**, not a general document-ingest API
  (module doc comment `crates/seshat/src/scripts.rs:1-11`). `POST
  /scripts/{name}` internally calls the same `ingest_document()` used by
  `seshat_ingest`, but wraps it in a slug-based delete-then-reinsert upsert
  (§3) that is *not* exposed for arbitrary documents. There is no
  `POST /documents` or similar generic REST ingest route — `/mcp` is the only
  path a document-ingest client is meant to use.
  `/mcp` is nested via `nest_service` at `crates/seshat/src/main.rs:175`.

## 2. Chunking model

- **Unit embedded**: one row in `pa_chunks` per chunk, embedded 1:1 into
  `pa_embeddings` (`crates/seshat/src/entity/pa_chunk.rs:1-36`,
  `crates/seshat/src/entity/pa_embedding.rs:1-34`). A chunk is plain UTF-8
  text (`pa_chunk.text`, `Text` column,
  `crates/seshat/src/entity/pa_chunk.rs:18-20`).

- **Default chunk size**: `DEFAULT_TARGET_CHUNK_CHARS = 1500` characters
  (`crates/seshat/src/ingest.rs:43`), chosen to stay under
  `all-MiniLM-L6-v2`'s 512-WordPiece-token cap with headroom
  (`crates/seshat/src/ingest.rs:24-27,37-42`). Floor is
  `MIN_TARGET_CHARS = 200` (`crates/seshat/src/ingest.rs:47`) — a caller
  passing `target_chunk_chars: 0` or a tiny value is silently bumped up to
  200 (`crates/seshat/src/ingest.rs:216`, test at
  `crates/seshat/src/ingest.rs:600-606`). This is a *character* budget, not
  tokens — no tokenizer runs during chunking, only at embed time.

- **No explicit overlap.** The chunker is paragraph-aware greedy packing with
  no sliding-window overlap between adjacent chunks: paragraphs are packed
  until the budget would overflow, then flushed
  (`crates/seshat/src/ingest.rs:209-275`). Long paragraphs fall back to
  sentence-boundary splitting (`crates/seshat/src/ingest.rs:358-389`), and
  sentences that still don't fit fall back to hard character-boundary splits
  (`crates/seshat/src/ingest.rs:429-442`). None of these fallbacks introduce
  overlap either — this is strictly a non-overlapping tiling.

- **Markdown headings**: for `format` values `"markdown"`/`"md"`, a leading
  ATX heading (`#`..`######`) is stripped off a paragraph and propagated as
  `heading` onto that chunk and all following chunks until the next heading
  (`crates/seshat/src/ingest.rs:214-238`, tests
  `crates/seshat/src/ingest.rs:559-651`).

- **`agent_session_transcript` format — the one relevant to us**:
  `crates/seshat/src/transcript.rs`. `chunks_for_request()`
  (`crates/seshat/src/ingest.rs:172-186`) dispatches to
  `transcript::chunk_agent_session_transcript()` when
  `format == "agent_session_transcript"`
  (`AGENT_SESSION_TRANSCRIPT_FORMAT`, `crates/seshat/src/transcript.rs:15`).
  This function:
  1. Parses the body as line-delimited JSON, one event per line
     (`crates/seshat/src/transcript.rs:78-149`). A parse failure on any line
     is a hard ingest error citing the 1-based line number
     (`crates/seshat/src/transcript.rs:85-90`, test
     `crates/seshat/src/transcript.rs:296-306`) — **the whole document is
     rejected**, not partially ingested, if any line is invalid JSON.
  2. Normalizes two envelope shapes: Codex `response_item` (unwraps
     `payload`, `crates/seshat/src/transcript.rs:95-102`) and
     Claude-style direct fields.
  3. Extracts **only** assistant message text and tool-result text
     (`SegmentKind::Assistant` / `SegmentKind::ToolResult`,
     `crates/seshat/src/transcript.rs:17-30`) — recognizing
     `type: "assistant"`/`role: "assistant"`, `tool_result`/`tool-result`
     content blocks nested in message content, and top-level
     `tool_result`/`function_call_output`/`custom_tool_call_output` events
     (`crates/seshat/src/transcript.rs:113-145`). Everything else (user
     messages/prompts, tool_use call arguments, IDs, timestamps, durations,
     protocol scaffolding) is dropped — confirmed by tests asserting absence
     of `session_id`, `tool_use_id`, `duration_ms`, `call_01`, raw shell
     commands, and the original human prompt text
     (`crates/seshat/src/transcript.rs:260-274,336-342`, and
     `crates/seshat/src/ingest.rs:526-528`). This is already a redaction
     step, but only redacts by *event-role*, not by *content* — a secret
     inside assistant prose or a tool result would still pass through
     untouched. **Gap**: there is no secret-scanning/PII-redaction pass
     anywhere in this file or module.
  4. Each extracted segment is independently paragraph-chunked
     (`chunk_paragraph_aware`, target = `target_chars` minus the
     "searchable context" prefix length, `crates/seshat/src/transcript.rs:59-73`).
  5. A compact **searchable-context prefix** is prepended to every chunk's
     text: `Work item: <id>`, `Executor: <x>`, `Outcome: <x>`, `Tags: <...>`
     — pulled from `metadata.work_item_id/executor/outcome` and the `tags`
     list (`crates/seshat/src/transcript.rs:201-230`). This means the raw
     chunk text itself (not just document metadata) carries these fields, so
     free-text semantic search can match on work-item id / executor /
     outcome even without using the structured filters.
  6. `chunk.heading` is set to `"Assistant"` or `"Tool result"`
     (`crates/seshat/src/transcript.rs:70`).

- **Embedder + dimensionality**: `all-MiniLM-L6-v2` via ONNX Runtime (`ort`)
  + HuggingFace `tokenizers`, mean-pooled + L2-normalized, **384 dimensions**
  (`EMBEDDING_DIM = 384`, `crates/seshat/src/embedder.rs:12`, full pipeline
  `crates/seshat/src/embedder.rs:50-166`). `MAX_TOKENS = 512`
  (`crates/seshat/src/embedder.rs:15`) — the tokenizer truncates to
  512 tokens; the model files (`model.onnx`, `tokenizer.json`) must be
  pre-placed in `~/.local/share/seshat/models/<name>/` or config
  `embedding.model_dir` — there is **no auto-download**
  (`crates/seshat/src/embedder.rs:176-198`, TODO at line 188). Storage
  column is `vector(384)` (`crates/seshat/src/entity/pa_embedding.rs:29`,
  raw SQL at `crates/seshat/src/migration/m20260409_000001_create_seshat.rs:123-132`).

- **Metadata that rides along per chunk and is queryable at search time**:
  - Document-level `tags` (JSONB array) and `metadata` (JSONB object) are
    stored once on `pa_documents` (`crates/seshat/src/entity/pa_document.rs:25-31`)
    and joined back onto every search hit
    (`RetrievedChunk.document_tags` / `document_metadata`,
    `crates/seshat/src/query.rs:25-26`, join SQL
    `crates/seshat/src/query.rs:105-120`) — **not duplicated per chunk row**.
  - Chunk-level columns returned by search: `chunk_id`, `document_id`,
    `document_title`, `text`, `chunk_index`, `similarity`, `heading`,
    `document_format` (`crates/seshat/src/query.rs:14-27`).
  - Three metadata fields are **structured/filterable** at query time via
    `SearchFilters` (`crates/seshat/src/query.rs:34-39`): `work_item_id`,
    `executor` (case-insensitive), `outcome` (case-insensitive) — implemented
    as parameterized `d.metadata ->> 'key'` predicates
    (`crates/seshat/src/query.rs:97-152`, SQL-injection-safety test at
    `crates/seshat/src/query.rs:168-191`). Exposed on the MCP `seshat_search`
    tool as `work_item_id`/`executor`/`outcome` params
    (`crates/seshat/src/mcp/mod.rs:56-65`). **Any other metadata key is not
    filterable** — only free-text vector search can reach it (or, for the
    transcript format, the searchable-context prefix baked into chunk text).
  - `chunk_index`, `chapter`, `section` exist on `pa_chunks`
    (`crates/seshat/src/entity/pa_chunk.rs:15-16,25-29`) but `chapter`/
    `section` are always `None` from the ingest path
    (`crates/seshat/src/ingest.rs:151-152`) — dead columns for our purposes.
    `query::surrounding_chunks()` (`crates/seshat/src/query.rs:325-348`) can
    fetch a `chunk_index` window around a hit for context expansion, but is
    **not wired to any MCP tool** — internal/library-only as of this reading.

## 3. Identity and idempotency

- **Primary key**: `pa_documents.id` is a server-generated `Uuid::new_v4()`
  chosen by `ingest_document()` itself (`crates/seshat/src/ingest.rs:114`),
  **not client-suppliable**. A caller cannot pin a document to their own
  stable id.
- **Slug**: `pa_documents.slug` has a **DB-level `UNIQUE` constraint**
  (`crates/seshat/src/migration/m20260409_000001_create_seshat.rs:36-40`).
  If `IngestParams.slug` is omitted, `derive_slug()` appends an 8-char UUID
  suffix to the title-derived slug specifically so **repeated ingests never
  collide** (`crates/seshat/src/ingest.rs:189-207,201-206`; the docstring
  says this outright: "Appends an 8-char UUID suffix to keep slugs unique
  when the same title is ingested more than once",
  `crates/seshat/src/ingest.rs:190-191`).
- **There is no upsert path in `ingest_document()` or in the `seshat_ingest`
  MCP tool.** Every call is an unconditional `INSERT`
  (`crates/seshat/src/ingest.rs:140,158-160`). Consequences for a
  re-runnable harvester:
  - If the harvester lets `slug` auto-derive (the default), **re-ingesting
    the same transcript N times creates N separate `pa_documents` rows and
    N×chunk-count duplicate chunks/embeddings** — silent duplication, not an
    error, not a replace.
  - If the harvester supplies a **stable, deterministic slug** (e.g.
    `claude-code/<session_id>` or a hash of `(source, session_id)`), a
    second ingest with the same slug fails at the DB unique constraint. That
    surfaces as a SeaORM error, which `ingest_document` does not special-case
    — it propagates as `SeshatResult::Err` up through
    `seshat_ingest_impl` and becomes `McpError::internal_error(format!("ingest: {e}"), ...)`
    (`crates/seshat/src/mcp/mod.rs:553-555`). The caller gets a generic
    internal-error tool response, not a structured "already exists"/conflict
    signal.
  - The **only place in this codebase that implements delete-then-reinsert
    upsert-by-slug** is `scripts::save_script_inner()`
    (`crates/seshat/src/scripts.rs:180-218`, calling
    `delete_document_and_chunks()` at `crates/seshat/src/scripts.rs:239-275`).
    That helper is private to the `scripts` module and wired only to the
    `POST /scripts/{name}` REST route for ELIZA KDL scripts — it is **not**
    reachable from `seshat_ingest` or any other MCP tool, and there is no
    generic "ingest-or-replace-by-slug" tool today.
- **Implication for the harvester**: to get idempotent re-derivation (the
  stated requirement — "re-runnable, will re-derive projections when the
  normalizer improves"), the harvester must either (a) implement its own
  delete-then-reinsert against Seshat by first calling `seshat_documents`
  (or a direct slug lookup we'd need to add) to find the prior document id
  for a given natural key, then... **there is no MCP delete tool to call**
  (see §4) — so (a) is not achievable purely as an external MCP client
  today; or (b) accept duplicate accumulation and treat old rows as
  superseded-but-present (search noise) until a manual/administrative purge;
  or (c) request Seshat grow a `seshat_upsert`/`seshat_replace` tool that
  wraps `delete_document_and_chunks` + `ingest_document` transactionally and
  is generalized past the `eliza-script` use case. This gap should be
  called out explicitly to whoever owns `crates/seshat/`.

## 4. Deletion / expunge

- **No MCP tool deletes anything.** Grepping the whole tool set
  (`crates/seshat/src/mcp/mod.rs`) turns up only
  `seshat_search`/`seshat_wish`/`seshat_status`/`seshat_documents`/
  `seshat_ingest`/`seshat_embed_now` — no `seshat_delete`, `seshat_expunge`,
  or `seshat_purge`.
- **No REST route deletes an arbitrary document.** The only exposed routes
  are the three `/scripts*` routes (§1); none accept a document id.
- **The only deletion code path**, `delete_document_and_chunks()`
  (`crates/seshat/src/scripts.rs:239-275`), is private, reachable only via
  `save_script_inner()`, and is scoped to the ELIZA-script slug convention.
  It does correctly cascade: it deletes `pa_embeddings` rows for the
  document's chunks, then the `pa_chunks` rows, then the `pa_documents` row,
  all inside one transaction (`crates/seshat/src/scripts.rs:245-274`). This
  is also naturally covered by the FK `ON DELETE CASCADE` from
  `pa_chunks.document_id` and `pa_embeddings.chunk_id`
  (`crates/seshat/src/migration/m20260409_000001_create_seshat.rs:93-98,126`),
  so a raw `DELETE FROM pa_documents WHERE id = $1` at the SQL level would
  also clean up correctly.
- **Precedent for exactly this need** (retracting leaked/garbage content)
  exists as a one-off **migration**, not a runtime API:
  `m20260521_000001_purge_tecton_garbage`
  (`crates/seshat/src/migration/m20260521_000001_purge_tecton_garbage.rs`).
  It hand-writes a `DELETE FROM pa_documents WHERE lower(title) ~ '<regex>'`
  for specific contaminated titles/patterns, relies on cascade for the child
  tables, and also cleans up now-orphaned `pa_microtheory_cache` rows whose
  `chunk_ids_used` JSONB array no longer intersects any surviving chunk
  (`crates/seshat/src/migration/m20260521_000001_purge_tecton_garbage.rs:49-61`).
  Its own `down()` is a deliberate no-op — "Purged data was garbage — restoration
  is neither possible nor desired"
  (`crates/seshat/src/migration/m20260521_000001_purge_tecton_garbage.rs:66-69`).
- **Straight answer to Q4**: yes, a document (and its embeddings) *can* be
  removed, and cleanly (FK cascade, no orphaned vectors), but **only by
  someone with direct database/migration access** — there is no
  self-service MCP/REST verb a harvester or any other remote client can call
  to retract a specific document by id or slug today. If a secret leaks into
  a projected transcript that Seshat has already ingested and embedded, the
  fix path as of this reading is: an operator runs a manual
  `DELETE FROM pa_documents WHERE id = '<uuid>'` (or a purge-style migration
  like `m20260521_000001`) directly against the `seshat` Postgres database.
  This is a real gap against the stated requirement ("gates whether a
  transcript can ever be retracted after a secret leaks") — flag it as
  something the harvester project should either accept (redact before
  ingest, never rely on after-the-fact retraction) or fix upstream (add a
  `seshat_expunge(document_id | slug)` MCP tool built on the existing
  `delete_document_and_chunks` pattern, generalized out of `scripts.rs`).

## 5. What search returns

- **MCP tool `seshat_search`** — `crates/seshat/src/mcp/mod.rs:261-317`.
  Params (`SearchParams`, `crates/seshat/src/mcp/mod.rs:47-66`): `query`
  (required), `top_k` (default 10), `min_similarity` (default 0.3, cosine),
  `work_item_id`/`executor`/`outcome` (optional exact/case-insensitive
  metadata filters, §2).
- **Result shape**: a JSON array of `RetrievedChunk`
  (`crates/seshat/src/query.rs:14-27`), pretty-printed as the tool's text
  content (`crates/seshat/src/mcp/mod.rs:313-316`):
  ```jsonc
  {
    "chunk_id": "<uuid>",
    "document_id": "<uuid>",
    "document_title": "<string>",
    "text": "<chunk text, possibly with the searchable-context prefix>",
    "chunk_index": 0,
    "similarity": 0.91,
    "heading": "Assistant" | "Tool result" | null,
    "document_format": "agent_session_transcript" | "markdown" | ...,
    "document_tags": ["..."],
    "document_metadata": { "work_item_id": "...", "executor": "...", "outcome": "...", ... }
  }
  ```
  Ranking is by pgvector cosine distance ascending
  (`ORDER BY e.embedding <=> $1::vector`, `crates/seshat/src/query.rs:148`),
  filtered to `similarity > min_similarity` before the limit
  (`crates/seshat/src/query.rs:120`), so `top_k` is applied *after* the
  threshold, not a hard cap independent of it.
- **No native `(source, session_id, seq)` pointer field exists anywhere in
  the schema or the result type.** The closest things are:
  - `document_id` + `chunk_index` — an internal Seshat-native address, into
    *Seshat's own* chunking, not into our transcript's original JSONL line
    numbering. `chunk_index` is the index of a chunk *within a segment's
    packed output* for transcripts (assistant/tool-result text runs through
    `chunk_paragraph_aware` independently per segment,
    `crates/seshat/src/transcript.rs:64-73`) — it has no direct arithmetic
    relationship to a JSONL line number in the source file, and multiple
    original JSONL lines can be merged into one chunk or one line's content
    split across chunks.
  - `document_metadata` — this is exactly where our own addressing (source,
    session_id, and — if we choose to add it — a seq/line-range) could ride
    through **as an opaque, uninterpreted JSONB blob**, since `metadata` is
    stored and returned verbatim (`crates/seshat/src/ingest.rs:135`,
    round-tripped in `RetrievedChunk.document_metadata`). Nothing in Seshat
    validates, types, or indexes arbitrary metadata keys beyond the three
    hardcoded filter fields (§2); it will pass through whatever we put there
    (e.g. `{"harvester_source": "claude_code", "session_id": "...", "seq_range": [12, 47]}`),
    and it comes back on every hit for us to read client-side and hand off
    to our own exact `get_lines` MCP tool.
  - There is no `line_start`/`line_end`/`seq` column on `pa_chunks` today
    (`crates/seshat/src/entity/pa_chunk.rs:8-36` — only `chapter`/`section`,
    both unused/`None` per §2). **Gap**: if we want per-chunk (not just
    per-document) addressing back to exact lines, Seshat's schema and
    chunker would need a new column threaded through
    `ChunkDraft`/`pa_chunk`/`RetrievedChunk` — not present today. Document-
    level addressing via `document_metadata` is achievable now with zero
    Seshat changes; chunk-level line-range addressing is not.

## 6. Auth model for a non-akh client (the harvester)

- **Single mechanism: shared-secret bearer token, mandatory on every HTTP
  route.** Implemented in `crates/seshat/src/auth.rs` as an axum middleware,
  `require_bearer` (`crates/seshat/src/auth.rs:99-124`), layered over the
  whole router (`/mcp` **and** `/scripts*`) in
  `crates/seshat/src/main.rs:174-180`.
- **No per-client identity, no OAuth, no mTLS, no API keys-per-caller** —
  every accepted token is equally privileged; the harvester would be
  indistinguishable from any other holder of the shared secret. There is no
  akh/workspace scoping visible in `crates/seshat/src/auth.rs` at all — the
  "non-akh client" framing in the question doesn't change anything here: the
  server has exactly one authorization concept, a shared bearer secret, and
  no notion of caller type.
- **Token source**: `SharedSecretAuth::from_env()`
  (`crates/seshat/src/auth.rs:35-42`) reads either
  `SESHAT_SHARED_SECRET_FILE` (path to a file, e.g. a K8s projected Secret —
  preferred) or `SESHAT_SHARED_SECRET` (inline value) — file wins if both
  set. The file/value can contain **multiple newline-separated tokens**, all
  accepted, so credential rotation can overlap old/new
  (`crates/seshat/src/auth.rs:60-77`, rotation test
  `crates/seshat/src/auth.rs:194-233`). The file is **re-read on every
  request** (`crates/seshat/src/auth.rs:63-65`), so rotating the shared
  file's contents takes effect without restarting `seshd`.
- **Protocol**: standard `Authorization: Bearer <token>` header.
  Missing header → `401` with `WWW-Authenticate: Bearer realm="seshat"`
  (`crates/seshat/src/auth.rs:104-106,126-133`). Present-but-wrong or
  malformed (non-ASCII, missing `Bearer ` prefix, whitespace-padded token) →
  `403 Forbidden` (`crates/seshat/src/auth.rs:107-115,135-137`). Token
  comparison is constant-time (`crates/seshat/src/auth.rs:87-96`). If the
  secret source itself is unreadable at request time (e.g. K8s Secret file
  briefly missing), the server fails closed with `503 Service Unavailable`
  rather than allow-by-default (`crates/seshat/src/auth.rs:119-123`).
- **Startup posture**: `main()` validates the auth source *before* opening
  the DB connection or binding the listener, specifically so there is never
  a window where the service is reachable unauthenticated
  (`crates/seshat/src/main.rs:55-60`).
- **For the harvester**: getting the shared secret (via whatever channel
  distributes `SESHAT_SHARED_SECRET`/`_FILE` today — not visible from this
  crate; likely a K8s Secret per the file-based path's doc comment,
  `crates/seshat/src/auth.rs:15`) is sufficient and the *only* thing needed.
  There is no separate registration, scoping, or rate-limiting tied to that
  credential (see §7 for the absence of rate limiting generally).

## 7. Constraints and gotchas for a transcript projection specifically

- **Whole-document JSONL validation is all-or-nothing.** A single malformed
  line anywhere in the transcript body aborts the entire ingest with a
  line-numbered error (`crates/seshat/src/transcript.rs:85-90`) — the
  harvester's normalizer must guarantee every line is valid JSON before
  calling `seshat_ingest` with `format: "agent_session_transcript"`, or
  fall back to a plain-text format for anything it can't fully normalize.
- **Empty-content rejection.** Both the generic body
  (`crates/seshat/src/ingest.rs:101-105`) and, for transcripts specifically,
  "no assistant or tool-result text found"
  (`crates/seshat/src/transcript.rs:52-57`) are hard errors — a session with
  only tool calls and no assistant text or tool output (e.g. a pure
  tool_use/no-response turn) will fail to ingest at all under this format.
- **Redaction is role-based, not content-based.** The transcript extractor
  filters by JSON event *shape* (assistant/tool-result vs. everything else)
  — it does not scan the retained text for secrets, tokens, credentials, or
  PII (§2 point 3). If a secret is embedded in assistant prose or a tool
  result body (e.g. a printed API key, a `cat`'d `.env` file), it passes
  straight into `pa_chunks.text` and gets embedded into `pa_embeddings`
  verbatim. **The redaction the harvester needs must happen before calling
  `seshat_ingest`** — Seshat provides none for content-level secrets.
- **No document-level dedup/idempotency** (§3) — repeated re-ingestion under
  the harvester's re-run/re-derive workflow will accumulate duplicate
  documents unless the harvester enforces its own stable-slug + external
  bookkeeping, and even then a second call with the same slug hits a raw DB
  unique-constraint error surfaced as a generic MCP internal error, not a
  clean conflict response.
- **No delete/expunge API** (§4) — a leaked secret cannot be retracted
  without direct DB access once ingested and embedded.
- **Synchronous ingest, asynchronous embedding — a visibility gap.**
  `seshat_ingest` returns as soon as the document/chunk rows are committed;
  chunks are **not searchable** until the background `embed_worker`
  (`crates/seshat/src/embed_worker.rs:31-88`) picks them up on its next poll
  (default `poll_interval_secs = 30`, `crates/seshat/src/config.rs:97-99`,
  default `batch_size = 32`, `crates/seshat/src/config.rs:93-95`), or until
  the harvester explicitly calls `seshat_embed_now` to force an immediate
  batch of up to 256 chunks (`crates/seshat/src/mcp/mod.rs:568-626,607-616`).
  For a harvester doing bulk backfill, either budget for up to ~30s
  visibility lag per batch or call `seshat_embed_now` after each ingest
  batch (be aware it processes *all* pending chunks server-wide, not just
  the harvester's own — could be slow/contended if other producers are also
  ingesting).
  The `embedding` cargo feature must also be compiled into the running
  `seshd` binary at all, or both embedding and `seshat_embed_now` are
  unconditionally disabled (`crates/seshat/src/mcp/mod.rs:253-259,592-598`,
  `crates/seshat/src/main.rs:91-113`) — **gap**: this note cannot confirm
  from source alone whether the live `seshd` deployment has `embedding`
  compiled in; that must be checked against its actual build/deploy config,
  not assumed.
- **No request body size limit found in source.** No `tower_http`
  body-limit layer, no axum `DefaultBodyLimit` configuration, and no
  seshat-specific size check on `IngestParams.body` anywhere in
  `crates/seshat/src/` (confirmed by grep across the crate). The `rmcp`
  0.17 `StreamableHttpServerConfig`
  (`~/.cargo/.../rmcp-0.17.0/src/transport/streamable_http_server/tower.rs:33-51`)
  exposes only `sse_keep_alive`, `sse_retry`, `stateful_mode`,
  `json_response`, `cancellation_token` — no body-size or request-size knob.
  **Gap / inference**: this means either (a) there truly is no cap short of
  whatever axum/hyper/OS defaults apply underneath (not verified from this
  crate's source — would need to check the `axum`/`hyper` dependency
  defaults, which this research did not do), or (b) a limit is enforced
  somewhere in front of `seshd` (ingress/reverse proxy) that is invisible
  from this repo. Large transcripts (some agent sessions run to many MB of
  JSONL) should be assumed **unbounded from Seshat's own code** until
  proven otherwise — a harvester pushing very large sessions should
  self-limit or split, not rely on Seshat to reject oversized bodies
  gracefully.
- **No rate limiting found anywhere in `crates/seshat/src/`** (confirmed by
  grep) — a bulk-backfill harvester will not be throttled by Seshat itself;
  it should self-throttle to avoid overloading the shared `embed_worker`
  batch/poll loop and the ONNX inference thread (embedding runs via
  `spawn_blocking`, one embedder `Mutex` shared across all callers —
  `crates/seshat/src/embedder.rs:200-202`, lock acquired per batch in
  `crates/seshat/src/embed_worker.rs:108-121` and per query in
  `crates/seshat/src/mcp/mod.rs:225-249` — so concurrent ingest-heavy
  harvesting and interactive `seshat_search`/`seshat_wish` calls from other
  akhs will serialize on that one lock).
  The embed worker's own scaling knob is `EmbeddingConfig.batch_size`
  (default 32, `crates/seshat/src/config.rs:93-95`) and
  `poll_interval_secs` (default 30, `crates/seshat/src/config.rs:97-99`) —
  configured server-side, not something a harvester can influence per-call
  except via `seshat_embed_now`'s fixed batch of 256
  (`crates/seshat/src/mcp/mod.rs:610`).
- **Chunk index is per-segment, not per-line** (§5) — do not assume
  `chunk_index` can be inverted back to an original JSONL line number
  without carrying our own addressing in `metadata`.
- **`ingested_by` is a free string, not validated** — a good place for the
  harvester to stamp its own identity (e.g. `"transcript-harvester/claude-code"`)
  for provenance/debugging, but it's unindexed/unfilterable at search time
  (only `work_item_id`/`executor`/`outcome` are structured filters, §2).

## Summary for the projection contract

Concretely, for pushing a redacted/derived transcript projection into
Seshat today:

1. Use `seshat_ingest` with `format: "agent_session_transcript"` — it
   already does role-based extraction (assistant + tool-result text only)
   and chunking; do **not** reinvent that layer.
2. Perform **content-level secret redaction ourselves before calling
   `seshat_ingest`** — Seshat does none.
3. Carry our own `(source, session_id, seq/line-range)` addressing in the
   `metadata` JSON object — it round-trips verbatim to every search hit as
   `document_metadata` and is the only place a stable pointer back to our
   own exact-line MCP server can live; there is no native chunk-level line
   addressing to rely on instead.
4. Accept that **there is no idempotent upsert and no delete/expunge** as
   exposed capabilities — either compute and pass a stable deterministic
   `slug` and treat a unique-constraint failure as "already ingested, skip"
   (cheap but doesn't support re-deriving with an improved normalizer), or
   escalate to whoever owns `crates/seshat/` for a `seshat_upsert`/
   `seshat_expunge` tool built by generalizing the existing
   `delete_document_and_chunks` pattern out of `scripts.rs`. This is the
   single biggest open gap against the stated requirements (re-runnable
   harvester; secret-leak retraction).
5. Budget for the ingest/embed visibility gap (§7) and either poll
   `seshat_status`'s `pending_embedding` count or call `seshat_embed_now`
   after a backfill batch.
6. The shared-secret bearer token (§6) is sufficient auth for the
   harvester; no separate registration is needed, but there is also no
   per-client accountability if that secret is what gets audited later.
