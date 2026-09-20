# 52a.1 brief — Sokar MCP surface (`/mcp` on sokard)

Implementer: Codex `gpt-6-astra` (local `codex exec`), branch `sokar/52a1-mcp` from main. Reviewer: Fable architect session.
Authority: [ADR 108](../../decisions/108-sokar-artifact-daemon.md) D7 (agents reach Sokar through one tool surface), [52a review](52a-review.md) (accepted foundation), Anima WI `52a.1` (Akh-Medu project). Operator ask: Sokar must be usable from Codex, Claude Code and every akh through MCP.

## Deliverable

`sokard` serves an MCP server (rmcp, streamable-HTTP) at `/mcp` on the API listener, behind the same principal bearer auth as `/v1/*`. Consumers register `http://sokar.lab.home/mcp` with `Authorization: Bearer <fleet token>`. Same repo rules as 52a: only `crates/sokar`, `crates/sokar-client`, `charts/akh-medu`, `Cargo.lock`, `docs/ops/sokar.md` (new) and `docs/ai/phases/phase-52-sokar/52a1-mcp-report.md` may change. No push. Conventional commits with `Co-Authored-By: Codex gpt-6-astra <noreply@openai.com>`.

## Pattern to copy

`crates/seshat/src/mcp/` + `crates/seshat/src/auth.rs` + `crates/seshat/src/main.rs`: rmcp `0.17` with the streamable-HTTP server transport mounted as an axum service, `#[tool_router]`/`#[tool]` macros with `schemars` parameter structs, `ServerHandler` with instructions, bearer middleware in front. Match the rmcp version and features seshat uses (check its Cargo.toml) so the workspace keeps one rmcp.

## Design

- `crates/sokar/src/mcp/mod.rs` (+ `tools.rs`): `SokarMcp { state: Arc<Sokar>, principal: PrincipalId }` created per session from the authenticated principal (the existing `auth::authenticate` result; tools use it as `created_by`/`principal` exactly like the HTTP handlers). Mount: `Router::new().nest_service("/mcp", streamable_http_service(...))` layered with the same `api_boundary`/auth middleware; keep `problem+json` for non-MCP errors and let rmcp return tool errors (`CallToolResult::error`) that carry the `SokarError` diagnostic code + help text.
- Config: `[server] public_api_url: Option<Url>` (used to build `content_url`s; when absent, tools return relative paths `/v1/artifacts/{id}/content`); `[mcp] enabled = true` (default), `path = "/mcp"`, `max_inline_bytes = 8_388_608` (put/read limits). `deny_unknown_fields` stays; update `examples/local.toml`.
- Tools (all names prefixed `sokar_`; JSON parameter structs with doc comments — those become the schema descriptions the models read, so write them for an agent):
  - `sokar_collection { scope: CollectionScope }` → create-or-get; returns the collection.
  - `sokar_put_text { collection_id, path, text, media_type = "text/plain; charset=utf-8", kind = "document", labels? }` → `{ artifact, outcome }`.
  - `sokar_put_base64 { collection_id, path, base64, media_type, kind = "other", labels? }` → same; reject > `max_inline_bytes` decoded with a helpful error.
  - `sokar_list { collection_id, prefix?, kind?, limit = 50, cursor? }` → page.
  - `sokar_get { artifact_id }` → metadata + `content_url`.
  - `sokar_find { collection_id, path }` → latest artifact at path (404 → tool error).
  - `sokar_read_text { artifact_id, max_bytes = 1_048_576 }` → text; if the bytes are not valid UTF-8, error telling the caller to use `sokar_read_base64`.
  - `sokar_read_base64 { artifact_id, max_bytes = 8_388_608 }` → base64 + media_type + size.
  - `sokar_delete { artifact_id }` → `{ tombstoned: true }`.
  - Reuse the service layer (`service.rs`) — no duplicated logic; uploads go through the same `stage`/`finish_upload` path (write the bytes into the staged upload from memory).
- Server `instructions` (short, for the model): what Sokar is, that artifacts are content-addressed and immutable per version, that `path` is unique per collection with latest-wins, and that `content_url` needs the same bearer token.
- Chart 0.1.17: `sokard.mcp.enabled`, `sokard.mcp.path`, `sokard.config.publicApiUrl` → rendered into `[mcp]` / `[server] public_api_url`; nothing else changes (the HTTPRoute already forwards `/`).
- `docs/ops/sokar.md`: how to run, the two listeners, the MCP endpoint, and registration snippets:
  - Codex `~/.codex/config.toml`: `[mcp_servers.sokar]\nurl = "http://sokar.lab.home/mcp"\nbearer_token_env_var = "SOKAR_MCP_TOKEN"`
  - Claude Code: `claude mcp add --scope user --transport http sokar http://sokar.lab.home/mcp --header "Authorization: Bearer $SOKAR_MCP_TOKEN"`
  - akhomed / akh-session-services upstream MCP config (find the exact key names in `src/config.rs` `UpstreamsSection` and `crates/akh-session-services` and show a snippet for each; do not modify those crates).

## Tests / exit condition

1. Gates: scoped fmt, `cargo clippy --locked -p sokar -p sokar-client --all-targets -- -D warnings` (rustc 1.98 clippy is what CI runs — keep `SokarError` < 128 bytes), `cargo test --locked -p sokar -p sokar-client`.
2. Integration test with an in-process rmcp client over streamable-HTTP against the running router (fs store, ephemeral index): `initialize` → `tools/list` returns the 9 tools with schemas; `sokar_collection` → `sokar_put_text` → `sokar_find` → `sokar_read_text` round-trips the text; `sokar_put_base64` of 3 bytes then `sokar_read_base64` matches; `sokar_delete` then `sokar_get` → error; a request without bearer → 401 before MCP; oversized `put_base64` → error naming the limit.
3. `sokard --check` unchanged; `--config examples/local.toml` boots with `[mcp]`.
4. Report `52a1-mcp-report.md`: what landed, deviations, and the three registration snippets verified by `helm template` output of `sokar.toml`.
