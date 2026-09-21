# 52a.1 review — Sokar MCP surface

Reviewer: Fable architect session, 2026-09-15. Subject: `sokar/52a1-mcp` at `60bd018b`. Reviewer gates: scoped fmt, clippy `-D warnings`, 38 tests pass. Design accepted: per-principal `StreamableHttpService` + `LocalSessionManager` selected by the authenticated request extension (sessions cannot cross principals), tools reuse the service layer, `[mcp]` + `public_api_url` config, chart 0.1.17.

## Round 2 — required fixes

- **R1 Session expiry** (`mcp/mod.rs`): `LocalSessionManager::default()` uses `SessionConfig { keep_alive: None }` = never expire; reconnecting agents (crash, network drop, no DELETE) leak a worker task + channels per session forever. Add `[mcp] session_idle_secs` (default 900) and build each manager with `SessionConfig { keep_alive: Some(Duration::from_secs(..)) }`. Test: a session that is idle past the TTL answers 404 on reuse and a fresh initialize succeeds.
- **R2 Body limit on the MCP subtree** (`http.rs`): rmcp buffers and parses the whole POST before any tool runs, so a 200 MB `text` argument is fully materialised twice before `InlineLimit` fires. Wrap `crate::mcp::service(..)` with `RequestBodyLimitLayer::new(max_inline_bytes * 4 / 3 + 64 KiB)` before `nest_service`; test that an oversized MCP POST is rejected with 413 before parsing.
- **R3 Idempotency for uploads** (`mcp/tools.rs`): add optional `idempotency_key: Option<String>` to `sokar_put_text` and `sokar_put_base64`, threaded into `UploadOptions` exactly like the REST `Idempotency-Key` header; describe it in the schema ("reuse on retry to avoid a duplicate version"). Test: same key + same content → same artifact id; same key + different content → `IdempotencyMismatch` tool error.
- **R4 Do not rewrite rmcp error bodies** (`http.rs` `api_boundary`/`normalize_problem`): responses originating under `mcp.path` keep rmcp's status + body (400 "Session ID is required", 404 "Session not found", 406/415/422); only the auth 401/403 stays Sokar's problem+json. Test: a POST without `Mcp-Session-Id` on a non-initialize request returns rmcp's text.
- **R5 Authorize before decoding** (`mcp/tools.rs` `sokar_put_base64`, `sokar_put_text`): parse the collection id and call `authorize()` first, then `check_inline`/decode (pre-auth work amplification once 52b lands). Also align permissions with REST: `sokar_list` uses the same permission REST `GET …/artifacts` uses (Read), and `sokar_find` authorizes the same resource string REST does for the by-path route. Introduce a small typed `Resource` enum shared by `http.rs` and `tools.rs` that renders both the REST path and the MCP resource so they cannot drift (keeps 52b's grant evaluation consistent).
- **R6 Config diagnostics** (`config.rs`): replace the single `mcp.path` boolean disjunction with a `RoutePath` newtype (`FromStr`/`Deserialize`) that reports a typed reason (`NotAbsolute | TrailingSlash | DotSegment | IllegalByte | ReservedV1`) with per-variant help; validate only when `mcp.enabled`. Tests per variant.
- **R7 One page default** (`mcp/tools.rs`): `ListParams.limit: Option<u32>` passed straight to `index::page_limit()` so REST and MCP share the default (100) and clamp; drop `list_limit()`.
- **R8 Report + ops doc**: update `52a1-mcp-report.md` ("Round 2" section) and `docs/ops/sokar.md` for `session_idle_secs`, the MCP body limit and the idempotency parameter. Gates as before (clippy 1.98 clean, `SokarError` < 128 bytes).

## Accepted as-is

- Raw `String` tool parameters re-parsed per tool: deliberate so malformed ids surface as the `{code,message,help}` tool-error envelope the contract tests assert, not as JSON-RPC invalid_params. Document this choice in the report.
- MCP tool-error envelope `{code,message,help}` differs from REST `problem+json`: acceptable for a tool result; keep the `code` values identical.
- `docs/ai/architecture.md` is updated by the reviewer at merge (outside the implementer's file scope).
