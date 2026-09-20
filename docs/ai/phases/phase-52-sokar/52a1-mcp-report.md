# 52a.1 — Sokar MCP implementation report

Date: 2026-09-15. Implementer: Codex gpt-6-astra.
Branch: `sokar/52a1-mcp`. Authority: [52a.1 brief](52a1-mcp-brief.md),
[accepted 52a foundation](52a-review.md), [ADR 108](../../decisions/108-sokar-artifact-daemon.md),
Anima WI 52a.1 (Akh-Medu).

## Commits and scope

- `6e09a2a0` — `feat(sokar): expose authenticated artifact tools over MCP`:
  implementation, integration tests, chart 0.1.17, and operator guide.
- This report is committed separately as
  `docs(sokar): record 52a.1 implementation and validation`.

Both commits carry the requested Codex co-author trailer. No push, deployment,
tracker mutation, or changes to consumer configuration were performed. All changed
files are within `crates/sokar`, `crates/sokar-client`, `charts/akh-medu`,
`Cargo.lock`, `docs/ops/sokar.md`, and this report. The existing phase brief
supplies the plan; broader architecture/ADR documents were left outside this
slice's explicit file scope.

## What landed

- The API router mounts rmcp 0.17 streamable HTTP at `/mcp` by default, under the
  same `api_boundary`, bearer authentication, request IDs and HTTP body limit as
  REST. Admin routes remain on their separate listener.
- Nine tools with described JSON schemas: `sokar_collection`, `sokar_put_text`,
  `sokar_put_base64`, `sokar_list`, `sokar_get`, `sokar_find`, `sokar_read_text`,
  `sokar_read_base64`, `sokar_delete`. Defaults match the brief. Collection scope
  and artifact kind schemas derive from the existing wire enums through an
  optional `sokar-client/schema` feature.
- Uploads use the existing `Store::stage` and `Sokar::finish_upload`, preserving
  content deduplication, version chains, cleanup, and bytes → manifest → index
  ordering. Authenticated principal identity supplies collection ownership and
  artifact `created_by`. No caller-supplied identity field is exposed.
- Reads share `Sokar::read_inline`, check manifest size before content I/O,
  bound the accumulated bytes while streaming, and reject a size mismatch.
  Effective read cap is `min(max_bytes, mcp.max_inline_bytes)`; reads never
  truncate. Base64 uploads bound encoded allocation before decoding, then check
  the decoded length. Text uploads count UTF-8 bytes.
- Service failures become `CallToolResult::error` with JSON `{code, message,
  help}`. New typed `InlineLimit` and `NotUtf8` diagnostics identify the effective
  byte limit and direct binary callers to `sokar_read_base64`. Diagnostic-code
  parity and the existing `<128 bytes` error-size regression tests pass.
- `[mcp]` defaults: enabled, `/mcp`, 8,388,608 inline bytes. Optional typed
  `[server] public_api_url` yields absolute content URLs, retaining any proxy
  prefix; absence yields relative paths. Config validation rejects ambiguous
  route syntax, `/v1` overlap, zero inline limits, and non-HTTP(S)/credentialed
  public URLs. `deny_unknown_fields` remains intact.
- Chart version 0.1.17 exposes `sokard.mcp.enabled`, `sokard.mcp.path`, and
  `sokard.config.publicApiUrl`. No HTTPRoute change was needed.
- [Operator guide](../../../ops/sokar.md) documents both listeners, local boot,
  token files, tool results and bounds, registration, and the consumer limitation
  below. The local example explicitly enables MCP.

## Implementation choices / deviations

1. **Principal binding with rmcp's zero-argument factory.** The current daemon's
   identity type is `authz::Principal { id: String }`, not `PrincipalId`. Each
   configured principal gets a separate `StreamableHttpService` and
   `LocalSessionManager`. A dispatch service selects it using the authenticated
   request extension; its factory creates a fresh `SokarMcp` for each session.
   Thus a different valid principal cannot reuse a session ID through POST,
   GET/resume, or DELETE. Credentials still reload on every request, including
   existing sessions. Tool handlers also consult the existing authorizer with
   their actual resource and permission.
2. **Schema version.** Runtime rmcp version/features match Seshat exactly:
   `0.17`, `server`, `macros`, `transport-streamable-http-server`. Like Seshat's
   implementation, parameter structs use `rmcp::schemars` (1.x). The optional
   wire-schema dependency uses that same 1.x version; Seshat's separate 0.8
   dependency is not the one its MCP module imports. Cargo.lock retains one
   rmcp 0.17.0; only dependency edges changed, with no package upgrades.
3. **akhomed upstream gap.** `src/config.rs::UpstreamsSection` supports only
   `anima` and `seshat`. It has no typed Sokar field or dispatch, and its existing
   parser ignores unknown entries. A purported working `[upstreams.sokar]`
   snippet would therefore mislead operators. This slice documents the gap and
   the functioning session-services route; it does not modify out-of-scope
   engine/session-services crates or repurpose another upstream slot.

## Validation

All Rust gates used `rustc 1.98.0 (88d9e12ae 2026-08-18)` via an explicit
`+1.98.0` toolchain, without changing the workspace/default toolchain.

| Gate | Result |
| --- | --- |
| `cargo +1.98.0 fmt -p sokar -p sokar-client --check` | Pass |
| `cargo +1.98.0 clippy --locked -p sokar -p sokar-client --all-targets -- -D warnings` | Pass |
| `cargo +1.98.0 test --locked -p sokar -p sokar-client` | 38 reported passed, 0 failed, 1 ignored; PostgreSQL caveat below |
| In-process rmcp client → TCP router, fs store + ephemeral index | All 4 new transport tests pass |
| `sokard --check`, `--check-stores`, local reindex CLI | Existing contracts pass |
| Updated local example: `--check`, then real daemon boot | Pass; both listeners accept connections (temporary token/root/ports substituted) |
| Helm 0.1.17, default/custom/disabled MCP variants | All render; extracted TOML parses and passes `sokard --check` with local fixture credentials/storage |
| Operator TOML / shell snippets | Every TOML block parses; all shell blocks pass `sh -n`; Claude flags match installed CLI help |
| `git diff --check`, file-scope audit, branch check | Pass before implementation commit |

The four new MCP contracts cover initialize/instructions, exactly nine tools and
schema descriptions/defaults, all scope variants in the schema, collection
create-or-get, text and three-byte binary round trips, non-UTF-8 recovery guidance,
metadata/relative and absolute URLs, dedup/version chains, latest-path lookup,
filtered cursor pagination, tombstone/get errors, missing paths, missing/invalid
bearers before MCP parsing, custom/disabled routes, text/base64 upload limits,
requested read limits, server caps against larger REST uploads, per-principal
session isolation and token-file rotation. Existing 100 MiB HTTP streaming and
restart/reindex contracts remain green.

**Coverage limits:** the existing `postgres_drop_schema_reindex_fs_contract`
returns early because `SOKAR_TEST_DATABASE_URL` is unset, despite appearing as a
pass in Cargo's totals. Thus 37 test bodies actually exercised their contracts.
The MinIO contract remains explicitly ignored. No live Postgres, S3/SeaweedFS,
Codex/Claude registration, akh-session-services process, CI, or deployment proof
is claimed. The required MCP integration uses the real rmcp transport with fs
storage and an ephemeral index.

Local run logs: `/tmp/sokar-52a1-clippy.log`, `/tmp/sokar-52a1-tests.log`,
`/tmp/sokar-52a1-mcp-tests.log`. These are convenience evidence, not committed
artifacts; the commands and tests above reproduce the checks.

## Registration and Helm evidence

For all three supported consumers, the configured fleet endpoint is
`http://sokar.lab.home/mcp` and the bearer must match one Sokar principal.

### Codex

```toml
[mcp_servers.sokar]
url = "http://sokar.lab.home/mcp"
bearer_token_env_var = "SOKAR_MCP_TOKEN"
```

Key semantics verified against the
[official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

### Claude Code

```sh
claude mcp add --scope user --transport http sokar http://sokar.lab.home/mcp --header "Authorization: Bearer $SOKAR_MCP_TOKEN"
```

Verified against installed `claude mcp add --help`; registration was not executed.

### akh-session-services

```toml
[endpoints.sokar]
url = "http://sokar.lab.home/mcp"
token_file = "/etc/akh-session-services/sokar-token"
allow_insecure_http = true
```

The guide includes an explicit `[[akhs.thoth.tools]]` grant with
`destination = { kind = "mcp", service = "sokar", tool = "sokar_collection" }`.
Key names were read from `Endpoint`, `AkhProfile`, `ToolSpec` and
`ToolDestination::Mcp`; `Peer::request` uses the endpoint URL verbatim. This is a
source-verified configuration snippet, not a live broker integration test.
Direct akhomed registration is the documented deviation above; the guide shows
its actual existing upstream shape without claiming a Sokar slot exists.

### Rendered configuration

Reproduce with a minimal local rendering fixture (use real stores/principals
for deployment):

```sh
helm template sokar-check charts/akh-medu \
  --set sokard.enabled=true \
  --set sokard.database.enabled=false \
  --set 'sokard.stores[0].name=local' \
  --set 'sokard.stores[0].service=fs' \
  --set 'sokard.stores[0].options.root=/tmp/sokar-chart-objects' \
  --set sokard.config.publicApiUrl=http://sokar.lab.home \
  --show-only templates/sokard-configmap.yaml
```

The extracted `sokar.toml` contains:

```toml
[server]
api_bind = "0.0.0.0:9400"
admin_bind = "0.0.0.0:9401"
public_api_url = "http://sokar.lab.home"

[mcp]
enabled = true
path = "/mcp"
```

This agrees with all three snippets' host/path. Additional renders verified
`--set sokard.mcp.path=/agents/sokar` and
`--set sokard.mcp.enabled=false --set sokard.config.publicApiUrl=`. Each extracted
TOML was parsed, its fields asserted, and checked by the real daemon after
substituting a temporary token path and fs root. Helm's pre-existing local
`helm-secrets` plugin warning did not affect template exit status or output.

## Open question for the reviewer

Which later slice should add the typed `upstreams.sokar` configuration and engine
dispatch for direct akhomed consumers? The requested file boundary prevents that
consumer change here. Session-services MCP routing is available with explicit
tool grants; Sokar's nine-tool MCP surface is implemented and locally validated.

## Round 2

Authority: all R1–R8 in [the review](52a1-mcp-review.md). Work remains on
`sokar/52a1-mcp`, within the original file boundary, with no push.

### Required fixes

| Item | Change and evidence |
| --- | --- |
| R1 | `[mcp] session_idle_secs` defaults to 900; every principal's `LocalSessionManager` now sets `SessionConfig.keep_alive`. The real transport test uses a one-second idle TTL, verifies reuse returns 404, then initializes and lists tools successfully in a fresh session. |
| R2 | The MCP subtree is wrapped in `RequestBodyLimitLayer` with `max_inline_bytes * 4 / 3 + 64 KiB`, before mounting. A bounded body adapter collects before rmcp parses JSON so unknown-length/chunked overflow also returns 413 instead of rmcp's body-read 500. Tests send oversized invalid JSON both with and without Content-Length. The global HTTP limit still applies. |
| R3 | Both upload schemas expose optional `idempotency_key` with retry guidance and pass it into `UploadOptions`. For both text and base64, tests verify same-key replay returns the original artifact ID, a REST replay returns that same ID, and changed content returns `sokar::idempotency_mismatch`. |
| R4 | Authenticated responses beneath the configured MCP mount bypass `normalize_problem`; authentication/authorization failures retain Sokar problem+json. Tests preserve rmcp's 400, 404, 406, 415 and 422 responses, including the exact missing-session POST body, and check custom mount matching does not capture sibling paths. |
| R5 | Upload tools parse collection IDs and authorize before size checks, decoding or staging. Shared typed `authz::Resource` renders REST-handler and MCP-tool authorization resources, including percent-encoded by-path segments; list permission is Read. A recording/denying authorizer verifies parity for list/find/get/content, including Unicode and reserved path characters, and proves denied text/base64 inputs return authz before inline/decode errors. REST and MCP endpoint 403 envelopes remain problem+json. |
| R6 | `RoutePath` implements FromStr/Deserialize with typed `NotAbsolute`, `TrailingSlash`, `DotSegment`, `IllegalByte`, and `ReservedV1` reasons and per-variant help. The TOML diagnostic text also includes corrective guidance. MCP deserialization defers path validation when disabled; `Config::validate` rechecks it if enabled later. Tests cover every reason, defaults, unknown fields, disabled invalid configuration, re-enabling, and positive inline/idle bounds. |
| R7 | `ListParams.limit` is Option<u32>, passed unchanged through ListArtifacts to the existing index page helper; `list_limit()` is removed. A 101-artifact transport test compares MCP and REST pages at the shared default 100 and explicit limits, and verifies invalid-limit tool errors. |
| R8 | This section, the operator guide, and the local TOML example document the changes. Scoped fmt, rustc 1.98 clippy, the test suite and error-size gate pass. |

### Accepted behavior retained and source clarifications

- Tool IDs and paths deliberately remain raw String parameters and are parsed
  inside each tool. Malformed values therefore retain the accepted
  `{code,message,help}` tool-error envelope instead of becoming JSON-RPC
  invalid_params. Diagnostic codes remain identical to REST; the accepted tool
  envelope, principal/session isolation and service-layer upload path remain.
- rmcp **0.17.0** returns **422**, with the exact text
  `Unexpected message, expect initialize request`, for a non-initialize POST
  without a session ID. Its missing-session GET returns **400** with
  `Bad Request: Session ID is required`. Tests preserve these actual upstream
  responses; no response is rewritten to match the review's illustrative 400.
- The existing `index::page_limit()` defaults to 100 and **rejects** values
  outside 1–1000; it does not clamp. MCP now shares that behavior with REST,
  preserving the foundation's validation semantics.
- Canonical resource authorization now occurs in the REST handlers after typed
  extraction, using the same Resource variants as the MCP tools. Bearer
  authentication remains at the API boundary before body processing. MCP also
  retains its endpoint authorization before transport/body processing.
- The chart retains its accepted 0.1.17 settings; the new idle setting uses the
  daemon's default unless configured directly in TOML. The reviewer-owned
  architecture document and all other out-of-scope files are untouched.

### Validation and limits

All Rust gates use `rustc 1.98.0 (88d9e12ae 2026-08-18)` via `+1.98.0`.

| Gate | Result |
| --- | --- |
| `cargo +1.98.0 fmt -p sokar -p sokar-client --check` | Pass |
| `cargo +1.98.0 clippy --locked -p sokar -p sokar-client --all-targets -- -D warnings` | Pass |
| `cargo +1.98.0 test --locked -p sokar -p sokar-client` | 44 reported passed, 0 failed, 1 ignored |
| MCP transport contracts | 9 passed, including the 5 added Round 2 contracts |
| Typed config diagnostics and CLI checks | 6 passed, including the added per-reason test and local example boot |
| `SokarError` size / diagnostic-code parity | Pass; error remains below 128 bytes |
| `git diff --check`, branch and file-scope audit | Pass |

The existing Postgres test returns early when `SOKAR_TEST_DATABASE_URL` is unset,
so 43 test bodies exercised their contracts; the MinIO test remains explicitly
ignored. No live external database/object-store, CI, deployment or consumer
registration validation is claimed. Registration snippets and chart templates
were unchanged in Round 2; their earlier evidence remains recorded above.

Logs: `/tmp/sokar-52a1-round2-clippy.log` and
`/tmp/sokar-52a1-round2-tests.log` (local convenience evidence).

### Round 2 commits

- `7a6bed75` — `fix(sokar): address round two MCP review findings`:
  implementation, regression tests, local example and operator documentation.
- This report is committed separately as
  `docs(sokar): record round two MCP fixes and validation`.

Both commits end with the required
`Co-Authored-By: Codex gpt-6-astra <noreply@openai.com>` trailer. No push.
