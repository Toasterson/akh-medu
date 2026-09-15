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
