# Sokar operations and MCP registration

Sokar stores content-addressed artifact bytes and UUID-keyed manifests in an
OpenDAL store. PostgreSQL is a rebuildable metadata index. See
[ADR 108](../ai/decisions/108-sokar-artifact-daemon.md) and
[52a.1 brief](../ai/phases/phase-52-sokar/52a1-mcp-brief.md) (Anima WI 52a.1).

## Run locally

From the repository root, create a private token file and filesystem store:

```sh
install -d -m 700 /tmp/sokar-local
(umask 077; openssl rand -hex 32 > /tmp/sokar-local/token)
export SOKAR_FS_ROOT=/tmp/sokar-local/objects
cargo run --locked -p sokar --bin sokard -- --config crates/sokar/examples/local.toml --check
cargo run --locked -p sokar --bin sokard -- --config crates/sokar/examples/local.toml
```

The example omits `[database]`: its ephemeral index is rebuilt from manifests at
startup and supports a **single daemon per store prefix**. Idempotency receipts
are lost on restart. Configure `[database] url = "env:SOKAR_DATABASE_URL"` for
PostgreSQL. `sokard --migrate`, `--check-stores` and `reindex --store local` retain
their existing behavior. Secret values use `env:NAME` or `file:/absolute/path`;
each principal token file contains one token of at least 24 characters with no
whitespace. Tokens reload on every request; matching must identify exactly one
principal.

Two listeners:

| Listener | Default | Routes / authentication |
| --- | --- | --- |
| API | `127.0.0.1:9400` | `/v1/*`, `/mcp`; configured principal bearer token required |
| Admin | `127.0.0.1:9401` | `/healthz`, `/readyz`, `/metrics`; keep this unauthenticated listener private |

```toml
[server]
api_bind = "127.0.0.1:9400"
admin_bind = "127.0.0.1:9401"
public_api_url = "http://sokar.lab.home" # optional; may include a proxy prefix

[mcp]
enabled = true
path = "/mcp"
max_inline_bytes = 8388608
session_idle_secs = 900
```

Omitting `public_api_url` returns relative `/v1/artifacts/{id}/content` URLs.
The configured base is used for absolute URLs; it must use HTTP(S) without
credentials, query or fragment. Every content URL requires the same bearer token.
Custom MCP paths must be literal absolute paths outside `/v1`, with no trailing
slash. Invalid paths report a specific reason and corrective help. Path and MCP
limit validation applies only when MCP is enabled. Disabling MCP leaves the REST API available. Unknown config fields fail
validation.

## Tools

The streamable-HTTP endpoint publishes nine tools:

| Tool | Purpose |
| --- | --- |
| `sokar_collection` | Create or get a collection by typed scope |
| `sokar_put_text` | Upload UTF-8 text; default kind `document`, MIME `text/plain; charset=utf-8` |
| `sokar_put_base64` | Upload standard padded base64; required MIME, default kind `other` |
| `sokar_list` | List live versions; optional prefix/kind/cursor, default page size 100 (valid range 1–1000, shared with REST) |
| `sokar_get` | Return `{artifact, content_url}` for a version |
| `sokar_find` | Find the latest live version at a collection path |
| `sokar_read_text` | Return full UTF-8 text; default maximum 1 MiB |
| `sokar_read_base64` | Return `{base64, media_type, size}`; default maximum 8 MiB |
| `sokar_delete` | Tombstone a version; return `{tombstoned: true}` |

Tool results contain JSON text (including a JSON string for `sokar_read_text`).
Upload results contain `{artifact, outcome}`. Both upload tools accept an optional
`idempotency_key`: reuse it on retry to avoid a duplicate version. The same key
with identical content and metadata returns the original artifact ID; changed
content or metadata returns `sokar::idempotency_mismatch`. This uses the same
semantics and key storage as the REST `Idempotency-Key` header. Errors use `isError: true` and
`{code, message, help}`. Invalid MCP arguments/protocol messages follow rmcp's
protocol errors; MCP transport errors preserve rmcp's HTTP status and body.
Authentication/authorization failures and REST HTTP rejections use `problem+json`.

Inline limits count decoded bytes, including UTF-8 bytes for text. Reads use the
smaller of `max_bytes` and the server cap, reject oversized content, and never
truncate it. Use the REST upload/content endpoints for larger artifacts.
The MCP subtree also limits the entire HTTP request body to
`max_inline_bytes * 4 / 3 + 64 KiB` (11,250,346 bytes with defaults), allowing
base64 expansion plus protocol/metadata overhead. The global `limits.max_body_bytes`
cap still applies. Oversized requests are rejected with HTTP 413 before JSON
parsing; use REST for larger uploads. JSON escaping and large metadata count
toward this request limit.
Non-UTF-8 text reads direct callers to `sokar_read_base64`.

Paths are relative NFC paths, unique for latest-wins lookup within a collection.
Every upload creates an immutable version; identical bytes deduplicate. Listing
includes live historical versions, and deleting the newest version exposes the
previous live version in path lookup. Deletion retains tombstone manifests and
blob bytes for later reconciliation/GC. MCP sessions are isolated per principal;
another principal cannot call, resume, or delete your session. Idle sessions
expire after `[mcp] session_idle_secs` (default 900 seconds); reuse returns HTTP
404 and the client must initialize a fresh session. Both `session_idle_secs`
and `max_inline_bytes` must be positive when MCP is enabled.

## Register consumers

Use `http://127.0.0.1:9400/mcp` for the local example. The following snippets use
the private fleet endpoint `http://sokar.lab.home/mcp`; supply its fleet token as
`SOKAR_MCP_TOKEN` in the launching client's environment. Public deployments should
use their HTTPS URL.

### 1. Codex

In `~/.codex/config.toml`:

```toml
[mcp_servers.sokar]
url = "http://sokar.lab.home/mcp"
bearer_token_env_var = "SOKAR_MCP_TOKEN"
```

The environment variable holds the raw token, without `Bearer `.
See the [official Codex MCP configuration documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

### 2. Claude Code

```sh
claude mcp add --scope user --transport http sokar http://sokar.lab.home/mcp --header "Authorization: Bearer $SOKAR_MCP_TOKEN"
```

This command registers the expanded header value in Claude Code's user config;
keep that config private and update the registration after token rotation.

### 3. akh consumers

#### akh-session-services

Add this endpoint to the existing service config; its MCP client uses the full
URL verbatim (it does **not** append `/mcp`):

```toml
[endpoints.sokar]
url = "http://sokar.lab.home/mcp"
token_file = "/etc/akh-session-services/sokar-token"
allow_insecure_http = true # explicit permission for the private non-loopback HTTP endpoint
```

Grant tools explicitly in the relevant existing `[akhs.<name>]` profile. For
example, append this entry for collection creation:

```toml
[[akhs.thoth.tools]]
name = "sokar_collection"
description = "Create or get an artifact collection by scope"
read_only = false
input_schema = { type = "object", required = ["scope"], properties = { scope = { type = "object", required = ["type"], properties = { type = { type = "string" } } } } }
destination = { kind = "mcp", service = "sokar", tool = "sokar_collection" }
```

For the full tool set, use the exact `tools/list` input schemas and add one grant
per tool. Mark the five metadata/content lookup tools (`list`, `get`, `find`,
`read_text`, `read_base64`) read-only. The token file must contain a raw fleet
token matching a Sokar principal. If `sokar` is included in the top-level
`session_mcp_endpoints` array, the broker requires provisioned per-session
credentials and never falls back to this daemon token file; configure that mode
only when those credentials have been provisioned and are accepted by Sokar.

Verified key sources: `crates/akh-session-services/src/config.rs` (`Endpoint`,
`AkhProfile`, `Config`), `src/mcp.rs` (`Peer::request`), and
`crates/akh-session-types/src/lib.rs` (`ToolSpec`, `ToolDestination::Mcp`).

#### akhomed: direct Sokar upstream registration needs a later slice

`src/config.rs::UpstreamsSection` currently contains **only** `anima` and `seshat`.
For comparison, the existing supported upstream shape is:

```toml
[upstreams.seshat]
url = "http://seshat.lab.home" # that client appends /mcp
auth_header = "env:SESHAT_MCP_TOKEN"
```

There is no functioning `[upstreams.sokar]` registration in this tree. An unknown
entry is ignored by that existing config parser; adding it would falsely appear
to work. Do not repoint Anima/Seshat at Sokar. Use the session-services route above
for akh sessions. Adding a typed Sokar upstream plus its engine dispatch is an
out-of-scope follow-up, recorded in the slice report.

## Helm 0.1.17

The API HTTPRoute already forwards `/`; it needs no change. Configure:

```yaml
sokard:
  enabled: true
  mcp:
    enabled: true
    path: /mcp
  config:
    publicApiUrl: http://sokar.lab.home
```

Keep the existing store, principal and database settings for your deployment.
These values render `[mcp] enabled/path` and `[server] public_api_url` in the
`sokard-config` ConfigMap's `sokar.toml`. Omitting `publicApiUrl` keeps content URLs
relative. The inline limit retains the daemon's 8 MiB default.
