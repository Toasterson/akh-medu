# Private session MCP capabilities

`akhsessd` can require a distinct MCP capability for each session. This supports
Anima's session-bound result tools without copying its token into automation
events, runtime configurations, model prompts or the public session journal.

In the session daemon configuration, list the configured MCP endpoint aliases
under `session_mcp_endpoints`. For example:

```toml
session_mcp_endpoints = ["anima_results"]

[endpoints.anima_results]
url = "https://anima.example/mcp"
token_file = "/private/unused-for-session-scoped-mcp"
```

The alias must already exist in `endpoints` and have a fixed URL. The endpoint's
token file remains part of the shared configuration format but is never read for
session-scoped MCP calls. An Akh profile must explicitly grant tools whose MCP
destination uses that alias. The daemon pins the endpoint configuration digest
in each new session. Changing its policy or destination refuses new dispatch and
tool access; it never falls back to a shared token. Existing runtime outcomes can
still be reconciled and cancelled.

The session owner, with the additional `session_credentials` role, posts to
`/v1/sessions/{session}/mcp-credentials/{service}` with `request_id` and `token`.
Administrators may also provision. Use authenticated HTTPS outside loopback.
The request must retain the same identity and token across retries. No user or
event payload can create a credential slot or choose the endpoint URL. Controller
leases govern conversational input; this separate capability role governs private
provisioning. Ordinary client and runtime principals cannot provision.

The response is a versioned receipt containing the session, service, endpoint
digest, request ID, token digest, principal and original timestamp. The journal
stores that receipt only. Private files reside under the daemon's data directory
in `session-mcp-credentials`, with directory mode 0700 and file mode 0600. Reads
reject symlinks, nonregular files, wrong ownership, permissive modes and corrupt
contents. Complete synced files are published without overwriting another intent.
A crash before the journal commit retains the original intent in the private
file; exact retry completes its acknowledgement. Secret rotation requires a new
session or a future explicit rotation contract.

Required credentials gate the first dispatch and later queued turns. Local
credential resolution precedes marking an MCP operation sent. If the file is
missing, exact reprovisioning can restore it; no MCP effect is inferred. Once an
MCP exchange starts, the existing unknown-outcome rule still prohibits automatic
tool replay. The token remains inside the broker throughout the MCP exchange.

Archive reconciliation removes the session's private capability files and any
interrupted temporary writes. Historical provisioning retries return the original
receipt without restoring credentials to a retiring or archived session. Anima
must separately revoke its upstream token through its canonical token authority;
local file removal alone is not upstream revocation or proof of sandbox release.

Local tests cover dispatch gating, owner/role enforcement, crash and restart
recovery, immutable intent, endpoint drift, broker authentication, absence from
journal/runtime payloads, archive cleanup and lost/mismatched HTTP replies.
Anima's independent client also exercises the real locally built daemon. These
checks do not enable Anima scheduler selection or mint a production capability.

An automation caller can include `bootstrap_requirements` with an explicit
`repository` and exact `mcp_credentials` alias list. Akh validates those gates
inside the admission transaction and requires a fresh per-occurrence target.
An incompatible profile, shared target or notification-only automation leaves
no new session or turn behind. The immutable occurrence retains these requirements
on retry. Existing callers may omit them; Anima's new preparation path requires
them. This prevents a misconfigured automation from dispatching before the
required repository and capability can be provisioned.

Anima migration 137 now revokes canonical session-bound tokens in the transaction
that closes a central owner. Its relay persists only token identity and exact
provisioning receipts, recovers the original live capability, and refuses changed
routing bindings. Local database/native-daemon tests cover the lost-response and
restart boundary. Scheduler selection and result ingestion remain separate work.
