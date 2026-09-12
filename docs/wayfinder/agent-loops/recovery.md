# Session platform recovery

An uncertain outcome stays `unknown` until its owner receives an explicit
resolution. Restarting a daemon never repeats a possibly effective operation.
Runtime hosts with unknown jobs refuse new work; executors fence mutations in
the affected workspace. Read the actual provider state, executor receipts and
process inventory before resolving an outcome.

Recovery endpoints require an `admin` principal. They record the authenticated
principal, exact request digest, evidence references, resolution and timestamp
in the owning daemon's database. A repeated request ID must have identical
content. `quiescent: true` is the operator's assertion that previous owners and
orphan processes cannot continue effects; it is not a daemon-generated proof.
Live local workers and helper locks independently prevent resolution.

| Owner | Endpoint | Resolution |
| --- | --- | --- |
| `akhrtd` | `POST /v1/jobs/{id}/reconcile` | Terminal status; completed jobs require the inspected output. Provider continuation identity stays unchanged. |
| `akhexecd` | `POST /v1/operations/{id}/reconcile` | Inspected `ToolResult`; success must agree with terminal status. The executor assigns the receipt ID. |
| `akhsessd` | `POST /v1/runtime/tools/reconcile` | `{call, resolution}` for an uncertain external MCP, cognitive or client result. Workspace effects must be resolved at the executor. |
| `akhsbxd` | `POST /v1/allocations/{session}/reconcile` | `{resolution, observed_binding}` for a command provisioner. An observed acquired binding still passes executor attachment/readiness. An observed release has no binding. |
| `akhexecd` | `POST /v1/services/{id}/reconcile` | Certify that a stopped/failed helper and its upstream process are quiescent. This does not launch a replacement. |

The common resolution shape is:

```json
{
  "request_id": "operator-resolution-unique-id",
  "expected_request_sha256": "digest-of-the-original-request",
  "status": "failed",
  "quiescent": true,
  "evidence": ["operator-owned-reference-to-the-inspection"],
  "output": null,
  "result": null
}
```

Fingerprint the original job/operation/allocation request, tool call or helper
manifest, not the enclosing status response. `akh-fingerprint request.json`
prints the digest without printing the request. It hashes compact serde JSON
with recursively sorted object keys; array order is significant. Build it with
`cargo build -p akh-session-services --no-default-features --features admin --bin
akh-fingerprint`. Store evidence references rather than credentials in records.

Resolve executor/external tool effects first, then the runtime job. The session
broker checks executor receipts again before accepting the terminal turn. It
also verifies the exact original dispatch and previously accepted runtime-event
history. Internal-model continuations interrupted mid-turn can be retired with
a failed/cancelled resolution; create a new explicit session after that. The
runtime never invents missing tool results or silently replaces that history.

## Helpers and config revisions

`POST /v1/services/{id}/stop` asks the current helper to stop its upstream
process. Wait for `stopped`. An expired heartbeat requires inspection and
explicit reconciliation. Then an admin can request:

```json
{
  "request_id": "unique-restart-id",
  "expected_manifest_sha256": "digest-of-the-current-helper-manifest",
  "profile": null
}
```

Send it to `POST /v1/services/{id}/restart`. A null profile keeps the current
configuration; a supplied profile may change its revision, command and managed
files after draining. The service ID and client kind remain fixed. The existing
private HOME/config and upstream environment identity are retained. Managed
files edited outside the helper cause a conflict. A durable restart intent
precedes launch, so a lost response does not cause a second process on retry.

## Leases and archival

`akhsessd` renews open sessions' executor leases through `akhsbxd` every minute.
Leases last five minutes. The executor checks expiry before admitting effects;
its watchdog cancels processes and stops upstream helpers after expiry. Expired
leases cannot be silently renewed or acquired again. Inspect and archive the
old session, then create a new allocation explicitly. Poll/cancel remain
available for settling an expired workspace's process receipts.

`POST /v1/sessions/{id}/archive` preserves the transcript and refuses new input.
Outstanding or uncertain turns must be cancelled/resolved first. The session
manager stops renewing and reconciles workspace release. Release stops upstream
services and revokes executor access even when the profile retains its sandbox.
The existing-workspace provisioner always retains files for separate, deliberate
cleanup. A Kubernetes deletion uses the originally observed resource UID as a
precondition. Command release failures require explicit outcome reconciliation.

SIGTERM stops new HTTP admission, starts runtime/executor cancellation, and
bounds HTTP draining to ten seconds. Any work without a durable terminal result
is fenced on restart. Bearer credentials are reread from mounted files. An
endpoint may specify an absolute `ca_file`, trusted only by that endpoint's
HTTP client, for a private Kubernetes or service CA.

## Internal-loop interruption and connection handoff

An internal-loop error remains unknown even when cancellation was requested.
The worker may have persisted a tool plan without obtaining its receipt;
requesting cancellation does not settle that gap. Reconcile the tool effects
first, then the runtime job. Resolving an unfinished internal job as failed or
cancelled retires its private continuation; archive that session and create an
explicit new one. There is no automatic replay or invented tool response.

`akh-connect --reconnect` requests a fresh upstream pairing link for the same T3
environment ID. An offer or successful browser-command exit is not proof of
pairing. Failed handoffs persist an unavailable offer and require explicit
reconnect; pairing secrets are never journaled. HTTP responses and launcher
execution are bounded. Service profiles cannot override their private HOME/XDG
roots. Optional `version_command` plus `version_output_contains` verifies the
installed upstream version before launch. A health endpoint already serving
before launch is rejected, preventing an unrelated existing server from making
this helper ready. This check does not replace native server identity validation.
