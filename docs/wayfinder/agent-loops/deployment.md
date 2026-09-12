# Deploying the session platform

The topology separates the cognitive core, central sessions, runtime workers,
sandbox allocation, workspace execution and event ingress. Runtime machines can
be different from the workspace hosts. Each service owns its own state directory;
none opens another daemon's database.

## Artifacts and placement

`docker/session-platform.Dockerfile` builds one selected daemon/helper profile.
For example, build with `SERVICE=akhsessd PROFILE=session`, `SERVICE=akhrtd
PROFILE=runtime`, or `SERVICE=akhexecd PROFILE=executor`. The executor image also
contains `akh-service`; the runtime contains its profile helper. Build each target
architecture natively through buildx. This Dockerfile does not install provider
binaries, login credentials or editors. Add the selected pinned upstream tools
in the corresponding image pipeline. The core uses the existing akhomed build
recipe with `FEATURES=core-server` and `--no-default-features`.

`docker/session-workspace.Dockerfile` copies the executor and service helper into
a selected unmodified upstream workspace image. The images must match CPU and
libc ABI. The provisioner supplies the UID, read-only config, credentials, writable
state/workspace mounts and the actual sandbox isolation. A writable directory on
the host is only the `existing` development backend, not an isolation boundary.

The `charts/akh-session-platform` chart accepts explicit session/runtime/sandbox/
event service entries, digest-pinned images and independently mounted credentials.
It uses one replica, `Recreate`, retained PVCs, non-root processes and a bounded
shutdown interval. Redb is a single-owner store: do not add an HPA or overlapping
replicas. Executor/editor pods are created by the sandbox provisioner, not this
central-services Deployment chart. Runtime images/private account state can be
placed using per-service node selectors. Existing ingress or the service mesh
terminates TLS; backend ports remain private. ACP clients connect through WSS.
`akh-acp --ca-file FILE` supports a private work/homelab CA without disabling
certificate verification.

For Archibald, `contrib/systemd/akh-platform@.service` runs a selected daemon as
`akh`. On the Mac mini, `contrib/launchd/it.akh.akhrtd.plist` is a per-user example;
create its config/log directories and use the intended user's launchd domain.
These files are templates, not evidence that a host service was installed.
Run `akhrtd --check --config FILE` and `akh-runtime-profile` with the same private
profile before enabling a worker. Provider versions, logins and conformance
artifacts belong to that runtime profile, not to the editor or session manager.

## Credentials and state

Each endpoint references a bearer token file and optional private CA file. Use
distinct capabilities for each caller/boundary, constrained by role, Akh and
source. A receiving principal can set `token_sha256: true` and mount only the
lowercase SHA-256 verifier in its `token_file`. This is especially useful inside
executor sandboxes: reading the verifier does not yield a credential that can
impersonate the central session manager. The caller retains the original random
high-entropy token. Do not place reusable central client credentials in an
editor's workspace. Runtime login files, session journals and helper state have
separate ownership and lifetimes.

Service helpers use private HOME/XDG directories, revision-check managed files,
keep upstream environment identity across restart, and can verify a version
command before launch. Their advertised endpoint never contains a pairing secret.
A health endpoint serving before launch is rejected. `/healthz` reports that the
service is listening; capacity, provider readiness, unresolved effects and actual
turn completion come from their respective job/allocation/receipt records.

The dispatcher owns independent bounded session tasks. Slow cognitive/runtime/
sandbox endpoints do not serialize unrelated sessions. Tool/recovery admission
locks one call identity at a time. Shutdown drops the dispatcher's owned task set;
provider/executor owners retain responsibility for effects already sent.

## Kubernetes Sandbox API

The provisioner accepts an administrator-owned namespaced CRD collection path
and manifest. Select the installed API version explicitly. The current upstream
[Sandbox v1beta1 API](https://github.com/kubernetes-sigs/agent-sandbox/blob/v1.0.0/api/v1beta1/sandbox_types.go)
reports readiness and observed generation on the Ready condition. The adapter
requires a current Ready condition, pins the observed object UID and uses a UID
precondition on deletion. An old Ready condition cannot authorize a new spec.
Workspace binding still requires the executor handshake and expiring lease.

A template must provide its executor Service/endpoint, mounted credentials and
workspace path `/workspaces/<allocation-id>`. Use the command provisioner for
sites whose supervisor additionally creates per-allocation credentials, routes,
VMs or host sandboxes. An uncertain command is reconciled from observed resources,
never automatically executed a second time. The existing fixtures validate API
and UID behavior, not a running Sandbox controller or kernel confinement.

## Released kagent BYO

The chart's optional `kagent` section creates a `kagent.dev/v1alpha2` BYO Agent
using the stateless `akh-kagent` binary on port 8080. Its JSON config binds one
Akh and one central integration principal. The facade fetches and rewrites the
central agent card, then streams the A2A JSON-RPC body without changing task,
context or message IDs. It never forwards caller identity/Authorization headers,
never stores sessions and never retries a lost POST.

The BYO hop uses the kagent trusted cluster network. `trust_kagent_network: true`
is explicit and the chart requires a non-empty same-namespace caller label
selector in an ingress NetworkPolicy. Run it only with a CNI enforcing that
policy. This is one shared integration identity, not per-user authorization.
Users requiring separate Akh ownership use separately scoped facades or the
centrally authenticated ACP/A2A endpoint. Do not expose this facade publicly.
The credential Secret contains `token` and `ca.pem` for central HTTPS.

The target is [kagent v0.10.1](https://github.com/kagent-dev/kagent/releases/tag/v0.10.1),
not its unreleased native harness. The upstream BYO translator can advertise both
A2A 0.3 and 1.0; this facade supports **0.3 only**, and clients must negotiate that
version. Discovery, typed message requests, idempotency, polling, cancellation and
SSE pass against the actual A2A Python SDK 0.3.22. The test does not run a kagent
controller or prove its 1.0 negotiation behavior. See
`scripts/session-platform/a2a-sdk-smoke.py` and the Rust A2A/facade tests.

## Schedulers, upgrades and recovery

`examples/session-platform/cronjob.yaml` provides a source-scoped trigger, Job
metadata RBAC and Downward API environment. Replace its example images/endpoints
and mount the selected capability. Cron owns the clock; `akhsessd` owns the run
ledger and dedupe. See [event contracts](events-and-mcp.md).

Drain active work before changing a provider profile or replacing a runtime.
Keep installed executables, bridge/SDK versions and conformance hashes aligned.
An unknown runtime, tool or allocation outcome is an operational fence, not a
reason to mint a fresh request ID. Use the [recovery API](recovery.md), retain its
evidence, and reconcile leases and editor owners before replacement. Archive
completed sessions to revoke their workspace access and stop retained helpers.
Archival admission is not cleanup completion. Poll `/v1/sessions/{id}/lifecycle`
and retain the external dispatch fence until `workspace_released` is true.

Local evidence: isolated feature builds, service/provider tests, six-process
smoke, actual A2A SDK requests, private-CA WSS initialization/reconnect, Helm
lint/render and launchd plist validation. The sandbox-daemon Linux arm64 image
builds and starts locally; the other images are not yet built. Run
`scripts/session-platform/t3-native-smoke.py --t3 /absolute/path/to/t3` for the
installed upstream 0.0.40 bootstrap/restart/pairing-link check. This checks an
editor-only profile and does not perform a model turn or browser registration.
`akh-service --manifest FILE --action pair` runs the same private profile's
bounded native pairing command; treat its stdout as a secret.
Image publication, chart installation, cross-host access, real Sandbox/RabbitMQ
operation and native-provider remote-tool certification are separate gates.
