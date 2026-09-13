# Selectable editor sandbox images and Anima routing simplification

Date: 2026-09-11. Status: design clarification and source trace only.
Parent: [deployment plan](k8s-acp-t3-deployment.md).

## Accepted direction

The user prefers launching different sandbox images containing the remote
tool/server they want, co-located with the Akh executor. Sessions remain
centrally managed. The alternatives include a T3 Code server, VS Code-compatible
remote server and Zed headless remote server. Anima should simplify by consuming
the central session/execution platform and gradually retiring its own router.

This changes the earlier T3 proposal: a central T3 server with distributed
filesystem/Git/terminal adapters is no longer the preferred deployment.
The sandbox-local editor server uses its normal filesystem and toolchain.
`akhrtd` may still run the agent loop on another machine, as previously requested.

The latest clarification requires unmodified upstream external services. Our
helpers own launch, supported configuration/login flows and connection automation.
Separate adapters use supported interfaces; the earlier proposed T3 source
changes are superseded. [Service profiles and tool bindings](upstream-services-tool-bindings.md)
defines the shared model and runtime-specific native-tool restrictions.

## Image/profile contract

Use an authorized catalog of `WorkspaceProfile` revisions. Example profile
names are illustrative, not images that already exist:

| Profile | Contents and endpoint |
| --- | --- |
| `rust-t3` | `akhexecd`, unmodified pinned upstream T3, Akh launcher/config helper, Git and Rust tools; T3 HTTP/WSS endpoint |
| `rust-openvscode` | `akhexecd`, chosen browser-editor distribution, extensions and Rust tools; authenticated editor HTTP/WSS endpoint |
| `rust-zed` | `akhexecd`, SSH service, compatible Zed remote-server installation/cache and Rust tools; SSH project endpoint |
| `headless` | `akhexecd` and selected toolchain; agent execution without an editor server |

Profiles pin image digest(s), platform/architecture, toolchain, service
entrypoints and health checks, workspace mount, process identity/permissions,
resource quota, network policy, persistence and supported client versions.
Services may share an image or run as coordinated containers in the same
sandbox. Neither arrangement requires privileged access to the host's container
daemon. A native macOS workspace needs a host/VM template implementation;
Linux OCI images can run inside a Linux VM on the Mac mini.

The catalog identifies software and capabilities. It contains no provider
login, long-lived user token or connection secret. Runtime identity and image
choice do not grant access to an arbitrary repository or central session.

Zed's [remote protocol](https://zed.dev/docs/remote-development) is SSH with a
client-version-matched headless server. A profile can cache versions and admit
only supported clients, or materialize the matching server through the supported
workflow. A single baked binary cannot promise compatibility with every desktop.
This is separate from Zed's ACP agent integration.

For a browser VS Code-like image,
[OpenVSCode Server](https://github.com/gitpod-io/openvscode-server) is one
candidate. Microsoft [VS Code Server](https://code.visualstudio.com/docs/remote/vscode-server)
has a distinct distribution and supported use/connection model. Pick the
intended distribution explicitly before publishing an image. An enabled
editor endpoint must enforce the allocated user's access policy.

## Launch and attachment flow

1. Anima or another client requests a central session, selecting an authorized
   Akh and workspace profile, repository/base revision, retention and placement
   constraints. Alternatively it asks to attach an editor to an existing session.
2. `akhsessd` durably accepts the request and resolves its policy. Repeated
   requests with the same operation ID return the same admission/allocation.
3. `akhsbxd` acquires the selected sandbox/profile and the workspace volume or
   worktree. It registers the sandbox lease with its generation and owner.
4. `akhexecd` materializes/verifies the workspace; editor servers use the same
   root and compatible permissions. Avoid separate clones for T3 and the agent.
5. Readiness requires the executor handshake plus the selected editor's native
   endpoint and version checks. Return a `WorkspaceAccessDescriptor` listing
   workspace/session IDs, endpoint kind, supported client, display metadata and
   auth-exchange mechanism. Keep secrets out of catalog records and ordinary URLs.
6. A small per-user process on the connecting machine consumes the ready
   descriptor and adds the bootstrapped environment through the upstream pairing
   flow. It reconciles duplicate events and changed endpoints, retaining explicit
   offered/paired/connected states. See the
   [machine-local connection helper](upstream-services-tool-bindings.md#machine-local-environment-connection-helper).
   Where upstream
   supports central agent attachment, a separate adapter attaches over ACP/WSS
   or the protocol actually accepted by its configured command/endpoint. Record
   workspace access and central chat attachment as separate capabilities; stock
   T3's latter path still requires verification. Creating an attached T3 thread
   must not create a second session when launch already supplied a binding.
7. `akhsessd` admits turns and selects an eligible runtime through `akhrtd`.
   Agent workspace tools reach the bound `akhexecd`; editor operations execute
   locally in the same sandbox. Runtime placement remains independent.

The descriptor is metadata plus a scoped access flow, not an arbitrary URL that
becomes trusted on receipt. A standard ingress, SSH gateway or private network
can carry editor traffic. This connectivity layer does not need to interpret
agent turns or live inside Anima.

Closing the editor detaches a client. Stopping a turn, releasing compute,
retaining a workspace and deleting user data are separate operations. Image
replacement preserves central session identity and retained application data,
or runs an explicit rehydration/migration. Editor switching initially uses a
controlled handover; simultaneous writable editor servers are a separate policy.

## Upstream services and our integration boundary

Keep upstream file browsing, Git, worktrees, checkpoints and terminal execution
local to the sandbox. Our helpers manage configuration, process lifecycle,
auth/endpoint handling and connection automation. A separate supported-interface
bridge must supply central-session binding/discovery, history projection,
approval/question/cancel mapping and restart reattachment. T3's internal driver
types do not establish a public extension API. Validate its supported surfaces
before claiming that bridge exists; do not patch its server to fill a gap.
A central launcher lists sandboxes and provides T3 environment connection
information without becoming a second T3 server. Test actual onboarding;
endpoint metadata alone is not a finished UX.

T3 retains its local database/environment ID and application records as needed.
`akhsessd` remains the canonical agent-session owner. Replaying central events
must not replay user commands. Chat parity in another editor requires its own
adapter; simply opening the same worktree does not transfer its conversation UI.

Co-location guarantees a shared execution location, not exclusive use of
`akhexecd`. Editor saves, Git operations and native terminals are legitimate
separate writers. Serialize destructive handovers/checkpoints against active
agent work, retain file-version/conflict checks, and attribute their activity
separately. Provider-native operations on a different runtime machine remain
the existing remote-execution conformance problem.

## Anima source trace and proposed simplification

Inspected checkout: `/Users/toasty/ws/anima`, 2026-09-11. Historical design ADRs
describe intent; the following files establish the current code seams. This
read-only inspection did not query Anima's database or mutate tracker state.

| Current Anima responsibility | Source | Target |
| --- | --- | --- |
| Publish lane work and choose AMQP versus gRPC fallback | `crates/anima-server/src/execution_fabric.rs` | One central session admission call with durable idempotency; no Anima choice of physical runner transport for migrated sessions |
| Fleet/executor capacity, reservations, runner mapping and addressed delivery | `crates/anima-server/src/services/runner.rs` (`RunnerRegistry`, `send_to_runner`) | Runtime/sandbox services own placement and leases; Anima consumes admission and capacity/status summaries |
| Thread-to-session creation and prompt dispatch | `crates/anima-server/src/services/bridge.rs` | Small `akhsessd` client plus durable Anima-thread/central-session mapping |
| Fleet-global protection against duplicate live execution | `crates/anima-server/src/services/live_session.rs` | Preserve the invariant through central admission and authoritative ownership; retire the runner-specific registry only for migrated paths |
| Lane consumption and per-session broker bindings | `crates/anima-runner/src/execution_fabric.rs` | Central services own runtime delivery; runner broker translation can leave the migrated lane |
| Provider subprocess/ACP transport and remote Akh gateway discovery | `crates/anima-runner/src/acp.rs`, `crates/anima-runner/src/attach.rs` | `akhrtd` owns provider workers; `akhsessd` supplies discovery/session endpoints |
| Session status and transcript-facing APIs | `crates/anima-server/src/services/session.rs`, `proto/anima/v1/session.proto` | Compatibility API/projection backed by central events as needed, with Anima-specific audit/work-item associations |

Anima keeps work-item dependency scheduling, project policy, approved briefs,
phase/attempt intent, base-revision identity, scoped Anima tools, review gates,
Change/PR association and final result verification. It may keep project-level
concurrency budgets; physical runtime/executor capacity has one central owner.

Conceptually its integration becomes `StartOrAttach`, `SubmitInput`,
`RespondToInput`, `Cancel`, `GetStatus` and `WatchEvents`, plus requesting an
editor attachment. These are proposed service operations, not existing Anima
API names. Anima can render launch links or a lightweight session projection;
it need not proxy every file, terminal, ACP frame or provider heartbeat.

Removing Anima's execution router means moving responsibility to a single
shared owner. Standard networking and the central session-to-runtime map still
exist. It does not mean removing Anima's web router, its work-item scheduler,
or deleting all RabbitMQ infrastructure: `akheventd` and other consumers remain.

## Cutover without duplicate execution

Current dispatch code distinguishes confirmed AMQP delivery, definite
non-delivery and an indeterminate publish; the last state forbids dispatching
again through gRPC. Preserve that principle across the new boundary. A timeout
from central admission is not evidence that no session was created: query by
the stable operation ID before considering another attempt or backend.

Keep work-item/phase/base-revision exclusivity distinct from delivery dedupe
and authorized attempt numbers. Anima's business intent remains durable; the
central owner enforces the live execution claim. Trace every central event back
to that intent before changing work-item state or accepting a Change/review.

Migrate one selected lane/project through a compatibility adapter. Record the
execution backend on each attempt/session; continuations and cancellation use
that same backend. Existing legacy sessions drain through their current owner.
Never dual-publish a real task to compare old and new paths. Read-only event
projection comparisons are possible without running another agent.

Retire routing modules only after no active session or enabled consumer depends
on them. Rollback of new admissions does not transfer ownership of already
running central sessions. An Anima restart should reconnect projections and
recover its command outbox while central sessions continue.

Acceptance requires one session under retries, cancellation with known/unknown
outcomes correctly represented, exact-revision review/result gates, a verified
editor/executor shared workspace, retained state after image replacement,
cross-user isolation, and no duplicate work during rollout or rollback.

## Delivery order

1. Central session contract and one sandbox profile containing T3 + `akhexecd`.
2. Verify upstream T3 attachment through supported configuration/extension
   interfaces, with a separate bridge if feasible. Pass attach/resume against
   the central session; keep native local workspace features. Report a missing
   upstream capability explicitly if this cannot be completed without patches.
3. Anima compatibility client, launch links and central event projection for
   one migrated lane; keep existing review/result semantics.
4. Zed and selected VS Code-compatible profiles, plus Mac mini placement.
5. Independent runtime placement conformance and provisioning/recovery tests.
6. Retire migrated Anima routing/gateway responsibilities after drain and
   verification. Keep kagent/A2A and event ingress on their shared-service seams.

This document changes the plan only. It does not build/publish images, modify
Anima code, launch sandboxes, migrate sessions or remove a running router.
