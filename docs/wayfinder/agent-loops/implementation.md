# Agent session platform implementation worktree

Prepared: 2026-09-11. Updated: 2026-09-12. Status: provider port and the first
authenticated five-process session/execution slice implemented and locally tested.
Full platform implementation and native/client/deployment validation remain in progress.

## Workspace boundary

Implement in `/Users/toasty/ws/akh-medu-agent-platform`, branch
`feat/agent-session-platform`. Base is freshly fetched `origin/main` at
`a5d1d4ad28260ba78b8aec3edf9da13919f6ab9a`.

The original `/Users/toasty/ws/akh-medu` checkout is shared with other sessions.
Leave it alone. Do not edit, stage, commit, reset, stash or clean that checkout.
Run repository commands with this dedicated worktree as their working directory.
Preserve other sessions' changes and do not use their identities or sessions.

The user's tracker rules also apply here: no direct Anima Postgres access;
tracker operations use the supported Anima MCP interface. The runner gateway
credential is not an MCP PAT. Preparation did not update tracker state, launch
providers, provision sandboxes or deploy services.

## Accepted design

The [deployment plan](assets/k8s-acp-t3-deployment.md),
[daemon boundaries](assets/daemon-boundaries.md),
[sandbox and Anima migration](assets/sandbox-images-anima.md), and
[upstream services and tool bindings](assets/upstream-services-tool-bindings.md)
are the implementation inputs. Later clarifications in those documents take
precedence over earlier alternatives in the research notes.

Keep external services on unmodified upstream versions. Use shared versioned
data contracts, specialized daemon ownership, upstream-compatible profile and
login helpers, and a small per-user environment connection process. Sessions
are central, editor servers share the executor's sandbox, and provider runtimes
can live on different machines. Anima keeps business scheduling and review
semantics while migrated execution routing moves to the shared services.

## Existing provider prototype

The shared checkout contains an uncommitted Codex/Claude prototype that is absent
from the `main` baseline. A snapshot was preserved in this worktree's private
Git metadata under `task-context/`: `provider-prototype.patch`, the untracked
source/bridge files under `provider-prototype/`, and a manifest with source
revision and content hashes. Locate it with:

```sh
git rev-parse --git-path task-context
```

This is a local reference snapshot, not committed implementation and not a
portable dependency of the repository. It does not track later edits by other
sessions. Port its relevant behavior deliberately onto `main`, accounting for
the newer context-assembly changes, rather than copying whole modified source
files over the baseline. Preserve the existing native-execution restriction
until provider-specific remote execution conformance passes. The imported
research documents contain source references to this earlier prototype; those
paths may remain absent here until the port is implemented.

## Implementation sequence

The table describes the full scope; the evidence below records partial progress.
Detailed acceptance conditions are in the deployment plan; a new binary or a
healthy endpoint alone does not pass them.

| Order | Scope | Required evidence |
| --- | --- | --- |
| P0 | Port the Codex app-server and Claude SDK prototype onto main, preserving current behavior and auth scope | Focused provider fixtures, cancellation/resume tests and existing remote-executor guard |
| D0 | Pin deployment targets, upstream profiles, supported host platforms and config revisions | Explicit work/homelab boundaries, versioned source/profile inventory and known capabilities |
| D1 | Shared contracts and durable `akhsessd` ownership/admission/journal | Stable identities, duplicate admission, input origin, ordered replay, approval recovery and fencing |
| D2 | ACP/WSS plus supported upstream T3 central-session attachment | End-to-end prompt, streaming, input, cancellation and reconnect; explicit missing capability if upstream is insufficient |
| D3 | `akhexecd`, workspace profiles and small client connection helper | Shared editor/executor files; upstream pairing; duplicate environment prevention; reconnect without stopping the prestarted server |
| D4 | Extract `akhrtd` and shared service/config/login helpers; reduce `akhomed` dependencies | Provider ownership, exact continuation, scoped effective config, drain/recovery and measured dependency separation |
| D5 | Semantic tool bindings and provider-specific native-tool restrictions | Successful brokered operations across hosts, denied native paths, correct receipts and no local fallback on broker failure |
| D6 | Released kagent BYO/A2A integration | Central task/context mapping, progress/input/cancel/artifacts and retry behavior |
| D7 | `akhsbxd` provisioning drivers and `akheventd` ingress | Kubernetes and selected host sandbox backends; RabbitMQ/webhook/Djehuty/cron delivery, durable dedupe, session targeting and cleanup |
| D8 | Deployment manifests, recovery, observability and operations | Independent restarts, rolling upgrades, retained state, site/user isolation and meaningful readiness |
| D9 | Anima compatibility client and gradual routing retirement | Stable work-item/attempt mapping, unchanged review/result gates, no duplicate dispatch and drained legacy owners |
| Follow-on | Optional AG-UI endpoint and A2UI payload integration | Same central session/tool authority and appropriate client conformance; not required for the first ACP/T3 deployment |

Run T3 supported-interface feasibility and one provider's remote-tool experiment
early and independently. D3 can start with manual allocation before D7. Encode
real implementation prerequisites as dependencies when updating the tracker;
this local checklist is not evidence of scheduled work or live dispatch.

Anima implementation changes require a separate dedicated worktree based on
that repository's main branch. Keep external upstream applications unmodified;
do not create T3 source patches as an implicit workaround.

## Verification and delivery

Use focused tests for contracts, persisted state, provider transports and failure
recovery, then cross-process integration tests. Distinguish fixture tests,
native provider tests, browser/client connection tests and actual cross-host
deployment evidence. Keep unsupported combinations explicit. Do not mark the
platform complete until its required ownership, execution and client behavior
has been demonstrated.

Keep changes in reviewable implementation commits in this worktree. Worktree
creation and local development do not authorize production restarts, account
changes or publication. Complete concrete local artifacts before any necessary
deployment approval step.

## Local implementation evidence (2026-09-12)

- P0 is committed: Codex app-server and Claude SDK adapters ported onto main.
  The original akhomed remote-executor guard remains. Ten provider transport
  tests passed; two credential-dependent native tests remain ignored. The
  existing akhomed server target checks successfully.
- Added engine-independent `akh-session-types`, `akh-runtime`, and
  `akh-session-services`. The services currently share a build package; final
  per-binary feature/dependency separation and akhomed extraction are pending.
- Added durable session admission, per-conversation creation, event ordering,
  exact dispatch requests, pending approvals, controller leases, runtime jobs,
  execution receipts and unknown-outcome fences. Credential files authenticate
  each service boundary. Tool schemas use JSON Schema validation with external
  reference retrieval disabled.
- Added the five service binaries, REST routes, deterministic webhook outbox,
  RabbitMQ ingestion, a cursor-based event-source adapter, and initial existing,
  command and Kubernetes provisioners. RabbitMQ and Kubernetes integrations are
  implemented but have not yet been tested against real servers. The event-source
  page contract is not yet a verified native Djehuty adapter.
- Added ACP v1 over WebSocket and an upstream-editor stdio bridge (`akh-acp`).
  The transport is separate from T3's native RPC. MCP catalog changes and non-text
  prompts are explicitly unsupported in this first transport implementation.
  Questions/client tools use Akh extension notifications and the durable response
  API; standard ACP permission requests are supported.
- Rust tests cover atomic persistence, duplicate/concurrent admission, approval
  recovery, stale generations, workspace traversal and symlinks, revision checks,
  process streaming/stdin/cancel, restart fencing, provider resume, authenticated
  ACP WebSocket permissions, and replay before the load response.
- `python3 scripts/session-platform/smoke.py` passed with five separate private
  processes: authentication, eight duplicate requests admitting one turn,
  approval before a workspace write, session-manager restart during approval,
  execution receipt, exact provider continuation, cancellation, webhook dedupe
  and delivery, and ordered replay. This uses a deterministic provider fixture
  on one host; it is not native-provider or cross-host isolation evidence.
- Four Node bridge tests passed. Brokered native runtime admission requires a
  conformance document bound to the installed profile, executable/bridge hashes
  and host platform. No native remote-execution profile has been certified yet.
- Fable review was attempted through local `claude --model claude-fable-5` after
  the user reported signing in again. The CLI still reported no authenticated
  session and made no model call. Clarification about the logged-in profile is
  pending; no Fable review is claimed.

Next required work includes upstream editor service/config/login supervision,
connection-helper behavior, T3 central-session feasibility, runtime/core
extraction, kagent A2A, recovery operations, deployment packaging, and the Anima
compatibility migration in a separate worktree. None of those are completed by
this first process slice.
