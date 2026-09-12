# Agent session platform implementation worktree

Prepared: 2026-09-11. Updated: 2026-09-12. The service split, durable session/tool
ownership, private runtime profiles, upstream workspace helpers, event adapters
and deployment artifacts are implemented in the dedicated worktrees. Local
cross-process, native T3 bootstrap, private-CA WSS and A2A SDK checks pass.

Full deployment remains in progress. T3 central-thread attachment, native provider
remote-tool certification, the enabled Anima scheduler/PWA migration, and real
cross-host validation are not completed by these checks. See the latest
checkpoint below and [deployment guide](deployment.md).

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

These are successive checkpoints; later sections supersede earlier pending work.

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

### Second local process slice

- `akh-service` now supervises upstream commands inside the executor's environment,
  checks configured readiness, preserves separately edited managed files, and
  uses a private service HOME for upstream login state. An interactive login
  action runs the selected upstream command directly. `akh-connect` maintains a
  per-user/profile environment-offer ledger and uses upstream-generated pairing
  links; it never writes T3's private connection database or marks an offer as a
  confirmed client connection. A browser pairing-link handoff is implemented; automatic native
  desktop registration still requires a supported upstream interface.
- A2A 0.3 JSON-RPC routes implement task/context mapping, message dedupe,
  streaming, pending-input responses, cancellation and resubscription. Contract
  tests pass; this has not been exercised through a deployed kagent controller.
- The provider implementations were removed from the original agent module.
  Its compatibility host now calls `akh-runtime` behind `embedded-runtime`.
  Nine legacy integration tests pass, with two native-provider tests ignored;
  the blocked-writer timeout test moved into the shared runtime package.
- Added an `akhomed` `core-server` build profile, selected with
  `--no-default-features --features core-server`. It serves scoped cognitive
  context, cognitive tools and accepted-turn projection endpoints. It starts no
  ACP listener or resident workspace agent loops. Session context is prepared
  before dispatch; completed turns reach a durable cognitive journal according
  to the existing transcript-capture policy.
- The core build and its scoped-tool/transcript-opt-out test pass. Its normal
  dependency graph contains none of `akh-runtime`, `agent-client-protocol`,
  `russh`, or `akh-session-services`. Shared record storage moved to the small
  `akh-service-store` crate; databases remain separately owned.
- The six-process smoke test passes with the real cognitive core, five session
  services, a provider fixture and a colocated editor fixture. It also checks
  that the editor reads the executor's file and that cognitive context and
  accepted-turn receipts appear in the central journal. Native providers,
  upstream editor clients, cross-host confinement and deployed kagent remain
  distinct validation work.
- The core test build exposed two pre-existing test compilation problems on the
  main baseline: a missing `DelegationTask.target_workspace` initializer and a
  metric test written against a newer OpenTelemetry API than the pinned 0.27.
  The initializer and test adapter were corrected; the bounded-label metric
  test passes against the pinned dependency.

The implementations remain in progress: native conformance,
explicit recovery operations, upstream T3 central-session bridging, real
provisioner/event-source checks, deployment artifacts and Anima migration are
still required. No native conformance, deployment, or Fable review is implied
by these fixture results.

### Independent build profiles and private runtime state

- `session`, `runtime`, `executor`, `sandbox`, `events`, `helpers`, and `client`
  features select the implementation each binary needs. The routers live in
  separate modules. Wire requests, helper status and provider profile types
  moved into `akh-session-types`. `scripts/session-platform/check-profiles.py`
  passes all nine daemon/helper builds and verifies their normal dependency
  graphs: only the runtime feature includes `akh-runtime`, only events includes
  RabbitMQ, and none includes the cognitive engine.
- Dedicated runtime workers require an explicit `isolation.directory`. They
  clear inherited environment variables and use private HOME, XDG config/cache,
  CODEX_HOME and CLAUDE_CONFIG_DIR. `akh-runtime-profile --config FILE --profile
  NAME prepare|login|status|version` uses the same profile and exclusive owner
  lock. Config files are installed below its `config/` directory; external edits
  cause conflicts and unmanaged login files remain provider-owned. Secret values
  are read from configured files at launch and are not persisted in wire records.
- The internal runtime now runs a bounded OpenAI-compatible chat-completions
  tool loop in `akhrtd`. It supports SSE text/tool deltas, broker callbacks,
  durable private continuation, exact resume, cancellation and byte/token/turn
  bounds. Cognitive context continues to come from `akhomed`. A truncated stream,
  unavailable continuation, or unfinished turn is not retried as a new turn.
  This transport targets the documented [Ollama compatibility interface](https://docs.ollama.com/api/openai-compatibility).
  It currently accepts text and event data; artifact access uses workspace tools.
- Native brokered conformance additionally covers configured immutable artifacts,
  Node and launcher executable hashes. Non-Claude native aliases are explicitly
  rejected. Runtime hosts quarantine new admissions when any job has an unknown
  outcome, pending explicit reconciliation.
- Shared-crate tests pass, including streamed model/tool fixtures with exact
  continuation, private profile locking/config preservation, and bounded helper
  output with timeout teardown of descendant processes. These are local protocol
  and failure tests, not native-provider certification.

### Recovery, leases and upstream restarts

- Added [explicit operator recovery](recovery.md) for runtime jobs, executor
  receipts, external tool outcomes, command allocations and helper processes.
  Resolution records bind the authenticated admin and evidence to the original
  request. They never rerun the old operation. Session ingestion checks the
  original dispatch and the digest of previously accepted runtime events.
- Workspace leases now expire at the executor. Open sessions renew through the
  sandbox manager; expiry fences effects and stops owned processes/helpers.
  Archival preserves session history, blocks new input and reconciles release.
  Retention still revokes executor access. Kubernetes deletion is UID-fenced.
- Helpers support explicit stop, reconcile and idempotent restart with an
  optional new profile revision. Their existing upstream environment identity
  and private login/config directory survive restart. Administrative subprocess
  input/output and process-group teardown are bounded.
- Added per-endpoint private CA support, runtime event/answer byte budgets and
  bounded HTTP shutdown. The `akh-fingerprint` helper computes request digests
  for recovery; its separate `admin` feature carries no provider or broker.
- Local tests pass for executor recovery and lease expiry, immutable runtime
  replay, and a Kubernetes API fixture that rejects deletion of a replacement
  UID. The six-process smoke additionally passes helper stop/restart/deduplication,
  environment identity preservation, lease renewal and archival cleanup. The
  Kubernetes fixture is not a real controller/sandbox deployment.

### Native event sources and MCP exchange

- Added the native Djehuty thread-message adapter, independent durable cursors,
  identity/order validation and outbox-before-cursor advancement. Added
  `akh-trigger` for external cron and Kubernetes Job occurrences. Scheduler
  identity/time survives retry; session admission remains the only run ledger.
- MCP tools now use a bounded negotiated Streamable HTTP exchange with typed
  protocol results, JSON/SSE handling and cursor-based GET recovery. An uncertain
  tool POST is never automatically repeated. See [contracts and configuration](events-and-mcp.md).
- All service tests pass with `admin,trigger`; all eleven isolated build profiles
  and dependency checks pass. Native Djehuty, MCP and Kubernetes occurrence
  fixtures pass; real service deployment remains separate validation.
- Anima work continues in `/Users/toasty/ws/anima-agent-platform`, branch
  `feat/akh-session-client`, based on freshly fetched main
  `95d7cc1301cf40a1593a33f8e30d9c5166be06e0`. Its original checkout is untouched.

### Deployment and upstream client checkpoint

- The dispatcher now runs bounded, independent session tasks; slow cognitive
  services no longer serialize other sessions. Tool settlement and execution
  share a per-call lock. Archived sessions expose a cleanup acknowledgement at
  `GET /v1/sessions/{id}/lifecycle`; an allocation created during cancellation
  must still be released, even if it never became a bound workspace. Downstream
  404s do not count as release evidence. Tests cover these races and stale
  Kubernetes Ready generations.
- Receiving principals can use a SHA-256 verifier instead of a reusable bearer
  credential. `akh-acp` supports a private CA without bypassing verification.
  `scripts/session-platform/wss-smoke.py` passes actual WSS rejection of an
  untrusted CA, authenticated ACP initialization, creation and reconnect/load.
- Runtime tool events retain native structured completion details. Unsupported
  input attachments fail at admission. Interrupted internal tool plans remain
  unknown until audited reconciliation, including when cancellation was requested.
- Helpers verify configured upstream versions and isolate HOME/XDG config/cache.
  The native `t3@0.0.40` package passes `t3-native-smoke.py`: startup, environment
  identity, fresh pairing links, restart and shutdown. The profile disables
  native provider probes for this editor-only check. Pair secrets stay in memory.
  Desktop handoff preserves the GUI after the dispatcher returns successfully;
  a failed handoff remains retryable with a fresh link for the same environment.
  Neither browser pairing nor central T3 agent attachment is claimed.
- Added the stateless `akh-kagent` BYO facade, twelve isolated build profiles,
  per-service container recipes, Helm chart, systemd/launchd examples and external
  CronJob trigger. Actual A2A Python SDK 0.3.22 discovery, task/context identity,
  dedupe, polling, cancellation and SSE pass through the facade. It requires an
  explicitly trusted, network-restricted kagent hop and supports A2A 0.3 only.
- A native Linux arm64 `akhsbxd` image builds with Rust 1.94.1 and starts as
  UID/GID 10001. Its local image is approximately 35.9 MB. Other daemon images,
  image publication, chart installation and controller operation remain separate
  checks. Helm lint/render and launchd plist validation pass.
- The six-process smoke passes after the lifecycle changes, using the real
  cognitive core plus a deterministic provider/editor fixture. The four Claude
  bridge tests and all twelve isolated feature checks pass. These fixtures do
  not certify native provider execution or kernel sandbox confinement.
- Anima's new `anima-akh` client/outbox pins the work-item/phase/head/attempt and
  central occurrence. Its durable owner fences legacy dispatch even after legacy
  cleanup removes a transient claim. A model completion is separate from Change
  ingestion, review, CI and phase advancement. The client/outbox is not yet wired
  into the canonical scheduler or PWA; physical routing has not been retired.

### Unadmitted dispatch recovery

- Added scheduler-scoped, audited automation revocation. Admission and revocation
  share the central transaction: a never-admitted occurrence receives a durable
  tombstone; an already-admitted occurrence returns its session/turn identity and
  still requires normal cleanup. Concurrent admission/revocation and restart
  tests pass.
- Anima persists revocation intent before HTTP and can recover a central identity
  whose admission response was lost. An expired, never-admitted assignment can
  release its transport fence only after central revocation. The real local
  akhsessd plus disposable PostgreSQL test passes for both outcomes, ordinary
  cancellation, journal import into the outbox, and workspace-release fencing.
- The complete Anima migration chain, including the two new additive migrations,
  boots successfully on a fresh disposable PostgreSQL 17 database. This is local
  database evidence; no tracker data or deployed service was changed.

### Native Kubernetes allocation and cleanup

- `scripts/session-platform/k8s-native-smoke.py` passed with the upstream Agent
  Sandbox v1.0.0 controller in a disposable kind v0.33.0 cluster (Kubernetes
  v1.35.8). It uses an explicit test context, scoped service-account token and
  verified API CA, and loads a locally built Linux arm64 executor image.
- A real executor pod accepted a workspace write through an authoritative receipt;
  the file was also read from that pod's filesystem. Repeating the operation
  preserved the receipt. The test held foreground cleanup, restarted `akhsbxd`,
  and verified release only after the owned resource disappeared. Cancellation
  before executor attachment also completed cleanup.
- Lost create responses remain fenced, replacement UIDs are rejected before
  deletion, and executor release tombstones prevent delayed attachment. Separate
  sessions make progress while another executor is slow. Audited observed absence
  can close an uncertain or previously failed allocation without reprovisioning.
- This is native controller and pod evidence on one disposable local cluster.
  It does not demonstrate gVisor/Kata confinement, work/homelab deployment,
  cross-machine provider tools, or NetworkPolicy enforcement.

### Released kagent controller interoperability

- The actual upstream kagent 0.10.1 controller, CRDs and development database ran
  in the disposable kind cluster. The repository's BYO Helm template created the
  native Linux facade deployment and passed upstream readiness. Its central hop
  used a private CA and server certificate, with separate central credentials.
- The test exposed two requirements beyond a simple A2A proxy: the controller
  reads task queries from its REST-backed store, and clients can create a kagent
  context before the first prompt. `akh-kagent` now projects central tasks through
  the released session/task REST APIs and persists context aliases in central
  conversation keys. Each facade binds one explicit kagent user to a dedicated
  central principal. It owns no local session database or model process.
- `scripts/session-platform/kagent-native-smoke.py` passes actual controller
  discovery/A2A 0.3 traffic, immutable-message dedupe, context follow-ups, another
  user's rejected cancellation, task polling/cancellation, and SSE completion.
  While the facade was stopped, the test cancelled a task centrally; the new
  facade projected that state after restart. A different, explicitly deleted
  kagent task view did not block this refresh. Upstream deleted IDs stay reserved.
- The chart rejects credential Secrets that collide with the controller's own
  generated Secret. Three adapter tests and the standalone A2A Python SDK 0.3.22
  regression pass. A bounded projection cache periodically refreshes unchanged
  tasks to repair a stale view left by overlapping rolling replicas; that repair
  is covered by a focused test. Shutdown drains for at most ten seconds.
- These tests use the upstream development authentication mode and kindnet.
  They do not prove authenticated production kagent configuration, NetworkPolicy
  enforcement, a browser UI, native provider tools, or cross-host deployment.
  No images or changes were published. See the [deployment guide](deployment.md)
  for the required identity, projection and context-mapping configuration.

### Native RabbitMQ ingress and reconnect

- The disposable RabbitMQ 4.3.5 test passes actual AMQP consumption and broker
  acknowledgement after durable outbox receipt, even while central sessions are
  unavailable. It checks dead lettering of malformed, forged-source and
  conflicting-ID events, ingress restart recovery, and exact central admission.
- A broker restart reconnects the consumer. Republishing the original event is
  acknowledged without creating another central turn. Test ports are explicit:
  Docker may change automatically assigned host ports when restarting a container.
  The broker, owned volume and generated test credentials were removed afterward.
- The tested image digest is pinned in `rabbitmq-native-smoke.py`. This is a real
  local broker test; it does not prove provider execution or production delivery.

### Durable Anima continuation commands

- Anima now persists immutable prompt, pinned-turn cancellation and generation-
  conditioned input responses separately from assignment admission. Its real
  local PostgreSQL/`akhsessd` test passes lost follow-up acknowledgement recovery,
  concurrent retries, prompt order, and closure fencing while commands or later
  turns remain outstanding. The complete additive migration chain boots locally.
- Closing intent is durable before archive/cleanup and prevents new commands.
  Exact historical command receipts survive closure. Entry-point routing and
  journal projection into Anima's existing UI remain separate migration work.
- Akh replays an already-recorded input acknowledgement after its turn advances;
  this does not reapply permission or revive a turn. A supplied input generation
  is checked before the decision. Stale or different answers remain rejected.

### Anima follow-up entry points and recovery worker

- Anima's thread messages and work-item description edits recognize existing
  central owners and queue immutable commands using their persisted source IDs
  and authenticated actor. An uncertain central/AMQP delivery cannot fall back
  to another physical owner or a future assignment. Attachments are rejected
  explicitly on the central text-only route.
- `ANIMA_AKH_SESSIONS_CONFIG` enables a bounded continuation recovery worker for
  already-authorized outbox records. Exact configured contracts select credentials;
  stored endpoints cannot redirect them. The worker retries admission, commands,
  revocation and closing intent, and imports the journal into the existing outbox.
- Server checks and local HTTP/database tests cover the actual prompt router,
  restarted background worker and cancellation progress during uncertain prompt
  delivery. The new scheduler allocation route, UI Stop/permission controls,
  canonical result/retirement path and activity projection remain to be wired.

### Installed Codex harness and supplied workspace tools

- The actual Codex CLI 0.153.4 app-server passed thirteen requests from a local
  synthetic Responses peer. Supplied write/read/process tools reached the real
  executor, duplicate operations preserved receipts, and a restarted app-server
  resumed the same provider thread. No account credentials were used.
- Forced native execution and patch calls were unsupported and absent from the
  advertised tool catalogs. A supplied-tool failure could not fall back to a
  native workspace write. Provider planning/goal and skill tools remain present.
- The strict resume check found that the adapter must disable environments on
  every turn: `thread/resume` has no such field. The fix keeps native patch and
  image tools unavailable after process restart; the native and fixture checks pass.
- This is native harness behavior with a model fixture, not live-model or
  cross-host conformance. The daemon's native admission gate remains in place.
  See [native provider checks](native-provider-checks.md) for commands and limits.

### Anima transcript projection

- Anima commit `cbb9b2cb` adds the existing UI's journal projection. Activity rows,
  final thread replies and the source cursor commit together; concurrent/restarted
  workers cannot duplicate replies. Existing authorized reads and stream buses
  consume the projection, while Anima's business lifecycle remains separate.
- A full-schema disposable PostgreSQL test passes injected reply-write rollback,
  replay, thread fan-out and equal-timestamp history. Rust/PWA shared wire examples
  pass; 97 focused UI regressions and the additional shared-wire check pass.
  Native browser and production streaming proof remain outstanding.
- Central text deltas are retained in both the cached source and UI projection
  during compatibility migration. Measure that storage cost before high-volume
  deployment. Stop/input controls and canonical retirement are the next wiring;
  the scheduler still does not create new central allocations.

### Durable permission decisions and Anima controls

- `input-decisions` atomically persists accepted and rejected permission/input
  intent. Stale or missing targets cannot activate on a later retry. Controller
  lease conflicts remain retryable; historical receipt reads apply no decision.
  Eleven local session tests pass, including concurrent decisions and restart.
- Anima validates the exact central decision before settling its command outbox.
  A real session-daemon/disposable-PostgreSQL test passes lost negative receipt
  recovery, concurrent retries and historical rejection replay after closure.
- Anima's permission card now resolves the immutable source event and generation,
  uses the authenticated actor and stable command identity, and waits for a
  receipt. HTTP uncertainty leaves a retryable UI action. Full-schema/HTTP tests
  pass loss/retry and stale-choice behavior; Rust/PWA shared wire examples pass.
  Stop, general question forms and canonical session retirement remain separate
  integration work. These checks do not enable a project or deploy any daemon.

### Pinned Anima Stop and atomic controller authority

- Anima commit `c0221e5e` connects Stop to a live, authorized central turn read.
  The command retains its original turn across lost responses and concurrent UI
  clicks; a queued follow-up cannot replace the first nonterminal turn. Full-schema
  HTTP recovery, native daemon turn-order/cancellation and three UI tests pass.
- Prompt, Stop, legacy input response and archive now check controller ownership
  in their store transaction. REST, ACP and A2A therefore cannot use a stale
  transport-level read to mutate after another attachment takes control.
  Twelve session tests and the ACP WebSocket/A2A integration tests pass locally.
- Whole-session retirement remains separate from Stop. Browser runtime and
  production deployment remain unverified; the central scheduler is not enabled.

### Audited whole-session retirement

- Akh now records immutable retirement intent, fences new inputs and cancels the
  outstanding turn set atomically. Unknown execution is preserved until runtime
  recovery confirms an outcome. Thirteen session tests plus ACP/A2A tests pass;
  the native Anima client also verifies the actual daemon's retirement receipt.
- Anima's canonical CancelSession/ReconcileSession paths persist retirement for
  its configured background relay. They retain the physical owner while prior
  commands, unknown admission, execution or workspace cleanup remain pending.
  Normal successful closure still requires a Change; operator retirement is an
  audited abort and explicitly does not accept the model's business result.
- Full-schema PostgreSQL/HTTP tests pass lost replies, delayed cleanup, audit-write
  rollback and concurrent retry. The terminal session outcome, audit, central
  owner closure and exact claim release commit together. Existing business phase,
  review and head gates remain unchanged. Replays preserve later claims.
- Scheduler selection and result-tool/workspace bootstrap remain migration work;
  no central project allocation, production daemon or infrastructure was enabled.

### Provider questions and live client input controls

- Shared question forms preserve provider IDs, choices, multiple selection,
  free text and explicit Skip; runtime adapters construct native Codex/Claude
  answers. Secret answers cannot enter this journaled channel. Claude's brokered
  bridge enables only its question tool alongside supplied MCP tools; native
  workspace denials remain enforced. New bridge hashes require new conformance.
- Anima validates the original question source and answers before its durable
  input-decision command. Its live-input reader binds pending inputs to cached
  journal events, replacing activity-tail inference for central permissions.
  Forms retain their source session and freeze uncertain answers for exact retry.
- Fourteen session tests, five runtime/transport tests, five SDK tests and the
  shared form check pass. Full-schema Anima recovery and 97 UI checks pass locally.
  Native question elicitation and browser/deployment proof remain unverified.
  See [the question contract](questions.md) for protocol and source details.

### Private session result capabilities

- Session-scoped MCP endpoints now pin their credential policy and destination
  at admission. Required capabilities gate new runtime dispatch. The broker
  resolves credentials before recording a sent tool call and never falls back
  to a shared token. Only secret-free receipts enter the journal.
- Private provisioning requires the session owner and a dedicated role. Exact
  retry survives interrupted file/journal commits and process restart; archive
  removes private files without allowing historical receipts to restore them.
- Sixteen session tests, three transport/lifecycle unit tests and A2A pass.
  Anima's client tests cover lost and mismatched acknowledgements; its native
  credential test passes against the locally rebuilt daemon through restart and
  archive. See [session credentials](session-credentials.md).
- Central scheduler allocation, canonical token issuance/revocation and exact
  repository bootstrap remain integration work. No production token was read or
  minted, and no project was enabled for central dispatch.

### Literal executor arguments

- `exec_start` now accepts either the existing `command` string or an `argv`
  array with an absolute executable. The latter passes each argument literally,
  preserving whitespace and shell metacharacters without shell expansion.
  Workspace ownership, leases, cancellation, timeouts and operation receipts
  continue to use the same executor lifecycle. Six executor tests pass locally.
- This is the execution primitive for repository bootstrap; it does not itself
  select a repository, create a checkout or certify a sandbox image.

### Repository bootstrap before editor and runtime readiness

- An owner with the bootstrap role can pin an allowed repository and exact commit
  to a new session. The sandbox manager validates its own allowed sources and
  pins the Git profile before coordinating seven executor operations. Editor
  startup and runtime dispatch require the resulting workspace/commit evidence.
- Native Git/executor HTTP checks pass lost responses, manager restart, unknown
  operation fencing with audited executor recovery, exact commit verification and
  preservation of pre-existing files. Seventeen session tests, two sandbox tests
  and six executor tests also pass locally. Anima now has an independent client
  method for the immutable repository intent. See [repository bootstrap](repository-bootstrap.md).
- Central scheduler selection, authoritative base-commit resolution, prior Change
  materialization and canonical result-token issuance/revocation remain migration
  work. No private Git login, production allocation or project dispatch was run.

### Atomic Anima admission primitives

- Anima's canonical claim can now compose a central session with its transport
  owner, attempt, scoped token, prompt metadata and context snapshot in one
  transaction, with no synthetic RunnerId. Network artifacts can be staged
  before claiming; the attempt is accepted only if its expected value still holds.
- The canonical claim now also refuses an open central owner after legacy-only
  cleanup. Full-schema PostgreSQL tests pass rollback, concurrent claims,
  transactional visibility and stale attempts. All sixteen existing phase,
  exact-head review, CI and dependency-gate tests pass. Test containers were
  owned fixtures; the shared-host global stale-container sweep was disabled.
- The central route, staging coordinator, result lifecycle and provider/runtime
  deployment remain integration work. These primitives do not enable a project.


### Anima bootstrap recovery and capability closure

- Added optional atomic bootstrap requirements to automation invocation. Akh
  validates the requested repository and exact session credential slots before
  committing a fresh per-occurrence session/turn. Eighteen session tests pass,
  including refusal without any admitted work and immutable retry.
- Anima now pins route/config identity, token ID, checkout and request IDs in the
  same canonical admission transaction. The relay provisions both intents and
  stores only receipts. Full-schema/native-daemon tests pass a lost committed
  credential response, restart, exact retry and routing-drift refusal. The native
  journal contains one credential event and no token value.
- Migration 137 revokes an owner's canonical session tokens atomically with every
  closure path. Tests verify rollback and preservation of another session's
  token; the existing operator-retirement regression also passes.
- Central scheduler selection/staging, authoritative repository resolution,
  preservation of prior Changes, and result ingestion/lifecycle still need their
  integration. No project route or production service was enabled. Fable's local
  Claude CLI still reports logged out, so its design review remains outstanding.


### Native Akh-to-Anima MCP compatibility

- Found and fixed Anima's legacy-only MCP version negotiation, which the modern
  Akh broker correctly refused. Anima now negotiates 2025-06-18 for Streamable
  HTTP, retains explicit legacy negotiation, and validates origin/version headers.
- The actual Akh MCP client now passes a scoped Work Item read against the actual
  Anima HTTP router and an owned full-schema local database. The router fixture
  also checks notifications, ping, filtered tools, auth/revocation and GET 405.
  No production token or endpoint was used. This proves read transport; the
  result submission and scheduler lifecycle integrations remain outstanding.
