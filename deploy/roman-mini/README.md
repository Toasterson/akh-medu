# Roman mini teammates

## Provider failure recovery

The Janus escalation outbox distinguishes command delivery from execution.
`astra_escalation_status` reports `running`, `turn_completed`,
`blocked_authentication`, `blocked_quota`, `blocked_rate_limit`, `failed`, or
`status_unavailable`. A completed turn still requires reading and verifying the
result. A live turn takes precedence over a retained older provider error.
Provider failures retain the incident key and thread; the operator repairs
access or waits for the usage reset, then resumes that same thread after
checking live ownership and partial work. The broker never automatically
replays an agent turn or changes provider/model.

The manager observer records provider failures separately for Janus, Minerva,
and Vesta. HTTP 429 means rate limiting unless the provider explicitly reports
exhausted quota. Consecutive provider failures defer scheduled wakes and
unsubmitted events for 30 minutes, 1 hour, 2 hours, then at most 4 hours.
After cooldown, the existing scheduler can admit a fresh bounded wake to
re-read current work. It does not replay failed actions. Already admitted
deliveries and running turns keep their ownership. Pending event identities
and coalesced changes are retained across restarts.

Authentication failures require operator repair and a later successful turn;
they do not expire on a timer. A successful turn clears only that manager's
block. Replayed or older receipts cannot extend cooldowns or override a newer
success. Inspect `provider_availability` through `monitoring_snapshot` for the
failure class, retry time, and next owner. These controls cover observer-owned
scheduled admission; direct user turns and already accepted events remain
under the session service's existing admission rules. Provider-specified reset
times are not currently exposed in manager error receipts, so cooldowns use a
bounded local policy rather than claiming to know an account's reset time.

Install `provider_failure.py` alongside both `observer.py` and
`janus_escalation.py`; the activation/configuration scripts copy it. Preserve
installed runtime monitoring customizations when applying a focused update.

Janus owns project execution, Minerva product clarity, and Vesta incident
coordination. They share a dedicated provider runtime and cognitive core.

## Local persistence

The cognitive core uses the existing SeaORM relational backing-store path with
SQLite restored. Configure it with:

```toml
[database]
url = "sqlite:///absolute/path/akh.sqlite3?mode=rwc"
migrate_on_start = true
pool_size = 1
wal_flush_interval_secs = 1
snapshot_interval_secs = 30
```

`[postgres]` remains a compatible section name and PostgreSQL stays supported.
Local SQLite uses WAL, full synchronous commits, serialized migrations, and
per-akh OS file locks paired with durable residency epochs. A released or stale
owner cannot append mutations. Do not unlink lock files: the OS releases locks
when their owning process dies. Use a local filesystem, not shared network storage.

SQLite holds the authoritative relational akh state, snapshots and mutation
log. The reasoning engine still uses its existing local redb working cache;
recovery reconstructs that cache from SQLite. This restores the backing-store
architecture instead of replacing the engine's internal storage abstractions.

The session, provider-runtime, sandbox, executor, and cognitive operation ledgers
also use private SQLite databases. All service endpoints bind to loopback and
require separately generated service credentials for protected operations.

## Installation and awakening

`prepare.py` renders the four service configurations. `configure_core.py` renders
the core: since Phase 53 (ADR 113) that is one **full akh** akhomed hosting
janus, minerva and vesta with their daemon loops, in-process turns (GLM-5.3 on
Ollama Cloud), full tool registries, per-akh MCP servers (`[[mcp.servers]]`,
tools named `mcp__<server>__<tool>`), reactors, ACP and the additive `/v1` core
API. Neither script starts launchd or invokes a model. The launchd definitions
live under the deployment root until installed into `/Library/LaunchDaemons`.
The operator runbook for moving the live deployment onto the full akh is
[CUTOVER.md](CUTOVER.md).

`awakening.json` defines Roman archetypes. The offline `roman_awaken` example runs
the actual `ritual_of_awakening`, publishes a fenced SeaORM snapshot, then recovers
the awakened psyche into a different cache. It refuses to replace an existing
durable identity. The ritual may choose a Latin self-name different from the
archetype; the service IDs remain janus/minerva/vesta.

Build only from a clean checkout of `origin/main`
(`.claude/rules/deploy-from-main.md`); the stripped `core-server` profile was
deleted in Phase 53h and `server` is the one akhomed profile:

```sh
git fetch origin && git switch --detach origin/main && git status --porcelain  # must be empty
cargo build --release --locked --features server --bin akhomed --example roman_awaken
cargo build --release --locked -p akh-session-services --features client --bin akh-acp
python3 deploy/roman-mini/configure_core.py --root "$ROOT" --binaries target/release --repo .
```

`configure_core.py` refuses unless the checkout is clean and `HEAD` is an
ancestor of `origin/main`, installs `bin/akhomed-main-<sha>` (and
`bin/akh-acp-main-<sha>`), and records the commit in `config/core-release.json`.
The core plist execs through `daemon_credentials.py core`, which resolves the
Ollama and MCP credentials into the environment and execs that binary.

The helper takes `DATABASE_URL INPUT_JSON WORK_DIRECTORY`. Its work directory
holds the initial/recovered caches and ritual evidence. No provider credentials
or cloud calls are involved in awakening.

## Activation status

Consult `evidence/prepared.json` and the other verification records under the
installed root for actual deployment status. A listening daemon or an awakened
identity does not prove an operational autonomous team.

`activate.py ROOT` renders enabled automations and the event/observer launchd
jobs after authenticated service integration. Janus (PM) reviews every 15
minutes, Minerva (PO) every hour, and Vesta (incident coordination) every 15
minutes. Addressed Djehuty delegations are polled every 15 seconds; status
messages do not wake teammates. Native event ingress persists before inbox
acknowledgement. One runtime serializes inference; this is not a guaranteed
low-latency incident-response service. Older turns waiting on runtime capacity
take precedence over newer wakes, so retry backoff cannot let a busy role
continually jump ahead of its teammates.

The observer polls Anima through MCP, rotating eight active projects per minute.
Work-item and plan changes wake both Janus and Minerva, coalesced to at most 40
changed entities per role every five minutes. Initial observations establish a
baseline. Capped work-item lists are expanded by phase where possible; coverage
and remaining truncation are visible, and missing rows never imply deletion.
Fresh sessions share each Akh's durable memory and limit accumulated context.
Each role admits one active automation turn at a time.

Monitoring changes wake Vesta. A separate deterministic watchdog posts daemon,
feed, and workflow failures even when inference cannot run. VictoriaMetrics and
Grafana are separate sources: unavailable Grafana means unknown alert state,
not zero alerts. The monitoring MCP exposes only timestamped observations.
Vesta has Anima and Board coordination tools, not host or Kubernetes remediation
commands. Event data never expands role authority.

The observer stores observations, immutable event outbox entries, schedule
occurrences and delivery outcomes in private SQLite. Repeated ingress attempts
reuse the same identity. Scheduled downtime does not replay historical ticks.
Session and observer LaunchAgents require the logged-in user's keychain;
remaining services run as system LaunchDaemons under that user. Bots themselves
have no filesystem or shell tools. Provider keys remain in process environments.

Grafana uses its HTTPS route with the authenticated cluster's public PKI CA;
its failing kubelet port-forward is not required. Alert suppression metadata is
preserved. Sokar's initially unavailable artifact backend was rechecked during
the GLM repair: a scoped verification artifact was written and read back exactly.
Internal notes remain independent of the archive backend. A known model output-limit
stop fails its turn without quarantining unrelated jobs; uncertain tool outcomes
retain reconciliation gates. Complete Ollama tool calls that reuse stream index
zero remain separate by their full call identities. The entire tool plan is
validated before persistence or execution, so malformed arguments cannot trigger
a partial batch or quarantine the runtime as an uncertain tool outcome.

The provider profiles use Ollama Cloud and OpenRouter's chat-completions APIs.
Routine wakes use Ollama Cloud `glm-5.3` with thinking enabled by the provider.
Each response has a 32768-token budget and each turn a 1200-second deadline.
Reasoning is preserved verbatim in its provider field across tool rounds and
streamed separately as ACP `agent_thought_chunk`; empty content chunks are ignored.
Each role has a dedicated runtime profile, so an uncertain internal tool outcome
quarantines that role rather than unrelated roles. Such uncertainty never expires
merely because time passed. Native child runtimes retain their global quarantine.
Model-only interruptions retire the continuation without replaying past effects.
Queued cancellations are persisted as runtime tombstones before admission.
Capacity waiting does not count as a stalled running session.
See [GLM thinking requirements](https://docs.z.ai/guides/capabilities/thinking-mode).
Board message payload schemas are narrowed to objects for the roles' structured
coordination protocols, avoiding stringified JSON payloads. Wake instructions
bound work to eight tool calls and require an early durable checkpoint.
Provider keys are resolved from the authorized 1Password references by
`daemon_credentials.py` when the runtime starts. Only environment variable names
are stored in runtime configuration; keys are never copied to files or T3.
The session launcher resolves the existing Anima operator PAT, archive service
credential, and each Akh's own Djehuty identity into its private environment.
Anima mutations remain subject to the operator PAT scopes and server gates.
OpenRouter is an alternate profile, not an automatic
fallback. Model availability and tool use require authenticated verification.

Local durable storage preserves state during outages. Cloud inference still
requires connectivity.

Coordination is recorded using the
[Agent Board coordination skill](/Users/toasty/.codex/skills/agent-board-coordination/SKILL.md)
under this Codex thread's own registered identity.


## Service and T3 integration

`configure_integrations.py ROOT` imports authenticated MCP catalogs from
`ROOT/evidence/{anima,seshat,sokar,board}-mcp-catalog.json`. It grants cognitive
and MCP tools only. Filesystem, shell, client, and workspace tools are absent;
the current session allocator still maintains an internal opaque workspace.
Do not grant executor tools to these roles. Notes belong in internal memory,
Seshat, Anima documents, or Sokar artifacts.

The installed T3 build has no generic ACP provider setting. `t3_entrypoint.py`
adapts its existing Grok ACP transport to `akh-acp --t3-compat`, using separately
scoped Roman client credentials. Provider instances display the Roman role
names; the transport's `grok-build` alias means the daemon-configured model
(GLM-5.3 via Ollama), not an xAI model. No T3 project files or filesystem tools
are forwarded. T3 settings are watched live; preserve unrelated instances.

Dedicated launchd-managed localhost Kubernetes tunnels route Djehuty and Sokar
while their lab ingress routes fail. A working MCP catalog is not proof of
artifact storage availability: Sokar currently reports backend SeaweedFS errors.

The session daemon runs in `gui/UID` as a LaunchAgent because the existing Anima
PAT lives in the login keychain. It starts at login; the other services remain
system LaunchDaemons. A logged-out reboot does not start the session daemon.

Install T3 entries with `configure_t3.py ROOT ~/.t3/userdata/settings.json`, then
restart the Roman user session service to load new client credentials. The T3
adapter discards editor-injected MCP servers and additional directory grants.

Initial integration checks (before automated-wake model tuning): all three Akhs
completed Ollama turns, called Anima/Board/Seshat,
and wrote internal memory. OpenRouter `z-ai/glm-5.3` returned `OK`. Broker probes
rejected file and shell operations with `tool_disabled`, and a cross-Akh T3 read
returned 403. Sokar's correctly typed call reached the backend and failed with
`store lab-seaweed: write manifest failed (Unexpected)`. Its schema definitions
are expanded before cloud use; authoritative broker validation is retained.

The original integration probe failed before its first tool could run due to a
callback principal mismatch. It was reconciled through the runtime API against
the exact request hash and retired through the session API; failed evidence is
retained. The renderer now uses the required `sandboxes` alias and runtime
callback principal.

### Autonomous activity in T3

`configure_t3.py` also installs three activity ACP providers and a
`t3-activity` LaunchAgent. Its supported T3 auth CLI issues a short-lived token
held only in memory; HTTP orchestration creates the **Roman autonomous activity**
project and one thread per role. Private SQLite records submitted session IDs,
and deterministic command IDs make retries idempotent. It never edits T3 tables.
The ACP viewer reads only sessions owned by that role's scoped T3 principal,
streams reasoning and tool receipt events, and reports terminal outcomes. Watching
or cancelling a feed does not start, repeat, or cancel the bot's actual job.
Messages typed into an activity thread are forwarded to that bot as scoped user
turns. Durable request IDs prevent duplicate admission after response loss. The
initial acknowledgment confirms delivery, not memory storage; the bot must record
the correction and its tool receipt provides proof. Watching alone starts no work.
Use the original role providers for longer direct conversations. Historical attempts and
errors remain visible. Model credentials and project filesystem access are never
provided to these viewers.

### Vesta GitOps sandbox and homelab diagnostics

Vesta alone can use `homelab_inspect` (live Kubernetes get/describe/pod logs) and
`gitops_exec` (commands and file edits in the fixed `roman-vesta-gitops` Colima
container). The container has a dedicated `toasterson/talos` clone at `/workspace`,
a non-root UID, read-only root filesystem, no network, dropped capabilities,
no Docker socket or host credentials, and bounded CPU, memory, process and command
budgets. Git, Python, jq, yamllint and kubectl/kustomize are installed. Persistent
edits live in the dedicated checkout; host home directories are not exposed.
`vesta_ops.py` checks the pinned container/image identity and confinement on each
execution. Cluster credentials remain in the separate host diagnostics broker.

`configure_vesta.py ROOT` registers only Vesta's grants and writes the
`dev.roman-akhs.vesta-ops` LaunchAgent after `ROOT/config/vesta-ops.json` records the
provisioned container. `activate.py` preserves this configuration on rerender.
The image recipe is `vesta/Dockerfile`; its kubectl download is checksum-verified.
Validate sandbox writes, denied host/credential/socket access, denied network and
live node reads before enabling new sessions. Local edits alone do not publish or deploy them. With action grants provisioned,
`gitops_publish` exports an exact commit through a temporary bare repository and
pushes only to `toasterson/talos`, without loading sandbox Git configuration on the
host or exposing SSH credentials. Force pushes and arbitrary remotes are absent.

### Bounded wake cadence

Janus checks periodically after 30 minutes of rest; Minerva and Vesta after an
hour. Active or queued work suppresses timer wakes. Anima changes accumulate by
entity and are batched no more often than 15 minutes for Janus and 30 minutes for
Minerva, with at least five minutes of rest. Alert state changes wake Vesta after
existing work completes and five minutes of rest. Volatile elapsed fields and
alert annotation values do not create new work. Concrete inbox delegations remain
durable and are never coalesced.

The event service optionally coalesces older, definitely unadmitted timer/alert
notifications while retaining their original events and supersession receipts.
Ambiguous transport failures and admitted turns are preserved. Compact session
and pending-delivery queries support activity checks; T3 starts its live activity
feed when a bot is running or has finished, rather than while it is queued.
Board write tools expose the registered status/delegation schemas explicitly.

### Vesta action grants and Janus merge authority

Run `configure_vesta_actions.py ROOT` to install the loopback Grafana MCP tunnel,
discover the live tool catalog, pin the cluster context and render Vesta grants.
The broker forwards authenticated `grafana__*` tools to the existing homelab MCP,
including write tools; Grafana credentials stay in that deployment. The broker
exposes named Kubernetes apply/patch/delete/scale/restart/cordon operations via
`homelab_write`. It never executes arbitrary host shell commands. Reload the broker
and session daemon at a quiescent boundary. Janus and Minerva get none of these
tools. Verify using a temporary branch, ConfigMap lifecycle, and Grafana annotation.

`configure_janus.py ROOT` records the operator's explicit standing authorization
for Janus to use its existing `anima__approve_and_merge` grant. Anima enforces the
live head, preceding review and exact-head CI. Explicit holds and human gates remain
in force. Janus can discover active workers on Board and send an addressed, bounded
handoff when a technical boundary requires it. `activate.py` preserves both role
configurations. No actual application PR is merged merely to test tool availability.

## Janus → Astra escalation in T3

Run `configure_janus_escalation.py ROOT` after other session renderers, install its
`dev.roman-akhs.janus-escalation.plist` in the user's LaunchAgents, and reload only
the Roman session controller at a quiet boundary. The broker binds loopback port
18438 and grants only Janus `escalate_to_astra` and `astra_escalation_status`.
Project IDs come from the live T3 orchestration snapshot; the model is the
installed `codex` provider's `gpt-6-astra` with high reasoning effort.

Janus escalates after two bounded unsuccessful attempts or an urgent blocker
outside its tools. A stable incident key owns one thread and one initial turn.
The broker commits the exact T3 commands to private SQLite before delivery,
reuses command IDs after lost replies/restarts, and retries delivery every ten
seconds while T3 is unavailable. Accepted/queued does not mean running. Four new
incidents per rolling day limit repeated escalation; identical retries do not
consume that quota. Janus reads the resulting assistant replies with the status
tool, records the receipt and yields that incident to Astra while doing other
work. New facts for an existing incident do not automatically start another turn.

T3 authentication is issued locally by the installed CLI and retained in process
memory. Janus receives neither T3 credentials nor filesystem access. The broker
uses T3 HTTP orchestration APIs and never edits T3's database. Existing project
ownership, exact-head review/CI and human gates remain in the Astra brief.
Legacy lane labels and zero witnesses are not sufficient stall evidence: inspect
central turns. Completed central turns with retained Anima ownership are a
separate reconciliation blocker and are appropriate escalation evidence.

Validation: `python3 -m unittest discover -s deploy/roman-mini -p test_janus_escalation.py`.
Tests cover restart after a lost turn-admission reply, one-turn deduplication,
offline durable admission, project/key validation and the escalation storm limit.
