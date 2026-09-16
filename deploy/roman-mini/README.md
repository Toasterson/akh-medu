# Roman mini teammates

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
the dedicated core, isolated XDG directories and a versioned binary path. Neither
script starts launchd or invokes a model. The launchd definitions live under the
deployment root until installed into `/Library/LaunchDaemons`.

`awakening.json` defines Roman archetypes. The offline `roman_awaken` example runs
the actual `ritual_of_awakening`, publishes a fenced SeaORM snapshot, then recovers
the awakened psyche into a different cache. It refuses to replace an existing
durable identity. The ritual may choose a Latin self-name different from the
archetype; the service IDs remain janus/minerva/vesta.

Build the core and helper with:

```sh
cargo build --locked --release --no-default-features --features core-server --bin akhomed --example roman_awaken
```

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
preserved. Sokar's artifact backend remained unavailable during activation;
consult current evidence before claiming artifact writes healthy. Internal notes
remain available. A known model output-limit
stop fails its turn without quarantining unrelated jobs; uncertain tool outcomes
retain reconciliation gates.

The provider profiles use Ollama Cloud and OpenRouter's chat-completions APIs.
Routine Ollama wakes request `reasoning_effort: none`, use an 8192-token output
budget and a 420-second turn deadline. This uses Ollama's documented
[reasoning control](https://docs.ollama.com/api/openai-compatibility).
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

Live checks: all three Akhs completed Ollama turns, called Anima/Board/Seshat,
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
