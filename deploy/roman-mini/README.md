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

Provider credentials, project-system grants, teammate handoffs, scheduled
admission and alert ingestion must be connected and verified before enabling
the three automations. They are disabled in the template. One runtime serializes
provider turns; urgent incident admission and an independent outage watchdog
still need to be connected before unattended operation.

The provider profiles use Ollama Cloud and OpenRouter's chat-completions APIs.
Private provider keys belong at `secrets/ollama.key` and
`secrets/openrouter.key` under the deployment root, mode 0600. Do not paste keys
into source, chat, logs or process arguments. The preparer never creates fake
provider credentials. OpenRouter is an alternate profile, not an automatic
fallback. Model availability and tool use require authenticated verification.

Local durable storage preserves state during outages. Cloud inference still
requires connectivity.

Coordination is recorded using the
[Agent Board coordination skill](/Users/toasty/.codex/skills/agent-board-coordination/SKILL.md)
under this Codex thread's own registered identity.
