# Workspaces

Workspaces isolate engine instances with separate data, configuration, and
agent sessions. Each workspace has its own knowledge graph, symbols, skills,
and compartments.

## XDG Directory Layout

akh-medu follows XDG Base Directory conventions:

```
~/.config/akh-medu/                     # XDG_CONFIG_HOME
    config.toml                         # global config
    workspaces/
        default.toml                    # per-workspace config
        project-alpha.toml

~/.local/share/akh-medu/                # XDG_DATA_HOME
    workspaces/
        default/
            kg/                         # oxigraph, redb, hnsw data
            skills/                     # activated skill data
            compartments/               # compartment data
            scratch/                    # agent scratch space
        project-alpha/
            kg/
            ...
    seeds/                              # installed seed packs

~/.local/state/akh-medu/                # XDG_STATE_HOME
    sessions/
        default.bin                     # agent session state
        project-alpha.bin
```

Override any path via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `XDG_CONFIG_HOME` | `~/.config` | Configuration files |
| `XDG_DATA_HOME` | `~/.local/share` | Persistent data |
| `XDG_STATE_HOME` | `~/.local/state` | Session state |

## Workspace Configuration

Each workspace has a TOML config file:

```toml
# ~/.config/akh-medu/workspaces/default.toml
name = "default"
dimension = 10000                       # hypervector dimension
encoding = "bipolar"                    # encoding scheme
language = "auto"                       # default grammar language
max_memory_mb = 1024                    # hot-tier memory budget
max_symbols = 1000000                   # symbol registry limit
seed_packs = ["identity", "ontology", "common-sense"]  # auto-applied on init
shared_partitions = []                  # mounted shared partitions
```

## Managing Workspaces

### CLI

```bash
# List all workspaces
akh-medu workspace list

# Create a new workspace
akh-medu workspace create my-project

# Show workspace info
akh-medu workspace info default

# Delete a workspace (removes all data)
akh-medu workspace delete my-project
```

### Using a Specific Workspace

Pass `-w` or `--workspace` to any command:

```bash
# Initialize a named workspace
akh-medu -w my-project init

# Query in a specific workspace
akh-medu -w my-project query --seeds "Dog" --depth 2

# Run agent in a workspace
akh-medu -w my-project agent run --goals "..." --max-cycles 10
```

### REST API

The [server](../server/overview.md) manages multiple workspaces simultaneously:

```bash
# List workspaces
curl http://localhost:8200/workspaces

# Create workspace
curl -X POST http://localhost:8200/workspaces/my-project

# Delete workspace
curl -X DELETE http://localhost:8200/workspaces/my-project

# Workspace status
curl http://localhost:8200/workspaces/default/status
```

## Workspace Manager

The `WorkspaceManager` handles CRUD operations:

```rust
use akh_medu::workspace::WorkspaceManager;

let manager = WorkspaceManager::new(paths);

// Create a workspace
let ws_paths = manager.create(config)?;

// List all
let names = manager.list();

// Get config
let config = manager.info("default")?;

// Resolve paths
let paths = manager.resolve("default")?;

// Delete
manager.delete("my-project")?;
```

## Auto-Seeding

When a workspace is initialized, the seed packs listed in its config are
applied automatically:

```toml
seed_packs = ["identity", "ontology", "common-sense"]
```

This bootstraps the workspace with foundational knowledge on first creation.
See [Seed Packs](../getting-started/seed-packs.md) for details.

## Shared Partitions

Workspaces can mount [shared partitions](partitions.md) to access knowledge
that lives outside any single workspace:

```toml
shared_partitions = ["shared-ontology", "company-knowledge"]
```

Shared partitions are read-write named graphs stored independently of any
workspace, making them accessible from multiple workspaces simultaneously.
