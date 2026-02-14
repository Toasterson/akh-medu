# akh-medu Server

Multi-workspace server hosting N engine instances with REST and WebSocket APIs.

## Building

```bash
cargo build --release --features server --bin akh-medu-server
```

## Running

```bash
# Default: 0.0.0.0:8200
akh-medu-server

# Custom bind/port via env vars
AKH_SERVER_BIND=127.0.0.1 AKH_SERVER_PORT=9000 akh-medu-server
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AKH_SERVER_BIND` | `0.0.0.0` | Bind address |
| `AKH_SERVER_PORT` | `8200` | Listen port |
| `RUST_LOG` | `info` | Log level filter |
| `XDG_DATA_HOME` | `~/.local/share` | XDG data directory |
| `XDG_CONFIG_HOME` | `~/.config` | XDG config directory |
| `XDG_STATE_HOME` | `~/.local/state` | XDG state directory |

## Directory Layout

```
~/.config/akh-medu/
    config.toml                 # global config
    workspaces/
        default.toml            # per-workspace config

~/.local/share/akh-medu/
    workspaces/
        default/
            kg/                 # oxigraph, redb, hnsw
            skills/             # activated skill data
            compartments/       # compartment data
            scratch/            # agent scratch space
        project-alpha/
            kg/
            ...
    seeds/                      # installed seed packs

~/.local/state/akh-medu/
    sessions/
        default.bin             # agent session state
```

## REST API

### Health

```bash
curl http://localhost:8200/health
```

Response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "workspaces_loaded": 2
}
```

### List Workspaces

```bash
curl http://localhost:8200/workspaces
```

Response:
```json
{
  "workspaces": ["default", "project-alpha"]
}
```

### Create Workspace

```bash
curl -X POST http://localhost:8200/workspaces/my-project
```

Response:
```json
{
  "name": "my-project",
  "created": true
}
```

### Delete Workspace

```bash
curl -X DELETE http://localhost:8200/workspaces/my-project
```

Response:
```json
{
  "deleted": "my-project"
}
```

### Workspace Status

```bash
curl http://localhost:8200/workspaces/default/status
```

Response:
```json
{
  "name": "default",
  "symbols": 142,
  "triples": 89
}
```

### Apply Seed Pack

```bash
curl -X POST http://localhost:8200/workspaces/default/seed/identity
```

Response:
```json
{
  "pack": "identity",
  "triples_applied": 18,
  "already_applied": false
}
```

### Preprocess Text

```bash
curl -X POST http://localhost:8200/workspaces/default/preprocess \
  -H 'Content-Type: application/json' \
  -d '{"chunks": [{"id": "1", "text": "The Sun is a star."}]}'
```

### List Equivalences

```bash
curl http://localhost:8200/workspaces/default/equivalences
```

### Equivalence Stats

```bash
curl http://localhost:8200/workspaces/default/equivalences/stats
```

## WebSocket Protocol

Connect to `ws://localhost:8200/ws/{workspace}` for a streaming TUI session.

### Client Messages

**Input (natural language):**
```json
{
  "type": "input",
  "text": "What is the Sun?"
}
```

**Command:**
```json
{
  "type": "command",
  "text": "status"
}
```

Available commands: `status`, `goals`.

### Server Messages

The server streams `AkhMessage` JSON objects back. Each message has a `type` field:

```json
{"type": "fact", "text": "Sun is-a Star", "confidence": 0.95, "provenance": null}
{"type": "system", "text": "Connected to workspace \"default\"."}
{"type": "tool_result", "tool": "kg_query", "success": true, "output": "Found 3 triples."}
{"type": "goal_progress", "goal": "Explore Sun", "status": "Active", "detail": null}
{"type": "error", "code": "ws", "message": "workspace not found", "help": null}
```

## Systemd Example

```ini
[Unit]
Description=akh-medu Knowledge Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/akh-medu-server
Environment=AKH_SERVER_BIND=127.0.0.1
Environment=AKH_SERVER_PORT=8200
Environment=RUST_LOG=info,egg=warn
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
