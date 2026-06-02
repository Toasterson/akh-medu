# Deployment and Awakening

This chapter covers the full deployment pipeline: building the binaries,
running `akhomed` as a system service, initializing a workspace, awakening
the agent's identity, and having your first conversation.

## Overview

akh-medu ships two binaries:

| Binary | Purpose | Feature Flags |
|--------|---------|---------------|
| `akhomed` | Multi-workspace daemon (REST + WebSocket + background learning) | `server`, optionally `nlu-full` |
| `akh` | CLI client (thin client talks to `akhomed`, or standalone with embedded engine) | `client-only` for thin client, none for standalone |

The recommended setup is **akhomed as a system service** with the **thin
client** `akh`. This gives you background learning, multi-workspace support,
and a lightweight CLI.

## Building the Binaries

### Prerequisites

| Component | Minimum Version | Notes |
|-----------|----------------|-------|
| Rust toolchain | Edition 2024 | `rustup update` |
| C compiler | gcc or clang | Required by some dependencies |
| cmake | 3.x | Required for `nlu-full` (ONNX runtime) |

### Build commands

```bash
# Server with full NLU pipeline (recommended)
cargo build --release --features server,nlu-full --bin akhomed

# Thin client (delegates to akhomed)
cargo build --release --features client-only --bin akh

# Install to ~/.local/bin/
cp target/release/akhomed ~/.local/bin/
cp target/release/akh ~/.local/bin/
```

If `nlu-full` fails due to missing cmake or C++ compiler, fall back to
server-only (grammar-based NLU still works):

```bash
cargo build --release --features server --bin akhomed
```

### Feature flag reference

| Feature | What It Adds | Build Dependencies |
|---------|--------------|-------------------|
| `server` | REST + WebSocket daemon, tokio runtime | None extra |
| `client-only` | Thin client (strips embedded engine) | None extra |
| `nlu-ml` | ONNX-based NER model | cmake, C++ compiler |
| `nlu-llm` | Local LLM translator (Qwen2.5-1.5B) | ONNX runtime |
| `nlu-full` | Both `nlu-ml` and `nlu-llm` | cmake, C++ compiler, ONNX runtime |
| `wasm-tools` | Wasmtime for WASM-based agent tools | None extra |

### NLU model paths

When built with `nlu-ml` or `nlu-full`, the NLU pipeline looks for ONNX
models in the workspace's data directory:

```
~/.local/share/akh-medu/workspaces/{name}/kg/
    ner_model.onnx          # Named entity recognition model
    ner_tokenizer.json      # Tokenizer config
```

Models are optional -- the NLU pipeline gracefully degrades to grammar-only
parsing if models are missing.

## Deploying with systemd (Linux)

### Install the service

```bash
# Copy the reference unit from the repo
mkdir -p ~/.config/systemd/user
cp contrib/systemd/akhomed.service ~/.config/systemd/user/

# Reload systemd, enable auto-start, and start now
systemctl --user daemon-reload
systemctl --user enable --now akhomed
```

### Verify

```bash
# Check service status
systemctl --user status akhomed

# Health check
curl -s http://127.0.0.1:8200/health
# {"status":"ok","version":"0.5.4","workspaces_loaded":2}
```

### View logs

```bash
journalctl --user -u akhomed -f
```

### The unit file

The reference unit at `contrib/systemd/akhomed.service` binds to localhost,
enables systemd hardening (read-only home, no new privileges, private /tmp),
and restarts on failure:

```ini
[Unit]
Description=akh-medu daemon — neuro-symbolic reasoning server
Documentation=https://akh-medu.dev
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=%h/.local/bin/akhomed
Restart=on-failure
RestartSec=5

Environment=RUST_LOG=info,egg=warn,hnsw_rs=warn
Environment=AKH_SERVER_BIND=127.0.0.1
Environment=AKH_SERVER_PORT=8200

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/.local/share/akh-medu %h/.config/akh-medu %h/.cache/akh-medu
PrivateTmp=true

[Install]
WantedBy=default.target
```

## Deploying with launchd (macOS)

On macOS, use the built-in service management command:

```bash
akh service install     # Creates ~/Library/LaunchAgents/dev.akh-medu.akhomed.plist
akh service start       # launchctl load
akh service stop        # launchctl unload
akh service uninstall   # Removes the plist
```

Pass `--port 9000` to `akh service install` to override the default port.

## Initializing a Workspace

With `akhomed` running, create a workspace and seed it with foundational
knowledge:

```bash
# Create the default workspace
akh init

# Or a named workspace
akh -w my-project init

# Apply seed packs (foundational knowledge)
akh seed apply ontology       # Core relations (is-a, has-part, causes, ...)
akh seed apply common-sense   # Animals, materials, spatial/temporal concepts
akh seed apply identity       # Who akh-medu is

# Verify
akh seed status
```

Seeds are idempotent -- applying the same seed twice is a no-op.

## The Awakening Pipeline

The `akh awaken bootstrap` command runs an 8-stage pipeline that transforms
a purpose statement into a fully awakened agent with identity, knowledge,
and competence:

```
 Purpose Statement
       │
       ▼
 ┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────┐
 │ 1. Parse     │ --> │ 2. Resolve│ --> │ 3. Expand │ --> │ 4. Prereq    │
 │ purpose      │     │ identity  │     │ ontology  │     │ discovery    │
 └─────────────┘     └──────────┘     └──────────┘     └──────────────┘
                                                              │
 ┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌─────▼────────┐
 │ 8. Assess    │ <-- │ 7. Learn  │ <-- │ 6. Discover│ <-- │ 5. ZPD       │
 │ competence   │     │ (ingest)  │     │ resources  │     │ classify     │
 └─────────────┘     └──────────┘     └──────────┘     └──────────────┘
       │
       ▼
 Awakened Agent (psyche, knowledge, competence)
```

### Running bootstrap

```bash
akh awaken bootstrap "You are a programming assistant expert in Rust and systems design based on Thoth"
```

**Expected output:**

```
Bootstrap Result
================
  Domain:          Rust and systems design based on Thoth
  Target level:    expert
  Chosen name:     Merhu
  Learning cycles: 9
  Target reached:  true
  Final Dreyfus:   expert
  Final score:     0.81
  Recommendation:  ready
```

The bootstrap pipeline:

1. **Parse** -- Extracts domain ("Rust and systems design"), target level
   ("expert"), and identity reference ("Thoth") from natural language.
2. **Resolve** -- Maps "Thoth" to a known archetype (Egyptian god of wisdom),
   generates a persona name, traits, and Jungian psyche configuration.
3. **Expand** -- Queries ConceptNet and Wikidata for domain concepts, building
   a skeleton ontology of the target domain.
4. **Prerequisite** -- Discovers prerequisite relationships between concepts
   (e.g., "ownership" requires "memory management").
5. **ZPD classify** -- Classifies each concept by Vygotsky's Zone of Proximal
   Development: mastered, proximal (learnable now), or distant.
6. **Resources** -- Discovers learning resources (URLs, papers) for
   ZPD-proximal concepts.
7. **Ingest** -- Fetches and ingests discovered resources in curriculum order.
   Runs multiple learning cycles until the target competence is reached.
8. **Assess** -- Evaluates competence on the Dreyfus scale (novice through
   expert). If below target, more learning cycles run.

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--plan-only` | Parse and show the plan without executing | off |
| `--resume` | Resume an interrupted bootstrap session | off |
| `--max-cycles <N>` | Maximum learning cycles | `10` |
| `--identity <NAME>` | Override the identity reference | (from statement) |
| `--status` | Show current bootstrap session status | off |

## Running Individual Stages

For debugging or manual control, run each stage separately:

```bash
# Parse a purpose statement
akh awaken parse "You are a Rust expert based on Thoth"

# Resolve identity (maps archetype, generates psyche)
akh awaken resolve

# Expand domain ontology via external knowledge sources
akh awaken expand

# Discover prerequisites
akh awaken prerequisite

# Discover learning resources
akh awaken resources

# Ingest resources in curriculum order
akh awaken ingest

# Assess competence
akh awaken assess

# Check current status at any point
akh awaken status
```

## Your First Conversation

After awakening, connect to the agent via the TUI:

```bash
akh chat
```

The TUI connects to `akhomed` via WebSocket. The agent's psyche (generated
during awakening) shapes its communication style -- a Thoth-based agent will
be wise, analytical, and methodical.

TUI commands (prefix with `/`):

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/grammar narrative` | Switch to narrative output style |
| `/grammar formal` | Switch to structured/academic style |
| `/grammar terse` | Minimal, facts-only style |
| `/workspace` | Show workspace info |
| `/goals` | List active goals |
| `/quit` | Exit |

Type natural language to set goals or ask questions. The agent runs OODA
cycles in the background and synthesizes findings using the active grammar.

## Getting the Agent to Write Code

The `code_gen` tool generates Rust source files using template-based
generation informed by the knowledge graph:

```bash
# From the TUI, set a goal:
> Write a hello world program in Rust

# Or use the agent directly:
akh agent run --goals "Generate a hello world Rust program" --max-cycles 5
```

The generated code is written to the workspace scratch directory:

```
~/.local/share/akh-medu/workspaces/default/scratch/
```

Code generation uses templates stored in `data/templates/` and enriches them
with type information from the knowledge graph.

## NLU Tiers

The NLU pipeline has four tiers that activate based on available features
and models. Each tier is a superset of the previous:

| Tier | Feature Flag | What It Does | Fallback |
|------|-------------|--------------|----------|
| **1. Grammar** | (always) | Rule-based relational pattern matching, 5 languages | None (base) |
| **2. ML NER** | `nlu-ml` | ONNX named entity recognition for better entity extraction | Tier 1 |
| **3. Local LLM** | `nlu-llm` | Qwen2.5-1.5B translates ambiguous input to structured form | Tier 2 |
| **4. VSA Ranker** | (always with tiers 2-3) | Re-ranks NLU candidates using VSA similarity | Tier 1 |

### Model download

Models are not bundled with the binary. Download them separately:

```bash
# NER model (requires nlu-ml feature)
# Place in: ~/.local/share/akh-medu/models/ner/
mkdir -p ~/.local/share/akh-medu/models/ner
# (Download your ONNX NER model here)

# LLM model (requires nlu-llm feature)
# Place in: ~/.local/share/akh-medu/models/llm/
mkdir -p ~/.local/share/akh-medu/models/llm
# (Download Qwen2.5-1.5B ONNX model here)
```

The pipeline auto-detects available models at startup and selects the
highest available tier.

## Troubleshooting

### Stale PID file

**Symptom:** `akh` reports "no akhomed server found" even though `akhomed`
is running.

**Fix:** Remove the stale PID file:

```bash
rm ~/.local/state/akh-medu/akhomed.pid
systemctl --user restart akhomed
```

### Health check timeout

**Symptom:** `akh` commands fail with "health check failed" or "timed out
reading response".

**Cause:** The server is busy with daemon tasks (learning cycles,
consolidation). The client health check has a 10-second timeout.

**Fix:** Wait a few seconds and retry. If persistent, check server logs:

```bash
journalctl --user -u akhomed --since "5 min ago"
```

### NLU models missing

**Symptom:** NLU falls back to grammar-only parsing despite building with
`nlu-full`.

**Fix:** Ensure ONNX model files are in the expected paths. Check the
startup log for model loading messages:

```bash
journalctl --user -u akhomed | grep -i "nlu\|model\|onnx"
```

### ConceptNet unavailable

**Symptom:** `akh awaken expand` produces few concepts or errors.

**Cause:** The ConceptNet API (`api.conceptnet.io`) is rate-limited or down.

**Fix:** The expansion stage is best-effort -- it falls back to local
ontology triples when external APIs are unavailable. You can retry later:

```bash
akh awaken expand
```

### Bootstrap interrupted

**Symptom:** Bootstrap was interrupted (Ctrl+C, network failure, crash).

**Fix:** Resume from where it left off:

```bash
akh awaken bootstrap --resume
```

Or check current progress and run individual stages:

```bash
akh awaken bootstrap --status
akh awaken ingest     # Re-run just the ingestion stage
akh awaken assess     # Re-assess competence
```
