# Installation

## Prerequisites

- **Rust toolchain**: Edition 2024 (`rustup update`)
- **Platform**: Linux, macOS, or Windows (WSL recommended)
- **RAM**: 512 MB minimum, 2 GB recommended for large knowledge graphs

## Build from Source

```bash
git clone https://github.com/Toasterson/akh-medu.git
cd akh-medu

# Full server with NLU (recommended)
cargo build --release --features server,nlu-full --bin akhomed

# Thin client (talks to akhomed over HTTP/WS)
cargo build --release --features client-only --bin akh

# Both binaries land in target/release/
```

## Feature Flags

| Feature | Flag | What It Adds |
|---------|------|--------------|
| **Server** | `--features server` | REST + WebSocket daemon binary (`akhomed`) |
| **Client-only** | `--features client-only` | Thin CLI that delegates to a running `akhomed` |
| **NLU (ML)** | `--features nlu-ml` | ONNX-based NER model for natural language understanding |
| **NLU (LLM)** | `--features nlu-llm` | Local LLM translator (Qwen2.5-1.5B) |
| **NLU (Full)** | `--features nlu-full` | Both `nlu-ml` and `nlu-llm` |
| **WASM Tools** | `--features wasm-tools` | Wasmtime runtime for WASM-based agent tools |

```bash
# Build server with all NLU tiers
cargo build --release --features server,nlu-full --bin akhomed

# Build thin client
cargo build --release --features client-only --bin akh

# Build standalone CLI (embedded engine, no server needed)
cargo build --release --bin akh
```

## Binary Targets

| Binary | Path | Feature Gate | Description |
|--------|------|--------------|-------------|
| `akh` | `src/main.rs` | None (always built) | CLI with embedded engine |
| `akh` | `src/main.rs` | `client-only` | Thin client (requires running `akhomed`) |
| `akhomed` | `src/bin/akhomed.rs` | `server` | Multi-workspace daemon |

## Initialize a Workspace

akh-medu uses XDG directory conventions for data, config, and state:

```bash
# Create the default workspace
akh init

# Create a named workspace
akh -w my-project init
```

This creates:

```
~/.config/akh-medu/
    workspaces/default.toml       # workspace config

~/.local/share/akh-medu/
    workspaces/default/
        kg/                       # knowledge graph data
        skills/                   # activated skill packs
        compartments/             # knowledge compartments
        scratch/                  # agent scratch space

~/.local/state/akh-medu/
    sessions/default.bin          # agent session state
```

## Verify Installation

```bash
# Show engine info (in-memory, no persistence)
akh info

# Show engine info with persistence
akh -w default info
```

## Run Tests

```bash
cargo test --lib
```
