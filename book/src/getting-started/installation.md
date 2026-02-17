# Installation

## Prerequisites

- **Rust toolchain**: Edition 2024 (`rustup update`)
- **Platform**: Linux, macOS, or Windows (WSL recommended)
- **RAM**: 512 MB minimum, 2 GB recommended for large knowledge graphs

## Build from Source

```bash
git clone https://github.com/Toasterson/akh-medu.git
cd akh-medu

# Core CLI binary
cargo build --release

# The binary is at target/release/akh-medu
```

## Feature Flags

| Feature | Flag | What It Adds |
|---------|------|--------------|
| **Server** | `--features server` | REST + WebSocket server binary (`akh-medu-server`) |
| **WASM Tools** | `--features wasm-tools` | Wasmtime runtime for WASM-based agent tools |

```bash
# Build with server support
cargo build --release --features server

# Build with everything
cargo build --release --features "server wasm-tools"
```

## Binary Targets

| Binary | Path | Feature Gate |
|--------|------|--------------|
| `akh-medu` | `src/main.rs` | None (always built) |
| `akh-medu-server` | `src/bin/akh-medu-server.rs` | `server` |

## Initialize a Workspace

akh-medu uses XDG directory conventions for data, config, and state:

```bash
# Create the default workspace
akh-medu init

# Create a named workspace
akh-medu -w my-project init
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
akh-medu info

# Show engine info with persistence
akh-medu -w default info
```

## Run Tests

```bash
cargo test --lib
```
