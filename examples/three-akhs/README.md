# Three-Akh starter (Tecton · Thoth · Gemma)

Starter config for running three akhs that share one Postgres backing store, each
with a different chat/agentic backing (ADR 070):

- **Tecton** — GLM over `ollama.com` cloud (structure/engineering persona)
- **Thoth** — best reasoning model over `ollama.com` cloud (careful reasoner)
- **Gemma** — Gemma 4 12B in-process via **Candle** (fully local, GPU-ideal)

## Layout

```
config.toml                 → ~/.config/akh-medu/config.toml        (host-wide: [postgres], [acp])
workspaces/tecton.toml      → ~/.config/akh-medu/workspaces/tecton.toml
workspaces/thoth.toml       → ~/.config/akh-medu/workspaces/thoth.toml
workspaces/gemma.toml       → ~/.config/akh-medu/workspaces/gemma.toml
```

## Quick start

```sh
scripts/pg-standalone.sh up                         # standalone CloudNativePG
cp examples/three-akhs/config.toml ~/.config/akh-medu/config.toml
cp examples/three-akhs/workspaces/*.toml ~/.config/akh-medu/workspaces/
export OLLAMA_API_KEY=sk-…                           # for the cloud akhs
cargo build --release --features "server,candle-backend"
OLLAMA_API_KEY=$OLLAMA_API_KEY ./target/release/akhomed &
akh -w thoth chat
```

Full walkthrough (secrets, the Gemma GGUF, the snapshot/sleep/relocate workflow,
and going multi-host over SSL to Archibald/Hephaistos): see
[`docs/ops/three-akhs-postgres.md`](../../docs/ops/three-akhs-postgres.md).
Bring up + validate Postgres with [`scripts/pg-standalone.sh`](../../scripts/pg-standalone.sh)
and [`scripts/run-pg-tests.sh`](../../scripts/run-pg-tests.sh).
