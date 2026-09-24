# Akh-Medu Agent Guide

This document covers the six Ennead agents, their psyche profiles, and the
LoRA fine-tuning pipeline that encodes each personality into model weights.

---

## Hard Rules (persist across sessions)

- **No psql, ever.** Never query the Anima Postgres directly (no `psql`, no
  `kubectl exec ... -- psql`). All tracker access goes through the Anima MCP
  server.
- **Anima MCP bootstrap.** The MCP tools (`mcp__anima__*`) may be deferred.
  If absent, self-bootstrap: mint a header with
  `/Users/toasty/.local/bin/anima-pat get-auth-header` and POST JSON-RPC
  `tools/call` to `https://anima.wegmueller.it/mcp` with
  `Accept: application/json, text/event-stream`. This works from any harness
  (opencode, claude, shell) — if one lacks access or tooling, check another's
  config (e.g. `~/.claude.json` mcpServers) and copy the pattern.
- **The runner token in `~/.config/anima-runner/gateway.env` is NOT an MCP
  PAT** — it lacks `workitem.read`; don't use it for tracker calls.
- **Deploy only from `main`.** Every binary or image that runs an akh or a
  platform daemon (akhomed, akh CLI, akhrtd/akhsessd/akhexecd/akheventd/akhsbxd,
  container images, Mac LaunchDaemons, cluster) is built from a commit on
  `origin/main` and carries its short SHA. Never deploy from a feature, `wip/*`,
  `codex/*` or worktree branch; merge the PR first. Full rule:
  `.claude/rules/deploy-from-main.md`.
- **Every akh is fully featured (ADR 113).** Never deploy a stripped akh (no
  daemon loop, no in-process turn, reduced tool catalog). Side-effecting tools from
  any caller route through `akhsbxd`/`akhexecd`.

---

## The Six Ennead Agents

| Agent  | Deity  | Primary    | Secondary | Domain                                |
|--------|--------|------------|-----------|---------------------------------------|
| Tecton | Ptah   | Sage       | Creator   | Systems design, ontology, architecture |
| Thoth  | Thoth  | Sage       | Explorer  | Research, inference, epistemology     |
| Horus  | Horus  | Hero       | Guardian  | Task execution, security, coordination |
| Maat   | Maat   | Guardian   | Healer    | Ethics, balance, conflict resolution  |
| Set    | Set    | Rebel      | Explorer  | Adversarial testing, innovation       |
| Khnum  | Khnum  | Creator    | Sage      | Code generation, prototyping          |

Each agent has a `Psyche` (defined in `src/compartment/psyche.rs`) that biases
tool selection, shapes communication style, and constrains shadow behaviors.
The LoRA adapter encodes these same personality dimensions into model weights so
that the base Gemma model responds in the agent's voice without runtime prompting.

---

## LoRA Fine-Tuning Pipeline

### Architecture

```
seeds.jsonl (Rust QaGenerator)
      │
      ▼
generate_qa_pairs.py  ──── teacher LLM (Ollama) ────▶  expanded.jsonl
      │                                                       │
      └─────────────────────────────────────────────────────┘
                              │
                              ▼
                      train_lora.py
                    (MLX or llama.cpp)
                              │
                              ▼
              adapters/{agent}/adapter.safetensors
              adapters/{agent}/meta.json
                              │
                              ▼
                    benchmark_adapter.py
                    (keyword F1 + load time)
                              │
                              ▼
              adapters/{agent}/benchmark.json
```

### Quality bar (from `scripts/lora_config.json`)

| Metric                 | Target      |
|------------------------|-------------|
| Eval accuracy (F1≥0.5) | ≥ 60%       |
| Adapter size           | 10–50 MB    |
| Loading overhead       | < 100 ms    |
| Training pairs         | ≥ 500       |
| Training time          | < 2 h/agent |

### Step-by-step

#### 1. Export seed QA pairs (Rust)

```bash
# Writes <data-dir>/qa/<agent>/seeds.jsonl for all agents.
# 6–10 hand-crafted persona-consistent pairs per agent.
cargo run --bin akh -- training export-seeds --all --output ~/.akh-medu/qa
```

#### 2. Expand to 500+ pairs with a teacher model

```bash
# Requires Ollama running locally.
python scripts/generate_qa_pairs.py --all --teacher ollama --model llama3.2

# Without a teacher (seeds only):
python scripts/generate_qa_pairs.py --all --teacher file
```

#### 3. Train LoRA adapters

```bash
# Apple Silicon (recommended):
python scripts/train_lora.py --all --backend mlx --data-dir ~/.akh-medu

# Single agent:
python scripts/train_lora.py --agent thoth --backend mlx

# llama.cpp (cross-platform fallback):
python scripts/train_lora.py --agent thoth --backend llamacpp \
  --llamacpp-dir /opt/llama.cpp/bin

# Resume interrupted training:
python scripts/train_lora.py --agent set --backend mlx --resume
```

#### 4. Benchmark adapter quality

```bash
# Keyword F1 only (no model needed):
python scripts/benchmark_adapter.py --all

# With Ollama inference (realistic, but requires adapter loaded as an Ollama model):
python scripts/benchmark_adapter.py --agent thoth --backend ollama \
  --adapter-model thoth-lora

# Loading overhead only:
python scripts/benchmark_adapter.py --agent tecton --overhead-only
```

Results are saved to `~/.akh-medu/adapters/<agent>/benchmark.json`.

#### 5. Load adapters in the Rust engine

```rust
use akh_medu::training::{AgentId, LoraRegistry};

let registry = LoraRegistry::new(data_dir.join("adapters"));

// Activate an adapter for an agent (< 100 ms):
let adapter = registry.activate(AgentId::Thoth)?;
println!("Loaded {} bytes", adapter.size_bytes());

// Hot-swap to a different agent:
let adapter = registry.activate(AgentId::Maat)?;

// Release cached adapter:
registry.deactivate();
```

---

## File Layout

```
~/.akh-medu/
  qa/
    tecton/
      seeds.jsonl       ← Rust QaGenerator output
      expanded.jsonl    ← teacher-expanded pairs
    thoth/ ...
  adapters/
    tecton/
      train.jsonl       ← training split (90%)
      eval.jsonl        ← eval split (10%)
      adapter.safetensors
      meta.json         ← LoraAdapterMeta (Rust-readable)
      benchmark.json    ← benchmark results
    thoth/ ...
  models/
    llm/
      gemma2-2b-instruct-q4_K_M.gguf
```

---

## Rust API Reference

### `training::lora`

| Type              | Description                                         |
|-------------------|-----------------------------------------------------|
| `AgentId`         | Enum of 6 agents; `as_str()`, `archetypes()`, `all()` |
| `LoraConfig`      | Hyperparameters (rank, alpha, target_modules, …)    |
| `LoraAdapterMeta` | Sidecar metadata written by training pipeline       |
| `LoadedAdapter`   | Loaded bytes + metadata; `agent_id()`, `size_bytes()` |
| `LoraRegistry`    | `activate()`, `deactivate()`, `available_agents()`  |

### `training::qa_generator`

| Type                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `QaCategory`          | Identity, Domain, Ethics, Style, Collaboration   |
| `QaPair`              | `{input, output, agent, category}`               |
| `AgentPersonaProfile` | Static persona data for each agent               |
| `QaGenerator`         | `seed_pairs()`, `write_seeds()`, `write_all_seeds()` |

### `training::benchmark`

| Type                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `BenchmarkConfig`     | `max_pairs`, `accuracy_threshold`, `stop_words`  |
| `PairResult`          | Per-pair precision/recall/F1/correct             |
| `AgentBenchmarkResult`| Aggregate accuracy, mean F1, `meets_bar`         |
| `AdapterBenchmark`    | `load_eval_set()`, `evaluate()`                  |

---

## Development Notes

- **Burn LoRA blocked**: Burn 0.21 has a libsqlite3-sys conflict with Seshat.
  The Python scripts use MLX (preferred) or llama.cpp until Burn stabilises.
  See `Cargo.toml` `burn-training` feature for the deferred wiring.

- **Adapter format**: `.safetensors` is the canonical output format.
  The `LoraRegistry` reads raw bytes; the candle/llama-cpp backends
  deserialize into their own tensor types.

- **Hot-swap guarantee**: `LoraRegistry::activate()` returns an error if
  loading takes > 100 ms.  50 MB at SSD speeds (~3 GB/s) ≈ 17 ms, well
  within budget.  If you hit `LoraError::SlowLoad`, the adapter is likely
  stored on a slow device or is larger than expected.

- **Adding a new agent**: add a variant to `AgentId`, add seeds in
  `qa_generator.rs`, add a profile to `PROFILES`, add an entry in
  `AGENTS` in the Python scripts, and increment `AgentId::all()`.
