# Runbook — Three Akhs on a shared Postgres (Tecton · Thoth · Gemma)

A worked, end-to-end setup that exercises the **whole Phase-38 Postgres
workflow** — durable truth, WAL, snapshot/sleep/wake, relocation, the
split-brain guard — with three akhs that each use a *different* chat/agentic
backing (ADR 070):

| Akh      | Backing                | Model                              | Host needs            |
|----------|------------------------|------------------------------------|-----------------------|
| Tecton   | Ollama **cloud**       | GLM (user: GLM 5.2) `glm-…:cloud`  | network + cloud key   |
| Thoth    | Ollama **cloud**       | best reasoning `gpt-oss:120b-cloud`| network + cloud key   |
| Gemma    | **Candle** (in-proc)   | Gemma 4 12B GGUF                   | the GGUF (+GPU ideal) |

Truth for all three lives in one Postgres database; each daemon keeps an
in-process cache. Cloud-backed akhs (Tecton, Thoth) are freely relocatable;
the Candle-backed akh (Gemma) is pinned to a host that has the GGUF, but its KG
still relocates to any other Candle-capable host.

Starter config lives in [`examples/three-akhs/`](../../examples/three-akhs/).

---

## 0. Prerequisites

- Docker (for the standalone Postgres) and a Rust toolchain.
- An `ollama.com` API key for the cloud akhs.
- For Gemma: a Gemma 4 12B GGUF, and (optional but recommended) an AMD GPU +
  ROCm to build with `--features rocm`.

## 1. Bring up Postgres (standalone CloudNativePG)

```sh
scripts/pg-standalone.sh up      # pulls ghcr.io/cloudnative-pg/postgresql:16
# prints: AKH_TEST_PG_URL=postgres://akh:akh@127.0.0.1:5432/akhmedu?sslmode=disable
```

The homelab production cluster runs a custom CNPG image (CNPG + AGE + pgvector,
Anima WI `019dc614`); this standalone uses the upstream CNPG image. The only
PG-specific migration step (the TimescaleDB hypertable) and AGE are *guarded* —
they no-op when the extension is absent — so the relocation workflow is fully
exercised on the base image (AGE projection is Phase 38d, deferred).

## 2. Validate the workflow (optional but recommended)

Before wiring real models, prove the durable-truth machinery against this
Postgres:

```sh
AKH_KEEP_PG=1 scripts/run-pg-tests.sh
# ALL POSTGRES INTEGRATION TESTS PASSED
```

This runs `live_pg` (ensure_akh / WAL append / membership / advisory-lock
split-brain guard / **concurrent advisory-locked migrators**), `snapshot_live`
(**wake reconstructs an identical akh**), `wal_live`, and the `relocation`
property test (random WAL tail recovers exactly).

## 3. Install the configs

```sh
mkdir -p ~/.config/akh-medu/workspaces
cp examples/three-akhs/config.toml              ~/.config/akh-medu/config.toml
cp examples/three-akhs/workspaces/*.toml        ~/.config/akh-medu/workspaces/
```

Edit `~/.config/akh-medu/config.toml` if your Postgres DSN differs. Confirm the
cloud model tags in `tecton.toml` / `thoth.toml` against your account
(`ollama list` or <https://ollama.com/library>).

## 4. Secrets + the Gemma model

```sh
export OLLAMA_API_KEY=sk-…                       # resolved via env:OLLAMA_API_KEY
# Gemma: drop the GGUF where Candle discovers it
mkdir -p ~/.local/share/akh-medu/models/llm
cp gemma4-12b-it-q4_k_m.gguf ~/.local/share/akh-medu/models/llm/
export AKH_LLM_MODEL=gemma4                       # disambiguate if >1 GGUF present
```

## 5. Build + start the daemon

```sh
# Cloud akhs only need `server`; add `candle-backend` for Gemma, `rocm` for GPU.
cargo build --release --features "server,candle-backend"     # + ,rocm on the AMD box
DATABASE_URL=                                                  # (optional override)
OLLAMA_API_KEY=$OLLAMA_API_KEY ./target/release/akhomed
```

On start the daemon connects to Postgres and (because `migrate_on_start = true`)
applies migrations **under a Postgres advisory lock** — so you can start several
hosts at once against one fresh cluster DB without a `seaql_migrations` race.

## 6. Talk to each akh

`akh` is a pure ACP client; `-w` selects the akh:

```sh
akh -w tecton acp status        # symbol/triple counts, backing
akh -w thoth  chat              # TUI chat with the reasoning akh
akh -w gemma  chat --headless   # plain stdin/stdout with the local Candle akh
```

## 7. Exercise the durable-truth workflow

```sh
# ... teach an akh some facts via chat ...
akh -w tecton akh snapshot      # write relational snapshot to Postgres now
akh -w tecton akh sleep         # snapshot + flush + free memory (deep sleep)
akh -w tecton akh wake          # rebuild the in-memory cache from Postgres
akh -w tecton akh relocate      # prepare to move hosts (snapshot + flush)
akh migrate                     # run migrations on demand (advisory-locked)
akh projection refresh          # AGE read-model rebuild (Phase 38d — stub)
```

See [`relocate-akh.md`](relocate-akh.md) for graceful vs dead-host recovery and
the split-brain guard details.

---

## 8. Going multi-host (Archibald + Hephaistos over SSL)

Goal: Tecton and Thoth (cloud-backed, so portable) homed on **Archibald** and
**Hephaistos**, with truth in the cellar Postgres reached over SSL. Gemma stays
on whichever host has the GGUF + GPU.

1. **Expose Postgres over TLS.** On the cluster, the homelab CNPG already serves
   TLS; for the standalone, regenerate with server certs and require SSL. Each
   remote host's `~/.config/akh-medu/config.toml` then uses:
   ```toml
   [postgres]
   url = "postgres://akh@<cellar-host>:5432/akhmedu?sslmode=require"
   ```
   (Prefer client-cert auth or a per-host role; never ship the cellar superuser.)
2. **Same identity, different host.** Copy the akh's workspace TOML to the target
   host. Because truth is in Postgres, `akh -w tecton akh wake` on Archibald
   reconstructs the same Tecton.
3. **Split-brain is guarded.** Two hosts cannot own one akh: the `akh.last_seen`
   heartbeat + a session advisory lock (`pg_try_advisory_lock(hashtext('akh:'||
   akh_id))`) make `wake`/`relocate` refuse fast if another host is live. See the
   relocation runbook.
4. **Archibald = OpenIndiana VM host.** The akhomed daemon there runs inside an
   OpenIndiana guest. Build for illumos/amd64; only `libpq`/TLS + outbound
   network to the cellar Postgres and to `ollama.com` are required (Tecton/Thoth
   are cloud-backed, so **no GPU needed** on Archibald). Candle/`rocm` is only
   relevant on whichever host runs Gemma.

> Concurrent `migrate_on_start` across these hosts is safe — the advisory-locked
> migration runner serializes them. If you prefer, run `akh migrate` once from a
> single host before starting the rest.

## 9. Tuning + known limitations (live findings, RX 7900 XTX, 2026-06-20)

Env knobs (set in the daemon's environment):

- **`AKH_GPU_MATMUL=1`** + **`AKH_GPU_QUANT=1`** — route the Candle akh's linear
  GEMMs through the AMD GPU (rocBLAS, int4-resident). Needs the `rocm` feature
  build. Verified: gemma4-12B → ~7 GB VRAM resident, GPU active during decode.
- **`AKH_LLM_MODEL=gemma4`** — pick which GGUF the Candle paths load.
- **`AKH_NLU_LLM=0`** — skip the NLU tier-3 LLM load. **Required on a
  memory-constrained host running a Candle chat akh**: otherwise the daemon loads
  the large model *twice* (NLU + chat backing — they don't yet share one instance)
  and can OOM (two gemma4-12B ≈ 30 GB RSS).
- **`AKH_AGENT_MAX_TOOLS=N`** — cap the agentic tool catalog. The full catalog is
  several thousand prompt tokens; for a small-context in-process model it dominates
  prefill and makes turns very slow. `0` runs the akh tool-free (pure chat). Cloud
  akhs (Tecton/Thoth) don't need this — the cloud handles the full prompt fast.

**One daemon serves all akhs.** `akh -w <name> <chat|acp status|akh …>` routes to
`<name>` regardless of the daemon's `[acp].workspace` — the ACP server resolves
each request's workspace on demand and any configured akh loads on first use. So
a single `akhomed` serves Tecton, Thoth, and Gemma together; `[acp].workspace` is
just the default when `-w` is omitted. (Verified live: one daemon answered all
three via `-w`, and a same-host `akh akh sleep` → `akh akh wake` round-trips
cleanly.)

Remaining caveats:

- **Candle GPU offload is partial** (linear GEMMs only; attention/lm_head still
  CPU), so gemma4-12B agentic turns are slow — pair it with `AKH_AGENT_MAX_TOOLS`
  to keep prompts short. Cloud akhs are unaffected.
- **A data-plane call resolves the akh's agent**, so the first request to a
  Candle akh (even `acp status`) triggers its model load (~tens of seconds for
  12B). Engine-only ops could skip the agent load — a later optimization.
