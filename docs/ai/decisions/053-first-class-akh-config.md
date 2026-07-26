# ADR 053 — First-class `AkhConfig` (`akh.toml`): pinned model, explicit prompt, related workspaces

> Date: 2026-06-14
> Status: Accepted
> Phase: P1 (Pure-ACP Agentic Harness pivot)
> Depends on: ADR-027 (shared-workspace Agent), ADR-046 (delegation config)
> Enables: P2 (ACP transport), P5 (per-akh automation), P6 (federation)
> Anima project: `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")
> Plan: [`docs/ai/plans/2026-06-14-pivot-acp-agentic-harness.md`](../plans/2026-06-14-pivot-acp-agentic-harness.md)

## Context

In the pivot to a pure-ACP agentic harness, the unit of agency is an **akh**.
An akh must behave **consistently**: the same pinned model, the same system
prompt, the same memory workspaces, run after run. Today none of that is a
first-class, persisted thing.

What an akh *is* today is implicit and scattered:

- **Identity / workspace** = `WorkspaceConfig` (`src/workspace.rs:133-161`), a TOML
  file under `$XDG_DATA_HOME/akh-medu/workspaces/{name}/`.
- **Behaviour** = an `Agent` (`src/agent/agent.rs:162`) plus a `Psyche`
  (`src/compartment/psyche.rs`).
- **System prompt** = **derived at runtime, never stored** — assembled in (at
  least) three places: `build_system_prompt` (`src/agent/model_manager.rs:394`),
  `PsychePromptSerializer` (`src/compartment/prompt.rs:68`), and the agent's own
  assembly path. The psyche is the de-facto prompt source.
- **Model** = **global**: `OllamaConfig.model` (`src/agent/ollama.rs:13`), with a
  local `base_url` only — no auth, no cloud (`ollama.com`) endpoint. Every
  workspace on a daemon shares one model. `AgentModelManager` exists but is wired
  only in tests.
- **`akh.toml`** is mentioned in delegation help text but **never parsed**.

This blocks the pivot's core promises:

1. **Per-akh model pin** — different akhs need different small models (one tuned
   for code, one for prose), and a model swap must not silently change behaviour
   mid-session.
2. **Operator-hosted *or* cloud Ollama** — the daemon must reach
   `ollama.com` with a Bearer key resolved from the environment, not just a local
   `base_url`.
3. **Explicit, authoritative system prompt** — locked decision #3 of the pivot:
   the prompt is supplied by the akh, replacing the psyche-derived text. Psyche
   becomes optional flavour.
4. **Related workspaces** — locked decision #4: an akh names the peer workspaces
   it federates reads across (consumed by P6).
5. **Automation off-by-default** — P5 needs a per-akh `[autonomy]` block.
6. **Delegation from config** — `DelegationConfig` is hardcoded; it must come
   from the akh's TOML.

These are all properties of *one akh*, and they must persist on the PVC across
restarts and relocations. There is no single struct that owns them.

## Decision

Introduce a first-class **`AkhConfig`** persisted as **`akh.toml`** next to the
existing workspace files. `AkhConfig` **embeds** `WorkspaceConfig` (via
`#[serde(flatten)]`) so the existing workspace TOML is a strict subset — old
files keep loading untouched — and adds akh-level fields:

```toml
# akh.toml  (all new fields optional; #[serde(default)])
name = "thoth"

# --- WorkspaceConfig fields flatten in here (id, partitions, …) ---

# Model pin — the akh always answers with exactly this model.
model = "qwen2.5:7b"

[ollama_endpoint]              # one of: local | cloud
kind = "cloud"                 # default "local"
base_url = "https://ollama.com"
api_key = "env:OLLAMA_API_KEY" # resolved like UpstreamEndpointConfig
# failover = false             # Ollama→fallback is opt-in per akh

# Authoritative system prompt. When set, REPLACES the psyche-derived prompt.
system_prompt = """
You are Thoth, keeper of records…
"""

is_main = true                 # MAIN memory workspace vs related
related_workspaces = ["egypt-history", "linguistics"]

[autonomy]                     # P5 — default master switch is OFF
enabled = false
# tasks = { consolidate = true, dream = false, … }

[delegation]                   # P-aligned with ADR-046; from TOML now
enabled = false

path_mode = "local"            # local (prefix+sandbox) | remote (ACP fs/*)
```

Concrete commitments:

- **`AkhConfig` embeds `WorkspaceConfig`** with `#[serde(flatten)]`; **every new
  field is `#[serde(default)]`** so existing `workspace.toml` and bincode
  sessions on the PVC keep loading. (Pattern already proven by `shared_partitions`
  + the legacy-seed-packs test, `workspace.rs:540`.)
- **The model pin moves off global `OllamaConfig` onto the akh.** `AkhConfig.model`
  + `AkhConfig.ollama_endpoint` are the source of truth. The per-workspace
  `AgentEntry` carries the resolved `OllamaClient`; `AgentModelManager` is wired
  into akhomed per workspace (it exists but was test-only).
- **`OllamaClient::{generate, health_check}` gain a cloud branch.** A new
  `OllamaEndpoint { Local { base_url } | Cloud { base_url, api_key } }` drives an
  optional `Authorization: Bearer <key>` header. The key is resolved with the
  **same `env:VAR` idiom** as `UpstreamEndpointConfig::resolved_auth` — no new
  secret mechanism.
- **Ollama→fallback failover is opt-in per akh** (`failover` flag), not implicit.
- **One system-prompt assembly function.** The three current assembly sites
  collapse into a single function (`assemble_system_prompt`) whose contract is
  (REVISED 2026-07-26): the operator section (`AkhConfig.system_prompt` when
  set, otherwise psyche-derived) composes with machine-owned sections
  (tool-operations + operating conventions). Psyche stops being the mandatory
  prompt source; the explicit prompt is no longer verbatim-replacement but the
  operator-controlled persona/voice authority that precedes tool documentation.
- **`related_workspaces`, `[autonomy]`, `[delegation]`, `path_mode`** are parsed
  and stored now but **consumed by later phases** (P6, P5, P4). P1 only lands the
  schema + loader + the model/prompt wiring; it does not change automation or
  federation behaviour.
- **Loading site:** akhomed loads `akh.toml` lazily at `get_engine`/`get_agent`
  (`src/bin/akhomed.rs`), the same place the workspace is resolved to an
  `Arc<Engine>` / `Arc<Mutex<Agent>>`. Missing `akh.toml` ⇒ defaults ⇒ today's
  behaviour exactly.

## Alternatives considered

### Keep model + prompt global; add only `related_workspaces`

**Why considered.** Smallest change; federation (P6) is the loudest near-term
need.

**Why rejected.** Defeats the pivot's central promise — *consistent per-akh
behaviour*. Two akhs on one daemon could not pin different models or carry
different prompts, which is precisely the multi-akh harness we are building.
Cloud Ollama would also remain impossible.

### A separate `model.toml` / `prompt.txt` per akh (multiple files)

**Why considered.** Each concern in its own file; easy to hand-edit a prompt.

**Why rejected.** Fragmentation — an akh's identity would span three or four
files with no single loader, exactly the sprawl this pivot is removing on the
transport side. One `akh.toml` per akh is the discoverable, atomic, relocatable
unit. (A long prompt can still use TOML multi-line strings.)

### New `SymbolKind`/Entity for "akh" in the KG

**Why considered.** Akh-medu models most concepts as Entity symbols (the agent
module already does this for goals/memories).

**Why rejected.** Config is boot-time, file-backed, operator-authored truth; it
must load *before* the Engine is constructed (the model pin selects the
`OllamaClient` the Agent is built with). Bootstrapping config out of the KG it
configures is circular. The KG remains the store for *learned* knowledge; config
stays a file.

### Make `system_prompt` additive (psyche + explicit, concatenated)

**Why considered.** Preserves psyche-derived flavour while letting operators add
instructions.

**Why rejected initially, then REVISED 2026-07-26.** Locked decision #3 was
explicit: the prompt **replaces** the psyche-derived text, so behaviour is
exactly what the operator wrote. However, this created a critical gap: models
were never briefed on available tools, invocation syntax, or operating
conventions. **Revision**: `system_prompt` now forms the **operator section**
(persona/voice authority), which composes with **machine-owned sections**
(tool-operations from the manifest registry + operating conventions). The
explicit prompt no longer replaces the entire assembly — it composes, ensuring
every akh receives tool documentation. See WI-250 ("mechanism that builds the
system prompt so models know how akh-medu tools work").

## Consequences

### Positive

- **Consistent per-akh behaviour** — pinned model + explicit prompt persisted on
  the PVC; the same akh wakes the same way.
- **Cloud Ollama unlocked** — `ollama.com` reachable via the existing `env:VAR`
  secret pattern; no new secret plumbing.
- **One prompt path** — three assembly sites collapse to one, killing a class of
  "which prompt won?" bugs.
- **Schema in place for P4/P5/P6** — `path_mode`, `[autonomy]`, `related_workspaces`
  are parsed now, so later phases wire behaviour without re-touching config
  loading.
- **Back-compat preserved** — `#[serde(default)]` + flatten means zero migration
  for existing workspaces; absence of `akh.toml` reproduces today's behaviour.

### Negative

- **`AgentModelManager` wiring touches the hot construction path** in akhomed
  (`get_engine`/`get_agent`); a regression here affects every workspace. Mitigated
  by the now-runnable integration suite (P0) and a dedicated two-akh test.
- **Config surface grows** — `[autonomy]`/`[delegation]`/`path_mode` exist before
  their behaviour does, so an operator could set a flag that is silently inert
  until its phase lands. Mitigated by documenting per-field "consumed in P_x" and
  by `path_mode`/autonomy defaulting to today's behaviour.
- **Secret handling spreads** — a second site (Ollama cloud) now resolves
  `env:VAR`. Mitigated by reusing the exact `resolved_auth` helper rather than
  re-implementing.
- **Failover semantics change** — making failover opt-in could surprise a
  deployment that implicitly relied on it. Mitigated: the default reproduces the
  prior single-endpoint behaviour (no silent fallback existed to lose).

## Implementation

P1 of the pivot plan. Order:

1. New `src/akh_config.rs`: `AkhConfig` (flatten `WorkspaceConfig` + new
   `#[serde(default)]` fields), `OllamaEndpoint` enum, `[autonomy]`/`[delegation]`
   sub-structs, `load`/`save`, defaults. Round-trip + legacy-load tests.
2. `src/agent/ollama.rs`: `OllamaEndpoint` cloud branch on `generate` /
   `health_check`; Bearer header via `resolved_auth` idiom; opt-in failover.
3. `src/agent/model_manager.rs`: collapse the three prompt-assembly sites into one
   function; explicit prompt authoritative, psyche fallback.
4. `src/bin/akhomed.rs`: load `akh.toml` at `get_engine`/`get_agent`; per-workspace
   `AgentModelManager`; move the model pin onto `AgentEntry`.
5. Tests: two akhs, different pinned models + explicit prompts, each answers with
   its own prompt; cloud endpoint reachable with an `env:VAR` key (mocked).

**Exit criteria:** two akhs with different pinned models + explicit prompts load
from `akh.toml` and answer with their own prompt; cloud endpoint reachable with
`env:VAR` key; `cargo test` stays green.

## References

- Plan: `docs/ai/plans/2026-06-14-pivot-acp-agentic-harness.md`
- ADR 027 — Shared-workspace Agent (one `Arc<Mutex<Agent>>` per workspace)
- ADR 046 — Delegation & External Scheduling (`DelegationConfig`)
- `src/workspace.rs:133` — `WorkspaceConfig` (embedded)
- `src/agent/ollama.rs:13` — `OllamaConfig` (model pin moves off this)
- `src/agent/model_manager.rs:394` — `build_system_prompt` (one of 3 sites)
- `src/compartment/prompt.rs:68` — `PsychePromptSerializer`
- `UpstreamEndpointConfig::resolved_auth` — the `env:VAR` secret idiom mirrored
  for cloud Ollama
