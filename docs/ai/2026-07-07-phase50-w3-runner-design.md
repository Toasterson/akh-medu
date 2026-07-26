# Phase 50 / W3 runner — design blueprint

- Status: Design (2026-07-07), ready to build once the Agent backend seam merges
- Plan: [`docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md`](plans/2026-07-05-phase50-feature-test-benchmarks.md) (W3)
- ADR: [`086`](decisions/086-feature-test-vm-harness-benchmarks.md) §2
- Corpus: `benchmarks/illumos-e2e/` (15 tasks, 5 families, ready)
- Depends on: **Agent backend-injection seam** (`Agent::with_backend`, task #23) — hard;
  **48f audit envelope** (task #22) — soft, enriches manpage grade.

## What it is

A `--features remote-executor` example, `examples/phase50_w3_runner.rs`, that drives a
rented cloud model (deepseek / glm-5.2 / qwen) through the **48c RemoteBackend** against the
live `exec1` OpenIndiana guest, one illumos-sysadmin task at a time, and emits a typed
`harness_core::RunRecord`.

## Executor principal — decided

The autonomous runner logs the akh in as **`root@exec1`** (the baked `root_ed25519` key).
Rationale: every task's `prompt.md` says *"You have root access on this host"*, and the SMF /
RBAC tasks need root. `akh-exec` + pfexec is the **48b LocalDrop** path (separate, later) —
not W3-autonomous. The RemoteBackend `os_user` in the audit envelope will read `root`.

## Per-task loop

For each `tasks/<id>/`:
1. **setup** — `root@exec1` runs `setup.sh` (upload via RemoteBackend file-write to
   `/tmp/akh-e2e/<id>/setup.sh` then `sh setup.sh`, or scp-equiv). Idempotent by contract.
2. **drive** — build the akh (below), `agent.run_agentic_turn(&engine, "w3-<id>", &prompt_md, &sink)`
   with `prompt.md` as the task text. Enforce `meta.toml`'s `timeout_secs` as a wall-clock cap;
   on overrun, `TaskOutcome::Aborted { reason: "timeout" }`.
3. **verify** — `root@exec1` runs `verify.sh`; exit 0 ⇒ `Pass` (score 1.0), non-zero ⇒ `Fail`
   (0.0). The akh never sees `verify.sh` (upload it only *after* the turn ends, or keep it
   host-side and pipe over ssh — never into a path the akh browsed).
4. **manpage grade** — scan the turn's tool-invocation trail (see below) for a `man`/`apropos`
   invocation naming each `expected_manpages` entry. Record as a note / secondary score; does
   **not** gate the deterministic pass/fail (ADR 086 §2 keeps them separate).
5. **RunRecord row** — `TaskResult::new(TaskId, outcome, score, WallTimeSecs, ResourceUsage, notes)`.

Between tasks: rely on idempotent `setup.sh`. For known-destructive tasks (zpool create, role
create) `setup.sh` must also tear down its own prior artifacts (the corpus authors own that);
if a task is flagged un-resettable, `vmctl reload --name exec1` is the escape hatch (slow — avoid
unless required). vmctl has **no snapshot** subcommand; do not design around one.

## Building the akh (exact wiring — verified against the tree)

```rust
use akh_medu::agent::model_manager::{AgentModelManager, ModelManagerConfig};
use akh_medu::agent::ollama::OllamaConfig;
use akh_medu::agent::chat_backend::OllamaChatBackend;
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::backend::remote::{RemoteBackend, RemoteBackendConfig, HostKeyPolicy};

// 1. Cloud model backing (replicates examples/three-akhs/workspaces/tecton.toml).
let mm = ModelManagerConfig {
    ollama: OllamaConfig {
        base_url: "https://ollama.com".into(),
        model: model_tag.clone(),          // "deepseek-v4-pro" | "glm-5.2" | "qwen3.5:397b"
        api_key: Some("env:OLLAMA_API_KEY".into()), // resolved at call time
        timeout_secs: 300,
        ..Default::default()
    },
    system_prompt: Some(SYS_PROMPT.into()), // "You are an illumos system engineer …"
    ..Default::default()
};
let mgr = std::sync::Arc::new(AgentModelManager::new(mm));

// 2. RemoteBackend → root@exec1 (host-forward :10042).
let backend = std::sync::Arc::new(RemoteBackend::connect(
    RemoteBackendConfig::new(/* host */ "127.0.0.1", /* user */ "root", key_path)
        .with_port(10042)
        .with_host_key_policy(HostKeyPolicy::AcceptNew) // lab; document the deviation
        .with_roots(vec!["/".into()], vec!["/tmp".into(), "/opt".into(), "/var".into()])
        .with_connect_timeout(std::time::Duration::from_secs(15)),
)?);
// NB: confirm the RemoteBackendConfig::new arg order against src/backend/remote.rs:264 —
// this blueprint may have host/user swapped. The example phase50_remote_exec1.rs already
// constructs one from AKH_EXEC1_SSH/AKH_EXEC1_KEY; copy its exact construction.

// 3. Engine (scratch data_dir so the akh has a sandbox) + Agent with the remote backend.
let engine = std::sync::Arc::new(Engine::new(EngineConfig { data_dir: Some(scratch), ..Default::default() })?);
let mut agent = Agent::with_backend(engine.clone(), AgentConfig::default(), Some(backend))?; // ← the seam (task #23)
agent.set_chat_backend(std::sync::Arc::new(OllamaChatBackend::new(mgr)));

// 4. Attach an AuditLedger so the manpage grade can read the tool trail (see below).
```

`run_agentic_turn` signature (src/agent/agentic_chat.rs:197):
`agent.run_agentic_turn(&engine, agent_id: &str, task: &str, sink: &dyn MessageSink) -> outcome`.
Copy the `TraceSink`/`MessageSink` impl from `examples/agentic_flow.rs`.

## Capturing the command trail for manpage grading

Two sources, prefer whichever is cleaner:
- **Audit ledger** — attach an `AuditLedger` to the agent's `ToolRegistry`
  (`registry.set_audit_ledger(...)`), then after the turn read back `ToolInvocation` entries.
  Each carries `params_summary` (the shell command text) and, **once 48f lands**, an
  `ExecEnvelope { backend, host, os_user, correlation_id }` proving it ran on `exec1`.
  Grade: for each `expected_manpages` page P, pass if some `shell_exec` entry's params match
  `man …P` or `apropos …P`.
- **MessageSink** — a custom sink that records every `AkhMessage::ToolResult { tool, output, .. }`;
  simpler but only sees results, not the exact issued command. Prefer the audit ledger.

Because every `shell_exec` the akh issues runs through the RemoteBackend, any `man` it runs is
**on exec1** by construction — so even pre-48f the command text is a sound manpage-grounding
signal; 48f's envelope only makes it forensically airtight (host-attributed). Design the grade
to read the envelope if present and fall back to command-text-only if not.

## RunRecord provenance (harness_core::schema)

- `HarnessId::new("illumos-e2e")`, `GitSha::new(<HEAD sha>)`.
- `TaskSetHash` — blake3 over the corpus (reuse the hashing the harness already does for W7;
  see `benchmarks/harness-core/src/hash.rs`).
- `ModelIdentity { tag: ModelTag::new("deepseek"|"glm-5.2"|"qwen"), backing: ModelBacking::Cloud { provider_tag } }`.
- `GradingMode::Autonomous`.
- `HostTier::Vm(VmImageIdentity { registry: "code.aopc.cloud/CloudNebulaProject", recipe_hash: VmRecipeHash })` —
  hash the `openindiana-hipster-cloud-fixed.qcow2` recipe (the bake script) for `recipe_hash`.
- Write with `harness_core::store::write_run(results_root, &record)` →
  `benchmarks/results/<date>-illumos-e2e-<model>/run.jsonl`.

Add `harness-core = { path = "benchmarks/harness-core" }` to the main crate's `[dev-dependencies]`
so the example can link it.

## Budget / cost discipline

- Default to a **1-task smoke** (`--task smf-01-clear-maintenance`, one model) to prove the whole
  loop end-to-end cheaply before any full sweep.
- Full sweep = 15 tasks × 3 models = 45 turns; cap per-turn OODA cycles (`AgentConfig.max_cycles`)
  and a hard per-task request budget; `Aborted` on cap. Reuse the W5 budget posture (~110 reqs
  total was the whole W5 leg).
- Cloud creds: `OLLAMA_API_KEY` from `/home/toasty/env-file` (`OLLAMA_CLOUD_KEY` there — export it
  as `OLLAMA_API_KEY` for the run). Never commit the token.

## Out of scope for this runner

Advisory mode (W4), orchestration bench (W6), the 48b LocalDrop principal, Terminal-Bench.
This is autonomous-only, deterministic + manpage grade, one akh.
