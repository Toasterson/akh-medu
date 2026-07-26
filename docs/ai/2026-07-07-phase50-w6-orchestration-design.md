# Phase 50 / W6 — fleet-orchestration benchmark design

- Status: Design (2026-07-07), ready to build
- Plan: `docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md` (W6)
- ADR: 086 §3 (fleet leg); delegation background: ADR 046 / 014
- Depends on: W3 runner pattern (`examples/phase50_w3_runner.rs`), `Agent::with_backend`,
  `RemoteBackend`, `harness_core`. Substrate: live exec1 (root@127.0.0.1:10042).

## What it measures

Does splitting a decomposable illumos task across an **orchestrator akh + N worker akhs**
beat a **single akh** doing the whole thing? Two arms, same task, same model:

- **control** — one akh runs the whole composite task in one `run_agentic_turn` (sequential).
- **orchestrated** — an orchestrator akh decomposes the task into N independent subtasks,
  then N worker akhs each run one subtask **in parallel** (own thread, own `RemoteBackend`
  connection to exec1, shared `Arc<Engine>`+audit ledger), and the orchestrator joins + the
  composite `verify.sh` grades the end state.

The speedup is genuine even though all workers hit one 2-vCPU guest: the work is
**latency-bound** (SSH round-trips + cloud-model inference), so N workers overlap their
cloud-model wait → real wall-clock reduction. It is NOT a guest-CPU claim.

## The task (decompose-and-conquer, non-colliding)

`benchmarks/illumos-orchestration/tasks/relay-group-3svc/`:
- **setup.sh** (root) — installs **three independent, differently-broken SMF services**:
  `site/relay-a` (start method points at a missing script — needs the script created),
  `site/relay-b` (imported but `disabled` — needs enable), `site/relay-c` (online but a
  required config property is wrong/missing — needs `svccfg setprop` + refresh). Distinct
  FMRIs, distinct `/opt/akh-e2e-relay-{a,b,c}/` dirs, distinct manifests → **zero shared
  state**, so parallel workers never collide.
- **prompt.md** (composite) — "Three services site/relay-a, -b, -c are all unhealthy.
  Diagnose and fix each so all three report `online`. You have root."
- **verify.sh** (root) — passes iff all three FMRIs are `online` (exit 0). Echoes each.
- **meta.toml** — `id`, `family="orchestration"`, `subtasks=3`, `timeout_secs`,
  `expected_manpages`, `parallelizable=true`. Add a machine-readable **subtask list** the
  harness can hand to workers when NOT relying on the orchestrator's own decomposition
  (`[subtasks]` with `{id, fmri, prompt}` × 3) so the control-vs-orchestrated comparison is
  apples-to-apples and the run is reproducible even if the orchestrator decomposes poorly.

Author it in the same 4-file shape as `benchmarks/illumos-e2e/`. One task is enough for v1;
the harness must be N-general so more can be added.

## Harness: `examples/phase50_w6_orchestration.rs` (feature `remote-executor`)

CLI: `--task` (default the relay-group task), `--model`/`--model-slug`, `--exec1`
(default `root@127.0.0.1:10042`), `--key` (default the abs main-tree root key),
`--workers N` (default = task's `subtasks`), `--results`, `--dry-run`.

### Shared setup
- One `Arc<Engine>` with a scratch `data_dir` (so the audit ledger auto-wires — the W3
  runner confirmed `Agent::with_backend` wires `engine.audit_ledger()` when data_dir is Some).
  **All arms + all workers share this one Engine/ledger** so every command is queryable from
  one place, attributed by `ExecEnvelope.correlation_id` (each `RemoteBackend` connection has
  a distinct id).
- Cloud chat backing exactly as the W3 runner builds it (`AgentModelManager` +
  `OllamaConfig{base_url:"https://ollama.com", model, api_key:"env:OLLAMA_API_KEY"}` +
  `OllamaChatBackend`).

### control arm
`setup.sh` → build one `Agent::with_backend(engine, cfg, Some(RemoteBackend))` → time
`run_agentic_turn(composite_prompt)` → `verify.sh` → record `{wall, tokens, requests}`.

### orchestrated arm (re-run setup.sh first to reset state)
1. **decompose** — an orchestrator `Agent` runs ONE `run_agentic_turn` whose prompt asks it
   to split the composite task into independent subtasks and emit a JSON array of subtask
   prompts. Capture its `{tokens, requests}` = the **coordination tax**. Validate/clip to the
   task's declared `subtasks` list; if the orchestrator's decomposition is unusable, fall
   back to `meta.toml`'s `[subtasks]` (and note it) so the arm still runs.
2. **fan out** — for each of the N subtasks, spawn a `std::thread` that builds its own
   worker `Agent::with_backend(engine.clone(), cfg, Some(RemoteBackend::connect(...)))` (a
   FRESH RemoteBackend per worker — distinct SSH connection + correlation_id) and runs
   `run_agentic_turn(subtask_prompt)`. Join all. (N worker Agents, one shared Engine — each
   Agent owns its OODA state; Engine is `Arc` + interior-mutable, already shared this way in
   the daemon.)
3. **grade** — composite `verify.sh` → pass/fail. Record `{wall = decompose+max(worker walls),
   tokens = sum, requests = sum}`.

### Metrics (all from what exists — NO core audit change needed for v1)
- **success**: composite `verify.sh` exit for each arm.
- **speedup**: `control.wall / orchestrated.wall`.
- **coordination tax**: orchestrator decomposition `{tokens, requests}` + fan-out count N
  (the "messages" proxy — one delegated subtask = one coordination message).
- **redundant commands**: from the shared audit ledger — `list_by_kind(0,…)` (ToolInvocation),
  group by `ExecEnvelope.correlation_id` (=worker), count shell commands issued by >1 worker
  (identical `params_summary`). High redundancy = poor decomposition (workers re-doing shared
  discovery). Report as a ratio.
- **(defer)** delegation-quality rotation-judge rubric → W4/W6 follow-up; v1 is deterministic.

### Output
Emit RunRecords under harness id **`fleet-orchestration`** with the two arms as task rows
(`relay-group:control`, `relay-group:orchestrated`), the orchestrated row's `notes` carrying
`speedup=…, tax=… tok/… req, redundant=…/…`. `HostTier::Vm(...)` like W3. Write via
`harness_core::store::write_run`; regenerate LEADERBOARD.md + site.

## Constraints / gotchas
- Parallel SSH: N=3 concurrent root sessions to exec1's sshd — well under `MaxSessions`(10)/
  `MaxStartups`. Fine for small N; if N grows, batch.
- `--dry-run` runs both arms' setup/verify plumbing + the fan-out threading with NO cloud
  (skip `run_agentic_turn`); proves threading + ledger-attribution without spend. I (main
  loop) run the real 3-model comparison to control budget — the agent must NOT run cloud.
- Keep it ONE composite task, N-general harness. Cost of a real run ≈ (1 control + 1 decompose
  + N workers) turns per model.
- Worktree-isolate the build agent (`isolation: worktree`) — do not repeat the 48f main-tree mess.
