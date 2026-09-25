# Phase 53 implementation briefs

- **ADR:** [113](../../decisions/113-full-akh-always.md)
- **Plan:** [2026-09-24](../../plans/2026-09-24-phase53-full-akh-always.md)
- **Anima:** umbrella `01a0d43b-bf90`. Slices: 53a `01a0d469-c9a8`, 53b `01a0d469-de3f`,
  53c `01a0d469-edd7`, 53d `01a0d469-f8f1`, 53e `01a0d46a-0381`, 53f `01a0d46a-18db`,
  53g `01a0d46a-2850`, 53h `01a0d46a-33a9`. 53i (per-akh MCP servers) is added here.
- **Integration branch:** `phase53/full-akh`

These briefs came out of two read-only research passes on 2026-09-24. **Architect
decisions** are binding for implementers. Everything else is recon; verify line
numbers before editing, because they drift.

## Architect decisions (binding)

1. **One build profile.**
   - `server` = `daemon, acp, postgres, tracing-otel, remote-executor, embedded-runtime, core-api, delegation`.
   - `default` stays lean (WI-339).
   - Acceptance command: `cargo build --release --features server --bin akhomed`.
   - `core-server` becomes a deprecated alias of `server` in 53a and is deleted in 53h.
   - No `cfg(not(feature = "core-api"))` gate may remove daemon or ACP behaviour.
2. **`/v1` is served only when `AKH_CORE_CONFIG` is set.** A missing variable means no
   `/v1`, not a startup failure. The auto-start set becomes the union of `auto_start`,
   `AKH_AUTO_START` and every `principals[*].akhs` in `core.json`, logged at `info`.
3. **The tool surface is scoped.**
   - `/v1` resolves tools from the akh's live `ToolRegistry` (the same `Arc` the Agent
     uses). A per-principal `ToolScope` filters it.
   - `ToolScope` defaults to `LegacyCognitive`: the previous 9 names. Tools such as
     `shell_exec`, `delegate` and `agent_spawn` stay denied unless a scope or grant names
     them.
   - `read_only` is derived from `ToolManifest.danger`.
4. **Autonomy defaults.**
   - `AutonomyConfig.enabled = true` by default.
   - `idle_cycle` (OODA tool use) defaults on **only when the effective exec backend is
     not `InProcess`**. Otherwise it is off and logs `info` with the reason.
   - Budget defaults: `max_tokens_per_day = 250_000`, `max_turns_per_day = 40`, per akh,
     UTC day.
   - LLM-cost tasks: librarian, reactor `LlmTurn`, knowledge_extraction (only when
     `external_api` is set), knowledge_elicitation.
   - Reflection and goal generation make no LLM calls and are free.
   - The librarian also stays behind `AKH_LIBRARIAN=1` until WI-298
     (`fix/wi-298-shared-triple-retention`) merges.
5. **Harvest is its own task.** It runs on the sleep tick **before**
   `run_sleep_cycle` and independently of it (today a sleep-cycle error silently skips
   harvest). It loops up to `HARVEST_MAX_BATCHES_PER_TICK = 4`.
6. **Deployed mode is explicit.**
   - Set it with `AkhomedConfig.deployment = "deployed"` or `AKH_DEPLOYMENT=deployed`.
   - Deployed mode rejects exec `InProcess`, and rejects `ChatBacking::Codex|Claude` in
     process (those delegate via akhrtd). Diagnostic: `akh::config::in_process_forbidden`.
   - `LocalDrop` and `Remote` are allowed only with `allow_legacy_exec = true` plus a
     warning.
   - Development is the default, so the tests are unaffected.
7. **Native runtimes default to `Brokered`.** In deployed mode akhrtd refuses
   `Colocated`. akhomed's in-process bridge must not hardcode `Colocated` in deployed
   mode.
8. **Grants (53e).**
   - Deliver `GrantAuthority`, enforcement and `DerivationKind::DelegatedCall` (tag 108,
     appended), plus the ACP `_akh/delegation/grant/{mint,revoke,list}` methods.
   - **Also** deliver automatic minting: a `NativeSession` delegation target that creates
     an akhsessd session carrying the grant. This needs a small akhsessd client in
     akhomed.
   - Remove the dead `LocalClaude` `/mcp` target.
9. **53i: per-akh MCP servers.** This blocks the Roman cutover: 95 of Minerva's 104
   tools are external MCP.
   - Config: `[[mcp.servers]] name, url, auth = { token_env | token_file | header_env }, tools = [allowlist] (optional)`
     in the workspace TOML.
   - At agent init, wire `McpClient` (`src/agent/tools/mcp.rs`) into the akh's
     `ToolRegistry`, with names `mcp__<server>__<tool>` (the Claude convention, matching
     how the Romans' prompts name them).
   - A server that is unreachable at init is a warning and gets retried lazily; it is
     never fatal.
10. **Errors:** miette + thiserror everywhere, each variant with
    `#[diagnostic(code(...), help(...))]`. Wire types stay the existing ones; add new
    fields as `#[serde(default)]`.
11. **Tests:** every slice adds the tests named in its brief. Run clippy on the files
    you touch. The crate already fails `-D warnings` on newer toolchain lints; don't add
    new ones and don't do unrelated clean-up.

## Build notes

- Use `CARGO_TARGET_DIR` per worktree (disk has about 90 GB free; one `server` target
  dir is roughly 10–15 GB).
- Delete agent worktree `target/` dirs after merge.
- Build in the foreground. Background cargo builds stall subagents.
- CI settings: `CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0`. Locally,
  use more jobs.
- Test commands:
  - `cargo check --features server`
  - `env -u AKH_TRANSCRIPT_CAPTURE cargo test --lib --features server core_service -- --test-threads=1`
  - `cargo test -p akh-session-services --features session,core-integration-tests`
  - `cargo test --features server --bin akhomed`
  - `cargo test --lib --features server agent::daemon akh_config`
  - `cargo test --lib --features postgres agent::harvest`

---

## 53a — `core-api` becomes additive

- Nothing gets built twice: `ServerState` already caches one Engine and one
  `Arc<StdRwLock<Agent>>` per akh. `CoreBackend for ServerState`
  (`src/bin/akhomed.rs` ~6442) calls `get_agent` and then `get_engine`.
- **Cargo.toml:**
  - add `core-api = ["daemon", "dep:axum", "dep:akh-service-store"]`;
  - `server` gains `core-api` and `delegation`;
  - `core-server = ["server"]` (a deprecated alias);
  - `crates/akh-session-services/Cargo.toml` feature
    `core-integration-tests = ["session","dep:akh-medu","akh-medu/core-api"]`.
- **Gates** (the only ones in the tree):
  - `src/lib.rs` ~50: `#[cfg(feature="core-api")] pub mod core_service;`
  - `src/nlu/mod.rs` ~26: comment only.
  - `src/bin/akhomed.rs`:
    - ~727: delete the cfg (always resume the agent after a reconnect).
    - ~4420: replace `cfg!(feature="core-server")` with a runtime
      `core_cfg_path.is_some()`, reading `AKH_CORE_CONFIG` before the config load.
    - ~4533 (auto-start): delete the cfg.
    - ~4625, 4674, 4763: `#[cfg(feature="acp")]`.
    - ~4701-4745 and 4787-4791: `#[cfg(feature="core-api")]`. Hold the listener as
      `Option<(oneshot::Sender<()>, JoinHandle<()>)>`.
    - ~6442: `#[cfg(feature="core-api")]`.
  - Reorder startup: load `CoreConfig` right after the global config, then start the
    daemons, then ACP, then bind `/v1`.
  - The ACP fallback workspace must not self-bootstrap a `default` workspace when
    `auto_start` names akhs: use the first auto-start akh instead.
- **`src/core_service.rs`:**
  - `CoreBackend` gains
    `fn tools<'a>(&'a self, akh: &'a str) -> CoreFuture<'a, Arc<ToolRegistry>>`.
  - The ServerState impl runs `get_agent` and then, in `spawn_blocking`,
    `agent.read().tool_registry_handle()`. Make `Agent::tool_registry_handle` public.
  - Delete `registry()`.
  - `tool_call` and `catalog` use `backend.tools(akh)` filtered by
    `CorePrincipal.tools: ToolScope` (`#[serde(default)]`, tag `kind`, variants
    `LegacyCognitive` (default) and `Named { tools: BTreeSet<String> }`).
  - `tools()` derives `read_only` with `is_cognitive_read_only(manifest)`: `danger.level == Safe`
    and capabilities ⊆ {ReadKg, Reason, VsaAccess, ProvenanceAccess, MemoryAccess}.
  - Add a `CoreApiError` enum (Config, ToolOutOfScope, BackendUnavailable) with
    `impl From<CoreApiError> for Fault`.
  - An out-of-scope call returns HTTP 200 with `ToolResult{success:false}` and
    `data {code, help}`.
- **Tests:**
  - update both `CoreBackend` test impls (`core_service.rs` ~797, and
    `crates/akh-session-services/src/server/sessions.rs` ~2068);
  - `shell_exec` is still denied, and `agent_spawn` is denied under the legacy scope;
  - a `Named{memory_write}` scope succeeds;
  - the catalog shows only in-scope tools;
  - akhomed bin test: `engine` and `tools` return `Arc::ptr_eq` handles.

## 53b — budgeted autonomy

- **Current state:**
  - `DaemonConfig` is in `src/agent/daemon.rs` ~138-280. Intervals come from
    `[daemon.intervals]` (`config.rs` ~427).
  - `AutonomyConfig` is in `akh_config.rs` ~199 (`enabled`, default false; `tasks` map;
    `task_enabled` ~417). Task names are free strings (`KNOWN_TASK_NAMES`).
  - Token counts live in `LlmRequestMetrics` (`llm_metrics.rs` ~144).
  - Delegation has an in-memory `CostGuard`.
  - `src/metrics.rs` `AkhomedMetrics` (feature `tracing-otel`) has no per-task
    instruments.
- **New `src/agent/autonomy.rs`:**
  - `DaemonTask` enum, replacing `KNOWN_TASK_NAMES`, with `ALL`, `as_str` and
    `cost() -> TaskCost { Free, Network, Llm }`.
  - `AutonomyBudget { max_tokens_per_day, max_turns_per_day }` in `AutonomyConfig.budget`
    (`#[serde(default)]`).
  - `BudgetLedger`, persisted via `engine.store().put_meta(b"autonomy:budget:v1")`, as in
    `reactor.rs` ~610.
  - `AutonomyGovernor::{admit -> Admission { Admitted(BudgetTicket), Exhausted(BudgetExhausted) }, settle}`
    and `AutonomyError`.
- **Metering:** a thread-local `UsageScope` in `llm_metrics.rs` accumulates
  `prompt + completion` tokens.
- **Gate points:** `run_librarian_task`, `run_knowledge_extraction` (only if
  `external_api` is set), `run_knowledge_elicitation_task`, and the reactor `LlmTurn`
  arm (~1618). The reactor must use its deferral path so the occurrence survives.
- **Wrapper:** every task runs through one `run_task(task, f) -> DaemonTaskReport`
  recording:
  - `outcome` = `Ran | SkippedDisabled | SkippedBudget | SkippedContended | Failed`;
  - `duration`, `LlmUsage`;
  - `FactDelta`, from `AtomicU64` counters in `Engine::record_mutation` (~766), bumped
    before the sink check.
- **Metrics:**
  - `tracing::info!(target: "akh::autonomy", ...)`;
  - `akh_daemon_task_runs_total{workspace,task,outcome}`,
    `akh_daemon_task_duration_seconds`,
    `akh_daemon_task_llm_tokens_total{direction}` and
    `akh_daemon_task_facts_total{change}`.
- **Defaults:**
  - flip `AutonomyConfig.enabled` to true and fix the tests (`akh_config.rs` ~1159,
    1208, 1232, 1255);
  - add `budget` to the autonomy known tables (~862);
  - `idle_cycle` follows decision 4.
- **Harvest:** its own `DaemonTask::Harvest` on the sleep tick, before and independent of
  `run_sleep_cycle` (~1137-1190).
- **Tests:**
  - `tests/harvest_sqlite.rs`, a copy of `harvest_live.rs` using a tempdir
    `sqlite://…?mode=rwc` in the `sqlite_identity_roundtrip.rs` style. The watermark
    advances, a second pass is idempotent, and harvest runs even when the sleep cycle
    errors.
  - Governor rollover, exhaustion, and persistence across a reopen.
  - Daemon: an exhausted budget produces a skip report.
  - Risk: SQLite `append_mutations` may need residency fencing. Seed via `LocalSession`.

## 53c — `ExecdBackend`

- **What exists:**
  - Typed wire types in `crates/akh-session-types/src/lib.rs`: `WorkspaceOperation`,
    `OperationReceipt`, `Allocate`, `Allocation`, `WorkspaceBinding`, `RenewLease`.
  - LocalDrop and Remote backends (`src/backend/local_drop.rs`, `remote.rs`), wired in
    `akhomed.rs` ~2700-2800.
  - `Backend` (`src/backend/mod.rs` ~297, sync, extends `Filesystem` in
    `src/fs/mod.rs` ~170) adds `exec`, `handshake` and `locus`.
  - `ExecBackendConfig` (`src/akh_config.rs` ~241) =
    `InProcess | LocalDrop | Remote | None`.
- **akhexecd verbs** (`crates/akh-session-services/src/executor.rs` ~480-680):
  - `read{path,encoding}→{content,revision}`,
    `write{path,content,expected_revision,create}`,
    `edit{path,old_text,new_text,expected_revision}`, `mkdir`, `list→{entries}`,
    `search{pattern,glob,path}`;
  - `exec_start{command|argv,cwd,env,timeout_ms}`, then `exec_poll{process_id,wait_ms≤10000}`,
    `exec_stdin`, `exec_cancel`; the final result is
    `{stdout,stderr,exit_code,timed_out,cancelled,truncated}`;
  - receipts are idempotent on `id`; `TurnStatus::Unknown` fences further writes.
- **Auth:** a bearer token against `Principal{roles,akhs}`. Operations need the
  `session` or `sandbox` role. `POST /v1/allocations` (`Allocate{session_id,principal,profile}`)
  returns a `WorkspaceBinding` with a 300 s lease, renewed via
  `/v1/allocations/{id}/renew`. akhsbxd proxies `/v1/executors/{sandbox}/v1/...`
  (needs the `executor_proxy` role).
- **Build:**
  - `ExecBackendConfig::Execd { sandbox_url, sandbox_token_file, executor_url: Option<String>, executor_token_file, profile, principal, poll_wait_ms }`.
  - `src/backend/execd.rs` uses the blocking HTTP client already in the dependencies
    (`ureq` or reqwest blocking) with its own runtime for async code, following
    `remote.rs`.
  - Add a `stat` verb to akhexecd (read-only list in `executor.rs` ~390 and
    `broker.rs` ~253) for `exists`, `file_size` and `is_symlink`.
  - Add `OperationReceipt.principal: Option<String>` (`#[serde(default)]`), set from
    the authenticated principal in `server/executors.rs::operation`. The type uses
    `deny_unknown_fields`, so all services deploy together.
  - `ExecdError` variants: Allocate, Operation, UnknownOutcome, LeaseExpired, Transport.
  - `exec` loops `exec_poll` and sends `exec_cancel` at the deadline, mapping to
    `ExecError::Timeout`.
  - A lease renewer every 100 s.
  - `handshake()` reports `binding.confinement` honestly. The Mac `existing` provisioner
    is not isolation; don't claim `Sandboxed` for it.
  - Wire it in `akhomed.rs` next to `build_local_drop_backend`.
- **Deployed-mode checks** follow decision 6. Native runtimes follow decision 7:
  `crates/akh-runtime/src/codex.rs` and `claude.rs` already support
  `ExecutionMode::Brokered`, and `src/agent/runtime/bridge.rs` ~28,48 hardcodes
  `Colocated`. OpenCode is not a `RuntimeKind`; leave it out of scope and record it.
  *Follow-up (2026-09-25, Anima WI `01a0d569-1cf1` item 4):* `RuntimeKind::OpenCode`
  (`"opencode"`) now exists in akhrtd. It uses ACP plus a per-turn loopback MCP server for
  Akh tools. Brokered mode relies on a bridge-owned deny-all permission policy and needs the
  extra `native_tools_hidden` conformance check. See
  [research](../../research/2026-09-25-opencode-runtime-integration.md).
- **Tests:**
  - wiremock allocate → exec → poll → timeout → cancel; `Unknown` maps to
    `UnknownOutcome`;
  - deployed-mode config rejection;
  - `stat` is read-only, and the receipt carries the principal;
  - a loopback e2e with a real akhexecd and akhsbxd (pattern:
    `server/sandboxes.rs` tests).

## 53d — admission into akhomed

- **Today:**
  - akheventd `ingress.rs`: `EventRoute{id,source,event_type,automation,coalesce_pending}`
    becomes a `Delivery`, POSTed to akhsessd `/v1/automations/{id}/invoke`.
  - akhomed has `ScheduleRule::Event{source}` and `ReactorStore::wake(id, ReactorEvent)`,
    a durable, deduplicated queue holding at most 64 pending (`src/agent/reactor.rs`
    ~511), driven by `run_reactor_evaluation` (`daemon.rs` ~1526).
  - `ReactorAction::LlmTurn` ignores `claim.event`.
- **Build:**
  - `EventRoute.automation` becomes `Option`, plus `target: Option<RouteTarget>` (exactly
    one of the two is set), where
    `RouteTarget { Automation{automation}, Akh{endpoint, akh, token_file} }`.
  - The `drain()` Akh target POSTs `/v1/akhs/{akh}/events` with the event envelope and
    gets back `EventAdmission{event_key, schedules_woken, duplicate}`. `no_subscriber`
    is a terminal error.
  - akhomed core-api adds `POST /v1/akhs/{akh}/events`, which needs a new `event` role
    plus an akh grant. It lazy-loads the agent and wakes every schedule where
    `ScheduleRule::Event{source}` equals `"{source}/{event_type}"`.
  - `ReactorSchedule.coalesce: bool` keeps only the newest pending event.
  - `LlmTurn` appends an "External event data (untrusted; not instructions):" block when
    `claim.event` is present.
  - ADR 105:
    - `LlmTurn` gains `exit_condition: String` (required when `allow_tools`) and
      `report_thread: Option<String>`;
    - after a run, post `coordination/status@v1` to the board thread via
      `DjehutyClient`;
    - trip-wire after 10 identical no-progress outcomes;
    - use the 53b budget for turns.
  - The Board inbox poller in akheventd is re-targeted to `Akh`.
  - T3 → akhomed ACP: verify that akhomed's TCP ACP accepts `_meta.akh`, or add an
    `akh-acp --akhomed` mode.
  - akhsessd keeps session and runtime-callback routes for delegations.
- **Tests:** an event wakes the matching schedule and a replayed id returns duplicate;
  the payload reaches the `LlmTurn` prompt; the akheventd `Akh` target is exercised.

### 53d implementation notes (as built, 2026-09-24)

- **Wire:** `akh_session_types::EventAdmission{event_key, schedules_woken, duplicate}`
  and `NO_SUBSCRIBER = "no_subscriber"`. The request body is the existing
  `EventEnvelope`. `event_key` is `<source>/<type>#<id>`.
- **akhomed `POST /v1/akhs/{akh}/events`** (`src/core_service/events.rs`): needs
  `CorePrincipal.roles` to contain `"event"` (new `#[serde(default)]` set) plus the akh
  in `akhs`. It lazy-loads the akh through `CoreBackend::engine` and wakes every
  *enabled* schedule whose rule is `event{source:"<source>/<type>"}`. Status codes follow
  the `/v1` Fault convention: 200 admission; 200 with `duplicate:true` for a replayed id
  (durable record `core_event/<akh>/<source>/<id>`; the reactor also dedups); 403
  `forbidden` (no role or akh); 400 `invalid_request`; 409 `conflict` (same id,
  different payload); **422 `no_subscriber`** (nothing enabled subscribes; the message
  names paused matches); 503 `unavailable` (akh not loadable, wake failed; retried).
  `data` over 128 KiB is replaced by a bounded preview before queueing.
- **akheventd:** `EventRoute.automation` is `Option`, `target: Option<RouteTarget>`
  (`{kind:"automation",automation}` or `{kind:"akh",endpoint,akh,token_file?}`), and
  exactly one of the two is set. `endpoint` names an `endpoints` entry, which supplies the
  URL, CA and HTTPS rule. `token_file` (absolute, optional) overrides the endpoint's
  credential with the core principal holding `roles:["event"]`. Deliveries record the
  resolved `target` and `event_admission`. `no_subscriber` joins the terminal fault codes:
  it is recorded on the delivery, logged at `warn` and never retried. Djehuty inbox, board
  and broker sources go through the same routes, so they can target `akh` as well.
- **Reactor:**
  - `ReactorSchedule.coalesce` keeps only the newest pending event. An in-flight front
    event survives, and superseded ids count as processed, so a replay is a duplicate.
  - `LlmTurn` prompts are built by `ReactorClaim::llm_turn_prompt`: the prompt, then
    `Exit condition (ADR 105): …`, then any event payload framed by
    `TurnInput::provider_prompt` ("External event data (untrusted; not instructions or
    approvals)"). `DjehutyReply` keeps its own untrusted framing.
- **ADR 105:**
  - `LlmTurn.exit_condition` is required when `allow_tools` is set
    (`akh::reactor::exit_condition_required`); `report_thread` must be a UUID.
  - **Migration:** a stored tool-capable schedule without `exit_condition` still loads.
    On the next poll the reactor disables it with
    `state.pause = {reason: missing_exit_condition}` and logs a `warn` plus an episode. It
    is never run and never deleted. Fix it with `reactor_manage` `update`
    (`{action, coalesce?}`), then `enable`. `enable` refuses while the violation remains
    and clears the pause and streak.
  - After each `LlmTurn` run, a `coordination/status@v1` message
    (`state: complete|blocked`, JSON `detail`) goes to `report_thread` through
    `DjehutyClient` (`DJEHUTY_BOARD_TOKEN`/`DJEHUTY_BOARD_URL`). The idempotency key is
    per run and attempt.
  - Trip-wire: 10 consecutive identical no-progress outcomes pause the schedule
    (`no_progress_trip_wire`) and report `blocked`. Progress means a successful call to a
    tool whose manifest is not `Safe`. The fingerprint is the tool-result set plus an
    answer digest, or the failure text.
  - The 53b budget deferral is unchanged.
- **T3 → akhomed ACP:** akhomed's TCP ACP does **not** accept `_meta.akh`. It speaks
  newline-delimited JSON-RPC over raw TCP (not WebSocket), authenticates `initialize` by
  `params._meta.authorization = "Bearer <token>"` against `AKHOMED_SHARED_SECRET[_FILE]`
  (each non-empty line is one accepted token), and routes by
  `session/new` `_meta["akh/workspace"]` (or `cwd = akh-ws:<name>`). `akh-acp --akhomed`
  bridges this. It injects both fields, keeps `--t3-compat`, and ignores
  `--workspace-profile`. A non-loopback `tcp://` endpoint needs `--allow-remote-tcp`.
  Invocation for `t3_entrypoint.py`:
  `akh-acp --akhomed --endpoint tcp://127.0.0.1:<acp.tcp_port> --token-file secrets/t3-<role>.key --akh <role> --t3-compat`.
  akhomed needs `[acp] tcp_port = …` and `AKHOMED_SHARED_SECRET_FILE` listing the three
  T3 keys. Caveat: the ACP shared secret is not scoped per akh. Any of those keys can
  open a session on any akh the daemon serves, whereas akhsessd enforced
  per-principal akh grants.
- **53i follow-up:** `[[mcp.servers]] allow_insecure_http` (default false). A
  credentialed `http://` server is rejected (`akh::mcp::config::insecure_http`) unless
  its host is exactly `localhost`, `127.0.0.1` or `[::1]` (the same set as akheventd's
  `Endpoint::validate`, so `127.0.0.2` is rejected too) or the flag is set.

## 53e — delegation grants

- **Today:**
  - akhsessd calls the core as the shared principal `sessions`.
  - The broker forwards `{request_id, session_id, akh, tool, arguments}`
    (`broker.rs` ~299), and the core never sees the delegate.
  - `authorize` (`core_service.rs` ~127) only checks the akh list.
  - `DerivationKind` has 108 variants (tags 0-107, `KIND_NAMES:[_;108]`).
  - The `delegation` feature is in no profile; decision 1 fixes that.
- **Build, in `src/core_service/grants.rs`** (or `core_service_grants.rs` if
  `core_service` stays a single file):
  - `GrantSecret`: 32 random bytes, base64url, never stored; only its blake3 hash is.
  - `DelegationGrant { id: Uuid v7, akh, principal, service_principal, tools: BTreeSet<String>, session_id: Option<String>, issued_at, expires_at, revoked }`.
  - `GrantTools { ReadOnlyPlusMemoryWrite, Explicit }`.
  - `GrantAuthority::{mint, resolve, revoke}`, stored in CoreApp's `Store` under
    `key("delegation_grant", [blake3(secret)])`.
  - `GrantError` variants: Missing, Unknown, Expired, WrongAkh, WrongPresenter,
    SessionMismatch, ToolNotGranted (its help lists the granted tools), TtlTooLong,
    CoreApiDisabled.
  - `CorePrincipal.grants: GrantPolicy { Optional (default), Required }`.
- **Wire changes:** `CognitiveToolCall.grant: Option<String>` and
  `NewSession.grant: Option<String>`. akhsessd keeps the secret the way it keeps MCP
  capabilities, not in `SessionRecord`, and the broker adds `grant`.
- **Enforcement order in `tool_call`:**
  1. authenticate;
  2. authorize the akh;
  3. resolve the grant (or the legacy `ToolScope` when the policy is `Optional` and no
     grant is presented);
  4. check the permit.

  A denial returns `success:false` with the diagnostic and never calls `begin()`. Strip
  `grant` before computing the idempotency fingerprint.
- **Attribution:** run the call inside a thread-local
  `provenance::with_delegate_attribution(...)`.
  - `DerivationKind::DelegatedCall { principal, grant_id, tool, session_id, request_id }`
    is tag 108, `KIND_NAMES[108]="delegated_call"`, with no sources. Update any
    exhaustive matches (`explain.rs` ~1204).
  - `Engine::store_provenance` stores a sibling `DelegatedCall` record while an
    attribution is active.
  - `memory_write` uses `DelegatedCall` under attribution.
- **Mint call sites:**
  - ACP `_akh/delegation/grant/{mint,revoke,list}` on `ServerControl`;
  - automatic minting in a new `DelegationTargetKind::NativeSession{service, workspace_profile}`,
    which needs a small akhsessd client in akhomed (decision 8);
  - remove the dead `LocalClaude` `/mcp` target.
- **Tests:**
  - a required policy with no grant → `missing`;
  - `kg_mutate` outside the grant → `tool_not_granted`;
  - `memory_write` → provenance `DelegatedCall`;
  - expired (injected clock), wrong akh and session mismatch are denied;
  - the secret is absent from the store scan;
  - `tag_for_name("delegated_call") == Some(108)`.

## 53f — Roman cutover (operations; after merge and deploy)

- **Live layout:**
  - `dev.roman-akhs.core` runs `bin/akhomed-d563ba34a247` with
    `AKH_CORE_CONFIG=config/core.json` (127.0.0.1:18430, principal `sessions`,
    `sqlite:true`) and
    `DATABASE_URL=sqlite:///…/roman-akhs/state/core/akh.sqlite3?mode=rwc`.
  - XDG dirs point at `roman-akhs/core-*`.
  - `core-config/akh-medu/config.toml` has `[database]`, `[daemon] auto_start=[]` and
    `[exec] kind="none"`.
  - Workspace TOMLs are at `core-config/akh-medu/workspaces/{janus,minerva,vesta}.toml`,
    with `system_prompt` = `roles/common.md` + `roles/<role>.md`,
    `[ollama_endpoint] kind="local"` (wrong for cloud), `[autonomy] enabled=false` and
    `capture_transcripts=true`.
  - Turns today run in akhrtd with profile `ollama-<role>` (glm-5.3,
    `https://ollama.com/v1/chat/completions`, `ROMAN_OLLAMA_KEY` injected by
    `daemon_credentials.py` from
    `op://Homelab/Ollama API Key Akhrtd Command Group/api key`).
  - The `sessions.json` automation instructions become the reactor `LlmTurn` prompts.
- **Target TOML:**
  - `model="glm-5.3"`,
    `[ollama_endpoint] kind="cloud" base_url="https://ollama.com" api_key="env:ROMAN_OLLAMA_KEY"`;
  - `[autonomy] enabled=true` plus budget;
  - `[exec] kind="none"` until an execd sandbox profile exists;
  - `[[mcp.servers]]` for anima, board, sokar, seshat, observer and laya (53i).
- **Host config:** `deployment="deployed"`,
  `auto_start=["janus","minerva","vesta"]`. `daemon_credentials.py` gains a `core` role
  that injects `ROMAN_OLLAMA_KEY` and the MCP tokens, and the core plist execs through
  it.
- **Runbook:**
  1. Pause admission (bootout events, observer and laya-controller); wait for no running
     turns.
  2. Bootout core.
  3. `sqlite3 state/core/akh.sqlite3 ".backup backups/phase53-<ts>/akh.sqlite3"`, plus
     `shasum`. Copy `core-config/`, `config/` and the plists. Never unlink the `.lock`
     files.
  4. Install `akhomed-main-<sha>` through `configure_core.py`; keep the old binary.
  5. Start core; check ACP and `/v1` health.
  6. Create reactor schedules with `reactor_manage`; re-target the akheventd routes and
     restart events.
  7. Verify with these queries:
     - `select hex(akh_id),harvest_watermark,cycle_count from akhomed_akh_state;` —
       watermark > 0
     - `select count(*) from akhomed_episodes;` — > 0
     - `select count(*) from akhomed_sleep_metrics;` — > 0
     - `select kind,count(*) from akhomed_mutation_log group by kind;`
     - a Board message triggers a reactor run
  8. Hygiene: install the schema pack, merge the `next_check` spellings, run
     contradiction repair.
  9. Retire the `ollama-*` profiles and runtime plists after 24 h.

  Rollback: restore the backup and the old plist and binary.

## 53g — the Roman scripts

- **Already in the repo** under `deploy/roman-mini/`:
  - identical to live: `provider_failure.py`, `t3_activity.py`, `vesta_actions.py`,
    `vesta_ops.py`;
  - drifted from live: `daemon_credentials.py`, `janus_escalation.py`, `observer.py`,
    `t3_entrypoint.py`.
- **Not in the repo:** `laya_controller.py`, `laya_service.py`, `mergiraf_worker.py`.
  The Anima runner glue (`~/.config/anima-runner/roman-chat/`, `anima-activity.py`,
  `roman-chat.py`) needs its own inventory.
- **Step 1 (this phase):**
  - Sync every live script into `deploy/roman-mini/`. Live is the source of truth:
    copy it, then review the diff.
  - Add `core` to `daemon_credentials.py`.
  - Re-point `t3_entrypoint.py` at akhomed ACP.
  - Mark `t3_activity.py` obsolete after the cutover.
  - Add `deploy/roman-mini/INVENTORY.md` listing each script: purpose, IO, plist, and
    target replacement (akh reactor, skill, akheventd adapter, or deploy tooling).
- **Step 2 (akh-native):**
  - `observer.py`'s event production becomes a Rust akheventd source, and its
    monitoring snapshot becomes a Vesta tool.
  - `laya_controller` becomes a Janus reactor plus skill.
  - `janus_escalation` and `mergiraf` become Janus delegation (53e) plus skills.
  - `provider_failure` folds into akhomed's provider error taxonomy.

  File a work item for each of these conversions that doesn't land in this phase.

## 53h — docs and clean-up

- Delete the `core-server` alias and `docker/core-service.Dockerfile` (or rebuild it
  with `--features server`).
- Update `deploy/roman-mini/`: README build lines, `configure_core.py`, `prepare.py`.
- Add superseded notes (pointing at ADR 113) to:
  - `docs/wayfinder/agent-loops/implementation.md` ~179-189;
  - `deployment.md` lines 3 and 17;
  - `assets/daemon-boundaries.md` §1 (the akhomed, akhsessd and akheventd rows).
- Update:
  - `docs/agent-backends.md` ~29-30 and `docs/ops/akhomed-feature-profiles.md`;
  - `docs/components.md` ~27 ("9 tools");
  - `docs/ai/architecture.md`: add `core_service` and the deployment topology, and bump
    "last updated";
  - the CLAUDE.md roadmap (Phase 53 line; already added on `docs/full-akh-always`).
- Add a CI step: `cargo test --lib --features server core_service`.
- Leave the historical research and plan docs as they are.

## Integration fix-ups (branch `phase53/fixups`, 2026-09-24)

Three gaps found while integrating 53a–53e, 53g and 53i.

### 1. ACP TCP tokens are scoped per akh

akhomed's ACP TCP listener used to accept any `AKHOMED_SHARED_SECRET` token for
every akh (akhsessd scoped each principal to its akhs). Principals now live in
the **host config** (`config.toml`, `[acp]`), not in `core.json`: the ACP
listener runs whenever `[acp] tcp_port` is set, while `/v1` exists only with
`AKH_CORE_CONFIG`, so tying ACP auth to `core.json` would make it depend on an
unrelated surface and feature.

```toml
[acp]
tcp_bind = "127.0.0.1"
tcp_port = 8201

[[acp.principals]]
id = "t3-janus"
token_file = "/abs/path/secrets/t3-janus.key"  # absolute; one 24+ char token per line
akhs = ["janus"]                               # ["*"] alone = every akh

[[acp.principals]]
id = "operator"
token_file = "/abs/path/secrets/acp-operator.key"
akhs = ["*"]
control = true                                 # server-wide _akh/* control plane; default false
```

- A token maps to exactly one principal. Token files are re-read on every
  `initialize` (so rotation can overlap old and new lines), every comparison is
  constant-time, and two principals may not share a token.
- `session/new` (the effective workspace, including the default), `session/load`,
  `session/prompt`, `session/cancel` and every `_akh/*` data-plane request are
  checked against the principal's `akhs`. A refusal is JSON-RPC 403 with
  `data.code = "akh::acp::workspace_not_granted"` plus `help`.
- Control-plane methods (`CONTROL_METHODS`) need `control = true`
  (`akh::acp::control_not_granted`).
- Requests before a successful `initialize` are refused (`akh::acp::unauthenticated`).
- **Legacy:** with no principals configured, `AKHOMED_SHARED_SECRET[_FILE]` keeps
  working, grants every akh and the control plane, and startup logs a `warn` that
  it is unscoped. With principals configured the shared secret is ignored for TCP
  (a `warn` says so if it is set). The Unix socket is unchanged (local, unauthenticated).
- `akh-acp --akhomed` needs no change: `--token-file` must hold the token of a
  principal whose `akhs` include `--akh`.
- Code: `src/acp/auth.rs` (`AcpPrincipalConfig`, `AcpIngressAuth`,
  `PrincipalAuth`, `AcpAccess`, `AcpAuthError`); `AcpSection.principals` in
  `src/config.rs` (present only with the `acp` feature).

### 2. Each akh posts to the Agent Board as itself

53d built one process-wide `DjehutyClient` from `DJEHUTY_BOARD_TOKEN`, so all
akhs reported as one identity. The workspace TOML now carries a `[board]` table,
resolved per akh by the daemon for `coordination/status@v1` run reports,
`DjehutyReply` and board ingress:

```toml
# workspaces/janus.toml — keep every scalar above the first [table]
model = "glm-5.3"

[board]
url = "https://board.lab.home"                # http:// only for loopback or allow_insecure_http = true
auth = { token_env = "ROMAN_BOARD_JANUS" }    # or { token_file = "/abs/path/janus-board.key" }
agent_id = "<janus board agent UUID>"         # needed for inbox ingest (self-message suppression)
```

- `board` is a known table (`url`, `auth`, `agent_id`, `allow_insecure_http`);
  `auth` and the table itself use `deny_unknown_fields`. The URL, env-var name and
  absolute token path are validated at load (`akh::board::config::*`).
- The token is read on every use. If a configured `[board]` token cannot be read
  (`akh::board::auth::*`), the report is **not** posted; there is no silent fallback
  to the shared identity.
- Without `[board]` the daemon falls back to `DJEHUTY_BOARD_TOKEN` /
  `DJEHUTY_BOARD_URL` / `DJEHUTY_AGENT_ID` and warns once per akh.
- A `[board]` without `agent_id` still posts reports and replies. Inbox ingest is
  disabled for that akh, with a one-time warning.

### 3. Delegated native sessions are closed

`NativeSession` delegation now ends through one guard on every exit path. The guard
first revokes the grant and then closes the akhsessd session:
`POST /v1/sessions/{id}/archive`. If akhsessd refuses to archive because a turn is
still outstanding (after a timeout or a poll failure), the guard calls
`POST /v1/sessions/{id}/retire` instead. Retiring cancels the turns, and the
reconciliation sweep then archives the session. Archiving removes the 0600 grant
slot. A failure to close is logged at `warn` and never masks the delegation's result.

### Cutover implications (53f)

- Add `[[acp.principals]]` for the three T3 keys (`akhs = ["janus"]` etc.) and an
  operator principal if the ACP control plane over TCP is needed. Once principals
  exist, `AKHOMED_SHARED_SECRET_FILE` no longer opens TCP ACP.
- Add `[board]` to `janus.toml`, `minerva.toml` and `vesta.toml` with
  `token_env = "ROMAN_BOARD_JANUS|MINERVA|VESTA"`. `daemon_credentials.py`'s `core`
  role must export those three variables into akhomed's environment.

## 53j — memory hygiene (as built, 2026-09-24)

The operator's original ask ("memory self-improvement"). Before 53j nothing ever
retracted: every status update added a live value plus a `TemporalConflict`
warning, and one predicate accumulated under several spellings (`next_check`,
`next-check`, `next check`).

- **Schema-pack cardinality and aliases** (`src/schema/source.rs`, `compile.rs`):
  `RelationDef` gains `cardinality` (`multi` default | `single_valued`), `temporal`
  (only with `single_valued`) and `aliases`. They lower to
  `(r, onto:cardinality, onto:cardinality/single-valued[-temporal])` and
  `(alias, onto:alias-of, r)`, so old packs compile to identical triples and
  checksums. The blob format is unchanged. New diagnostics:
  `akh::schema::{temporal_requires_single_valued, alias_is_relation, duplicate_alias}`.
- **Read side** (`src/schema/cardinality.rs`): declarations are read straight from
  the live KG, with no cache and no persisted registry, so install, wake and
  relocation all make them visible. An alias counts only when its target is a
  declared relation `(r, is-a, relation)`. The ADR 081 promotion `onto:alias-of`
  edges point at properties and do not rewrite predicates.
- **Engine write path** (`Engine::add_triple_locked`):
  - A declared alias is stored under its canonical relation.
  - For a single-valued predicate, older live values in the same
    `(subject, compartment)` are retracted **after** the new edge is
    published, with `RetractionScope::CascadeIfOrphaned` (below).
  - A write older than the live value (`newest_first`: timestamp, then
    confidence, then object id) is recorded as superseded and not published.
  - Same-(s, p) contradictions are filtered out for these predicates, so they
    no longer warn.
  - Provenance: `DerivationKind::MemoryHygiene { action: HygieneAction, subject,
    predicate, successor, valid_until }` is **tag 109** (`memory_hygiene`,
    appended after 53e's 108). `valid_until` is set for temporal predicates.
  - New engine APIs: `supersede_triple` (retraction with provenance, scope
    `CascadeIfOrphaned`) and `rewrite_predicate` (assert the re-spelled edge,
    then remove the alias edge with scope `ExactTriple`, since the fact still
    holds).
  - **Retraction scope** (`tms::RetractionScope`,
    `Engine::remove_triple_scoped`): the TMS tracks support per *symbol*, so
    the historical `remove_triple` (`Symbol`) retracts the object symbol and
    cascades through every conclusion it supports even while other subjects
    still assert that value. Hygiene never does that: `ExactTriple` removes
    only the edge; `CascadeIfOrphaned` cascades only when no other live edge
    still has the object as its object. Superseding `(wi-569, status,
    resolved)` therefore leaves `resolved`, `was_retracted(resolved)` and
    anything derived from it alone while another subject is `resolved`.
    All scopes commit through the same exact-edge `RetractBatch` transition.
    *Follow-up (Anima WI 01a0d569-1cf1 item 6):* `Engine::remove_triple`, every
    bulk API (`remove_triples_batch`, compartment removal, pattern/SPARQL
    delete, purges), `kg_delete` and ACP `_akh/triples/remove` (hence
    `akh` CLI deletes) now default to `CascadeIfOrphaned` too. `Symbol` is
    only reachable through the explicit `Engine::retract_object_symbol`,
    whose one production caller is replay of a legacy single-edge `retract`
    WAL record (that writer relied on replay re-deriving the symbol cascade).
- **Structural keys** (`src/schema/canonical.rs`): `PredicateKey` = NFKC +
  Unicode lowercasing + separator runs (whitespace, `_`, dash punctuation)
  collapsed to `_`. There are no word lists, and namespaces stay distinct.
  Collision families pick the canonical spelling deterministically:
  1. the single declared relation;
  2. an engine-reserved name (`is-a`);
  3. the most live triples;
  4. the smallest label.

  A family with two declared relations, or with an alias belonging to another
  relation, is `Ambiguous`. A family made only of declared aliases of an outside
  relation is `DeclaredAliases`. Both are reported, never merged.
- **Hygiene pass** (`src/schema/hygiene.rs`, `run_hygiene`): bounded by a
  `MutationBudget`, it runs:
  1. an auto-merge of exact collisions;
  2. the single-valued backlog;
  3. functional-predicate repair per policy: `warn` reports, `replace` keeps
     the newest value, `reject` keeps the oldest.

  The `HygieneReport` lists merges, superseded and repaired values, provenance
  written, truncation, ambiguous families, pending declared aliases and the top
  20 undeclared multi-value predicates. The pass is idempotent. The explicit
  merge API is `execute_merge(MergeRequest { selection: Explicit | Declared |
  Collisions, dry_run, limit })`.
- **Daemon**: `DaemonTask::MemoryHygiene` (`memory_hygiene`, `TaskCost::Free`,
  autonomy-gated, unmetered) runs on the sleep tick. The order is harvest,
  then hygiene, then sleep cycle, and none of the three depends on another
  succeeding. It works on the engine handle, never the agent lock, and logs
  `memory hygiene pass` under `akh::autonomy`. The 53b `FactDelta` carries its
  retractions.
- **Config**: `[memory] contradiction_policy = "warn"|"replace"|"reject"`
  (default `warn`, unchanged) and `hygiene_batch = 1000` form a known table.
  akhomed's `get_engine` applies the policy to the engine's `Phase9Config`, and
  the daemon gets `hygiene_batch` through `DaemonAkhSettings.memory`.
  `ContradictionPolicy` now serializes lowercase and accepts the old
  capitalized names.
- **Surfaces**:
  - tool `kg_merge_predicates` (`Dangerous`, `WriteKg`);
  - **`kg_delete` is now registered** in the live registry. It was never
    excluded on purpose; it existed only in test registries. It is
    `Dangerous`, so plan mode denies it and `/v1` exposes it only by scope or
    grant;
  - ACP control methods `_akh/memory/hygiene` and `_akh/memory/merge_predicates`;
  - CLI `akh schema hygiene [--dry-run]` and
    `akh schema merge-predicates (--canonical X --alias Y… | --declared | --collisions)`.
- **Roman seed pack**: `deploy/roman-mini/schemas/roman-memory.json` declares 11
  temporal single-valued state predicates with their observed spellings. The
  evidence per relation is in its `description`. CUTOVER step 10 is rewritten
  around it.
- **Acceptance** (`tests/hygiene_accept.rs`, `#[ignore]`, `AKH_HYGIENE_ACCEPT_DB`
  pointing at a private copy): the test runs `snapshot::wake`, then
  registry admit, install and activation, then hygiene until not truncated,
  then the declared-alias merge, then two more passes. On a copy of the
  2026-09-24 Roman store every subject ends with exactly one live value for
  every declared predicate, and the second pass is a no-op. The `next_check`
  family results:

  | akh | before | after |
  |---|---|---|
  | janus | 262 live over 5 spellings, 127 subjects, 29 multi-valued, max 46 | 127 |
  | minerva | 125 / 56 subjects | 56 |
  | vesta | 103 / 54 subjects | 54 |

  In total the first pass auto-merged 142 exact-collision families (543
  triples; for example `is_a`/`is a` → `is-a` and `blocked by`/`blocked-by` →
  `blocked_by`). The run made 805 retractions and wrote 881 `memory_hygiene`
  records (555 `predicate_merge`, 326 `single_valued_supersede`). Each pass takes
  1–3 s; wake dominates the run at 50–150 s.



## Review fixes, core (branch `phase53/rfA-core`, 2026-09-24)

### Read-only is declared, never inferred

`DangerInfo` has a new field, `effect: ToolEffect` (`Mutating` | `ReadOnly`,
`#[serde(default)]` = `Mutating`). `is_cognitive_read_only` now needs three
things: an explicit `ReadOnly` declaration, `Safe`, and read-only cognitive
capabilities. It drives both the `/v1` catalog's `read_only` and the default
`ReadOnlyPlusMemoryWrite` grant.

- These tools were `Safe` but mutating, and could be granted by default. They are
  now `Cautious` and `Mutating`:
  - `core_memory_append` and `core_memory_replace` (also `WriteKg`);
  - `task_list`;
  - `inference_config`.
- These tools are marked `ReadOnly` after an audit of every `Safe` tool:
  `kg_query`, `similarity_search`, `reason`, `gap_analysis`, `memory_recall`,
  `archival_search`, `query_graph`, `entity_schema` (3), `study_note` catalog,
  list and read, `structured_task_list`, `tool_query`, `search_wiki`,
  `openclaw_discover`, and `agent_list`.
- A new tool is mutating until its author declares otherwise.

### Every delegated mutation is the delegate's provenance

- A `/v1` call of a mutating tool that stores no provenance of its own
  (`kg_mutate` → `add_triple`, `core_memory_*`, `task_list`) now gets exactly one
  call-level `DelegatedCall`, keyed by `request_id`. It derives from the first
  symbol the tool touched (the written subject), with the rest as sources. If
  the tool touched no symbol, the record hangs off the per-delegate entity
  `delegate:<principal>`.
- Siblings are deduplicated. A call gets at most one `DelegatedCall` per
  symbol, and a tool that stores its own `DelegatedCall` (`memory_write`) gets
  no sibling.
- Read-only calls and replays record nothing.

### Rollout order for the new wire shapes

`OperationReceipt.principal` (53c) and `DerivationKind::DelegatedCall`
(tag 108, 53e) break older readers. `deny_unknown_fields` stays, so deploy
**readers before writers**:

1. Restart akhsessd, akhsbxd and akhomed.
2. Restart akhexecd last.
3. Move every akhomed that can wake or relocate the same akh (ADR 049)
   together, before the first delegated `/v1` call.

Roll back in the reverse order. The operator runbook is
`deploy/roman-mini/CUTOVER.md` ("Wire compatibility").

### Smaller fixes

- **Lock hygiene.** akhomed caches each resident Agent's `Arc<ToolRegistry>`
  (weak handles tied to the Agent instance) when it creates the Agent. `/v1`
  tool resolution and `_akh/delegation/grant/mint` no longer take the Agent's
  std `RwLock`. The registry `Arc` is fixed for the Agent's lifetime.
  Registration, including lazy MCP and `sync_live_tools`, mutates it in place.
- **Legacy exec policy.** `allow_legacy_exec` is fixed at startup together with
  the process mode (`deployment::set_process_policy`). A replaced config can
  only tighten it (`ExecPolicy::only_stricter`).
- **Native delegation availability.** `DelegationGrantIssuer::is_available`.
  `CoreGrantIssuer` is available only once the `/v1` grant authority is
  published, so an akh without `AKH_CORE_CONFIG` no longer picks
  `NativeSession` and then fails every delegation with `GrantUnavailable`.
- **Grant binding.** An unbound grant binds atomically to the first session that
  presents it. If a create's answer is lost (a timeout after sending, or an
  unreadable reply), it is re-sent with the same `request_id` (akhsessd
  deduplicates creates) to recover the session id, and the session is then
  archived.
- **Event replay key.** The key is `(akh, source, type, id)`. One id under two
  event types is two events.
- **ACP.**
  - `session/prompt` authorizes the session's akh before checking the session
    out, the same as `session/load`.
  - The SET forms of `_akh/permission/mode` (a `mode` param) and
    `_akh/psyche/archetypes` (any weight) need `control`. The GET forms stay
    per-akh.

## Follow-up: ACP sessions belong to their creator (Anima WI 01a0d569-1cf1, item 2)

Per-principal scoping (fix-up 1) limited each connection to its akhs, but two
principals granted the same akh could still load, prompt or cancel each
other's sessions by id, and ids were nanosecond timestamps.

- **Owner recorded at `session/new`.** Each session stores a `SessionOwner`:
  the connection's principal id, `legacy` for the unscoped shared secret, or
  `local` for the Unix socket and in-process embeds.
- **Checked before any side effect.** `session/load`, `session/prompt` and
  `session/cancel` check the owner before any LRU touch, turn checkout or
  cancel flag, and then check the session's akh. No `_akh/*` method takes an
  ACP session id.
- **Session operators.** Callers that already hold the control plane may use
  any session on an akh they are granted: principals with `control = true`,
  the legacy shared secret and the local socket. An operator using another
  owner's session is logged at `info`. A control principal is still limited to
  its `akhs`.
- **No existence oracle.** A caller without `control` gets the same error for
  another principal's session and for an unknown id: `akh::acp::session_not_owned`,
  JSON-RPC invalid-params (the code an unknown session always had, so clients
  that fall back to `session/new` still do), and the same message and help.
  Because the owner is checked before the akh, a session on an akh that is not
  granted no longer answers `workspace_not_granted` either. Operators still see
  the plain unknown-session error.
- **Persistence.** ACP sessions are memory-only, so a daemon restart drops them
  and no stored session lacks an owner. `SessionOwner`'s default is `legacy`: a
  session restored from state that predates ownership can be used only by the
  legacy secret and by operators.
- **Unguessable ids.** Ids are now `akh-<UUIDv4>`, with 122 bits from the OS
  CSPRNG. The `akh-` prefix and the opaque suffix are unchanged.
- Code: `SessionOwner`, `SessionUse`, `AcpAccess::{Local, SharedSecret}` and
  `AcpAccess::authorize_session` in `src/acp/auth.rs`; `SessionRoute` and
  `AkhAcpAgent::authorize_session_access` in `src/acp/mod.rs`.


## Follow-up: long-period cadence survives restarts (Anima WI 01a0d569-1cf1, item 1)

The staggered start made every heavy task wait one full period after each
start, so a task whose period is longer than the gap between restarts (the
24 h librarian on an akh redeployed daily) never ran.

- **Which tasks.** `DaemonTask::cadence(period)`: a task with
  `FirstTick::AfterOnePeriod` and a configured period of at least
  `RESUMABLE_MIN_PERIOD` (15 min) is `Cadence::Resumable`. At the default
  intervals that is schema_discovery, gap_analysis, continuous_learning,
  knowledge_extraction, bridge_training, openclaw_sync,
  knowledge_elicitation, kg_compaction (opt-in), relation_enrichment,
  librarian, seshat_curation and heka_curation. Shorter tasks, infrastructure
  and the sleep tick (harvest, memory hygiene, sleep cycle) are unchanged.
- **Where.** A `LastRunLog` (task → `{at: unix secs, cause}`) in engine meta
  `daemon:last_run:v1`, in memory when the engine has no durable tier (same
  fallback as the budget ledger), and in snapshot `extra.autonomy.last_run`
  so relocation and wake keep it.
- **First tick.** `clamp(last.at + period - now, RESUME_MIN_DELAY (60 s), period)`:
  overdue tasks run after 60 s, so a restart storm cannot burst them, and a
  clock that stepped back never postpones a task past one period.
- **Never run.** At its first start the daemon writes a `first_seen` mark for
  every enabled resumable task without one. The task waits one period from
  that first start (as before), but later restarts no longer push it out.
  Gated-off tasks get no mark until they are enabled.
- **Marks.** Written when a tick finishes `Ran` or `Failed`. A failure is
  marked like a run so a failing heavy task is retried once per period, not
  after every restart. Skips (gated, over budget, contended) leave the mark.


## Follow-up: a built-in fleet memory pack for every akh (Anima WI 01a0d569-1cf1, item 7)

53j gave memory hygiene to akhs that have a schema pack declaring their state
predicates, and in practice only the Romans had one (`roman-memory`). Every
other akh still piled up status values. Now every akh gets a default pack,
`fleet-memory@1.0.0`, without an operator installing it.

### What the pack declares (`data/schema-packs/fleet-memory.json`)

The pack declares only predicates that hold one current value per subject by
construction:

| Relation | Cardinality | Evidence |
|---|---|---|
| `status`, `state`, `has_state`, `current_state`, `current_phase`, `next_action`, `next_check`, `planned_action` | single-valued, temporal | Roman store 2026-09-24 (the `roman-memory` descriptions): every multi-valued subject held successive updates of one state. The same aliases as `roman-memory` (separator variants, plus `has_next_check`). |
| `tool:danger_level`, `tool:schema` | single-valued | The engine writes these itself (`agent/tool_graph.rs`) each time a tool manifest is registered. In the 2026-09-25 store, 6 of 111 tool subjects per predicate held both the old and the new value, for example `tool:task_list` was `Safe` and `Cautious` at once after the review fixes reclassified it. |

These predicates are left out:

- **`agent:*` state** (`agent:has_status`, `agent:has_priority`). The
  namespace is reserved (`akh::schema::reserved_name`), and `agent/goal.rs`
  already removes the old value before writing a new one. The store has no
  multi-valued subject for either predicate.
- **Legitimately multi-valued predicates**: `agent:has_tag`, `agent:learned`
  (1,424 multi-valued subjects, which are real facts), `tool:access`,
  `psyche:*`, `is-a` and `part-of`.
- **Roman-specific state**: `observed_state`, `verified_state` and
  `boundary_state` stay in `roman-memory`. An observation log can legitimately
  hold several values, and "boundary" is Anima vocabulary.

The list is data in a reviewed file, and the code holds no word lists.

### Admission: built in, admitted by code review

ADR 081 decision 8 requires a human confirm before a pack is admitted to the
shared registry. The fleet pack never enters the registry. The binary embeds
it with `include_str!` (`src/schema/builtin.rs`, `BuiltinPack::FleetMemory`),
so the code review that merges the JSON is its admission, and a binary can
only install the revision it was built from. The registry, `akh schema
admit|install` and wake-time convergence are unchanged. No
`workspaces_active_microtheories` row is written, so `converge_workspace` never
looks for the pack in the registry.

A built-in pack may declare relations with `cardinality`, `temporal` and
`aliases` only. Types are global and immutable (decision 5), so a default tier
for them could give no precedence. `default_tier_triples` rejects any other
triple with `akh::schema::not_default_tier`.

### Precedence: the akh's own pack wins, in either install order

The KG stores one edge per `(s, p, o)` whatever the compartment. If the fleet
pack wrote `(status, onto:cardinality, …)`, that edge would merge into
`roman-memory`'s identical edge. It could not be told apart, and retiring the
fleet pack would delete `roman-memory`'s edge. The fleet pack therefore
installs on a **default tier** with its own predicates:

- `(r, onto:default-relation, relation)`
- `(r, onto:default-cardinality, …)`
- `(alias, onto:default-alias-of, r)`

These go under the compartment `mt:schema/builtin/fleet-memory@<version>`,
which cannot collide with a registry pack's `mt:schema/<name>@<version>`.
`schema::cardinality` reads the pack tier first:

- **Cardinality.** Any `onto:cardinality` edge replaces the default. A pack can
  now say `"cardinality": "multi"` explicitly, which lowers to
  `onto:cardinality/multi` and switches a default off for one relation.
  `RelationDef.cardinality` is now `Option`, so a pack that omits the field
  still compiles to the same triples and checksum.
- **Aliases.** A pack alias wins. A default alias does not rewrite a spelling
  that a pack declares as a relation.
- **Relations.** A default relation that a pack re-declares as an alias of its
  own relation stops counting as a relation.

A relation that the akh's pack declares without a cardinality keeps the
default. To opt that relation out, declare `multi`.

### Install, versions, opt-out

`schema::ensure_builtin_packs(engine, workspace, policy)` runs in akhomed
`get_engine` after the WAL is wired and before the engine is published. Hydrate
and wake both pass through `get_engine`, and the install is written to the WAL
like any other install. A failure is logged and never refuses the akh.

| Installed state | Result |
|---|---|
| None | `Installed`: declarations, the `akh:schema-installed` marker and the `ctx:specializes` activation edge from `mt:akh/<ws>-effective` |
| This version | `Current`, nothing written (idempotent) |
| An older version | `Upgraded`: the old compartment and activation edge are retired **first**, then the new version is installed, so the retirement cannot delete edges both versions share |
| A newer version | `NewerKept`: a rolled-back binary never downgrades an akh |
| `[memory] default_pack = false` | `Disabled`: installs nothing and retires any installed version |

`[memory] default_pack` defaults to `true`, is a known key, and is written back
only when it is `false`. Versions are dot-separated integers compared
numerically (`PackVersion`; `1.0` equals `1.0.0`). Bump the version in the JSON
to ship changed declarations.

### Acceptance (`tests/fleet_pack_accept.rs`, `#[ignore]`)

The run used a private copy of the 2026-09-25 Roman store, where
`roman-memory` was already active and hygiene had converged. For each akh it:

1. woke the copy (which converged `roman-memory` from the copy's registry);
2. ran `roman-memory` hygiene to its fixed point;
3. ensured the fleet pack;
4. ran hygiene to a fixed point again;
5. ran an idempotency pass.

| akh | fleet install | overlapping relations | conflicts with roman-memory | hygiene with both packs | roman-memory relations changed | last pass |
|---|---|---|---|---|---|---|
| janus | 35 triples | 8 | 0 | 4 mutations, 4 superseded, 0 merges | 0 | no-op |
| minerva | 35 triples | 8 | 0 | 4 mutations, 4 superseded, 0 merges | 0 | no-op |
| vesta | 35 triples | 8 | 0 | 4 mutations, 4 superseded, 0 merges | 0 | no-op |

- **Roman-only baseline.** The baseline pass was already a fixed point: 0
  mutations.
- **What the fleet pass superseded.** The 4 supersessions per akh are the
  stale tool manifests. `tool:danger_level` and `tool:schema` each went from
  39 live values on 37 subjects (2 subjects multi-valued) to 37 on 37.
- **Overlap.** Every overlapping relation kept `roman-memory`'s cardinality
  and alias targets, and every `roman-memory` relation kept its live triple
  count.
- **Wake time.** Wake took 65–175 s; the passes took under a second.

### Operator notes

- **Romans.** Nothing to do. `roman-memory` stays installed and wins on its
  relations. The fleet pack adds only the two tool predicates for them.
- **Other akhs.** The pack installs on the first open after the deploy. The
  next sleep-tick `memory_hygiene` pass clears the backlog, within
  `hygiene_batch` mutations per pass.
- **Opting out.** Set `[memory] default_pack = false` in the akh's TOML and
  reopen the akh. The pack is retired, but values it already superseded stay
  retracted: they are history, with `memory_hygiene` provenance.

### Code

- `src/schema/builtin.rs`: `BuiltinPack`, `PackVersion`, `DefaultPackPolicy`,
  `BuiltinPackOutcome`, `ensure_builtin_pack[s]`, `default_tier_triples`.
- `src/schema/cardinality.rs`: `DeclarationTier` and the two-tier reads.
- `src/schema/canonical.rs`: default relations anchor collision families.
- `src/schema/compile.rs` and `source.rs`: explicit `multi` and the
  default-tier vocabulary.
- `src/akh_config.rs`: `MemoryConfig.default_pack`.
- `src/bin/akhomed.rs`: the `get_engine` hook.
- New diagnostics: `akh::schema::{invalid_pack_version, not_default_tier}`.
