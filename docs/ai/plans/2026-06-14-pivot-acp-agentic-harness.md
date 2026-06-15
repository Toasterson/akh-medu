# Akh-Medu → Pure-ACP Agentic Harness

> Date: 2026-06-14
> Status: In progress (P0 complete)
> Anima project: `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")
> ADRs: 052 (ACP sole transport), 053 (first-class AkhConfig), 054 (peer-Engine
>       federation), 055 (filesystem abstraction), 056 (OpenClaw markdown import)
> Supersedes the "local model runner" framing of the daemon's interaction surface.

## Context

Akh-medu is shifting from a "local model runner" into a proper **agentic
harness**. The unit of agency is an **akh** (analogous to an OpenClaw "Agent").
Motivation: keep compute/environmental cost low by leaning on **small LLMs run
through Ollama** (operator-hosted or `ollama.com` cloud), and capitalize on a
wave of small-model research expected to land over the coming weeks. An akh must
behave consistently, so each akh pins exactly one model and carries its own
system prompt.

The daemon today is a single `axum` HTTP server (`akhomed`) exposing four
overlapping inbound surfaces — ~130 REST routes, a WebSocket chat (`/ws/{ws}`),
an `rmcp` MCP **server** (`/mcp`, `src/mcp/mod.rs` = 7009 lines / 85 tools), and
a bundled HTMX/Alpine web UI — plus a CLI (`akh`) that drives the daemon via ~90
typed `ureq` REST calls. There is **no gRPC** and **no ACP** anywhere. We are
collapsing all of that to a single transport: **the Agent Client Protocol
(ACP)** over a Unix socket and TCP (QUIC later). akhomed becomes the ACP
**agent**; the TUI, CLI, and anima become ACP **clients**.

Intended outcome: a small-LLM-backed, ACP-only, multi-akh harness with
Claude-Code-grade tools, per-akh model+prompt+memory configuration, federated
thematic memory, automation off-by-default, OpenClaw-style awakening, and a
green, trustworthy symbolic test suite.

### Decisions locked in (from clarifying questions)

1. **Rip-out scope = ACP-only.** Retire WebSocket chat, the inbound MCP server,
   **and** the REST API + web UI. The CLI becomes an ACP client. Outbound MCP to
   anima/seshat (`upstream_client.rs`, `agent/tools/mcp.rs`) is **kept**.
2. **OpenClaw import = local markdown files** (`MEMORY.md`, `SOUL.md`,
   `AGENTS.md`, `memory/YYYY-MM-DD.md`). Retire the bespoke `api.openclaw.dev`
   HTTP client.
3. **System prompt = explicit, authoritative, replaces the psyche-derived
   prompt.** Psyche becomes optional flavor, no longer the prompt source.
4. **Workspace assembly = full peer-Engine federation.** Related workspaces stay
   complete Engines; the akh queries across all of them (symbolic + vector);
   writes go to MAIN only.

### Defaults chosen (stated, not asked)

- akhomed = ACP **agent** (the thing connected into); TUI/CLI/anima = ACP
  **clients**.
- Transport order: **Unix socket + TCP first**, QUIC later (P8).
- Local TUI also runs through an ACP loopback over the Unix socket → exactly one
  "purely ACP" code path.
- ollama.com cloud key supplied via the existing `env:VAR` secret pattern
  (mirrors `UpstreamEndpointConfig::resolved_auth`); an akh pins
  `{endpoint + api_key + model}` as one unit.
- Data-plane CRUD is exposed as **custom `_akh/<group>/<op>` ACP JSON-RPC
  methods**, advertised via `agentCapabilities._meta`.

## Current state (grounded, with anchors)

- **Daemon/transport:** one `TcpListener` on `:8200` (`src/bin/akhomed.rs:3995`);
  router at `:3584-3965`; WS chat `ws_handler`→`handle_ws_session`
  (`akhomed.rs:1561-1674`); MCP nest `:3967-3991`. Discovery = PID file
  `{pid,port,bind}` + `GET /health` (`src/client.rs:65,111`).
- **Interaction core (reuse, don't rewrite):**
  `ChatProcessor::process_input(text,&mut Agent,&Engine)->Vec<AkhMessage>`
  (`src/chat.rs:127`) is transport-agnostic and shared by WS, MCP `chat`, local
  TUI, headless CLI. `AkhMessage` + `MessageSink` (`src/message.rs:15-65`) is the
  response envelope.
- **Multi-akh registry (reuse):** `ServerState` holds per-workspace
  `Arc<Engine>`, `Arc<Mutex<Agent>>` (one shared Agent per workspace,
  **ADR-027**), cached NLU pipeline, audit broadcast (`akhomed.rs:55-282`).
  `get_engine`/`get_agent`/`get_nlu_pipeline` are the lazy loaders.
- **No first-class akh:** an akh is implicitly `WorkspaceConfig`
  (`src/workspace.rs:133-161`) + `Agent` (`src/agent/agent.rs:162`) + `Psyche`.
  System prompt is **derived** at runtime (`build_system_prompt`
  `src/agent/model_manager.rs:394`; `PsychePromptSerializer`
  `src/compartment/prompt.rs:68`), not stored. Model is **global**
  (`OllamaConfig.model` `src/agent/ollama.rs:13`, local `base_url` only, no
  auth/cloud). `AgentModelManager` exists but is wired only in tests. `akh.toml`
  is named in delegation help text but never parsed.
- **Filesystem:** zero abstraction — all paths eagerly from XDG (`src/paths.rs`),
  raw `std::fs` everywhere; `FileIoTool` read path has no sandbox; no
  prefix/chroot.
- **Tools:** 26 native tools (`src/agent/tool.rs`, `src/agent/tools/`) cover
  Bash/Read/Write/WebFetch; **missing Edit, Glob, Grep, WebSearch, structured
  task-list**. `ToolInput` is a string-only `HashMap`.
- **OpenClaw:** `src/agent/openclaw.rs` fetches JSON journals from
  `api.openclaw.dev` (bespoke/fictional — real OpenClaw memory is local
  markdown). Two divergent markdown importers already exist
  (`src/agent/journal_ingest.rs` + `JournalIngestTool`).
- **Automation:** one `AgentDaemon::run` loop with 18 interval tasks + 2 startup
  bursts (`src/agent/daemon.rs:244`). "Does a daemon exist" is already opt-in
  (`auto_start`), but **no per-task switch**; `DelegationConfig` is hardcoded,
  never read from TOML.
- **Testing:** `cargo test --lib` is green (~2405 tests); **plain `cargo test`
  was un-compilable** until P0 (a non-feature-gated bin had 6 errors). P0 fixed
  this; the 121-test integration/persistence/hermeticity suite now runs.
  `src/store/mmap.rs` panicked on truncated headers (PVC hazard) — P0 hardened
  it to a typed `StoreError`.
- **External facts (high confidence):** ACP = official `agent-client-protocol`
  crate v0.14.x, JSON-RPC 2.0, newline-delimited, runtime-agnostic; its
  `ByteStreams` transport accepts any `AsyncRead/AsyncWrite` ⇒ Unix/TCP/QUIC with
  no protocol change. Client-provided `fs/read_text_file`/`fs/write_text_file`
  (gated by `clientCapabilities.fs.*`) are the mechanism for remote-workspace
  filesystem access; `session/request_permission` is the permission hook.

## Target architecture

`akhomed` runs one ACP agent (`agent-client-protocol`) served over a Unix socket
(XDG runtime dir) and TCP, behind a new `acp` feature. `session/prompt` →
`ChatProcessor` (run in `spawn_blocking`, lock dropped before await);
`AkhMessage`s stream out as `session/update` via a new `AcpSink: MessageSink`.
Data-plane CRUD → `_akh/<group>/<op>` custom methods over the same connection,
reusing `src/api_types.rs` structs as params. A first-class **`AkhConfig`**
(`akh.toml`, embeds `WorkspaceConfig`) carries model pin, explicit system prompt,
`related_workspaces`, `[autonomy]`, `[delegation]`, and path-mode. A
**`Filesystem`** trait branches Local (prefixed+sandboxed `std::fs`) vs Remote
(delegates to ACP `fs/*`+`terminal/*`). A **`Federation`** layer fans out reads
across MAIN + peer Engines, reconciling by label. Claude-Code-parity tools
(Edit/Glob/Grep/WebSearch/task-list) sit on the shared `Filesystem` backend
behind a harness-enforced permission layer. REST/WS/MCP-server/web UI are
deleted; outbound MCP stays.

## Key design decisions

- **ACP crate, not hand-rolled JSON-RPC.** Pin exact version (0.14.0). Bridge
  tokio↔futures with `tokio_util::compat` as needed.
- **Data-plane = `_akh/*` custom methods + `_meta` capability groups.** Rejected:
  admin-as-agent-tool (tool-calls are agent→client only) and a second control
  RPC (reintroduces the sprawl we're deleting). One connection, one transport,
  one discovery record.
- **Concurrency (ADR-027):** every Agent access goes through one `spawn_blocking`
  helper that locks the `std::sync::Mutex<Agent>` *inside* the closure and drops
  it before emitting `session/update`. Read-only Engine ops need no agent lock.
- **`AkhConfig` back-compat:** every new field `#[serde(default)]` so existing
  workspace TOML + bincode sessions on the PVC keep loading (pattern proven by
  `shared_partitions` + the legacy-seed-packs test, `workspace.rs:540`).
- **Federation reconciliation:** results are
  `FederatedHandle{engine_tag, SymbolId, label}`; merge/dedup on normalized label
  (`src/registry.rs:29,93`). Vector search gated by `compatible_vsa`
  (`dimension`+`encoding` match); "search by symbol" uses **label-anchored**
  per-peer query vectors (raw-vector broadcast would false-positive because VSA
  seed = raw id, `src/vsa/encode.rs:20`). Federate cleanly: KG fact lookup,
  SPARQL SELECT/ASK (sequential + label re-resolve), episodic recall, gated
  vector search. Do **not** federate: cross-engine traversal, analogy/binding,
  inference, aggregate SPARQL. Writes → MAIN only.
- **System prompt:** one prompt-assembly function; explicit `AkhConfig.system_prompt`
  is authoritative; psyche-derived text only if no explicit prompt. Collapses the
  3 current assembly sites.
- **Automation:** `[autonomy] enabled=false` master + per-task map on `AkhConfig`;
  build the effective ticker set at daemon construction (omit disabled tickers);
  gate the 2 startup bursts; flip `auto_start` default to `[]`; keep
  infra/session-persist always-on; user-authored Triggers remain opt-in and keep
  firing once added.

## Phased plan

### P0 — Unblock testing + de-risk transport (depends: none) ✅ COMPLETE

- Fixed `src/bin/generate_qa_dataset.rs` so plain `cargo test` compiles; the 121
  integration tests now run.
- Hardened `src/store/mmap.rs` truncated-header parse → typed `StoreError`
  instead of panic (+ 2 regression tests).
- Spiked the `agent-client-protocol` crate over a Unix socket wrapping
  `ChatProcessor` for one workspace (no rip-out) behind an `acp` feature
  (off by default); validated the lock-release-before-await invariant and
  `ByteStreams`. Test runs on a multi-thread runtime with hard timeouts.
- **Exit (met):** `cargo test` green; ACP spike echoes a `session/prompt` turn
  over a Unix socket. Commit `b1028f5`.

### P1 — First-class `AkhConfig` (`akh.toml`) (depends: P0) — IN PROGRESS

- New `AkhConfig` embedding `WorkspaceConfig`, adding: `name`, `model`,
  `ollama_endpoint {local base_url | cloud url + env-resolvable api_key}`,
  `system_prompt: Option<String>`, `is_main`, `related_workspaces: Vec<String>`,
  `[autonomy]`, `[delegation]`, `path_mode`. All `#[serde(default)]`.
- Wire `AgentModelManager` into akhomed per workspace; move model pin off global
  `OllamaConfig` onto `AkhConfig`/`AgentEntry`; add a Bearer-auth + cloud-`base_url`
  branch to `OllamaClient::{generate,health_check}` (mirror
  `UpstreamEndpointConfig::resolved_auth`). Make Ollama→fallback failover opt-in
  per akh.
- Unify the 3 system-prompt assembly sites into one function; explicit prompt
  authoritative.
- **Files:** new `src/akh_config.rs` (or extend `src/config.rs`);
  `src/workspace.rs`; `src/agent/ollama.rs`; `src/agent/model_manager.rs:394`;
  `src/bin/akhomed.rs` (load at `get_engine`/`get_agent`).
- **Exit:** two akhs with different pinned models + explicit prompts load from
  `akh.toml` and answer with their own prompt; cloud endpoint reachable with
  `env:VAR` key.

### P2 — ACP server transport (depends: P1)

- New `acp` feature; `src/acp/mod.rs` with `AkhAcpAgent` impl (`initialize`,
  `session/new`, `session/load`, `session/prompt`, `session/cancel`) +
  `ext_method` dispatcher for `_akh/*`; advertise capability groups in
  `agentCapabilities._meta`.
- Accept loop building `AgentSideConnection` per stream over a `UnixListener` +
  `TcpListener`, sharing `ServerState`. Runs **alongside** REST/WS during
  migration.
- `AcpSink: MessageSink` mapping `AkhMessage` → `session/update`
  (agent_message_chunk/tool_call/plan); map `audit_rx` onto `session/update`.
  `session/prompt` resolves Engine/Agent/NLU and runs `ChatProcessor` in
  `spawn_blocking`.
- New discovery record `{pid, socket, tcp, acp_version, protocol}` (superset of
  old); `initialize` replaces the health GET.
- Build the **mock-ACP test harness** (none exists): framing/handshake/lifecycle,
  `AkhMessage`↔ACP, Unix-vs-TCP parity.
- **Exit:** a raw ACP client completes a prompt-turn over both Unix and TCP;
  mock-ACP harness green.

### P3 — Pure-ACP TUI + ACP CLI client (depends: P2)

- Rewrite `src/tui/remote.rs` as an ACP client; collapse `ChatBackend` so
  **local also runs through an ACP loopback** (single path). Replace
  `tokio-tungstenite`.
- Add `AkhClient::Acp { conn, session_id }` beside `Local`/`Remote`; reimplement
  the ~90 CLI methods over ACP/`_akh/*` keeping signatures identical so
  `run_client_only` (`src/main.rs:4321`) is untouched. Port read-only groups
  first.
- Delete WS chat and the inbound `/mcp` nest + `src/mcp` server surface —
  **after** lifting reusable tool *bodies* into shared `Engine`/`Agent` functions.
- **Exit:** `akh chat`/`akh repl` work purely over ACP (local + remote); WS +
  inbound MCP deleted; outbound MCP intact.

### P4 — Filesystem abstraction + Claude Code tool parity (depends: P2)

- Daemon-config `path_prefix` (default `/data` on K8s) + a `Filesystem` trait:
  **Local** (`std::fs` joined under prefix + canonicalize+`starts_with` sandbox
  on **read and write**) and **Remote** (delegates to ACP `fs/*`+`terminal/*`,
  gated by `clientCapabilities.fs.*`). Route all raw `std::fs` callers through it.
- Add **Edit**, **Glob**, **Grep** (ripgrep), **WebSearch**, **structured
  task-list** tools on the shared backend; migrate `ToolInput` to typed/JSON args.
  Register the orphaned outbound `McpClient` tools.
- Harness-enforced **permission layer** (deny→ask→allow; modes
  default/acceptEdits/plan/bypass) wired to `session/request_permission`.
- **Exit:** Edit/Glob/Grep/WebSearch/task-list pass tests; FileIoTool sandboxed;
  a remote workspace reads/writes via ACP `fs/*`.

### P5 — Per-akh automation gating (depends: P1; parallel to P2–P4)

- `[autonomy]` master switch (default off) + per-task map on `AkhConfig`; build
  effective tickers at daemon construction, omit disabled; gate the 2 startup
  bursts (`daemon.rs:275,281`); flip `auto_start` default to `[]`; remove the
  "default"-sentinel hack; wire `DelegationConfig` from TOML; keep
  infra/session-persist always-on.
- **Exit:** fresh akh runs zero background tasks; enabling one task in `akh.toml`
  runs exactly that task.

### P6 — Peer-Engine federation (depends: P1)

- New `src/federation.rs`: `Federation{ main: Arc<Engine>, related: Vec<PeerEngine> }`,
  `FederatedHandle{engine_tag, id, label}`, assembled via
  `ServerState::get_federation` reusing the `get_engine` cache.
- Phase-1 federated reads: KG fact lookup label-based; SPARQL SELECT/ASK
  sequential fan-out + IRI→label re-resolve; episodic recall; gated vector search
  (`compatible_vsa` + label-anchored queries). Writes → MAIN only. Single-process
  only; cross-pod deferred.
- **Exit:** an akh with 2 related workspaces returns merged, engine-tagged,
  label-reconciled results for fact lookup, SPARQL, recall, and vector search.

### P7 — OpenClaw markdown import + OpenClaw-style Awaken (depends: P2, P4)

- Retire `src/agent/openclaw.rs` (HTTP) + `memory_sync` HTTP path; consolidate
  the two markdown importers into one `comrak`-based, per-akh, prefix-aware
  importer targeting `MEMORY.md`/`SOUL.md`/`AGENTS.md` + `memory/*.md`; fix
  `.md`→PlainText heading loss.
- Reshape awaken into an **interactive ACP-streamed sequence** (reuse the
  resumable `BootstrapSession` + `Checkpoint` enum) that produces a complete
  `AkhConfig`.
- **Exit:** importing a real OpenClaw workspace dir lands daily notes +
  MEMORY/SOUL/AGENTS into episodic/semantic memory; `akh awaken` runs as a
  streamed interactive ACP flow yielding a saved `akh.toml`.

### P8 — Retire REST + web UI; K8s transports; symbolic hardening (depends: P3–P7)

- After all data-plane groups are reachable over `_akh/*`: delete the REST
  router/handlers, web-asset handlers + bundled assets, `AkhClient::Remote` +
  ureq helpers, `data/web/*`. Drop Cargo features `server`, `mcp`; drop deps
  `axum`, `tower-http`, `tokio-tungstenite`, `rmcp`. Graceful shutdown drains
  **all** listeners.
- Helm: add an L4 TCP Service (+ Gateway) for ACP-TCP, UDP for QUIC follow-up;
  wire `path_prefix=/data`; preserve the Recreate single-writer PVC assumption;
  remove/repoint the L7 Ingress + web UI.
- Symbolic test hardening: adversarial integration tests for
  belief-revision/reasoning + VSA bridges; property-based round-trips for
  VSA/grammar/TMS; ACP parity tests.
- **Exit:** binary builds with no `axum`/`rmcp`; only ACP listeners; chart
  exposes ACP-TCP; symbolic suites green and meaningfully expanded.

## Verification

- **Per phase:** the explicit Exit criteria above; `cargo test` (must stay green
  from P0 onward) + `cargo test --lib`.
- **ACP end-to-end:** drive `session/prompt` from a raw ACP client over both Unix
  socket and TCP; assert `session/update` stream content; exercise
  `request_permission` and `fs/*` on a remote workspace; mock-ACP harness for
  framing/lifecycle parity.
- **Federation:** an akh with ≥2 related workspaces returns merged, engine-tagged,
  label-reconciled results; vector search excludes incompatible-encoding peers.
- **Tooling:** Edit/Glob/Grep/WebSearch/task-list unit + integration tests;
  FileIoTool sandbox escape attempts rejected.
- **Interop reality-check:** point an external ACP client (e.g. Zed) at the
  socket to confirm spec conformance.
- **Deploy:** Helm template renders ACP-TCP service; pod boots ACP-only with
  `path_prefix=/data` on a PVC.

## Top risks

1. **ADR-027 lock footgun** — concurrent ACP sessions deadlock if the
   `std::sync::Mutex<Agent>` is held across an await; funnel all Agent access
   through one `spawn_blocking` helper.
2. **ACP-only data-plane breadth** — ~130 routes / ~90 CLI methods must be
   re-expressed as `_akh/*` before REST can be deleted; mitigate with
   group-by-group migration and dual-serving until P8.
3. **Filesystem is net-new surface** (local prefix vs ACP-remote) over a PVC,
   with a current read-side sandbox hole and `canonicalize`-fallback symlink
   hazard.
4. **Federation correctness** — per-engine id collisions + VSA seed=raw-id make
   naive merges fabricate matches; the label-reconciliation + `compatible_vsa`
   gate is mandatory, and cross-pod federation is explicitly deferred.
5. **Back-compat + write-once identity** — new `AkhConfig` fields must be
   `#[serde(default)]`; the explicit-prompt path supersedes the write-once
   psyche-as-prompt coupling.
6. **K8s exposure** — Unix sockets don't cross pods; raw ACP-TCP needs an L4
   Service/Gateway and QUIC needs UDP, all without breaking the single-writer PVC.

## Documentation / ADR obligations (per CLAUDE.md)

- This plan doc (canonical record).
- ADRs, numbered from 052: 052 (ACP sole transport + `_akh/*` data-plane), 053
  (first-class `AkhConfig`), 054 (peer-Engine federation), 055 (filesystem
  abstraction), 056 (OpenClaw local-markdown import). ADRs are authored at their
  phase boundary (053 lands with P1; 052 with P2; etc.).
- Update `docs/ai/architecture.md` (transport change, new modules) at P2/P3.
- Anima work items per phase under project `019d9844-ce74-7993-a9f0-c953b8f04a4e`.
- Seshat ingest of this plan + each ADR.
