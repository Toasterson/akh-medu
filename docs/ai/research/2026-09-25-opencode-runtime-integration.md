# OpenCode as a native akhrtd runtime: integration surface and Brokered semantics

- **Date:** 2026-09-25
- **Anima:** WI `01a0d569-1cf1`, item 4 (Phase 53 follow-up)
- **Code:** `crates/akh-runtime/src/opencode.rs`, `crates/akh-runtime/src/opencode/{config,mcp}.rs`,
  `RuntimeKind::OpenCode` (`"opencode"`) in `crates/akh-session-types`
- **Earlier research:** [cursor-opencode.md](../../wayfinder/agent-loops/assets/cursor-opencode.md)
  (design evidence, pinned `f69becea`). This note re-reads current source and records what was built.
- **Pinned source:** `anomalyco/opencode` (formerly `sst/opencode`) release **v1.18.32**
  (`545f51d2`, 2026-09-21). The files below are byte-identical at `dev` `adee738d` (2026-09-25):
  `src/acp/{service,permission}.ts`, `src/permission/index.ts`, `src/session/llm/request.ts`,
  `src/config/config.ts`. No real binary was installed or run and no credentials were read.

## 1. Integration surfaces

| Surface | Transport | Tool injection | Verdict |
| --- | --- | --- | --- |
| `opencode acp` | NDJSON JSON-RPC 2.0 on stdio (ACP protocol 1, `@agentclientprotocol/sdk` 0.21) | Client-supplied `mcpServers` on `session/new` / `session/load` (`mcpCapabilities.http/sse = true`) | **Chosen** |
| `opencode serve` | HTTP API with OpenAPI spec, SSE `/event` stream | `POST /mcp` registration, permission replies over HTTP | Viable, but it needs an HTTP+SSE client and a daemon lifecycle; ACP is the same service behind a stdio facade |
| `opencode run --format json` | One-shot CLI, raw JSON events on stdout | Config only; no approval round-trip, no clean cancel | Not suitable: no host-mediated permissions |

ACP fits `akh-runtime`'s existing `Process` transport (bounded NDJSON, process-group teardown),
the same shape as the Codex app-server and the Claude bridge. `opencode acp` still starts its
internal HTTP server (`Server.listen`, loopback, port 0), which the ACP adapter calls through the
SDK. The bridge pins `--hostname 127.0.0.1 --port 0 --mdns=false` and sets a random
per-turn `OPENCODE_SERVER_PASSWORD`, so other local processes can't drive that session.

ACP methods the bridge uses: `initialize`, `session/new` or `session/load`,
`session/set_mode`, `session/set_config_option` (`model`, `effort`), `session/prompt`,
`session/cancel`, and the `session/update` notifications (`agent_message_chunk`,
`agent_thought_chunk`, `tool_call`, `tool_call_update`). Agent-to-client requests are
`session/request_permission`, which goes to host approval, and `fs/*` / `terminal/*`, which get
`-32601`. The client advertises `fs.readTextFile = fs.writeTextFile = terminal = false`.
`session/prompt` returns `stopReason` and `usage {inputTokens, outputTokens, thoughtTokens?, cachedReadTokens?}`.

Some source facts shape the design:

- **Replay on load.** `session/load` replays history as `session/update` before it responds. The
  bridge ignores updates until it sends `session/prompt`.
- **Model fallback.** `session/new` quietly falls back to another model when the configured one is
  missing (`selectDefaultModel`). The bridge therefore sets the model with
  `session/set_config_option {configId:"model"}`, which fails for an unknown model, and never
  relies on the config `model` key.
- **MCP registration.** `registerMcpServers` swallows registration errors (`Effect.ignore`). It
  connects and lists tools eagerly (`MCP.add` → `createAndStore`) before `session/new` answers.
  The bridge checks that its endpoint really served `tools/list` and fails the turn with
  `akh::runtime::opencode::mcp_not_connected` if it didn't.
- **Client writes.** When the client approves an `edit` permission, the permission handler also
  calls the client's `fs/write_text_file`. The bridge refuses that method.

## 2. Carrying Akh tools: a per-turn loopback MCP server

ACP has no client-side tool callback. Codex has `dynamicTools` and the Claude SDK has an
in-process MCP server; OpenCode has neither. The bridge serves the turn's Akh tool catalog as a
**Streamable-HTTP MCP server on `127.0.0.1:<ephemeral>/mcp`**:

- It is guarded by a per-turn 256-bit bearer token, compared in constant time and passed to
  OpenCode only in the ACP `mcpServers[].headers`.
- It is `POST`-only with JSON responses. `GET` and `DELETE` get 405, which the spec allows for a
  server that declines the optional SSE stream. Every response uses `Connection: close`.
- `tools/call` hands the call to the turn's own thread, which runs it through
  `Turn::tool` → `Host::tool`, the same broker path and denial rules as Codex dynamic tools. A
  name missing from the supplied catalog gets `ToolResult::denied`. Structured results (process
  handles, receipts) are serialized whole into the text content, as in the other bridges.
- OpenCode exposes these tools as `akh_<name>`. The bridge sets `experimental.mcp_timeout` to the
  turn timeout, so long Akh tools aren't cut off by the MCP client's 60 s default.

## 3. Brokered mode: what disables native side effects

OpenCode has **no structural switch** that removes its native tools, unlike Codex's
`features.shell_tool=false`. Brokered mode is built from OpenCode's permission engine, which is
pinned in source:

1. **Hidden from the model.** `llm/request.ts::resolveTools` drops every tool that
   `Permission.disabled` reports. That covers any tool whose last matching rule has pattern `*`
   and action `deny`; `edit`, `write` and `apply_patch` map to `edit`.
2. **Refused before effect.** Each native tool calls `ctx.ask` with the same ruleset before it
   acts. `deny` fails the call before anything runs.

The bridge makes that ruleset its own:

- **Its own agent.** `OPENCODE_CONFIG_CONTENT` is the highest-precedence file layer. It defines a
  primary agent `akh` whose `permission` is `{"*":"deny", <every native key>:"deny", "akh_*":"allow"}`.
  An agent's own `permission` is merged after the defaults and the user `permission` (`agent.ts`),
  so it wins. The bridge sets `default_agent = "akh"` and calls `session/set_mode akh` on every
  turn, because a loaded session restores the agent its history last used. A missing agent fails
  the turn.
- **The same rules everywhere else.** They also go into the top-level `permission` and into
  `OPENCODE_PERMISSION`, which is merged last, after MDM-managed preferences. Native keys are
  denied one by one, because a deep merge keeps the position of a `*` key another layer already
  defined.
- **Side channels off.** `snapshot = false`, `formatter = false`, `lsp = false`,
  `autoupdate = false` and `share = "disabled"`. The environment sets `OPENCODE_PURE=1` (no
  external plugins), `OPENCODE_DISABLE_PROJECT_CONFIG`, `OPENCODE_DISABLE_CLAUDE_CODE`,
  `OPENCODE_DISABLE_EXTERNAL_SKILLS`, `OPENCODE_DISABLE_LSP_DOWNLOAD` and
  `OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER`.
  `experimental.continue_loop_on_deny = true` returns a refused call to the model instead of
  ending the turn.
- **Refused profile inputs** (`akh_runtime::opencode::validate_profile`, run before launch, at
  akhrtd admission and by `akh-runtime-profile verify`):
  - no `isolation` directory, which would expose the ambient `~/.config/opencode`;
  - bridge-owned or other `OPENCODE_*` environment. Only `OPENCODE_DISABLE_MODELS_FETCH` and
    `OPENCODE_MODELS_{PATH,URL}` are allowed;
  - managed files under `opencode/` other than `opencode.json` and `AGENTS.md`;
  - an `opencode.json` that isn't strict JSON, or that has keys outside
    `$schema, model, small_model, provider, disabled_providers, enabled_providers, username,
    compaction, logLevel, tool_output`;
  - a provider `npm` package;
  - `node_binary` or `bridge_script`.
- **Approvals.** Brokered turns never approve: `Turn::approve` returns `false`, and the bridge
  picks the `reject_once` option.

**Status: Brokered is supported, gated by conformance.** The denial rests on OpenCode's permission
policy, not on a structural absence of tools. That is the same class of guarantee as the Claude
bridge (`disallowedTools` plus a denying `canUseTool`). For OpenCode, `validate_conformance`
requires one extra check beyond the common seven:

- `native_tools_hidden`: the tool catalog the model received contained only `akh_*` tools.

Residual risks that conformance measurement must cover and the profile must avoid:

- **Built-in auth plugins.** These still load in pure mode (`plugin/index.ts`). They make network
  calls for provider auth; they have no workspace effects.
- **`.opencode` directories.** Any `.opencode` directory under the private config or cwd triggers
  a background `npm install @opencode-ai/plugin` and a `.gitignore` write. These happen in
  private runtime state, not the workspace.
- **MDM-managed macOS preferences.** These override every file layer. `OPENCODE_PERMISSION` is
  still merged after them, but a managed `agent.akh` or `mcp` block would change the surface.
  Don't run brokered OpenCode on MDM-managed hosts without re-measuring.
- **Version drift.** The config schema has `additionalProperties: false`, so a release that drops
  one of the keys above fails at startup, which is fail-closed. A release that adds a native tool
  is still denied by `"*"`. Re-run conformance on every version change; the evidence pins the
  binary digest.

Colocated (development only; deployed services refuse it, per ADR 113 decision 7):

- Native reads (`read`, `glob`, `grep`, `list`, `lsp`) are allowed.
- `question`, `task` and `plan_*` are denied.
- Everything else is `ask`, answered by host approval.
- A `read_only` approval policy denies all non-read native tools.

## 4. Model, provider and authentication

- **Model.** `RuntimeBinding.model` must be `provider/model`, for example `openai/gpt-5`.
  Anything else fails with `akh::runtime::opencode::model_format`. `ProviderConfig.effort` maps
  to the OpenCode variant (`configId:"effort"`) and fails if the model has no such variant.
- **Credentials.** They stay in OpenCode's own flow. `isolation.login_command =
  ["opencode","auth","login"]` stores them under the private `$HOME/.local/share/opencode`.
  Alternatively, provider API keys can come through `isolation.environment_files`, for example
  `OPENAI_API_KEY`. OpenCode's provider guide says Claude Pro/Max subscription plugins were
  removed; don't transplant tokens from another harness.

## 5. Deploying a brokered OpenCode runtime (operator)

```toml
[runtimes.opencode]
kind = "opencode"
revision = "opencode-1.18.32-r1"
binary = "/opt/akh/opencode/1.18.32/bin/opencode"   # absolute, pinned; autoupdate is disabled
working_dir = "/var/lib/akhrtd/opencode/work"       # private runtime state, never the workspace
execution_mode = "brokered"                          # required in deployed mode
timeout_secs = 600
conformance_file = "/etc/akh/conformance/opencode-1.18.32-r1.json"

[runtimes.opencode.isolation]
directory = "/var/lib/akhrtd/opencode/profile"
login_command = ["opencode", "auth", "login"]
version_command = ["opencode", "--version"]
environment_files = { OPENAI_API_KEY = "/run/akh/credentials/openai" }  # or use login
managed_files = { "opencode/opencode.json" = '{"$schema":"https://opencode.ai/config.json","enabled_providers":["openai"]}' }
```

Steps:

1. Run `akh-runtime-profile --config FILE --profile opencode prepare`, then `login` (or install
   the environment files).
2. Measure conformance for this exact binary, profile digest and platform. Write the
   `Conformance` document (`crates/akh-session-services/src/runtime.rs`) with `provider_version`
   from `opencode --version`, a non-empty `evidence` list, and these checks:
   `native_write_denied_before_effect`, `native_exec_denied_before_effect`, `hooks_disabled`
   (external plugins off, `OPENCODE_PURE`), `supplied_tool_succeeds`, `exact_resume`,
   `cancel_teardown`, `workspace_semantics` and **`native_tools_hidden`**. No OpenCode wire-smoke
   script exists yet. `scripts/session-platform/codex-native-wire-smoke.py` is the template, and
   the ignored `real_opencode_round_trips_an_akh_tool` test in `crates/akh-runtime/tests/providers.rs`
   covers config acceptance and the MCP round trip.
3. Run `akh-runtime-profile --config FILE --profile opencode verify`.
4. Deploy from `origin/main` only (`.claude/rules/deploy-from-main.md`). Editing the profile
   invalidates its evidence.

The in-process akhomed compatibility bridge (`src/agent/runtime/bridge.rs`,
`backend = "codex" | "claude"`) does not get an OpenCode backend. It is development-only and
refused in deployed mode, so OpenCode work goes through akhrtd.
