# OpenAI, Codex and Claude backends

Select a backend per akh. OpenAI models drive Akh's existing tool loop. Codex
and Claude run complete coding-agent sessions, with their native tools plus
Akh's advertised tools, memory context and persona.

| `backend` | Interface | Who runs the tool loop |
| --- | --- | --- |
| `openai` | OpenAI-compatible Chat Completions API | Akh-Medu |
| `codex` | Installed Codex `app-server`, JSON-RPC over stdio | Codex |
| `claude` | Official Claude Agent SDK, through the Node bridge | Claude |

This follows the integration architecture used by T3 Code: its
[Codex session runtime](https://github.com/pingdotgg/t3code/blob/main/apps/server/src/provider/Layers/CodexSessionRuntime.ts)
uses app-server and its
[Claude adapter](https://github.com/pingdotgg/t3code/blob/main/apps/server/src/provider/Layers/ClaudeAdapter.ts)
uses the Agent SDK. Akh keeps its existing ACP client interface.

## Subscription access and the intended execution boundary

The follow-on design is tracked in the
[interchangeable agent-loop Wayfinder map](wayfinder/agent-loops/map.md).
It generalizes the session prototype to Internal, Codex, Claude, Cursor and
OpenCode while preserving Akh's session and execution services. The
[architecture draft](wayfinder/agent-loops/assets/architecture-draft.md)
is discussion material, not implemented configuration or an accepted ADR.

The [daemon-boundary addendum](wayfinder/agent-loops/assets/daemon-boundaries.md)
traces this prototype into `akhrtd` workers, `akhsessd` session ownership,
`akhomed` cognitive services, `akhsbxd` provisioning and `akhexecd` execution,
alongside `akheventd` ingress and automation triggers. It is a local design
proposal; the current implementation still runs through the existing daemon.
**Superseded in part** by [ADR 113](ai/decisions/113-full-akh-always.md) (Phase
53h): `akhomed` is always the fully featured akh (daemon loop, in-process turn,
full tool registry, ACP — one `server` build profile, `core-api`/`/v1` additive
on top; see `docs/ops/akhomed-feature-profiles.md`), and `akhsessd`/`akheventd`
contain only native-runtime delegation and event ingress, not the akh's own
automation or admission.

The [2026-09-11 deployment plan](wayfinder/agent-loops/assets/k8s-acp-t3-deployment.md)
traces these adapters into Kubernetes, ACP over WebSocket, upstream T3 Code and
kagent. It distinguishes the released kagent BYO integration from development
native harnesses, and keeps remote workspace execution as a provider-specific
verification requirement.

The [sandbox-image clarification](wayfinder/agent-loops/assets/sandbox-images-anima.md)
places the selected T3/VS Code-compatible/Zed remote server beside `akhexecd`,
keeps session management central, and traces retirement of Anima's duplicated
execution-routing responsibilities.

The [upstream-service and tool-binding plan](wayfinder/agent-loops/assets/upstream-services-tool-bindings.md)
keeps external services unmodified. Akh helpers own supported launch/config/login
and connection workflows; runtime adapters preserve tool semantics and enforce
native-call restrictions. These are proposed profiles, not changes to the
prototype's current tool permissions.

The desired integration is subscription access with Akh retaining its custom
tools, workspace abstraction and SSH execution. The current session prototype
does **not** yet meet that execution requirement: provider-native tools still
execute locally, so the remote-executor guard below is necessary.

Subscription credentials are not general-purpose model API keys.
[Codex authentication](https://learn.chatgpt.com/docs/auth) distinguishes
ChatGPT subscription access from API-key billing. The
[App Server](https://learn.chatgpt.com/docs/app-server) supports managed ChatGPT
login and host-executed dynamic tools. This provides a supported interface for
an Akh client while Codex retains the inference loop.

[Anthropic's authentication rules](https://code.claude.com/docs/en/legal-and-compliance)
allow end users to sign into the unmodified Claude Code binary with their own
subscription, including when a platform hosts that binary. They prohibit a
third-party application from collecting or intermediating Claude subscription
credentials or presenting its own Claude.ai login. The
[Agent SDK overview](https://code.claude.com/docs/en/agent-sdk) directs product
developers to API authentication unless approved otherwise. Therefore an SDK
smoke test succeeding is not proof that arbitrary subscription-backed product
usage is permitted. Keep authentication in the official binary's own flow.

For subscription-based sessions, the intended architecture is:

```text
Akh orchestration, persona and memory
  -> official Codex / Claude Code runtime with the user's own login
  -> advertised Akh tool request
  -> Akh permissions, workspace routing and execution backend
  -> local or SSH result returned to the provider runtime
```

Before enabling remote execution, the integration must ensure every workspace
read, write and command goes through Akh. Provider-native execution and inherited
tools/hooks must not bypass that route; a prompt asking the model to prefer Akh
tools is insufficient. Validate this using an SSH executor witness and a check
that no equivalent local operation ran. Preserve cancellation and completion
evidence through the same boundary.

This architecture preserves Akh's execution capabilities and outer orchestration,
but the provider retains its internal tool/model loop and context management.
Direct model calls driving Akh's existing `ChatBackend` loop require separately
billed model API access; these official subscription interfaces do not document
a drop-in model endpoint for that loop. Do not present the two modes as equivalent.

## Configure the akh

Edit the akh's configuration file on the **daemon host**. The filename is
`<config_dir>/workspaces/<name>.toml` (`AkhPaths::workspace_config_file`); this
per-akh configuration is also called `akh.toml` in architecture documents.
Keep top-level fields such as `name`, `backend`, `model` and `system_prompt`
above the first TOML table. Apply the config when the workspace is next loaded
by the updated daemon. These examples do not change a running deployment.

### OpenAI API

```toml
name = "thoth-openai"
backend = "openai"
model = "gpt-5"

[openai_endpoint]
base_url = "https://api.openai.com/v1"
api_key = "env:OPENAI_API_KEY"
timeout_secs = 600
```

Make `OPENAI_API_KEY` available to the daemon through your existing secret
management. Choose a model available to that API account and supporting Chat
Completions. This existing backend sends `/chat/completions` requests; it
does not use Codex's ChatGPT login. Leave `temperature` and `max_tokens`
unset unless the chosen model supports them. OpenAI recommends the
[Responses API for new integrations](https://developers.openai.com/api/docs/guides/migrate-to-responses);
the compatibility backend remains on Chat Completions in this change.

### Codex

Install Codex on the daemon host and authenticate as the account running the
daemon with `codex login`. Akh launches the official
[Codex App Server](https://developers.openai.com/codex/app-server) and uses
`thread/start`, `thread/resume` and `turn/start`. It streams messages and tool
events, forwards command/file approvals to the ACP client, and interrupts
cancelled turns. Akh tools use app-server's experimental dynamic-tool API.

```toml
name = "thoth-codex"
backend = "codex"
# Optional: model and reasoning_effort; omit to use the provider's selection.
# model = "your-available-codex-model"
# reasoning_effort = "high"

[agent_runtime]
working_dir = "/absolute/path/to/checkout"
binary = "/absolute/path/to/codex"
timeout_secs = 600
```

`binary` is optional when `codex` is on the daemon's PATH. Codex uses a
workspace-write sandbox and requests approval for escalation. Akh plan mode
selects read-only. Akh never enables the provider's permission bypass.

### Claude

Install the bridge dependencies in this checkout:

```sh
npm ci --prefix scripts/agent-backends --ignore-scripts
```

Configure authentication supported by the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) for the
daemon account. API credentials can be supplied through `ANTHROPIC_API_KEY`;
the provider also handles its supported local login/provider configuration.
Akh does not implement or extract a Claude OAuth login. Availability and
usage accounting depend on the authentication method and provider plan.

```toml
name = "thoth-claude"
backend = "claude"

[agent_runtime]
working_dir = "/absolute/path/to/checkout"
bridge_script = "/absolute/path/to/akh-medu/scripts/agent-backends/claude.mjs"
# Optional: use an installed Claude Code binary instead of the SDK's binary.
binary = "/absolute/path/to/claude"
# Optional if node is already on the daemon's PATH.
node_binary = "/absolute/path/to/node"
timeout_secs = 600
```

Keep the bridge and its `node_modules` together when installing them elsewhere.
The bridge uses SDK streaming input, session resume, an in-process MCP server
for Akh tools, and `canUseTool` for approval requests. The SDK remains responsible
for its own configuration, native tools and context management.

## Session and execution behavior

- Each ACP conversation owns its provider session ID. A follow-up resumes that
  exact session. Failed resumes report an error and never create a replacement.
- Provider identities are retained while the ACP session lives. They are not
  restored across daemon restarts in this implementation. Provider-side history
  remains managed by the provider.
- ACP `model/set` updates the selected provider for the next turn. Runtime
  model discovery is not exposed yet: `model/list` reports the configured model
  rather than querying Ollama. Set a provider model ID explicitly when needed.
- An ACP cwd must be inside the configured `working_dir`. Changing the provider,
  cwd or advertised tools requires a new ACP session.
- Akh tool calls run through the session's registry, approval policy, audit and
  tracker. Akh injects persona and retrieved memory context; accepted answers
  enter its conversation history and existing transcript capture path.
- Explicit Akh completion contracts still require Akh-observed tool evidence.
  Autonomous dispatch needs successful Akh execution evidence; provider-native
  activity alone does not satisfy that gate. Unsatisfied contracts refuse the
  turn rather than releasing provisional narration as accepted completion.
- Native provider tools execute under the daemon's local account. These backends
  refuse to start with `exec.kind = "none"`, `"remote"` or `"local-drop"`.
  Use them on a host whose intended executor is `"in-process"`.
- Cancellation, timeout, malformed protocol and early process exit terminate
  the provider process group on Unix. Detached provider background processes
  are not a supported way to run persistent services.
- Image prompts and arbitrary provider question/form dialogs are not yet
  supported. Approval requests are supported; other input requests are declined.
  This adds backend sessions, not T3's desktop interface or complete UI parity.

## Validation

```sh
cargo check --bin akhomed --no-default-features --features server
cargo test --lib --no-default-features --features acp agent::runtime
npm test --prefix scripts/agent-backends
```

The default tests use deterministic subprocess fixtures and a fake SDK. Optional
live tests use the installed providers and make two small model calls each:

```sh
cargo test --lib --no-default-features --features acp live_installed_codex_session -- --ignored
cargo test --lib --no-default-features --features acp live_installed_claude_session -- --ignored
```

Protocol schema checked against Codex CLI 0.153.4; Claude SDK dependency pinned
to 0.3.266. Provider updates should rerun the live session checks. Local checks
on 2026-09-09 passed for both installed providers, including an Akh tool call
and a follow-up in the resumed session. This is local interoperability evidence;
no running daemon was upgraded or restarted.
