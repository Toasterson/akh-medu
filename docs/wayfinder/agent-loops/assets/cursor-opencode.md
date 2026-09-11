# Cursor and OpenCode integration contracts

Research date: 2026-09-09. Research ticket: `01a086ef-8121-7a33-b729-1ddb28b6c15d`.

This is design evidence, not an implementation or interoperability claim. No model calls, credential access, dependency installation, or deployed service changes were performed. Documentation was checked live; source inspection pins OpenCode to `f69beceaffca94bed05a7669af93602125c37248` and Cursor SDK Bridge to `6716981f2649987c3ab4f5b82e71ec133826c528`. Neither source pin establishes compatibility with an installed release.

## Findings and candidate routes

Both providers have a credible `AgentLoop` adapter surface. Complete Akh ownership of workspace and execution remains a separate acceptance gate. Prefer a Cursor **local SDK Bridge** spike, with ACP as an alternative client protocol. Prefer a dedicated OpenCode **server or ACP subprocess** per Akh session until configuration and MCP isolation are proven. Keep model credentials with the selected provider runtime and execute Akh tool callbacks through the Akh session's existing services.

These are proposed integration choices. In particular, connecting MCP or answering a permission request does not establish that native file, shell, hook, plugin, or subagent execution traverses Akh.

## Protocol and capability matrix

| Concern | Cursor | OpenCode |
|---|---|---|
| Official integration surface | `agent acp`: NDJSON JSON-RPC over stdio, explicitly intended for custom clients. [ACP](https://cursor.com/docs/cli/acp) Also supported SDK Bridge for Rust and other languages, using Connect/protobuf over loopback HTTP/1.1. [Bridge](https://cursor.com/docs/sdk/bridge) | `opencode serve`: HTTP API with OpenAPI specification. [Server](https://opencode.ai/docs/server/) `opencode acp`: JSON-RPC stdio subprocess. [ACP](https://opencode.ai/docs/acp/) |
| Session and turn control | ACP supplies new/load/prompt/cancel and streaming updates. Its question and plan extension requests require responses. [ACP](https://cursor.com/docs/cli/acp) SDK Bridge exposes create/resume/send/observe/cancel and distinct agent/run IDs. [Pinned service contract](https://github.com/cursor/sdk-bridge/blob/6716981f2649987c3ab4f5b82e71ec133826c528/proto/sdk/v1/sdk_agent_service.proto) | Server supports session create/get, explicit-ID message/prompt_async, abort, children and fork. [Server](https://opencode.ai/docs/server/) Source ACP adapter implements load/resume/close/fork/list and model/mode configuration. [Pinned adapter](https://github.com/anomalyco/opencode/blob/f69beceaffca94bed05a7669af93602125c37248/packages/opencode/src/acp/agent.ts) |
| Streaming and reconnect | Bridge `ObserveRun.after_offset` resumes durable events; `Send` streams until completion. Local send/create idempotency is not promised by the proto; its idempotency-key comments specify cloud. [Pinned service contract](https://github.com/cursor/sdk-bridge/blob/6716981f2649987c3ab4f5b82e71ec133826c528/proto/sdk/v1/sdk_agent_service.proto) | SDK `event.subscribe()` exposes SSE. Subscribe before prompting; route by session identity. Durable replay across disconnect is not established by the cited SDK guide. [SDK](https://opencode.ai/docs/sdk/) |
| Custom tool execution | Bridge host implements `CallCustomTool`, receiving tool name, typed JSON arguments, agent ID and optional call ID; tool metadata is carried separately. This is an execution callback Akh can implement. [Pinned callback contract](https://github.com/cursor/sdk-bridge/blob/6716981f2649987c3ab4f5b82e71ec133826c528/proto/sdk/v1/sdk_custom_tool_callback_service.proto) | Remote and local MCP supported. An Akh MCP server can supply the registry without embedding custom JS execution into OpenCode. [MCP](https://opencode.ai/docs/mcp-servers/) Plugin-defined tools also exist, but execute in the provider process. [Plugins](https://opencode.ai/docs/plugins/) |
| Provider-native approvals | ACP emits `session/request_permission` when required; modes include agent/plan/ask. [ACP](https://cursor.com/docs/cli/acp) | HTTP permission replies and ACP mapping available; native permissions can allow, ask or deny actions. [Server](https://opencode.ai/docs/server/), [Permissions](https://opencode.ai/docs/permissions/) |
| Native execution replacement | No universal executor replacement established. Local Bridge does publish tool allowlisting; candidate Akh-only configuration described below. [Pinned options](https://github.com/cursor/sdk-bridge/blob/6716981f2649987c3ab4f5b82e71ec133826c528/proto/sdk/v1/sdk_messages.proto) | No universal executor replacement established. Permission restrictions plus Akh MCP are a candidate. ACP itself is not a remote filesystem/terminal layer: inspected service connection includes permissions and text writes, without general terminal/read callbacks. [Pinned service](https://github.com/anomalyco/opencode/blob/f69beceaffca94bed05a7669af93602125c37248/packages/opencode/src/acp/service.ts) |

Cancellation is a request, not proof that a tool stopped. Akh must revoke further dispatch, cancel active Akh execution, await a terminal witness and preserve completed side effects. This is a proposed common adapter contract, not a claim that either provider currently supplies all those guarantees.

## Subscription and authentication findings

Cursor SDK accepts user and service-account API keys, and user-key runs bill to that user's plan using the same request pools and pricing as IDE/cloud runs. Browser login can provision a key. This is Cursor billing, not permission to transfer a ChatGPT or Claude subscription into Cursor. Headless SDK runs lack a native human approval flow and are unrestricted by default. [SDK authentication, billing and sandbox sections](https://cursor.com/docs/sdk/typescript)

OpenCode's current provider guide documents its own ChatGPT Plus/Pro browser login and an API-key alternative. It explicitly says Claude Pro/Max plugins are prohibited by Anthropic and were removed from bundled OpenCode starting with 1.3.0. The same page retains an earlier contradictory Claude-login sentence; do not follow that stale sentence. Treat ChatGPT support here as an OpenCode-documented integration; verify upstream entitlement requirements separately before declaring Akh product/service use supported. Do not transplant tokens from another harness. [Providers](https://opencode.ai/docs/providers/)

The pinned OpenCode plugin loader includes its Codex authentication plugin. That is source evidence of implementation, not an independent policy grant. [Pinned plugin loader](https://github.com/anomalyco/opencode/blob/f69beceaffca94bed05a7669af93602125c37248/packages/opencode/src/plugin/index.ts)

## Candidate Cursor boundary

The local Bridge's `AgentOptions.tools` allowlist distinguishes omitted from empty, rejects unknown names, and supports public tool names and capability groups. Tool restrictions are local-only. [Pinned options](https://github.com/cursor/sdk-bridge/blob/6716981f2649987c3ab4f5b82e71ec133826c528/proto/sdk/v1/sdk_messages.proto)

**Proposed spike:** allow only `mcp`, inject Akh custom tools, exclude settings sources and use an empty provider working directory. Keep `task` unavailable. This exact combination is untested. Reapply tool restrictions and inline MCP when resuming: they are not persisted. Disallowing `mcp` also removes custom tools; enabled subagents retain their own curated toolsets. The SDK's default local MCP discovery includes only inline definitions, with project/user/plugins opt-in through `settingSources`. [SDK tools and settings](https://cursor.com/docs/sdk/typescript)

Native hooks remain a gate. The SDK documents file-based hooks with no programmatic callback. Hook commands themselves execute scripts; before-shell/MCP hooks default to allowing through on failure unless configured otherwise. Do not treat a hook deny rule as a complete execution sandbox or presume setting-source selection suppresses every hook layer. [SDK hooks](https://cursor.com/docs/sdk/typescript), [Hooks](https://cursor.com/docs/hooks)

The SDK Bridge contract is attractive because the Akh callback can block on Akh approval and then use the Akh workspace/SSH executor. That does not require a provider-native approval dialog for Akh tools. Call correlation must include the provider agent binding; the callback's optional call ID cannot alone identify an Akh tenant/session. [Pinned callback contract](https://github.com/cursor/sdk-bridge/blob/6716981f2649987c3ab4f5b82e71ec133826c528/proto/sdk/v1/sdk_custom_tool_callback_service.proto)

Cursor ACP is a useful alternative but its guide loads project/user MCP and does not promise full client-owned filesystem or terminal execution. CLI permissions cover Shell/Read/Write/WebFetch/Mcp, with deny taking precedence. Those controls do not constitute documented interception of startup scripts and every future native capability. [ACP](https://cursor.com/docs/cli/acp), [CLI permissions](https://cursor.com/docs/cli/reference/permissions)

## Candidate OpenCode boundary

**Proposed spike:** a dedicated provider instance, a minimal explicit agent, deny all tool permissions with an exception for the Akh MCP namespace, and no provider access to the actual workspace. Explicitly cover read/edit/glob/grep/bash/task/skill/lsp/web tools, not merely bash. Most native permissions otherwise default to allow, and agent-specific configuration participates in permission selection. These are tool permission checks, not a guarantee against executable configuration. [Permissions](https://opencode.ai/docs/permissions/)

Configuration files merge. `OPENCODE_CONFIG` is an additional layer rather than a complete replacement; project settings and `.opencode` directories follow it. Managed settings can override user configuration. Therefore a temporary JSON file alone does not isolate this provider. [Config](https://opencode.ai/docs/config/)

Plugins are automatically loaded from local/global directories, can execute shell commands through their context, and may install dependencies at startup. `tool.execute.before` can alter arguments or throw, and plugin tools may override native names. None proves all background, initialization or plugin code passes through the same tool gate. [Plugins](https://opencode.ai/docs/plugins/) The pinned loader suppresses external plugins in `pure` mode but handles built-in auth plugins separately; this is a source-level lead, not a documented complete isolation profile. [Pinned loader](https://github.com/anomalyco/opencode/blob/f69beceaffca94bed05a7669af93602125c37248/packages/opencode/src/plugin/index.ts)

There is a concrete session-isolation concern: ACP `registerMcpServers` tracks registration by session but calls `sdk.mcp.add` with **directory, name and config**, without a session ID, and ignores registration errors. Do not assume two Akh sessions sharing a provider directory receive different MCP credentials/tool scopes. Dedicated processes and provider directories are the conservative initial design, subject to a test. [Pinned service](https://github.com/anomalyco/opencode/blob/f69beceaffca94bed05a7669af93602125c37248/packages/opencode/src/acp/service.ts)

The ACP permission handler can read local files and issue a client text-write preview before replying to OpenCode's permission API. That is evidence of UI integration, not replacement of the native executor. [Pinned permission handler](https://github.com/anomalyco/opencode/blob/f69beceaffca94bed05a7669af93602125c37248/packages/opencode/src/acp/permission.ts)

## Process placement and unsupported claims

Treat `runtime_host` and `workspace_backend` as separate configuration dimensions. An isolated local provider process may call a remote Akh tool gateway while all workspace paths resolve on Akh's SSH backend. The local runtime directory is provider state/scratch, never an implicit local mirror of the remote workspace. Running the provider itself over SSH is another placement option; it does not establish Akh ownership of the provider's native tools.

Do not advertise the following as supported yet:

- Full Akh workspace/SSH exclusivity for either provider: untested, with startup/plugin/hook gaps still unresolved.
- Cursor cloud execution using the local tool-restriction contract: explicitly local-only in the published proto.
- OpenCode using Claude subscriptions: explicitly rejected by its current provider guide.
- Automatic substitution of a new provider session after resume failure: a new session is not exact continuation.
- Tool-call or shell observations as successful Akh evidence when they did not pass through Akh's executor.

## Required feasibility spikes and acceptance evidence

| Spike | Required evidence | Stop condition |
|---|---|---|
| Cursor Bridge contract | Pin a released binary/proto pair; fake callback returns a unique marker; stream correlates callback and run; only Akh MCP tools offered; unknown tool configuration fails. | Cannot restrict to Akh tools while preserving custom tools. |
| Cursor execution isolation | Canary hooks at user/project/plugin layers, native reads/writes/shell/subagents, inherited MCP and startup behavior. Trace filesystem/process activity, including before the first prompt. Repeat after exact-ID resume. | Any real-workspace access or unapproved executable extension bypasses Akh. |
| OpenCode configuration isolation | Pin a release; establish complete effective configuration and tool inventory; seed canary project/global plugins, custom tools, MCP, LSP and formatter behavior; cover startup and restart. | Merged settings re-enable paths or executable initialization occurs outside the permitted runtime envelope. |
| OpenCode session/MCP isolation | Two sessions with same tool names but distinct gateway credentials/results. Attempt cross-session calls, same-directory reuse, missing server, changed server config and resume. | Cross-session credential/tool bleed or silent unavailable-Akh-tool fallback. |
| Both: SSH routing | Remote workspace contains marker absent locally. Read/edit/execute through Akh; record executor host, cwd, request ID, approval and output. Compare local sentinel files/process traces. | Success can occur through provider-native execution without an Akh record. |
| Both: lifecycle | Cancel during model streaming, pending approval and long-running Akh tool; kill client then reattach; provider crash during side effect; wrong/missing resume ID; duplicate event delivery. | New dispatch after cancellation, duplicated side effects, silent session replacement or unverifiable completion. |

A successful spike should attach protocol transcripts with secret values redacted, exact version/digest, configuration manifest, tool-catalog digest, execution trace and local/remote canary results. Only then should the corresponding adapter advertise `akh_tools_only` and remote-workspace capability. The common trait can exist before every implementor qualifies for every capability.
