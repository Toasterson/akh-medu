# Codex and Claude loop integration: research findings

Status: research input, not an architecture decision or a completed integration.
Verified 2026-09-09 against fetched primary documentation and installed public
interfaces. No credentials were read and no model requests were made.

## Findings that affect the proposed boundary

The provider-loop/Akh-tools split has supported extension points in both
providers. **Exposing Akh tools is established; exclusively routing every
workspace operation through Akh is a separate conformance requirement.** In
particular, approval callbacks are not generic tool-result substitution APIs.
The evidence below supports a capability-negotiated adapter rather than a
promise that every adapter can enforce the same execution boundary immediately.
This paragraph is a research inference from the contracts and gaps below.

| Concern | Codex | Claude |
| --- | --- | --- |
| Official programmable loop | App Server is the rich-client interface; JSON-RPC over stdio or WebSocket. [App Server](https://learn.chatgpt.com/docs/app-server) | Agent SDK exposes Claude Code's loop and context management in Python/TypeScript. Other languages can launch the CLI subprocess. [SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) |
| Akh tool execution | Experimental `dynamicTools` produces `item/tool/call` requests for client execution, with content and success responses. [App Server](https://learn.chatgpt.com/docs/app-server#dynamic-tool-calls-experimental) | `tool()` plus `createSdkMcpServer()` registers application handlers; `mcpServers` exposes them as `mcp__akh__<tool>`. A remote Akh MCP server is another route. [Custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools), [MCP](https://code.claude.com/docs/en/agent-sdk/mcp) |
| Provider session | `thread/start`, `thread/resume`, `thread/fork`; dynamic tools persist in thread metadata. `turn/start` runs, `turn/steer` adds active input, `turn/interrupt` ends with interrupted status. [App Server](https://learn.chatgpt.com/docs/app-server) | Capture the provider session ID and use explicit `resume`; directory-based `continue` is unsuitable for concurrent users. Fork copies conversation, not files. Cross-host resume requires retained transcripts or a session store. [Sessions](https://code.claude.com/docs/en/agent-sdk/sessions) |
| Stream and terminal result | Item start/completion, message deltas, and `turn/completed` carry separate progress and terminal state. [App Server](https://learn.chatgpt.com/docs/app-server) | `includePartialMessages` adds raw stream events alongside complete assistant/result messages; adapters must avoid rendering both as duplicate text. [Streaming](https://code.claude.com/docs/en/agent-sdk/streaming-output) |
| Approvals and questions | Command/file approvals are server requests; permission grants, structured questions, and MCP elicitation have distinct response shapes. These authorize provider execution. [App Server](https://learn.chatgpt.com/docs/app-server#approvals) | `canUseTool` handles unresolved permission requests and `AskUserQuestion`. Its response allows/denies or modifies input; it does not provide an arbitrary replacement execution result. Pending callbacks require cancellation/lifecycle handling. [User input](https://code.claude.com/docs/en/agent-sdk/user-input) |

## Subscription and authentication boundaries

**Codex:** ChatGPT sign-in supplies subscription access; API-key sign-in supplies
usage-based API access. App Server's managed ChatGPT mode owns login, storage,
and refresh. Its experimental external-token mode is documented for hosts that
already own a ChatGPT authentication lifecycle; that is not a reason for Akh to
extract another application's tokens. Prefer provider-managed sign-in and keep
the Akh client/daemon transport credential separate. Neither source establishes
a general-purpose raw model API funded by an ordinary ChatGPT subscription.
[Authentication](https://learn.chatgpt.com/docs/auth),
[App Server authentication](https://learn.chatgpt.com/docs/app-server#authentication-modes)

**Claude:** the current legal page explicitly permits an end user to sign into
the **unmodified Claude Code binary** with their own subscription, including on
a platform hosting that binary. Hosting in products/services requires the
applicable Commercial Terms, preserving the binary and its authentication
methods, and direct end-user authentication/billing. The same page prohibits
third parties collecting, storing, or intermediating Claude.ai credentials and
routing subscription requests on users' behalf. It directs developers building
SDK products toward API keys or supported clouds.
[Legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)

The SDK overview separately says third-party products may not offer Claude.ai
login/rate limits without prior approval. Therefore, **do not present “SDK uses
my subscription” as universally authorized**. A provider-hosted binary launch
where the user completes Anthropic's own sign-in is distinguishable from an
Akh credential proxy, but the sources do not expressly settle every SDK-wrapped
hosted arrangement. Preserve that distinction in the design and resolve the
exact distribution/hosting arrangement before promising subscription support
there. This is an inference, not legal clearance.
[SDK overview](https://code.claude.com/docs/en/agent-sdk/overview),
[Legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)

Search-index snippets mentioned a June 2026 SDK subscription-credit change, but
the fetched current English overview/legal pages did not establish that credit
schedule. This research therefore makes no credit amount or billing-limit
promise. Authentication compatibility and contractual permission also require
separate evidence.

## Exclusive Akh execution: evidence and limits

### Codex

The configuration reference exposes individual controls including
`features.shell_tool`, `features.unified_exec`, `agents.enabled`,
`features.hooks`, `features.code_mode.enabled`, `web_search`, `tools.view_image`,
MCP enablement/tool lists, and app/plugin controls. It does **not** establish one
universal “only these client tools” switch. Turning off shell alone does not
prove removal of patching, image reads, browser/computer tools, MCP, hooks,
subagents, or later-added execution paths.
[Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)

`PreToolUse` covers shell/unified execution, patching, MCP, and many local
function tools. However, hosted tools do not use this path, specialized tools
may opt out, and `write_stdin` does not rerun the pre-hook. The documentation
explicitly warns that hooks are not a complete enforcement boundary. MCP hook
errors or missing servers do not block the operation. Therefore a deny hook
cannot by itself certify Akh-exclusive execution.
[Hooks](https://learn.chatgpt.com/docs/hooks#tool-coverage)

**Research conclusion:** support for an Akh custom-tool callback is high
confidence; complete native execution exclusion is **unproven**. A dedicated
runtime environment without the actual project filesystem, combined with
verified tool/config exclusion and an Akh-only broker, is a candidate to test.
An empty directory or read-only sandbox alone is insufficient evidence: reads,
inherited settings, and non-filesystem actions must be included in the test.

The installed `codex-cli 0.153.4` experimental generated schema includes
`ThreadStartParams.environments` and environment-native workspace roots. It
describes an empty environment list as disabling environment access. Public
App Server docs describe inspecting remote environments but do not establish an
arbitrary Akh executor implementation contract. Treat environment integration
as a separate optional spike, not as equivalent to callback tool routing.
Local evidence: `/private/tmp/akh-provider-schema/v2/ThreadStartParams.json`,
`definitions.TurnEnvironmentParams`; schema generated by the installed CLI.
[App Server environment inspection](https://learn.chatgpt.com/docs/app-server#inspect-an-execution-environment-experimental)

### Claude

`--tools ""` disables built-in tools while leaving MCP tools available. There
is a documented exception: `EndConversation` remains while another tool exists.
Treat that as provider loop control, not workspace execution. `--restricted`
removes command/code tools by default and limits settings/file access;
`--strict-mcp-config` restricts MCP configuration. These are documented knobs,
not yet a tested Akh launch profile.
[CLI reference](https://code.claude.com/docs/en/cli-reference)

`allowedTools` is an **auto-approval list**, not an availability allowlist.
Auto-approved calls skip `canUseTool`; a `PreToolUse` hook runs earlier.
Bare `disallowedTools` entries remove tools, while scoped rules deny matching
calls. Consequently Akh's authoritative permissions must remain inside its
actual tool handler even when the provider pre-approves the Akh MCP namespace.
[SDK permissions](https://code.claude.com/docs/en/agent-sdk/permissions)

`settingSources: []` excludes user/project/local settings but still reads
managed policy and global config. Auto-memory and Claude.ai connectors need
separate handling; `strictMcpConfig: true` suppresses inherited connectors.
Filesystem hooks and SDK callback hooks can coexist. A startup review must
include inherited hooks, skills, agents, plugins, MCP, memory, and managed
settings, not just the model's visible tool names. Managed policy cannot simply
be disabled by SDK options.
[Configuration loading](https://code.claude.com/docs/en/agent-sdk/claude-code-features)

Installed SDK **0.3.266** strengthens the technical case: its public `sdk.d.ts`
declares `tools: []`, `strictMcpConfig`, and `toolAliases`, with an explicit
example redirecting `Bash` to a workspace MCP tool. The alias documentation
warns that aliases cover model-name lookup only; `disallowedTools` is still
needed for harness-internal direct calls. This is a promising compatibility
mechanism, **not an isolation guarantee**. The same interface exposes
`abortController`, `Query.interrupt()` and `close()`; interrupt receipts may
report queued messages that still need cancellation. Verified in installed
`sdk.d.ts` around lines 1466–1508, 2075–2135, 2601–2615, and 2948–2950;
the TypeScript documentation fetch returned an error, so these details are
version-bound package evidence.
[Published SDK package](https://registry.npmjs.org/@anthropic-ai/claude-agent-sdk/-/claude-agent-sdk-0.3.266.tgz)

2026-09-11 follow-up: the published 0.3.266 declarations were fetched again and
confirm that alias API. A [Cowork issue](https://github.com/anthropics/claude-code/issues/91622)
reports native deny rules also denying alias destinations on one CLI version.
That report has not been reproduced here; alias and deny composition requires
an exact-version positive/negative test. Prefer explicit Akh tools with native
tools removed/denied for the first experiment, then test alias compatibility.
See [upstream services and semantic bindings](upstream-services-tool-bindings.md).

**Research conclusion:** a Claude profile exposing Akh MCP tools while excluding
native workspace execution is technically plausible with more explicit
controls than Codex currently documents. Neither this research nor the prior
successful custom-tool smoke test establishes full isolation. The installed
Claude CLI version observed was **2.1.263**; the implementation must pin/test
the SDK and the actual launched binary together.

## Bounded spikes and acceptance evidence

These are proposed experiments derived from the gaps above, not provider
capabilities already demonstrated.

1. **Execution boundary:** launch each pinned runtime with a declared Akh-only
   tool profile. Exercise read, search, edit, patch, shell, interactive shell
   input, hooks, inherited MCP/connectors, skills, browser/network tools and
   subagents. A local canary and a distinct remote canary must prove that all
   workspace effects and reads come through Akh. Trace startup as well as turns.
   Unknown tool/config capabilities must fail closed for this profile.
2. **Runtime placement:** verify provider authentication/session files remain
   available in a dedicated runtime environment without granting its native
   tools the Akh workspace or SSH credentials. Native provider infrastructure
   reads/writes are expected; “Akh owns everything else” should mean workspace
   operations and tool authority, not zero provider filesystem activity.
3. **Session contract:** bind Akh session, provider session, runtime host,
   workspace identity, executor identity and tool-catalog digest. Test exact-ID
   resume after restart, fork isolation, changed capabilities, lost provider
   state and concurrent sessions. Never substitute “most recent session.”
4. **Cancellation and approvals:** cancel during model streaming, an approval,
   a running Akh tool and a queued follow-up. Require terminal provider state
   plus tool/executor termination or explicit indeterminate status. Distinguish
   stream disconnection from cancellation and reconnect from replay.
5. **Subscription product surface:** document whether Akh launches the user's
   binary locally, hosts an unmodified binary per user, or embeds the SDK in a
   service. Validate the exact route against current provider terms; keep
   provider sign-in and account identity outside Akh's tool credentials.
6. **Loop variety:** test a normal tool loop, plan/review mode and delegated
   work against one Akh session/event contract. Provider-native delegation
   should remain disabled or separately marked until child tool authority,
   cancellation, session lineage and resource accounting are demonstrated.

The evidence favors an `AgentLoop` abstraction with explicit capability and
execution-policy metadata. It does not support assuming all implementations can
arbitrarily replace their inner reasoning algorithm, migrate hidden context
between providers, or supply identical mode semantics.
