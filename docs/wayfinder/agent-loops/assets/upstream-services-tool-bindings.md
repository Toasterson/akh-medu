# Upstream services, managed profiles and runtime tool bindings

Date: 2026-09-11. Status: proposed contracts and researched capability boundaries;
no services launched or provider conformance tests run.
Parent: [deployment plan](k8s-acp-t3-deployment.md).

## Accepted constraint

External services use unmodified upstream releases. Akh starts the required
remote server, prepares its supported configuration, manages supported login
flows when needed, and automates connecting the client. This supersedes the
earlier requirement to add an Akh driver to T3's source. Our own launchers,
protocol bridges and supported extensions can be separate artifacts; patching
upstream bundles or writing their private databases is outside this plan.

Editor servers remain beside `akhexecd` in selectable sandbox profiles.
`akhsessd` owns central sessions, and `akhrtd` may place their runtimes elsewhere.
Anima consumes central admission, events and attachment descriptors, retaining
its work-item and review responsibilities. See the
[image and Anima migration trace](sandbox-images-anima.md).

## One profile model, specialized owners

Share versioned data contracts and a small profile/configuration library.
Do not add another central management daemon or link every provider SDK into `akhomed`.
Use provider-specific adapters to render actual upstream configuration.

| Proposed record | Contents |
| --- | --- |
| `ServiceProfile` | Upstream artifact/version/digest, platform, service kind, supported launch arguments, configuration renderer revision, health/connect protocol, state retention and update policy |
| `ServiceInstance` | Profile revision, instance/owner IDs, sandbox or runtime binding, process lease/generation, observed revision, readiness and attachment capabilities |
| `ConfigRevision` | Non-secret desired settings, managed paths/keys, input and rendered hashes, precedence rules and activation/restart requirements |
| `AuthProfileRef` | Principal, site, service/account scope, supported authentication method and reference to provider-owned credential storage; no token value |
| `ToolBindingProfile` | Runtime/version, advertised names and schemas, semantic adapter versions, authoritative destinations, native-tool dispositions and conformance evidence |

`akhomed` supplies Akh identity and desired configuration references. `akhsbxd`
owns sandbox allocation and the desired lifecycle of sandbox-local services.
Its in-sandbox helper, hosted with `akhexecd`, applies profiles and supervises
editor processes. `akhrtd` performs the equivalent work for provider processes
on runtime hosts. Share the lifecycle/configuration library, not a second
controller of either process. Platform supervisors can implement restart policy
under the same instance lease.

`akhsessd` authorizes attachment, binds immutable profile revisions to a session,
and presents readiness/login-required/connect state. `akhexecd` continues to own
agent workspace operations. Service-control operations are separately authorized
management operations: an agent shell tool must not acquire the ability to
replace its own restrictions or read another service's login state.

The common lifecycle is prepare -> render/validate -> authenticate if required
-> start -> probe -> connect -> drain/update -> stop. Preparation and status
inspection do not start a billable agent turn. Login can pause for the user's
supported browser/device flow; helpers can initiate and resume it without
pretending every provider offers unattended login. Credential revocation and
expired login produce explicit state, not fallback to an unrelated account.

## Configuration and connection management

Use a private profile directory per service/account scope and the upstream
configuration-path options where supported. Keep generated configuration,
upstream writable application state, provider continuation/login state and
workspace files in separate retention classes. An image contains software and
templates; allocated instances receive their own configuration and auth scope.
Never change the host user's shared editor/provider profile as an implicit side
effect of launching a sandbox.

The renderer owns explicit files or keys. Validate the candidate, record its
revision, and apply it atomically. Preserve upstream/user-owned settings and
report conflicting changes instead of repeatedly overwriting them. If a program
rewrites an entire file, use its supported API or a stopped-process update with
a version check. File replacement alone cannot make concurrent writers safe.
Treat policy-bearing files as management-owned; arbitrary repository content
must not loosen enforced session policy. Capture effective configuration and
loaded settings sources, not just our intended overlay.

Resolve credentials at the owning service host through the supported auth
adapter. Keep token values out of profile records, images, launch URLs and
ordinary logs. Preserve the provider's supported credential store and account
scope through restarts; do not copy arbitrary host login directories into an
image. Only supported sharing/refresh flows may be used. Editor access, central
session access and model-provider login are distinct identities.

Return a `WorkspaceAccessDescriptor` with client type, workspace/session binding,
endpoint, compatible version, connection method and scoped auth-exchange flow.
The launcher can generate supported SSH/client entries, open an upstream deep
link or show the exact connection action. A descriptor must separately report
`workspace_access` and `central_agent_attachment`: a healthy editor endpoint
does not establish that its chat UI controls the central session.

Update profiles by revision and drain active processes before incompatible
changes. Disable unattended in-process upgrades where upstream supports that;
otherwise detect revision drift and stop admitting new sessions under the old
capability claim. Restart/rollback must preserve or explicitly migrate retained
state, including T3 environment identity and provider continuation handles.

## Machine-local environment connection helper

The user confirmed a small process on the connecting machine to add the remote
T3 environment once it has been bootstrapped. Implement this as a per-user
connection helper, with an on-demand command and optional background watcher.
It uses the shared profile/attachment contracts; it does not need a new central
daemon. Its responsibility ends at environment registration, connection and
reconnection. `akhsessd` remains the session owner, while `akhsbxd` and its helper
own the prestarted sandbox/server lifecycle.

Proposed flow:

1. The sandbox helper reports the executor/server handshake, persistent upstream
   environment ID, endpoint and allocation generation. `akhsessd` exposes the
   authorized attachment descriptor and a durable readiness event to the user's
   connection helper. Event delivery is a notification, not an access grant.
2. The connection helper checks reachability from the actual client machine and
   reconciles the requested environment against its local attachment records.
   Scope those records by user/client profile and upstream environment ID;
   repeated readiness events must not create duplicate entries. A changed route
   to the same environment is an update, not a new environment identity.
3. If the client is not paired, request a fresh device-specific pairing link
   through the authorized server helper and invoke T3's supported connection
   flow. Keep the secret out of readiness events and the helper's ordinary state
   or logs. Retain only non-secret mapping/status; T3 owns its client credentials.
4. Report `connection_offered`, `paired` and `connected` separately, using the
   evidence available from the supported client interface. Opening a URL is not
   proof of registration. If a confirmation/import API is absent, retain an
   explicit pending state and let the upstream UI finish pairing.
5. Reconnect using the client's existing authorization. After suspension or a
   missed event, reconcile current descriptors rather than replaying every
   bootstrap action. On release, close helper-owned tunnels and mark the entry
   unavailable; remove a managed connection only through a supported interface.
   Client disconnect must not stop the centrally owned remote server.

The inspected upstream [remote-access guide](https://github.com/pingdotgg/t3code/blob/2ebc9fa4ed24e389eef24039f27c73ae6633858f/docs/user/remote-access.md)
documents `t3 pair` for an already-running server and pairing through the
receiving client's Add environment flow. Its
[hosted pairing implementation](https://github.com/pingdotgg/t3code/blob/2ebc9fa4ed24e389eef24039f27c73ae6633858f/apps/web/src/hostedPairing.ts)
also provides a `/pair` route. Use upstream-generated links and preserve their
secret handling; do not invent an undocumented desktop URI scheme. Direct
pairing over the authorized private route is the initial design. T3 Connect
account-based discovery is an optional upstream route, not a new requirement.
Validate each selected release and client type, including duplicate addition,
expired pairing, endpoint replacement and removal; these sources establish the
upstream flow, not a tested unattended desktop import API.

This helper resolves environment onboarding. Central agent attachment inside
that environment still uses the supported runtime/bridge path described below;
registering a server does not itself change how T3 runs its provider sessions.

## Upstream T3 attachment feasibility

The inspected T3 [built-in registry](https://github.com/pingdotgg/t3code/blob/2ebc9fa4ed24e389eef24039f27c73ae6633858f/apps/server/src/provider/builtInDrivers.ts)
instantiates a static set of provider drivers. Internal `ProviderDriver` types
are not a public plugin loader; an arbitrary configured driver name does not
register Akh. Its ACP runtime spawns a process, so a central WSS URL alone does
not complete integration.

The existing [Codex driver](https://github.com/pingdotgg/t3code/blob/2ebc9fa4ed24e389eef24039f27c73ae6633858f/apps/server/src/provider/Drivers/CodexDriver.ts)
uses configurable binary/home paths and launch arguments. That is a concrete
surface to evaluate for a separately shipped bridge, not evidence that T3
already supports generic Akh sessions. A bridge must honor the protocol that
driver expects, including discovery, account/status probes, start/resume,
approvals, cancellation, artifacts and events. A stdio-to-WSS ACP relay is
sufficient only for an upstream client that actually launches arbitrary ACP
commands. Do not feed ACP to a Codex app-server connection.

First verify supported command/endpoint/extension configuration on the pinned
release. Prefer a generic ACP interface when upstream exposes one. If a bridge
over an existing provider protocol is feasible, keep it external, narrow and
explicitly versioned. It maps to an already authorized central session and must
not silently spawn a second loop, invent account capabilities, or let provider
maintenance mutate shared runtime installations. Test T3 thread recreation and
connection automation without editing its private database.

If stock T3 has no adequate supported attachment surface, mark central chat
attachment unavailable for that version, while its workspace server remains
usable. Record the upstream feature needed; adding a source patch is not an
implicit fallback. Full central T3 sessions remain a delivery requirement,
not a capability satisfied by launching its UI. Other ACP-capable clients and
Anima can validate the central-session contract independently.

## Preserve tool semantics at the runtime boundary

The session tool catalog is authoritative for the admitted session. `akhrtd`'s
provider adapter translates it into upstream dynamic tools/MCP definitions and
provider-native result envelopes. The broker in `akhsessd` authorizes calls and
routes them to `akhexecd`, `akhomed`, scoped MCP services or an authorized client.
The runtime cannot choose a different workspace or widen those grants.

Use a semantic adapter, not merely a familiar tool name:

| Surface | Contract the adapter must preserve |
| --- | --- |
| Shell/process | Argument units and defaults, workspace-relative cwd, environment scope, stdout/stderr and exit status, timeouts, streaming, process handles, stdin, cancellation and background completion |
| Read/search | Path mapping, offsets/limits, encoding, binary handling, glob/regex semantics and truncation indicators |
| Edit/patch | Exact-match or revision preconditions, atomicity expectations, conflict/failure reporting and resulting diff/revision |
| Control/tools | Permission requests, user questions, tool-use/result IDs, structured errors, parallelism/read-only annotations and turn termination |

Map native process/task handles to durable executor operation IDs; an exit code
cannot stand in for a still-running process. Preserve unknown outcomes on lost
connections and query by operation ID before retrying a mutation. A tool receipt
records requested and resolved tool names, policy/binding revision, session/turn,
executor/workspace generation and outcome. No receipt claims an operation ran
merely because the provider emitted a tool call.

Native control tools that terminate a turn or ask a question need appropriate
session mappings; they are not automatically workspace tools to disable.
Provider-private configuration/authentication reads remain necessary. In remote
workspace mode the runtime sees only its authorized private state; workspace
data and execution capabilities are provided through the bound broker.

## Substitute first, deny residual native paths, explain the correction

1. Advertise the supplied tools and concise workspace instructions at session
   start; restore the binding context on resume/compaction using supported
   runtime mechanisms. Updated policy takes effect in enforcement independently
   of whether the model has received a new reminder.
2. Prefer a supported native-name alias/replacement when behavior is compatible.
   Remove unneeded native definitions using tool-selection controls. Otherwise
   expose distinctly named Akh tools and instruct the model to use them.
3. Deny forbidden remaining native workspace calls before they execute. Apply
   the same policy to subagents, hooks, plugins, resumed jobs and inherited MCP
   configuration. Host isolation limits the runtime's actual capabilities where
   tool controls do not cover every path; it must still access its permitted
   provider state and model endpoint.
4. Return a concise provider-compatible error identifying the available tool
   and its required arguments. For example: “Native Bash is disabled for this
   session. Use mcp__akh__bash; it runs in the bound workspace. Nothing ran.”
   Only claim nothing ran when interception preceded execution. A compatible
   alias can route the original call directly and avoid a corrective round trip.
5. Bound repeated denied attempts and report a configuration/capability failure
   if the runtime cannot proceed. Never enable the forbidden tool as recovery.

The model can adapt within the conversation from instructions and tool errors.
That is not persistent training, and neither fast adaptation nor perfect tool
selection is guaranteed. Blocking remains effective on the first attempt.
Use capability tests to verify native denial **and** successful replacement;
a profile that denies every tool is not a working integration.

## Provider evidence and acceptance

Claude Agent SDK 0.3.266's published type declarations expose `tools`,
`strictMcpConfig` and `toolAliases`, including built-in-to-MCP routing. Aliases
resolve model-emitted names; the declarations separately identify native direct
calls as requiring denial controls. This supports the proposed bridge design,
not a verified combination of permissions on every CLI version.
[Published package](https://registry.npmjs.org/@anthropic-ai/claude-agent-sdk/-/claude-agent-sdk-0.3.266.tgz).

Claude's [permission documentation](https://code.claude.com/docs/en/agent-sdk/permissions)
distinguishes tool availability, explicit denies and auto-approval. `allowedTools`
is not an exclusive tool list. A [reported Cowork regression](https://github.com/anthropics/claude-code/issues/91622)
describes deny rules also matching alias destinations; it is an upstream user
report, not a reproduced result here. Test the exact SDK/CLI/permission pairing.
If alias composition fails, use explicit Akh tool names with native tools
removed/denied, while preserving independent containment. SDK capabilities do
not automatically exist in every CLI adapter or kagent harness.

Codex app-server supports experimental `dynamicTools`, with thread persistence
and resume behavior. This is a tool-registration surface, not a universal
native-tool replacement switch.
[App-server documentation](https://learn.chatgpt.com/docs/app-server).
Its `PreToolUse` hooks cover many local tools but exclude hosted tools and some
specialized paths; `write_stdin` does not repeat the pre-hook for an existing
process. Hooks alone therefore cannot enforce all remote workspace restrictions.
[Hook coverage](https://learn.chatgpt.com/docs/hooks#tool-coverage).
Keep per-version disable controls, configuration-source checks and OS/process
restrictions in the profile. Existing [Codex/Claude research](codex-claude.md)
and the current remote-executor guard remain applicable.

Acceptance for one upstream profile requires: unchanged upstream artifact;
repeatable configuration with retained user state; supported scoped login and
reconnect; the same central session after restart; an executor-only canary read,
edit and command from a runtime on another host; denied native paths including
continuations/subagents; a successful supplied-tool path after a denial; and
receipts with correct workspace/version/cancellation outcomes. Test broker
unavailability before admission and during a turn: no local execution fallback.
Run the same checks on upgrades before advertising the profile as compatible.

Implement this as refinements of D0/D2/D3/D4/D5 in the deployment plan. The first
provider experiment should use the existing Akh Claude SDK bridge with explicit
Akh tool names, then evaluate aliases as a compatibility improvement. This
separates central remote execution proof from T3's attachment feasibility.
