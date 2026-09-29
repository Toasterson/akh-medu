# Lilith C4: the teacher arm through a sealed akh

- **Date:** 2026-09-29
- **Status:** Proposed (implemented on branch `feat/lilith-c4-teacher-via-akh`, not deployed, no live
  teacher run yet).
- **Work item:** Anima C4 `01a0e270-475e-74d0-9701-c1806eab79d0` (Plan `01a0e26f-d032`).
- **Related:** [ADR 116](../decisions/116-lilith-chief-of-staff-guided-rl.md) (Lilith CoS, C4 dojo),
  [ADR 110](../decisions/110-kaine-character-guided-reward-live-environment.md) (guided reward, live
  environment), [ADR 113](../decisions/113-full-akh-always.md) (full akh always; this plan adds one
  explicit exception), [exam workflow](2026-09-27-lilith-exam-and-evaluation-workflow.md) §2
  ("a teacher run through the same broker"), `scripts/lilith/dojo/README.md`.
- **Operator decision (2026-09-29):** the teacher makes no batch or direct model-API calls. It runs
  on akh runtimes and existing akhs.

## 1. Decision

The teacher is a dedicated **`dojo-teacher` akh on the Mac core's akhomed**. The dojo runner reaches it
over akhomed's loopback ACP listener (`tcp://127.0.0.1:18444`) as a new model kind, `--model-kind acp`.

- **Backing model:** an Ollama Cloud model, called with the Romans' key (`ROMAN_OLLAMA_KEY`). The default
  is the Romans' `glm-5.3`. The operator picks the model in `config/dojo-teacher.json`.
- **Replay tools:** the episode's replay servers stay inside the runner process. A loopback MCP gateway
  serves them, and the runner attaches them to each ACP session as client-pushed HTTP MCP servers.
- **The akh side is sealed:** with `tool_surface = "session-only"`, the akh keeps no tool of its own.
  The only tools a turn has are the ones the runner pushes for that episode.

### Evidence for the choice

| Question | Finding |
|---|---|
| Does akhomed's `session/new` honour client `mcpServers`? | Yes, for the HTTP transport. `src/acp/mod.rs:5027-5040` records them per session. `src/acp/mcp_bridge.rs:1-11,36-58`: every prompt connects to them and registers proxies `mcp__{server}__{tool}` (`mcp_bridge.rs:190`). stdio and SSE servers are counted but not connected, so `replay_server.py serve` over stdio cannot be attached. The runner serves the replay servers over HTTP instead (§2). |
| Can a client list the akh's tools? | Yes. `_akh/tools` returns the agent's registry (`src/acp/mod.rs:3978-3982`, `tools_json` at `4482-4488`). Session proxies never show up there (they are thread-local to the turn, `src/agent/tool.rs` `ToolSessionState`). |
| Is "exec none + no MCP servers" enough isolation? | **No.** Every akh registers about 40 built-in tools, including `http_fetch` and `web_search` (`src/agent/agent.rs:532-533`), whatever the config says. There was no per-akh allowlist. The only switch was the daemon-wide `AKH_AGENT_MAX_TOOLS=0`, which would also drop the pushed proxies. A teacher with `http_fetch` could read the public Forgejo history of `akh-medu` and learn which commits landed after the snapshot. The fix is the **sealed tool surface** below (a Rust change). |
| Could the teacher be Claude/Codex inside akhomed? | No. Deployed akhomed refuses `backend = "claude"` and `"codex"` (`src/deployment.rs:142-147`, `src/agent/runtime/bridge.rs:14-26`). |
| akhsessd + akhrtd Claude runtime (Opus on the Claude subscription)? | Possible, but not the smaller correct path today. akhsessd refuses client `mcpServers` (`crates/akh-session-services/src/acp.rs:211-218`) and advertises `mcpCapabilities.http = false` (`:191`). Tools come only from the server-side Akh profile (`config.rs:188-225`). Profile tools can have `{"kind":"client"}` destinations, which the ACP client answers via `_akh/input_required` / `_akh/respond`. But: (a) the Claude bridge hard-codes the server name `akh`, so the model sees `mcp__akh__<name>`, not `mcp__anima__...` (`scripts/agent-backends/claude.mjs:15,34`); fixing that changes the bridge hash and needs re-certified runtime conformance; (b) the tool specs are static profile config, while the replay contract is per snapshot; (c) the profile lives in the Talos repo (`clusters/archibald/akh-platform/fleet.json`), not in this repo; (d) both Claude runtimes (`fable-review-1/2`) are on a provider quota hold (`fable-quota-hold.json`). Kept as the follow-up for an Opus teacher (§7). |
| An existing Roman akh (janus/minerva/vesta)? | No. They carry live `[[mcp.servers]]` (anima, seshat, board, ...; `deploy/roman-mini/full_akh.py:40-54`). A Roman teacher would read the live tracker, which is the leak this design must make impossible. |

## 2. How an episode runs

`scripts/lilith/dojo/acp_teacher.py` (`AcpEpisodeRunner`) has the same `run_traced(spec, tools) ->
RunResult` as the in-process `ModelEpisodeRunner`, so `baseline.run_batch`, `families.verify`,
`reward.reduce_episode` and `compare` are unchanged.

1. The batch opens one `mcp_gateway.ReplayMcpGateway`, an HTTP server bound to `127.0.0.1` on an
   ephemeral port.
2. For each episode, the runner builds the replay servers as before (`replay_server.episode_servers`:
   Anima replay + git oracle, one `ReceiptLog`). It opens them in the gateway under a fresh 256-bit token:
   `http://127.0.0.1:<port>/dojo/<token>/{anima,git}`, with `Authorization: Bearer <token>`.
3. ACP: the runner calls `initialize` with the bearer of principal `dojo-teacher` in
   `_meta.authorization`. It checks `agentCapabilities.mcpCapabilities.http`.
4. **Preflight:** `_akh/tools` must answer `stale: false` with no tools, and `_akh/health` symbol and
   triple counts are recorded. Only then does the runner call `session/new`
   (`cwd = akh-ws:dojo-teacher`, `_meta["akh/workspace"] = "dojo-teacher"`, `mcpServers` = the two
   gateway URLs).
5. One `session/prompt` carries three text blocks: a two-line preamble, the **dojo system prompt**
   (`runner.compose_system_prompt`: role + chief-of-staff procedures + dojo addendum), and the **task**
   (`runner.user_prompt`).
6. The akh runs its own agentic loop. Its bridge calls the gateway (`initialize`,
   `notifications/initialized`, `tools/list`, `tools/call`). Every call goes through
   `runner.Broker.call`: `cos_declaration` is stripped, the replay server issues the receipt, and the
   result is rendered in the one encoding every model reads.
7. The runner reads `session/update` notifications:
   - `agent_message_chunk` text is split by `tool_call` boundaries;
   - text before a call becomes that step's `why_this_move`;
   - text after the last call is the final answer;
   - thought chunks are kept in the transcript as reasoning.

   Permission requests are answered `allow` for replay tools only.
8. After the turn: `_akh/usage` counter deltas give the episode's tokens, and `_akh/health` must be
   unchanged. The gateway closes the token (later calls get 404).
9. The `Episode`:
   - one `StepRecord(ToolCall)` per gateway call that reached a server (receipt id from the server);
   - then `runner.finish_answer` on the final text (the same parser the in-process arm uses);
   - `Episode.receipts` is the servers' log, unmodified.

Episodes run one at a time (`--concurrency` > 1 is refused, because the memory check is exact only
then). A provider failure is recorded as an unscored error episode that the next run retries, as with a
model endpoint failure. This covers an `error[...]` chunk, a `Refusal` stop, a transport timeout or a
wrong token.

## 3. Isolation argument: why the teacher cannot read the future

The sealed evaluation is invalid if the teacher can observe anything recorded after the episode's
snapshot `t_k`. The recording itself answers only as of `t_k` (`replay_server.py`). Every other channel
is closed by construction and then checked:

1. **No tool of its own (capability, in Rust).**
   - `tool_surface = "session-only"` makes akhomed call `Agent::seal_tools_to_session` after the last
     registration of the workspace (`src/bin/akhomed.rs`, just before the transcript-capture wiring).
   - `ToolRegistry::seal_to_session_tools` (`src/agent/tool.rs`) empties the persistent catalog and sets
     a one-way flag. `register` then refuses every later tool (skill activation, reactors, lazily
     attached servers).
   - What remains is the turn-local overlay, which holds only the proxies of servers the ACP client
     pushed for that session.
   - Unit test: `agent::tool::tests::sealed_registry_keeps_only_session_pushed_tools`.
2. **No other channel in the config (capability, in Rust).**
   - `AkhConfig::validate_tool_surface` (`src/akh_config.rs`, run by `load_checked`, which akhomed
     uses for every workspace) refuses to load a session-only akh that has any of:
     - `[[mcp.servers]]`, `[board]` (inbox ingress), `[monitoring]`, `[delegation]`,
       `[chief_of_staff]` or `[agent_runtime]`;
     - a Claude/Codex backing, or an `[exec]` other than `none`;
     - `openclaw_import` or `extra_read_roots`;
     - `related_workspaces` (federated reads of other akhs' memory) or `shared_partitions`;
     - `context_strategy = "virtual"`, `capture_transcripts = true` or enabled autonomy.
   - Tests: `akh_config::tests::session_only_*`, `roman_fixture_tests::dojo_teacher_workspace_loads_sealed`.
3. **Only the episode's replay servers (network).**
   - The gateway binds `127.0.0.1` only.
   - It routes a request only if the path token names an open episode and the bearer matches
     (constant-time comparison).
   - It answers only the replay servers' own tools. An unknown tool is a protocol error with no receipt.
   - It forgets the token when the episode ends.
   - The akh cannot reach a different episode's servers, because it never learns their tokens.
4. **Checked at run time (the runner refuses, it does not warn).**
   - `IsolationRefusal` aborts the whole batch before anything is scored:
     - `_akh/tools` lists any tool at preflight, or answers `stale`;
     - `mcpCapabilities.http` is not `true`;
     - during the turn, a `tool_call` or `session/request_permission` names a tool outside the
       episode's replay set (the turn is cancelled first);
     - the akh's memory size changes across an episode or between episodes.
   - `cmd_run` probes the akh before it writes `provenance.json`. A leaky teacher therefore leaves no
     run directory.
   - The probe (daemon version, memory size) and the optional `--akh-config` digest are identity keys
     of the run, so a resume against a changed teacher is refused.
5. **No memory between episodes.**
   - The workspace keeps nothing: `capture_transcripts = false`, `autonomy.enabled = false`,
     `context_strategy = "passthrough"` (no history, KG or recall pre-loaded), one ACP session per
     episode.
   - The `_akh/health` check (item 4) proves at run time that nothing was written. A teacher that
     cannot remember episode A cannot carry its later-snapshot evidence into episode B.
6. **A scoped identity.**
   - Principal `dojo-teacher` (`secrets/dojo-teacher.key`) may name only the `dojo-teacher` akh. It has
     no control plane.
   - The teacher is not in the core `/v1` scopes, the events principal, T3 or reactor seeds
     (`deploy/roman-mini/configure_core.py`; test `DojoTeacherRenderTests`).
   - Working akhs cannot be pointed at by this key.

The backing model itself is a remote model with no tools of its own on this path. Ollama Cloud runs only
the tools the akh sends it, and the akh sends only the sealed catalog.

## 4. "Same broker" equivalence

| Aspect | In-process arm (parent, candidate) | Teacher arm (`acp`) | Delta |
|---|---|---|---|
| System prompt | `compose_system_prompt` as the system message | the same bytes, as the second text block of the first user turn. The akh's system prompt is the neutral `DOJO_TEACHER_PROMPT` plus akhomed's own "Available Tools" section built from the same catalog | Placement differs. `system_prompt_sha256` and `prompt_sha256` are identical, so `compare` does not warn. `provenance.teacher.prompt_delivery = acp-session-prompt` records the placement. |
| Task | `user_prompt(spec, now)` | the same bytes (third block). `user_prompt_sha256` is equal | none |
| Tool names | `mcp__anima__<tool>`, `mcp__git__<tool>` | akhomed names pushed proxies `mcp__{sanitize(server)}__{tool}` (`mcp_bridge.rs:190`). `anima` and `git` are unchanged by sanitizing | none. `tools_sha256` is equal (tested) |
| Tool schemas | recorded `inputSchema` verbatim | akhomed keeps `properties` + `required` (`agentic_chat.rs:1956-1966`) and drops undeclared arguments (`tools/mcp.rs:783-792`) | The recorded Anima contract (1 contract across all 20 snapshots, 80 tools) loses only `additionalProperties`, on 14 tools, and akhomed enforces it anyway by dropping undeclared arguments. |
| Result the model reads | `{"isError", "_meta": {receipt, digest, observed_at}, "result"}` | the gateway puts exactly that JSON (`Broker.call`) into `content[0].text`. akhomed passes on only the text (`upstream_client.rs:330-333,419-426`) and ignores `isError` (`mcp_bridge.rs:213-222`), but both are inside the text | none (tested byte for byte against the in-process arm) |
| Receipts | issued by the replay servers into one `ReceiptLog` | the same servers and log, in process | none |
| `cos_declaration` | stripped by the broker, recorded in the trace | akhomed drops it before the call as an undeclared argument, so the teacher's declarations are not observable | `trace.declarations` is empty for the teacher. Scoring does not read declarations. |
| Answer parsing | `finish_answer` on the message without a tool call | `finish_answer` on the text after the last tool call | none |
| Round cap | `--max-rounds` model turns | the runner cancels the turn after `--max-rounds` tool calls (each round makes at least one). The akh's own caps (context-scaled rounds, `AKH_AGENT_TURN_TIMEOUT_SECS` = 300 s by default) also apply and end as `round-cap` | slightly stricter for the teacher, never looser |
| Budget component | completion tokens per turn on the first call of the turn | per-turn tokens are not visible over ACP. All completion tokens of the episode (the `_akh/usage` delta) sit on the answer step, and `wall_ms` is measured between calls | The cumulative budget at the answer step, which sets the episode weight, is the same quantity. Intermediate steps differ. |

## 5. Cost and quotas

- **No API key and no direct model API.** The teacher's turns run on the **Romans' Ollama Cloud
  subscription** (`ROMAN_OLLAMA_KEY`, `deploy/roman-mini/full_akh.py:27-28`). They share its rate limits
  with Janus, Minerva and Vesta. Interactive ACP turns are not counted against the autonomy budget.
- Tokens per episode are the same order as the parent arm, about 90k-120k input at 4-6 rounds (see the
  README cost section). One episode at a time: the sealed set (278 episodes) is a long job that can be
  resumed. Schedule it outside the Romans' busy hours. The first `--limit 1` run measures the real
  latency.
- No Runpod or Claude quota is used. The Fable/Claude runtimes stay free for Anima reviews.

## 6. Rollout (supervisor; nothing here is deployed yet)

1. Review and merge the branch. Build `akhomed` from `origin/main` and install it per
   `deploy/roman-mini/CUTOVER.md` step 1 and `.claude/rules/deploy-from-main.md`. The seal is new
   akhomed code: without it the runner refuses, because `_akh/tools` would list about 40 tools.
2. On the Mac: `printf '{"model": "glm-5.3"}\n' > "$ROOT/config/dojo-teacher.json"` (or a stronger
   Ollama Cloud model).
3. `python3 deploy/roman-mini/configure_core.py --plan --root "$ROOT"`, review, render, restart the
   core. The render creates `secrets/dojo-teacher.key` (0600) and adds the principal and the auto-start
   entry.
4. Verify: the core log shows `tool registry sealed ... workspace=dojo-teacher`, and a runner
   `--dry-run` plus a `--limit 1 --split dev` smoke run succeed. The probe refuses any teacher that is
   not sealed.
5. Sealed run and compare. The exact commands are in the C4 report and `scripts/lilith/dojo/README.md`.

## 7. Follow-ups

- **An Opus teacher** through akhsessd + akhrtd's Claude runtime (Claude subscription). This needs:
  - `destination: client` tool specs for the replay contract in a dedicated Akh profile (Talos repo);
  - a bridge option to name the in-process MCP server per tool namespace (`anima`, `git`) instead of
    `akh`, with re-certified runtime conformance;
  - a matching `--model-kind akhsessd` runner that answers `_akh/input_required` from the same
    `Broker.call`.

  The gateway, `finish_answer`, the isolation checks and the `Episode` assembly carry over unchanged.
- Once the teacher has run, record the placement of the dojo system prompt and the `cos_declaration`
  delta in the C4 report, next to the per-family table.
- ADR 113 gets an amendment line naming `tool_surface = "session-only"` as the one exception
  (exam instruments only). This plan is the design record until then.
