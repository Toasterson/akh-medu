# ADR 062 — LLM-driven agentic chat loop over ACP `session/prompt`

> Date: 2026-06-17
> Status: Implemented
> Depends on: ADR-053 (first-class AkhConfig: per-akh model/endpoint/prompt pin),
>             ADR-060 (tool sandbox + permission layer), ADR-061 (ACP-only),
>             ADR-027 (agent lock discipline)
> Plan: [`docs/ai/plans/2026-06-17-llm-agentic-chat.md`](../plans/2026-06-17-llm-agentic-chat.md)

## Context

After the P8 rip-out (ADR-061) ACP is akhomed's only inbound transport, and the
akh's pinned model is configured per-workspace (`akh.toml` `ollama_endpoint`,
local or `ollama.com` cloud — ADR-053). But the chat path was never wired to it:
`session/prompt` ran only the **symbolic** `ChatProcessor` (NLU → KG facts /
queries / goals, with a small in-process NLU LLM for creative fallback), and the
only entry point to the pinned model — `Agent::generate_with_model` — had **zero
callers**. So configuring a model changed nothing, and the agent's tools
(Edit/Glob/Grep/code_gen/…) were never driven by an LLM. This was the parked "P2
LLM response generation" work.

## Decision

Add an **LLM-driven agentic turn** and route `session/prompt` through it, with
the symbolic pipeline as a fallback.

- `Agent::run_agentic_turn(engine, agent_id, user_prompt)` (`src/agent/agentic_chat.rs`)
  runs the loop: ask the akh's pinned model (via `AgentModelManager` → Ollama
  `/api/chat` with `tools`), execute any requested tool calls through
  `ToolRegistry::execute`, feed results back, and repeat until the model returns
  a final answer or a round cap is hit. Returns `Vec<AkhMessage>` narrating the
  turn (each tool call + result, then the answer).
- **Health-check-gated fallback.** It returns `None` when no model backend is
  reachable (no model manager, or `ollama_available()` is false); the ACP
  `run_turn` handler then falls back to `ChatProcessor::process_input`. So a
  daemon with no Ollama configured behaves exactly as before — the LLM path is
  purely additive.
- **Tool catalog.** Every registered tool's `ToolManifest` is exposed to the
  model as an Ollama/OpenAI function schema. Safe to expose all: the ADR-060
  permission policy still gates execution at dispatch, and `Ask` tools bridge to
  the ACP client via the existing per-turn permission asker.
- **Transport plumbing.** `OllamaClient::chat(messages, tools) -> AssistantTurn`
  (content + tool calls) and `AgentModelManager::chat_with_tools(agent_id,
  conversation, tools)` (prepends the akh's cached system prompt). Ollama-only —
  there is no remote-fallback tool-calling path yet.

## ADR-027 compliance

The whole turn is **synchronous** — blocking `ureq` HTTP to Ollama interleaved
with synchronous tool execution — and runs inside the existing `spawn_blocking`
closure in `AkhAcpAgent::run_turn` with the agent `Mutex` held. No guard crosses
an `.await`. The cost is that the agent lock is held for the duration of an LLM
turn (seconds to minutes for a multi-tool turn); acceptable for the primary
interactive path, where the turn *is* the priority and turns are serialized.

## Alternatives considered

- **Release the agent lock during Ollama HTTP, re-lock per tool call.** Lower
  lock-hold time, but breaks the single-lock ADR-027 invariant and complicates
  the permission bridge. Deferred — revisit if lock contention is observed.
- **Stream tool progress mid-turn** (emit `session/update` as each tool runs).
  Needs the `AcpSink` inside `spawn_blocking`, which would put an async sender
  next to held guards. v1 collects the turn's messages and emits them after the
  guards drop (matching the existing pattern). Live streaming is a follow-up.
- **Curate the exposed tool set.** Exposing all ~25 tools is token-heavy for a
  small model. Left as-is for v1 (permission-gated); curation can come later.

## Consequences

- A workspace with a reachable `ollama_endpoint` now gets real LLM chat with tool
  use over ACP; everything else stays symbolic.
- New surface: `OllamaClient::chat`, `AgentModelManager::chat_with_tools`,
  `Agent::run_agentic_turn`. `generate_with_model` remains for non-chat callers.
- Bounded to 8 tool-call rounds per turn (backstop against a looping model).
- Outbound MCP to Anima/Seshat (`UpstreamMcpClient`, ADR-061) is still not a tool
  — exposing it as an agent tool is separate, future work.
