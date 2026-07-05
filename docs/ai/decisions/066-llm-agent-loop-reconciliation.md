# ADR 066 — LLM Agent Loop: reconciliation with ADR 062

> Date: 2026-06-18
> Status: ACCEPTED (records reality + residual deltas; not a build plan)
> Phase: — (no new phase; the loop is built)
> Supersedes: this session's standalone LLM-agent-loop design draft (removed —
>             it was redundant by content with ADR 062)
> Related: ADR 062 (LLM agentic chat loop — *Implemented*), ADR 060 (tool sandbox
>          + permission), ADR 065 (Working-Context Assembly Runtime — assembles
>          into this loop), develop Phase 41 (Thinking Engine, ADR 052)

## Context

While drafting a context-management plan, this session designed an "LLM agent
loop" from scratch against a branch that predated develop. That was a mistake of
process: **the loop already exists on develop**, implemented and tested, as
[ADR 062](062-llm-agentic-chat-loop.md). A same-numbered ADR collision is *not* a
content collision — but here the *content* genuinely matched, so the draft was
redundant and has been removed. This ADR records the reconciliation so the work
isn't re-attempted, and captures the deltas ADR 062 does **not** cover.

## What ADR 062 already provides (do not rebuild)

`Agent::run_agentic_turn` ([`src/agent/agentic_chat.rs`](../../src/agent/agentic_chat.rs),
develop): asks the akh's pinned model via `OllamaClient::chat(messages, tools) ->
AssistantTurn { content, tool_calls }`, exposes every `ToolManifest` as a
function schema (`ollama_tool_catalog`), executes calls through
`ToolRegistry::execute`, feeds results back, loops to `DEFAULT_MAX_ROUNDS = 8`,
and falls back to the symbolic `ChatProcessor` when no backend is reachable. Tool
execution is gated by the **ADR-060** sandbox/permission layer (with `Ask` tools
bridging to the ACP client) — *not* a bespoke psyche-veto path.

This matches, point for point, the loop this session had proposed (Ollama
chat-with-tools, schema bridge, round loop, guards, fallback). No new build is
warranted.

## Residual deltas (tracked, not a phase)

Things ADR 062 explicitly leaves open, plus deltas from this session's design
that remain genuinely unaddressed:

- **LLM-driven OODA *autonomy* Decide.** ADR 062 wired only the chat path
  (`session/prompt`). Whether the autonomous OODA loop should gain an
  `DecideMode::LlmAgentic` (LLM proposes the next tool toward a goal) is open —
  and may instead be served by develop's Phase 41 Thinking Engine (ADR 052).
- **VSA-decide as recommender/validator** of the LLM's tool choice
  (`vsa_native_decide`, [`src/agent/vsa_decide.rs:370`](../../src/agent/vsa_decide.rs)) —
  augment vs. override (the long-standing RFD OQ1).
- **Safety-layer alignment.** ADR-060 is the canonical execution gate; any psyche
  `check_veto`/`check_bias` signal should layer as a *soft/bias* input, not a
  second hard gate.
- ADR 062's own follow-ups: mid-turn `session/update` streaming; tool-set
  curation/ranking for small models; exposing `UpstreamMcpClient` (Anima/Seshat)
  as a tool; remote-fallback (OpenAI-compatible) tool calling.

## Relationship to Phase 44

[ADR 065](065-context-assembly-runtime.md) (Phase 44) assembles the prompt this
loop consumes: its budget allocator and graduated-fidelity transcript feed
`chat_with_tools`, and its `SafetyEnforced` decorator guards what the loop's
window may drop — complementing ADR-060's execution gate.

## Lesson for process

Always reconcile against `develop` (and sibling worktrees) before designing a
subsystem — branch-local absence is not global absence. ADR/phase *numbers* can
collide without content colliding; judge redundancy by **content**.
