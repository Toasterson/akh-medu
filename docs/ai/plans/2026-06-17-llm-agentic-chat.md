# Plan — LLM-driven agentic chat over ACP

> Date: 2026-06-17
> Status: Implemented
> ADR: [`062-llm-agentic-chat-loop.md`](../decisions/062-llm-agentic-chat-loop.md)

## Goal

Make `session/prompt` drive the akh's pinned Ollama model (local or `ollama.com`
cloud) with an LLM tool-calling loop over the existing `ToolRegistry`, falling
back to the symbolic `ChatProcessor` when no model backend is reachable. This is
the parked "P2 LLM response generation" work.

## Steps (all landed)

1. **Ollama tool calling** (`src/agent/ollama.rs`): `OllamaClient::chat(messages,
   tools) -> AssistantTurn { content, tool_calls }`; parse `message.tool_calls`.
2. **Model manager** (`src/agent/model_manager.rs`): `chat_with_tools(agent_id,
   conversation, tools)` — prepend the akh's cached system prompt, call
   `ollama.chat`. (Reuses `ollama_available()` for the gate.)
3. **Agentic loop** (`src/agent/agentic_chat.rs`, new): `Agent::run_agentic_turn`
   — build the tool catalog from manifests, loop chat→execute→feed-back until a
   final answer or 8 rounds; `None` when no backend (caller falls back).
4. **ACP wiring** (`src/acp/mod.rs`): `run_turn` tries `run_agentic_turn`, falls
   back to `process_input`; the per-turn permission asker is installed for both.

## Verification

`cargo test --features server`: 2537 lib + integration suites, 0 fail (4 new
agentic_chat unit tests). Test agents have no model manager / no Ollama, so they
exercise the symbolic fallback — existing ACP/chat tests are unaffected.

## Follow-ups (not in this increment)

- Live mid-turn streaming of tool progress (`session/update` from inside the
  turn) — needs the sink across the spawn_blocking boundary.
- Curate / rank the exposed tool set for small models.
- Expose `UpstreamMcpClient` (Anima/Seshat) as an agent tool.
- Remote-fallback tool calling (OpenAI-compatible `fallback_api`).
