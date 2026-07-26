# ADR 070 — Selectable chat/agentic backing: Ollama or Candle (full tool calling)

> Date: 2026-06-20
> Status: Accepted (implemented)
> Supersedes: the "Ollama removed entirely" stance recorded in ADR 067 (the
>             candle-conversational-boundary work was done on a branch where
>             Ollama had been deleted; this ADR restores Ollama and makes both
>             backings co-equal)
> Related: ADR 062 (Ollama agentic loop), ADR 053 (first-class AkhConfig),
>          ADR 067/068/069 (candle boundary / ROCm GPU / Gemma-4 audio),
>          ADR 037 (LLM-VSA deep integration — the eventual InferenceBackend),
>          Phase 39 (distributed inference — full backend registry/router)

## Context

Two branches diverged in opposite directions on the LLM backend. `develop` kept
**Ollama** as the agentic chat backing (ADR 062), added `ollama.com` **cloud**
support (Bearer auth + `env:VAR`), then built Phase 38. A separate branch
(`worktree-chat-e2e`) pivoted to **candle-only**, *deleted* Ollama and llama.cpp,
and added a native Candle conversational boundary, a ROCm/HIP AMD-GPU backend,
Gemma-4 audio, vendored hidden-states forks, and a reasoning benchmark.

`ollama.com` cloud removes the local-compute cost objection that motivated the
candle-only pivot, so Ollama returns as the *common* backing. But Candle is not
merely an NLU helper — for a host that wants fully local inference it is a
first-class **chat/agentic backing**. Phase 44 already anticipates this
("Build when Candle lands in the loop"). The constraint is that **not every host
in the home network has a GPU**, so a Candle backing must run CPU-only too.

## Decision

The agentic turn loop is **backend-pluggable**. Each akh chooses its backing in
`akh.toml`:

```toml
backend = "ollama"   # default — local or ollama.com cloud (native tool calling)
# backend = "candle" # in-process GGUF, grammar-constrained tool calling
```

- **`ChatBackend` trait** (`src/agent/chat_backend.rs`): `available`,
  `prepare`, `chat_with_tools`, `chat_with_tools_streaming`, returning the same
  `AssistantTurn { content, tool_calls }` contract Ollama already produced — so
  `Agent::run_agentic_turn` is identical regardless of backend.
- **`OllamaChatBackend`** wraps the existing `AgentModelManager` — behaviour is
  exactly ADR 062 (native function-calling, cloud-capable). It is the default;
  `run_agentic_turn` wraps the model manager in it when no explicit backend is
  set, so nothing changes for existing akhs.
- **`CandleChatBackend`** is a full tool-calling peer. It renders the model's
  native tool template (Qwen2.5 `<tool_call>`, Gemma function call), parses the
  emitted calls, and — crucially — generates tool-call **arguments under a JSON
  grammar constraint** (`JsonConstraint`, `tool_call.gbnf`): at each decode step
  only token candidates that keep the output a valid JSON prefix are sampled. The
  model supplies the semantics (which tool, which arguments); the grammar
  guarantees the bytes parse, so a call is never dropped to malformed JSON. This
  is the same mechanism llama.cpp/Ollama enforce via GBNF — available to us
  because, with in-process Candle, **we own the sampling loop**.
- **Device-aware.** Candle selects GPU when a `rocm`/`metal`/`cuda` device is
  available, else CPU. GPU-less hosts still run Candle (slower). The GPU path is
  ADR 068.

## Consequences

- Candle is no longer just the NLU verbalizer (ADR 067); it can drive the whole
  agentic turn with tools. The NLU verbalize/dialogue path (ADR 067) is
  unchanged and still candle-first with an `nlu-llm` fallback.
- A Candle-backed akh currently loads its **own** GGUF instance for the chat
  backing (separate from the NLU pipeline's copy). Sharing the one in-process
  model across NLU + chat backing is a follow-up (memory optimisation), tracked
  toward Phase 39 / ADR 037.
- Tool-calling fidelity: output is *always* valid (grammar-constrained); the
  residual variable is small-model *judgment* (which tool/args), not syntax.
  Ollama (or a cloud model) remains the higher-judgment option for hard turns.
- This trait is deliberately shaped to forward-port into Phase 39's richer
  `InferenceBackend` (reservation, cost, remote) — it is the precursor, not the
  router.

## Alternatives considered

- **Raw `git merge` of the candle-only branch.** Rejected: it would re-delete
  Ollama, ACP, and Phase 38. We forward-ported the candle *additions* instead
  (the commit that precedes this ADR).
- **Candle = NLU only, never the chat backing.** Rejected: it contradicts the
  user's requirement and Phase 44/ADR 037, which both place Candle in the loop.
- **Best-effort (parse-and-pray) Candle tool calls.** Rejected: the user
  required *full* tool calling. Grammar-constrained decoding makes it reliable.
- **Full token-level GBNF engine now.** Deferred: the JSON-object constraint
  (the part that guarantees valid tool calls) is implemented and tested; a
  general GBNF compiler is Phase-39-scope if richer grammars are needed.

## Verification

- `JsonConstraint` unit tests (valid/invalid JSON prefixes, completion, escapes).
- `chat_backend` unit tests (tool-call block parsing, plain answers, malformed
  blocks skipped for constrained repair).
- `AkhConfig` test: `backend = "candle"` parses; absent → Ollama (back-compat);
  round-trips through serialization.
- Build: default (`nlu-candle,acp`) + akhomed bin (`server,nlu-candle`) compile;
  full multi-round Candle tool-calling against a real GGUF is verified on a host
  with a model present (and on the RX 7900 XTX with the `rocm` feature, ADR 068).
