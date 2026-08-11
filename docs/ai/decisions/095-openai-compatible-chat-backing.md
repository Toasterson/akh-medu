# ADR 095 — OpenAI-compatible chat backing, and per-akh context strategy

- **Status**: Accepted
- **Date**: 2026-08-11
- **Extends**: [ADR 070](070-selectable-chat-backing.md) (selectable chat backing),
  [ADR 065](065-context-assembly-runtime.md) (context-assembly runtime)
- **Anima work item**: _to be filed_ — project `019d9844-ce74-7993-a9f0-c953b8f04a4e` (Akh-Medu)

## Context

ADR 070 gave an akh two choices of chat/agentic backing: **Ollama** (local or
`ollama.com` cloud, native function calling) and **Candle** (in-process GGUF,
grammar-constrained tool calling). Both are full tool-calling peers behind the
`ChatBackend` trait.

Neither covers the largest category of inference hosts actually available to us.
vLLM, SGLang, llama.cpp's server, OpenRouter, DeepSeek's own API, and Hetzner's
free inference experiment all speak the **OpenAI `/chat/completions` contract**,
not Ollama's `/api/chat`. The immediate driver: Hetzner offers
`DeepSeek-V4-Flash-0731` free at `https://inference.hetzner.com/api/v1`, and we
want it as a third implementer lane on the Mac mini next to `tecton` (GLM) and
`imhotep` (Kimi K2.7 Code).

Separately, we want that lane to run the **Phase 44i MemGPT `VirtualContextManager`**
while the other two stay on `compacting` — a live A/B on the same task queue. The
daemon read `AKH_CONTEXT_STRATEGY` once per process, so all akhs in one daemon
necessarily shared a strategy. There was no way to have a control group.

## Decision

### 1. A third `ChatBackend` peer: `backend = "openai"`

`src/agent/openai_backend.rs` implements `ChatBackend` against the OpenAI
contract — non-streaming and SSE-streaming, with tool calling. Configured by a
new `[openai_endpoint]` table:

```toml
backend = "openai"
model = "DeepSeek-V4-Flash-0731"

[openai_endpoint]
base_url = "https://inference.hetzner.com/api/v1"
api_key = "env:HETZNER_INFERENCE_KEY"
context_window = 512000
```

`base_url` includes the version segment; the client appends `/chat/completions`
and `/models`. `api_key` supports the same `env:VAR` indirection as
`[ollama_endpoint]`, resolved at call time, and never resolves to the literal
`"env:VAR"` string — an unresolved reference yields no header rather than a
bogus token.

### 2. Message translation lives in the backend, not the loop

`agentic_chat` builds its conversation in **Ollama** shape, because that was the
only wire format when it was written: assistant `tool_calls` have no `id` and
carry `arguments` as a JSON *object*, and `role:"tool"` messages have no
`tool_call_id`. A strict OpenAI server rejects all three.

We translate at the boundary (`to_openai_messages`) rather than reshaping the
shared loop. Assistant tool calls gain deterministic ids (`call_{round}_{idx}`)
and stringified arguments; tool results are paired to those ids positionally, in
the order the loop appends them. The inverse runs on responses, parsing
OpenAI's stringified `arguments` back into objects.

**Why deterministic ids rather than UUIDs**: a replayed conversation must produce
byte-identical requests, or the A/B harness this backing exists to serve is not
comparing like with like.

### 3. `context_window` is configuration, not discovery

Ollama exposes the context window via `/api/show`; the OpenAI contract exposes
no equivalent. Rather than guess from the model name, the window is stated in
`[openai_endpoint].context_window` and feeds the Phase 44 budget directly.
`None` leaves the budget on its own default. This is a knob an operator must
set correctly — too high over-fills the prompt, too low wastes the window.

Worth checking per provider rather than assuming: vLLM-family servers report a
non-standard `max_model_len` on `GET /models`, which many OpenAI-compatible
hosts omit. Hetzner does expose it, and it caught a real error — this ADR
originally shipped `deepseek` with `context_window = 131072`, carried over from
the DeepSeek V3/R1 family. The endpoint reports **512000** for
`DeepSeek-V4-Flash-0731`, so the first deployment was quietly using a quarter of
the available window. Nothing failed; it simply under-filled. That is exactly
the failure mode this field's "configuration, not discovery" design invites, and
the argument for probing `/models` when standing up any new OpenAI-backed akh.

### 4. Misconfiguration is fatal for the akh, not a fallback

If an akh sets `backend = "openai"` without an `[openai_endpoint]` or without a
pinned `model`, the daemon **refuses to start that workspace**
(`AkhConfigError::MissingOpenAiEndpoint`).

This deliberately differs from the Candle branch, which falls back to Ollama
when no GGUF is present. Candle has a same-akh endpoint to fall back *to*; an
OpenAI-backed akh does not — it would silently answer from whatever the default
Ollama model is. "Which model produced this answer?" must stay answerable, and
a silent substitution invalidates every measurement taken against the akh
without failing anything.

### 5. Per-akh `context_strategy`, with the env var demoted to a default

New optional top-level scalar; precedence is
**`context_strategy` → `AKH_CONTEXT_STRATEGY` → `compacting`**.

One daemon process hosts several akhs, so a process-global env var could only
ever move all of them together. Comparing strategies requires a control group in
the same process, on the same queue, against the same tools. An unrecognized
value now logs a warning before falling back, because a typo that silently
downgrades the strategy corrupts a comparison rather than failing it.

Note this field is also the ADR 081 slice-7 **tier gate**: only akhs on
`virtual` receive the memory self-edit tools (`core_memory_append`/`replace`,
`archival_search`). Setting it is a claim about the model's capability.

## Consequences

**Good**

- Any OpenAI-compatible host is now a first-class akh backing — the largest
  category of inference endpoints, including every free/cheap one we have access
  to. Phase 50's public benchmark legs (deepseek / GLM-5.2 / qwen) get a native
  path instead of needing an Ollama shim.
- The MemGPT-vs-compacting comparison becomes a *live* experiment on real work,
  not a replay harness (`src/bench_phase44/`).
- The `ChatBackend` seam held: no change to `agentic_chat`, and the Ollama and
  Candle paths are untouched.

**Costs / risks**

- `context_window` can silently drift from reality if the provider changes the
  model behind a name. It is stated in one place per akh, so correcting it is
  cheap, but nothing detects the drift.
- Positional pairing of tool results to call ids assumes the loop appends tool
  results in call order. That is true today and covered by tests; a future loop
  that interleaves or reorders results would break it silently. The tests
  (`tool_results_are_paired_with_their_call_ids`, `ids_stay_unique_across_rounds`)
  are the tripwire.
- `available()` probes `/models` once per process and caches. A mid-session
  provider outage surfaces as a failed turn rather than an unavailable backing —
  acceptable, since the agentic loop already falls back on turn failure, and the
  alternative costs every turn a round-trip.
- Free/experimental endpoints have unstated rate limits. Nothing here implements
  backoff; a 429 surfaces as `OpenAiError::RequestFailed`. If the Hetzner lane
  hits limits under real load, retry/backoff is the follow-up.

## Alternatives considered

- **A third `kind` on `[ollama_endpoint]`** — rejected. The two speak different
  wire protocols; a table named `ollama_endpoint` configuring a DeepSeek
  endpoint would mislead every future reader.
- **Reshape `agentic_chat` to emit OpenAI-shaped messages natively** — rejected
  for now. It would force ids onto the Ollama path that it does not want, and
  churn the one loop all three backings share. Worth revisiting if OpenAI shape
  becomes the majority case.
- **Reuse `model_manager`'s existing `ExternalApiConfig` fallback path** — it
  already calls `/chat/completions`, but only for single-shot generation with no
  tool calling, and it is a *fallback* for a degraded Ollama rather than a
  selectable backing. Wrong shape for an agentic peer.
- **Per-akh strategy via a second env var** (`AKH_CONTEXT_STRATEGY_<akh>`) —
  rejected; config belongs in the akh's config file, which travels with the akh
  when it relocates (ADR 049).
