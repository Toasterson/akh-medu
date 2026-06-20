# Phase 44 — measure-first results (gate cleared for 44b/44c/44e)

- **Date:** 2026-06-20
- **Hardware:** hephaistos (AMD RX 7900 XTX, gfx1100), live 3-akh setup
  (Tecton=glm-5.2, Thoth=gpt-oss:120b over ollama.com cloud; Gemma=gemma4-12B via
  Candle). See [`docs/ops/three-akhs-postgres.md`](ops/three-akhs-postgres.md).
- **Plan:** [`plans/2026-06-18-phase44-context-assembly-runtime.md`](plans/2026-06-18-phase44-context-assembly-runtime.md)
  (§Sequencing — "measure first"). ADR 064/065.

The Phase 44 plan gated the quality-driven sub-phases on three measurements,
each unlockable only with akhs running real models — which we now have. Results:

## Q1 — Does conversation history improve NL quality?  ✅ YES (gates 44c/44e)

Coreference A/B against the live cloud model (glm-5.2), identical follow-up:

| Condition | Prompt | Answer |
|---|---|---|
| **No history** | `What city is it in?` | *"I don't know what you're referring to — could you provide more context?"* |
| **With history** | `[I'm thinking of the Eiffel Tower. OK][What city is it in?]` | *"The Eiffel Tower is located in Paris, France."* |

History resolves the reference; without it the turn fails. **Crucially, the akh's
agentic path currently sends no history** — `run_agentic_turn` builds
`conversation = vec![{user_prompt}]` (one turn). So every akh follow-up today
behaves like the "no history" case. → **Build 44c (graduated-fidelity history) +
44e (discourse selection / session seeding).**

## Q2 — Real token window + what dominates it  ✅ (gates 44b)

**Correction (the windows are huge — read from the model, not guessed).** The
8192 cap was a leftover; the real windows, now read from the model file/API:

| Model | Window | Source |
|---|---|---|
| gemma4-12B (Candle) | **262144** | GGUF `gemma4.context_length` |
| glm-5.2 (Tecton, cloud) | **1000000** | `/api/show` `model_info` |
| gpt-oss:120b (Thoth, cloud) | **131072** | `/api/show` `model_info` |

Real prompt with the **full tool catalog** (candle-tokenized, user prompt `"hi"`):
**6605 tokens** — user 1, system a few dozen, **tool catalog ~6540**. Against the
real windows that is **~2.5% of gemma4's, ~0.65% of glm-5.2's** — *not* a
constraint. (My earlier "80% of the window" was wrong because of the 8192 cap.)

Findings:
- **Window pressure is not the problem** — every deployed model has a 128K–1M
  window; the tool catalog + history + KG all fit comfortably. So 44b's
  original framing ("budget to fit the window") is **low-urgency**.
- **For the Candle akh the real constraint is COMPUTE, not window** — gemma4-12B
  agentic turns are slow because of the partial GPU offload (attention/lm_head on
  CPU) × the prefill of a ~6.6K-token prompt. So trimming the tool catalog still
  matters, but as a **latency** lever, not overflow avoidance. Cloud akhs are
  fast and unconstrained.
- **Implemented from this finding:** model capabilities are now read from the
  model — `CandleBackend::context_length()` (GGUF) and `OllamaClient::capabilities()`
  (`/api/show`: context window + `thinking`/`tools`); the agentic `TokenBudget`
  window is sized per-akh from them (verified live: gemma 262144, glm-5.2
  1000000), and `reasoning_effort`/`think` is now gated on `supports_thinking`.
- **Architectural note for 44b:** the tool catalog is assembled in the *backend*
  (`ollama_tool_catalog` in `run_agentic_turn`), *downstream* of the
  `ContextManager` (44a telemetry sees only the 1-token user prompt). A future
  tool-budgeting step would route the catalog through the manager (the allocator
  already orders `Tools` last = trimmed first) — but motivated by Candle latency,
  not window overflow.
- → **Re-prioritise:** 44b (window budget) is low-urgency; the data-justified win
  is **44c/44e (conversation history)** per Q1. A Candle-latency tool-trim is a
  separate, optional lever.

## Q3 — KG-recall quality baseline  ◑ baseline established; precision deferred (gates 44g)

The plan assumed Phase 42b's LongMemEval/LoCoMo harness would pick the recall
backend (VSA vs pgvector) — **but Phase 42 is absent on this branch**. Recall is
pure-KG: `recall_episodes` / `recall_recent_episodes` (`src/agent/memory.rs`) =
reverse-triple symbol-overlap lookup + mood-congruent bias.

So the baseline 44g must beat is **symbol-overlap ranking**. A precision@k number
needs a labeled query set, which does not yet exist on this branch. Decision:
**defer the recall-precision measurement to the 44g build**, where a small
constructed query set (LongMemEval-lite) compares VSA-similarity-blended ranking
against the symbol-overlap baseline. This does not block 44b/44c/44e.

## Gate status + next build order

- **Cleared:** Q1 + Q2. The measurements **re-prioritise** the plan: windows are
  huge, so 44b (budget-to-fit-window) is **low-urgency**, and the clear win is
  **conversation history (44c/44e)** — currently absent, demonstrably helpful.
- **Done as a direct consequence:** model capabilities are read from the model
  (GGUF / `/api/show`), the budget window is sized per-akh, reasoning is gated.
- **Next:** 44c — inject recent conversation turns into the agentic prompt
  (start of the `CompactingManager`), then 44e (discourse selection). A
  Candle-latency tool-trim and 44b's full allocator are optional follow-ups.
- **Deferred:** Q3's recall precision → measured during the **44g** build; 44d
  (delta KG), 44h/44i (Passive/Virtual), 44j (harness); 44k/44l recorded-deferred.
