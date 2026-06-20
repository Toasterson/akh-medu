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

Measured on the live Gemma akh (full tool catalog, user prompt `"hi"`):

| Quantity | Value |
|---|---|
| Model window (gemma4-12B, our cap) | **8192** tokens |
| Reserved for output | 1024 → **7168** available |
| Real prompt sent to the model (candle-tokenized) | **6605** tokens |
| ↳ user prompt | 1 token |
| ↳ system prompt (psyche/explicit) | ~few dozen |
| ↳ **tool catalog** | **~6540 tokens (~80% of the window)** |
| 44a ContextManager-assembled context (pass-through) | **1 token** |

Findings:
- **The tool catalog is the dominant context cost by far** — it alone nearly
  fills a small-context model's window, leaving ~no room for history/KG/recall.
  This is why the Candle Gemma akh needs `AKH_AGENT_MAX_TOOLS=0` to be usable; the
  cloud akhs (glm-5.2/gpt-oss, large windows) absorb the full catalog fine.
- **Architectural note for 44b:** the tool catalog is assembled in the *backend*
  (`ollama_tool_catalog` in `run_agentic_turn`, then rendered by each
  `ChatBackend`), *downstream* of the `ContextManager` — which is why 44a's
  telemetry sees only the 1-token user prompt. The budget allocator (44b) must
  route the tool catalog *through* the manager (as a `Tools` section) to budget
  it. The allocator's default priority already places `Tools` last (degraded
  first), which is the correct order.
- → **Build 44b**: one allocator over `{system, KG, history, recalled, tools}`
  sized to the measured window; the first, highest-value lever is **trimming /
  budgeting the tool catalog** (a measured, configurable cap rather than the
  blunt `AKH_AGENT_MAX_TOOLS=0`).

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

- **Cleared:** Q1 + Q2 → **44b → 44c → 44e** are unblocked (the core
  `CompactingManager`: budget allocator + graduated-fidelity history + discourse).
- **Deferred:** Q3's precision number → measured during the **44g** build.
- 44d (delta KG), 44h (Passive), 44i (Virtual), 44j (A/B harness) follow per the
  plan's dependency order; 44k/44l remain recorded-deferred.
