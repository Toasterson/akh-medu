# ADR 075 — Frontier-style agentic round budget (adaptive tool-loop termination)

> Date: 2026-06-23
> Status: Accepted (implemented)
> Related: `src/agent/agentic_chat.rs` (`RoundBudget`, `Agent::run_agentic_turn`),
>          ADR 070 (selectable chat backing), ADR 064/065 (context-assembly runtime),
>          memory: `agent-round-cap-memory-retrieval`

## Context

The agentic turn loop in `run_agentic_turn` capped tool-call rounds at a flat
`DEFAULT_MAX_ROUNDS = 8`. Measured 2026-06-23 against Tecton's imported episodic
memory (8 grounded probes, driven live over ACP): **0/8** correct — `memory_recall`
returned episode *summaries*, the specific fact lived in the KG triples, and the
recall→drill→synthesize loop ran out of rounds before answering ("stopped after 8
tool-call rounds without a final answer"). Raising the cap to 20 lifted it to
**6/8**. So the flat cap was wrong in both directions: too low strangled legitimate
deep work, while doing nothing to catch a genuine spin (it would also have let a
model uselessly repeat the same call up to the cap).

A quick stopgap (`AKH_AGENT_MAX_ROUNDS` env override) proved the point but isn't a
principled fix.

## What frontier harnesses do (research)

- **OpenHands / SWE-agent** set the iteration cap *generously* (~100 per instance)
  as a pure backstop, relying on other limits to terminate normally.
- Production agent-loop guidance: **defense in depth** — never one terminator.
  Overlap (a) an iteration cap (typ. 15–25 in production), (b) **repeated-tool-call
  detection** ("same tool + same args 3× → terminate"), (c) a **token/cost or
  wall-clock budget** (~300s), and a **forced final answer** on exhaustion.

Sources: OpenHands eval harness docs; SWE-agent (≤100 LLM calls/instance);
production loop-termination write-ups (iteration + repeat-signature + time/cost).

## Decision

Replace the flat constant with a `RoundBudget` that combines overlapping limits:

1. **Model-scaled iteration backstop.** Derived from the backend's real context
   window (already known at assembly time): GLM-class (~1M) / gemma4-12b (262k) →
   30; ≥32k → 20; small/unknown (constrained Candle) → 12. `AKH_AGENT_MAX_ROUNDS`
   overrides absolutely (keeps the stopgap working).
2. **Wall-clock budget.** Default 300s; `AKH_AGENT_TURN_TIMEOUT_SECS`.
3. **Repeat-loop detection.** Hash each `(tool, arguments)` signature; the same
   call issued `AKH_AGENT_REPEAT_LIMIT` (default 3) times ends the turn. This is
   what makes a *high* backstop safe — productive drill-downs run to the cap,
   genuine spins die at round 3.
4. **Forced final synthesis.** On exhaustion, append a "stop calling tools, answer
   now with what you have" nudge and make **one tool-free** model call, emitting
   its answer — instead of the old bare `(stopped after N rounds)` cutoff. Frontier
   "reflect and answer" behaviour: a hit budget becomes a best-effort answer, not a
   dead end.

## Why not

- **Just raise the constant.** Fixes the strangling but not spinning, and a single
  number can't fit both a 1M-context cloud model and a 4k-context in-process one.
- **Token/cost budget instead of wall-clock.** Per-turn token accounting isn't
  plumbed through the backend trait yet; wall-clock is a good, cheap proxy and the
  repeat detector already curbs the dominant waste (re-issued identical calls). A
  token budget is a clean future addition.

## Consequences

- Deep retrieval turns get the rounds they need (scaled to the model); spins are
  cut at the 3rd repeat; every turn ends with an actual answer.
- New env knobs: `AKH_AGENT_MAX_ROUNDS` (now an absolute override of the scaled
  backstop), `AKH_AGENT_TURN_TIMEOUT_SECS`, `AKH_AGENT_REPEAT_LIMIT`.
- Unit-tested: cap scaling by window, iteration-cap stop, repeat-loop stop-before-cap,
  distinct-calls-don't-trip. Deploy needs a release rebuild + daemon restart.
- Untouched (separate gap): `memory_recall` returns episode summaries, not the
  granular facts — a recall-quality follow-up worth doing so a single round suffices.
