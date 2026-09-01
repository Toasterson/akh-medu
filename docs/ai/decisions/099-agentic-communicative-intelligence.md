# ADR 099 — Communicative Intelligence in the Agentic Loop

> Date: 2026-09-01
> Status: Accepted
> Amends: ADR 044 (Communicative Intelligence)
> Depends on: Phase 34 (strategy selector, style transform), Phase 44 (context assembly runtime, ADR 065)
> Related: ADR 067/070 (candle grammar-constrained backing), ADR 064 (context runtime lessons)

## Context

ADR 044 wired communicative intelligence (response-strategy selection, psyche
style transform, content filter, shadow veto) into the grounded ChatProcessor
path only, and deliberately excluded the agentic LLM loop. That exclusion
rested on one premise: "trying to control response type via prompt engineering
with small LLMs is unreliable — the strategy should shape the input to the
verbalization pipeline, not hope the LLM follows instructions."

Three things have changed since March 2026:

1. **The models changed.** Akhs now run GLM/Kimi-class cloud models and the
   candle backing has full grammar-constrained tool calling (ADR 067/070).
   Structure can be *enforced* on the LLM side, not hoped for.
2. **The seam got built.** Phase 44 (ADR 065) routes every agentic turn
   through a swappable, budget-aware `ContextManager`
   (`src/agent/agentic_chat.rs` `run_agentic_turn`, assembly at the
   Phase 44a block). A strategy directive can be a first-class assembled
   context section instead of ad-hoc prompt splicing.
3. **The center of gravity moved.** The agentic loop is now the primary
   output surface (tracker work, delegation, tool-using chat). Default-on
   comm-intel on the grounded path shapes only a minority of real output —
   the psyche and the 18-strategy taxonomy have near-zero influence on what
   akhs actually say all day.

## Decision

Extend communicative intelligence to the agentic loop, **keeping strategy
selection symbolic and engine-side** (ADR 044's first rejection stands: the
LLM cannot see Dempster-Shafer belief intervals, Admiralty ratings, psyche
weights, or discourse trajectory, so it must not pick the strategy). Only the
**delivery mechanism** changes: the selected strategy is injected as context,
and the LLM plays the move the engine chose.

1. **StrategyContext from episode state.** Build the selector's
   `StrategyContext` from the agentic `EpisodeState` + conversation state
   (discourse trajectory tracking already exists in the conversation module).
   `StrategySelector::select` (`src/agent/response_strategy.rs`) runs
   unchanged, engine-side, per turn.
2. **Directive as an assembled context section.** The chosen strategy +
   psyche register render to a compact communication-directive block emitted
   through `ContextManager::assemble`, subject to the Phase 44
   `BudgetAllocator` and `Retention`/`SafetyEnforced` eviction rules. Under
   budget pressure the directive degrades before knowledge content is
   evicted (it is guidance, not truth).
3. **Shadow veto becomes post-hoc.** On the agentic path the shadow patterns
   inspect the drafted response and can soften/veto, recorded in the shadow
   journal as on the grounded path ("considered X, chose Y instead").
4. **Staged rollout behind a flag.** `AKH_COMM_INTEL_AGENTIC` gates the
   agentic-path injection, default **off** until measured; flip the default
   after a Phase 44j-style deterministic A/B run (`src/bench_phase44/`)
   shows no regression on task completion and a measurable strategy-adherence
   delta. `AKH_COMM_INTEL=0` continues to kill both paths.

## Rejected Alternatives

### LLM-based strategy selection
Still rejected, unchanged from ADR 044. Selection needs engine-side signals
the LLM cannot access; the scoring function stays simple, fast, deterministic,
and debuggable.

### Prompt splicing outside the ContextManager
Injecting the directive by string-editing the system prompt bypasses the
budget allocator, per-turn telemetry, and retention guarantees that Phase 44
exists to provide, and re-creates the ad-hoc prompt assembly ADR 065 removed.

### Immediate default-on
The grounded path shipped default-on because the grammar pipeline made
strategy adherence structural. On the agentic path adherence is model-
dependent; measure before flipping the default.

## Consequences

- `src/agent/response_strategy.rs`: `StrategyContext` builder from agentic
  episode/conversation state.
- Context runtime: new directive section type emitted during assembly; all
  three managers (compacting/passive/virtual) carry it or degrade it.
- `src/config.rs`: `AKH_COMM_INTEL_AGENTIC` flag.
- A/B evidence via the Phase 44j harness before default flip.
- ADR 044 status corrected to Accepted; its "agentic loop deliberately not
  hooked" consequence is superseded by this ADR. Update the CLAUDE.md
  Phase 34 roadmap line and `docs/ai/architecture.md` when the wiring lands.

## References

- ADR 044 — Communicative Intelligence (amended by this ADR)
- ADR 064/065 — Context Assembly Runtime + lessons
- ADR 067/070 — Candle conversational boundary, selectable chat backing
- Phase 44j A/B report: `docs/ai/2026-06-21-phase44j-strategy-ab.md`
- Seam: `src/agent/agentic_chat.rs` (`run_agentic_turn`, Phase 44a assembly block)
