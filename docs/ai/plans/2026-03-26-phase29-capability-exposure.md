# Phase 29 — Capability Exposure & User-Facing Enhancement

> Date: 2026-03-26
> Status: Planned
> Phase: 29
> Depends on: Core engine (Phases 1-23 complete)
> ADR: [039-capability-exposure](../decisions/039-capability-exposure.md)

## Motivation

The engine has deep capabilities that are invisible to users. Microtheories,
skillpacks, psyche dynamics, argumentation, hypothesis forking, causal chains,
evidence fusion, and source trust all exist as internal subsystems but have no
user-facing controls. Exposing them transforms akh-medu from a black-box engine
into a transparent, controllable reasoning partner.

Each sub-phase is independent and gets its own plan document for detailed design.

## Sub-phases

### 29a — Microtheory Management

**Currently**: Compartments with CWA/circumscription (Phase 9a, 9m). Used
internally for per-repo code scoping, skill isolation, and domain separation.

**Expose**:
- MCP: `list_microtheories`, `create_microtheory`, `query_microtheory`,
  `compose_microtheories` (overlay multiple for combined reasoning)
- CLI: `akh microtheory list|create|query|compose`
- NLU: "In the context of X, what do you know about Y?"
- Dashboard: Microtheory browser with triple counts, CWA status, composition viewer

**Key value**: Users can create isolated knowledge domains ("my cooking knowledge",
"project X architecture") and compose them for cross-domain reasoning.

Plan: `2026-xx-xx-phase29a-microtheory-exposure.md` (to be created)

---

### 29b — Skillpack Lifecycle

**Currently**: `src/skills/` with Cold/Warm/Hot lifecycle. Skills are internal
capability bundles.

**Expose**:
- MCP: `list_skills`, `activate_skill`, `deactivate_skill`, `skill_status`,
  `create_skill_from_microtheory`
- CLI: `akh skill list|activate|deactivate|status|create`
- NLU: "Learn how to do X", "Forget about Y", "What skills do you have?"
- Dashboard: Skill catalog with activation state, usage stats, dependency graph

**Key value**: Users can teach the agent new capabilities and manage what it knows
how to do. Skills can be shared between workspaces.

Plan: `2026-xx-xx-phase29b-skillpack-exposure.md` (to be created)

---

### 29c — Psyche & Shadow Dynamics

**Currently**: Read-only via `agent_psyche` MCP tool. Psyche locked after
ritual of awakening. Shadow patterns active but invisible.

**Expose**:
- MCP: `psyche_profile`, `shadow_active`, `archetype_weights`,
  `adjust_trait` (within bounds), `shadow_journal` (log of shadow vetoes)
- CLI: `akh psyche profile|shadow|archetypes|journal`
- NLU: "Why did you hesitate?", "What's your shadow telling you?",
  "Be more exploratory" (adjusts archetype weights)
- Dashboard: Psyche radar chart (OCEAN + archetypes), shadow veto timeline,
  integration score progression

**Key value**: Users understand why the agent behaves the way it does. Shadow
vetoes become visible ("I almost did X but my shadow pattern prevented it").
Archetype tuning allows personality calibration within the identity established
at awakening.

Plan: `2026-xx-xx-phase29c-psyche-exposure.md` (to be created)

---

### 29d — Argumentation & Structured Debate

**Currently**: `src/argumentation/` — pro/con framework with meta-rules,
verdicts, evidence chains. Internal to decision-making.

**Expose**:
- MCP: `argue_for`, `argue_against`, `debate_topic`, `argumentation_verdict`
- CLI: `akh argue "claim" [--for|--against|--balanced]`
- NLU: "Argue for X", "What are the arguments against Y?",
  "Give me both sides of Z"
- Dashboard: Argument tree visualization (pro/con branches, evidence links,
  strength indicators)

**Key value**: Users can request structured reasoning about any claim. The agent
marshals evidence from the KG, constructs pro/con arguments, and renders a
verdict with confidence. This is one of the most uniquely powerful capabilities
— no LLM-based system can do this with verifiable evidence chains.

Plan: `2026-xx-xx-phase29d-argumentation-exposure.md` (to be created)

---

### 29e — Transparent Reasoning Traces

**Currently**: `src/agent/explain.rs` — provenance-to-prose pipeline with
DerivationNode trees and 5 query types. Internal.

**Expose**:
- MCP: `explain_belief`, `explain_decision`, `reasoning_trace`,
  `provenance_chain` (already partially via `provenance_of`)
- CLI: `akh explain "why do you think X?"`
- NLU: "Why do you think X?", "How did you conclude Y?",
  "Show me the reasoning for Z"
- Dashboard: Interactive derivation tree (expandable nodes, source links,
  confidence at each step)

**Key value**: Full interpretability. Users can trace any belief back to its
source evidence through the provenance chain. This is the definitive advantage
over LLM-based systems.

Plan: `2026-xx-xx-phase29e-transparent-reasoning-exposure.md` (to be created)

---

### 29f — Hypothesis Explorer

**Currently**: `src/infer/superposition.rs` — hypothesis forking/merging with
constructive/destructive interference. Internal to inference.

**Expose**:
- MCP: `explore_hypotheses`, `fork_hypothesis`, `merge_hypotheses`,
  `hypothesis_confidence`
- CLI: `akh hypothesize "topic" [--max-depth N]`
- NLU: "What are the possible explanations for X?",
  "Explore alternatives for Y"
- Dashboard: Hypothesis tree with confidence bars, interference patterns,
  surviving vs. collapsed branches

**Key value**: Users can see the agent's internal deliberation — all the
hypotheses it considered, which reinforced each other, which cancelled out,
and why the dominant one won.

Plan: `2026-xx-xx-phase29f-hypothesis-exposure.md` (to be created)

---

### 29g — Causal Explorer

**Currently**: Event calculus (Phase 15b), causal world model (15a),
counterfactuals (15c). MCP has `holds_at`, `project_state`, `simulate_actions`,
`counterfactual` — but these are low-level primitives.

**Expose**:
- MCP: `causal_chain` (high-level: "what caused X?"),
  `what_if` (simplified counterfactual), `timeline` (event sequence vis)
- CLI: `akh causal "event" [--chain|--what-if|--timeline]`
- NLU: "What caused X?", "What would have happened if Y?",
  "Show me the timeline of Z"
- Dashboard: Causal graph visualization (events → fluents, initiates/terminates
  edges), counterfactual comparison view (actual vs. hypothetical timelines)

**Key value**: Users can explore causality interactively. "Why did this happen?"
traces back through initiating events. "What if?" shows alternative timelines.

Plan: `2026-xx-xx-phase29g-causal-exposure.md` (to be created)

---

### 29h — Evidence & Belief Dashboard

**Currently**: Dempster-Shafer (Phase 17), MCP has `add_evidence` and
`assess_evidence`. But users can't see the evidence landscape.

**Expose**:
- MCP: `belief_summary` (all assessed claims with intervals),
  `evidence_timeline` (evidence arrival over time for a claim),
  `conflict_report` (high-conflict claims needing attention)
- CLI: `akh evidence list|timeline|conflicts`
- NLU: "How confident are you about X?", "What evidence supports Y?",
  "Where are you most uncertain?"
- Dashboard: Belief interval bars ([belief, plausibility] ranges),
  evidence arrival timeline, conflict heatmap

**Key value**: Users see not just what the agent believes, but how confident it
is, what evidence supports/contradicts each belief, and where the greatest
uncertainty lies.

Plan: `2026-xx-xx-phase29h-evidence-exposure.md` (to be created)

---

### 29i — Source Trust Dashboard

**Currently**: Admiralty + Bayesian trust (Phase 18), MCP has `rate_source` and
`verify_source_claim`. Source trust is tracked but not surfaced.

**Expose**:
- MCP: `source_trust_report` (all sources with reliability scores),
  `trust_history` (how a source's trust evolved over time),
  `source_comparison` (rank sources by domain)
- CLI: `akh trust list|history|compare`
- NLU: "How reliable is source X?", "Which sources do you trust most for Y?",
  "Has source Z been accurate?"
- Dashboard: Source leaderboard, trust evolution charts, per-domain reliability
  matrix, verification history

**Key value**: Users can audit where the agent's knowledge comes from and how
much it trusts each source. Critical for high-stakes domains.

Plan: `2026-xx-xx-phase29i-source-trust-exposure.md` (to be created)

## Estimated Effort

| Sub-phase | Scope | Est. Lines | Priority |
|---|---|---|---|
| 29a — Microtheories | MCP + CLI + NLU + dashboard | ~600 | High |
| 29b — Skillpacks | MCP + CLI + NLU + dashboard | ~400 | Medium |
| 29c — Psyche & Shadow | MCP + CLI + NLU + dashboard | ~500 | Medium |
| 29d — Argumentation | MCP + CLI + NLU + dashboard | ~600 | High |
| 29e — Transparent Reasoning | MCP + CLI + NLU + dashboard | ~500 | High |
| 29f — Hypothesis Explorer | MCP + CLI + NLU + dashboard | ~500 | Medium |
| 29g — Causal Explorer | MCP + CLI + NLU + dashboard | ~500 | Medium |
| 29h — Evidence & Belief | MCP + CLI + NLU + dashboard | ~400 | High |
| 29i — Source Trust | MCP + CLI + NLU + dashboard | ~400 | Medium |
| **Total** | | **~4,400** | |

## Implementation Pattern

Each sub-phase follows the same structure:

1. **MCP tools** (2-5 new tools in `src/mcp/mod.rs`)
2. **CLI subcommands** (1-3 new subcommands in `src/main.rs`)
3. **NLU intents** (extend grammar parser with new intent patterns)
4. **Dashboard panel** (new htmx partial in `data/web/`)
5. **Documentation** (update architecture.md, add to MCP tool list)

The MCP tools are the primary interface — CLI and NLU delegate to the same
underlying engine methods. Dashboard panels fetch data via the existing
HTTP/MCP endpoints.

## Priority Order

Recommended implementation sequence (highest user value first):

1. **29e — Transparent Reasoning** (most requested: "why?")
2. **29d — Argumentation** (unique differentiator)
3. **29a — Microtheories** (knowledge organization)
4. **29h — Evidence & Belief** (confidence visibility)
5. **29g — Causal Explorer** (temporal understanding)
6. **29c — Psyche & Shadow** (personality transparency)
7. **29f — Hypothesis Explorer** (deliberation visibility)
8. **29i — Source Trust** (provenance trust)
9. **29b — Skillpacks** (capability management)
