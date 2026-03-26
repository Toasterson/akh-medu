# ADR 039 — Capability Exposure & User-Facing Enhancement

> Date: 2026-03-26
> Status: Proposed
> Phase: 29
> Depends on: Core engine (Phases 1-23 complete)
> Enhances: All existing subsystems

## Context

Akh-medu has accumulated powerful subsystems across 23+ phases of development:
microtheories, skillpacks, psyche/shadow patterns, argumentation, transparent
reasoning, superposition reasoning, causal reasoning, evidence theory, source
reliability, and more. However, most of these are **internal implementation
details** — they power the engine but are invisible to and uncontrollable by the
user.

The MCP server exposes 51 tools, but these are primarily focused on knowledge
manipulation (triples, provenance, search) and agent introspection (goals,
recall, psyche read-only). The deeper reasoning capabilities have no user-facing
controls.

A user cannot currently:
- Create, browse, or manage microtheories
- Inspect or activate skillpacks
- See or configure psyche/shadow dynamics
- Request argumentation for a claim
- Ask for transparent reasoning explanations
- Trigger superposition reasoning on a topic
- Explore causal chains visually
- Submit evidence and see belief updates live
- View source reliability assessments

This is like having a powerful car with most controls hidden under the hood.

## Decision

### Expose each capability group as user-facing operations

Each capability group gets:
1. **MCP tools** — for programmatic access (Claude Code, external integrators)
2. **CLI subcommands** — for terminal users
3. **Web dashboard panels** — for visual exploration
4. **NLU intents** — so users can invoke via natural language ("argue for X",
   "explain why you think Y", "what's your confidence in Z?")

### Phase 29 sub-phases

Each sub-phase is independently implementable and gets its own plan document
for detailed research and design.

| Sub-phase | Capability Group | Currently | Exposure Goal |
|---|---|---|---|
| 29a | Microtheories | Internal compartment system, CWA | Create/browse/query/compose microtheories |
| 29b | Skillpacks | Cold/Warm/Hot lifecycle, internal | Browse/activate/deactivate/create skills |
| 29c | Psyche & Shadow | Read-only via `agent_psyche` MCP | Configure traits, view shadow dynamics, archetype tuning |
| 29d | Argumentation & Debate | Internal pro/con framework | "Argue for/against X", structured debate |
| 29e | Transparent Reasoning | `explain` module (provenance-to-prose) | "Why do you think X?", reasoning trace visualization |
| 29f | Hypothesis Explorer | Superposition reasoning (internal) | "What hypotheses exist for X?", fork/merge visualization |
| 29g | Causal Explorer | Event calculus + counterfactuals | "What caused X?", "What if Y instead?", causal chain vis |
| 29h | Evidence & Belief | Dempster-Shafer (MCP: assess_evidence) | Live belief updates, evidence timeline, conflict alerts |
| 29i | Source Trust | Admiralty + Bayesian (MCP: rate_source) | Source credibility dashboard, trust history |

## Rejected Alternatives

### Expose everything via a single "advanced mode" toggle
Too coarse. Users need granular control. A researcher wants argumentation but
not psyche tuning. A casual user wants explanations but not hypothesis forking.

### Only expose via MCP (no CLI/NLU/dashboard)
MCP is for programmatic access. Users interacting via TUI, chat channels, or
the web dashboard need native integration points.

## Consequences

- ~9 sub-phases, each independently implementable (~300-600 lines each)
- Each sub-phase adds 2-5 MCP tools + 1-3 CLI subcommands + NLU intents
- Web dashboard gets new panels per sub-phase
- Grammar framework extended with new dialogue act types per capability
- Total estimated: ~4,000-5,000 lines across all sub-phases

## References

- Phase 9: Cyc-inspired HOL (microtheories, argumentation, CWA)
- Phase 12f: Transparent reasoning (explain module)
- Phase 15: Causal world model, event calculus, counterfactuals
- Phase 17: Dempster-Shafer evidence theory
- Phase 18: Source reliability, Admiralty rating
- Phase 23: Affective system (psyche, shadow)
