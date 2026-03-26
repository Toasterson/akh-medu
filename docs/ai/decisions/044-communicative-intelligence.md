# ADR 044 — Communicative Intelligence

> Date: 2026-03-26
> Status: Proposed
> Phase: 34
> Depends on: Phase 33 (verbalization tiers), Phase 23 (affective system/psyche)
> Enhances: Phase 29c (psyche exposure), Phase 12b (grounded dialogue)

## Context

The engine currently has one response mode: knowledge dump. Every query returns
linearized triples. There is no mechanism for the agent to CHOOSE how to
communicate — it cannot decide to explain, challenge, ask for clarification,
express uncertainty, or use an analogy. It also has a rich psyche (OCEAN traits,
Jungian archetypes, shadow patterns) that has zero influence on output.

Eliza's insight from 1966 remains relevant: the CATEGORY of response matters
more than the specific words. Eliza had ~100 patterns organized by response
category (reflection, clarification, redirection). The genius was in choosing
WHICH category, not in generating text.

Akh-medu can do this with far more depth because:
- It actually understands the query (NLU pipeline)
- It has real knowledge to draw from (KG)
- It has personality (psyche)
- It has confidence estimates (Dempster-Shafer)
- It has conversation history (dialogue manager)

## Decision

### 1. Separate "what type of response" from "what content" and "how to say it"

Three distinct stages:

```
Stage 1: WHAT TYPE?   → Response Strategy Selector (symbolic)
Stage 2: WHAT CONTENT? → Content Filter (KG query + selection)
Stage 3: HOW TO SAY?   → Psyche Style Transform + Verbalization (Phase 33)
```

### 2. Response Strategy Taxonomy

**Knowledge strategies** (when you have content):
- `DirectAnswer` — state facts concisely
- `Explanation` — provide reasoning chain, context, "why"
- `Contextualization` — place answer in broader framework
- `Comparison` — relate to something user already knows
- `Analogy` — express through metaphor from a different domain

**Epistemic strategies** (when confidence matters):
- `ConfidentAssertion` — high belief, multiple sources
- `HedgedStatement` — moderate belief, limited evidence
- `UncertaintyAdmission` — low confidence, honest about limits
- `EvidenceWeighing` — conflicting sources, present both sides
- `KnowledgeGap` — no triples found, optionally offer to learn

**Conversational strategies** (when interaction matters):
- `ClarificationRequest` — ambiguous query, need more info
- `ExplorationPrompt` — related interesting topic available
- `Challenge` — counter-evidence exists in KG
- `Redirect` — can't answer X but know about adjacent Y
- `Empathy` — recognized frustration/confusion in user

**Meta-cognitive strategies** (self-awareness):
- `SelfReflection` — discovered a new connection during query
- `ConfidenceCalibration` — explicit about certainty level
- `ReasoningTrace` — show derivation chain (provenance-to-prose)

### 3. Strategy selection is symbolic, not neural

The selector is a scoring function, not a classifier:

```
score(strategy, context) = Σ(signal_i × psyche_weight_i × affinity_i)
```

**Signals evaluated**:
- Query type (factual/procedural/causal/existential/meta)
- Answer quality (confidence, triple count, provenance depth)
- Evidence state (Dempster-Shafer belief interval, conflict level)
- Source reliability (Admiralty ratings of supporting sources)
- Conversation history (repetition, discourse trajectory, user mood)
- Superposition state (competing hypotheses?)

**Psyche modulation**:
- Archetype weights shift strategy preferences
  (sage → explanation, explorer → exploration, healer → empathy)
- OCEAN shifts confidence thresholds
  (high N → more hedging, low A → more challenging)

### 4. Psyche Style Transform shapes surface form

Once strategy and content are selected, the psyche shapes HOW it's expressed:

**OCEAN → communication register**:
- Openness: metaphor use, tangential connections, creative phrasing
- Conscientiousness: structure, explicitness, caveats
- Extraversion: length, enthusiasm, engagement
- Agreeableness: affirmation, hedging, supportiveness
- Neuroticism: caution, uncertainty markers, confidence

**Archetype → framing patterns**:
- Each archetype has opening moves, characteristic phrases, framing devices
- These become template variants in the grammar framework
- And style prefixes for T5/mT5 NLG prompts

**Shadow → communication inhibition**:
- Shadow patterns can veto or soften specific response strategies
- Shadow journal records "considered X, chose Y instead"

## Rejected Alternatives

### LLM-based strategy selection
Strategy selection requires evaluating KG confidence, evidence state, psyche
weights, and conversation history. An LLM can't access these. The scoring
function is simple, fast, deterministic, and debuggable.

### Fixed strategy per query type
Too rigid. The same factual query should get a different response depending on
confidence level, psyche state, and conversation history.

### Per-strategy LLM prompt engineering
Trying to control response type via prompt engineering with small LLMs is
unreliable — the same problem as enrichment. The strategy should shape the
input to the verbalization pipeline, not hope the LLM follows instructions.

## Consequences

- New module: `src/agent/response_strategy.rs` — strategy taxonomy + selector
- New module: `src/grammar/style_transform.rs` — psyche → surface form mapping
- Extended: grammar archetypes gain per-archetype template variants
- Extended: T5 NLG prompts gain style prefixes from psyche
- Extended: dialogue manager tracks response strategy per turn
- Extended: conversation module gains discourse trajectory analysis
- Phase 29c (psyche exposure) shows strategy selection reasoning

## References

- Weizenbaum, J. (1966). ELIZA — A Computer Program for the Study of Natural
  Language Communication Between Man and Machine
- Phase 23: Affective system (psyche, archetypes, shadow)
- Phase 33: Grounded verbalization (tiered NLG pipeline)
- Phase 12b: Grounded dialogue (conversation state)
- Phase 17: Dempster-Shafer evidence theory (belief intervals)
- Phase 18: Source reliability (Admiralty ratings)
