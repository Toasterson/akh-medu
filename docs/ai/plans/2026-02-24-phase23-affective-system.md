# Phase 23 — Affective System

> Date: 2026-02-24

- **Status**: Planned
- **Phase**: 23 (Lifeform Engine — Emotional Valence)
- **Depends on**: Phase 20 (Active Inference OODA Enhancement)
- **Enhances**: Agent OODA loop, decision-making, memory consolidation, goal priority

## Goal

Implement an internal affective (emotional) system that modulates agent behavior —
not sentiment analysis of external text, but genuine internal states that influence
decision-making, memory salience, and reasoning priority. Emotions as computational
shortcuts: "this feels wrong" as a fast-path heuristic before slow symbolic reasoning.

Inspired by Damasio's somatic marker hypothesis: emotions are not opposed to rational
decision-making; they are essential to it. Without affective valence, the agent has no
basis for preferring one equally-justified option over another.

## Architecture Overview

```
Stimuli (KG events, goal outcomes, contradictions, user interactions)
     │
     ▼
┌──────────────────────────────────────────┐
│ Affective Appraisal                      │
│ Event → relevance × novelty × valence    │
│ Maps to dimensional emotion space        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Mood State (slow-moving baseline)        │
│ EMA of recent appraisals                 │
│ Modulates: exploration rate, risk        │
│ tolerance, communication tone            │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Somatic Markers                          │
│ Learned valence associations on concepts │
│ VSA-encoded: bind(concept, valence_role) │
│ Fast-path: skip deliberation for         │
│ strongly-marked options                  │
└──────────────────────────────────────────┘
```

## Sub-phases

### 23a — Affective Appraisal & Dimensional Emotion (~800 lines)

Dimensional emotion model (Russell's circumplex): valence (pleasant–unpleasant) ×
arousal (activated–deactivated). More tractable than discrete emotions for a symbolic
system.

**Key types**:
```rust
struct AffectiveState {
    valence: f32,      // -1.0 (unpleasant) to 1.0 (pleasant)
    arousal: f32,      // 0.0 (calm) to 1.0 (activated)
    dominance: f32,    // 0.0 (submissive) to 1.0 (dominant)
}

struct Appraisal {
    trigger: SymbolId,
    relevance: f32,    // How relevant to current goals
    novelty: f32,      // How unexpected
    valence: f32,      // Positive or negative outcome
    coping: f32,       // Can the agent handle this?
}

struct MoodState {
    baseline: AffectiveState,
    recent_appraisals: VecDeque<(Appraisal, u64)>,
    ema_valence: f32,
    ema_arousal: f32,
}
```

**Appraisal triggers**:
- Goal completed → positive valence, reduced arousal
- Goal failed → negative valence, increased arousal
- Contradiction detected → negative valence, high arousal
- Novel concept discovered → positive valence (curiosity), moderate arousal
- User praise/correction → positive/negative valence
- Resource pressure → negative valence, high arousal

**Mood effects on behavior**:
- Positive mood → higher exploration rate, more creative analogies, broader search
- Negative mood → more conservative, deeper verification, narrower focus
- High arousal → faster decisions, less deliberation
- Low arousal → more reflection, longer consolidation cycles

### 23b — Somatic Markers & Valence Memory (~600 lines)

Learned associations between concepts and affective valence. VSA-encoded for fast
retrieval. Damasio's somatic markers as computational shortcuts.

**Key types**:
```rust
struct SomaticMarker {
    concept: SymbolId,
    valence: f32,          // Learned emotional association
    confidence: f32,       // How strong the association is
    source_episodes: Vec<SymbolId>,  // Episodes that formed this marker
}

struct SomaticMarkerRegistry {
    markers: HashMap<SymbolId, SomaticMarker>,
    valence_role: HyperVec,  // VSA role vector for valence binding
}
```

**Learning**: When a decision leads to a positive/negative outcome, the concepts involved
get their somatic markers updated (EMA). Over time, the agent develops "intuitions" —
concepts that feel good or bad before explicit reasoning.

**Decision integration**: During OODA Decide phase, somatic markers provide a fast-path
bias. Strongly positive markers boost utility; strongly negative markers penalize.
This is not overriding symbolic reasoning — it's providing a Bayesian prior from
experience.

### 23c — Affective Memory Salience (~400 lines)

Emotionally significant events are consolidated with higher priority into episodic
memory. The affective state at encoding time is stored alongside the episode.

**Integration points**:
- Memory consolidation: episodes with high |valence| or high arousal get priority
- Episodic recall: mood-congruent retrieval (current mood biases recall toward
  similarly-valenced episodes)
- Goal generation: negative valence on stalled goals triggers re-evaluation
- Communication: mood modulates grammar archetype selection (formal when anxious,
  narrative when calm, terse when urgent)

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/agent/affect.rs` | NEW — AffectiveState, Appraisal, MoodState, SomaticMarkerRegistry |
| `src/agent/ooda.rs` | Integrate mood into decision utility scoring |
| `src/agent/memory.rs` | Affective salience in consolidation priority |
| `src/agent/agent.rs` | AffectiveSystem field, lifecycle |
| `src/provenance.rs` | DerivationKind::AffectiveAppraisal |

## Success Criteria

1. Agent mood measurably shifts after goal completion/failure
2. Somatic markers develop over repeated interactions with concepts
3. Memory consolidation prioritizes emotionally significant episodes
4. Decision-making shows measurable influence from mood state
5. Communication tone adapts to current affective state

## Non-Goals

- Simulating specific human emotions (joy, anger, fear) as discrete states
- Expressing emotions to the user (internal system, not performance)
- Emotional manipulation or persuasion
