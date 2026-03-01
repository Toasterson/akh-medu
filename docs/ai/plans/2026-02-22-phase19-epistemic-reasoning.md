# Phase 19 — Epistemic Logic & Theory of Mind

> Date: 2026-02-22
> Research: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

- **Status**: Planned
- **Phase**: 19 (3 sub-phases: 19a–19c)
- **Depends on**: Phase 17 (evidence intervals), Phase 18 (source reliability), Phase 12d (interlocutor microtheories)
- **Provenance tags**: 68–70

## Goal

Give the agent formal epistemic reasoning: the ability to represent, track, and reason about what different agents know, believe, and don't know. The agent should model others' belief states (theory of mind), predict how information events change knowledge distributions, and use this to plan communicative actions and detect information asymmetries.

## Architecture Overview

```
                 ┌───────────────────────┐
                 │  19a Epistemic State   │  Knows(agent, proposition)
                 │  Representation       │  Believes(agent, proposition)
                 │                       │  Possible-worlds via microtheories
                 │                       │  Accessibility relations
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  19b Dynamic Epistemic │  Announcement updates
                 │  Logic Operations     │  Private communication
                 │                       │  Observation events
                 │                       │  E-graph epistemic rules
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  19c Theory of Mind    │  Level-k recursive belief modeling
                 │  Engine               │  Predict agent behavior from beliefs
                 │                       │  Information asymmetry detection
                 │                       │  Strategic communication planning
                 └───────────────────────┘
```

## Sub-phases

### 19a — Epistemic State Representation (~400 lines)

**New file**: `src/agent/epistemic.rs`

**Input**: Per-interlocutor microtheories (Phase 12d) + belief intervals (Phase 17)

**Output**: Formal epistemic state model with accessibility semantics

**Types**:
```rust
/// Epistemic modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicModality {
    /// Agent knows phi (factual, justified, true in all accessible worlds).
    Knows,
    /// Agent believes phi (may be false, true in most accessible worlds).
    Believes,
    /// Agent considers phi possible (true in at least one accessible world).
    ConsidersPossible,
    /// Agent is ignorant about phi (neither knows phi nor knows not-phi).
    IgnorantAbout,
}

/// An epistemic proposition: what an agent knows/believes about a triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicProposition {
    /// The agent whose epistemic state this represents.
    pub agent: SymbolId,
    /// The modality (knows, believes, etc.).
    pub modality: EpistemicModality,
    /// The proposition (as a triple reference or KG pattern).
    pub proposition: PropositionRef,
    /// Confidence in the epistemic assessment itself.
    pub confidence: f32,
    /// When this epistemic state was last updated.
    pub timestamp: u64,
}

/// Reference to a proposition (single triple or pattern).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropositionRef {
    /// A specific triple.
    Triple { subject: SymbolId, predicate: SymbolId, object: SymbolId },
    /// A KG entity (the proposition "entity exists with these properties").
    Entity(SymbolId),
    /// A pattern (the proposition "some triple matching this pattern exists").
    Pattern { subject: Option<SymbolId>, predicate: Option<SymbolId>, object: Option<SymbolId> },
}

/// Well-known KG predicates for epistemic reasoning.
pub struct EpistemicPredicates {
    pub knows: SymbolId,
    pub believes: SymbolId,
    pub considers_possible: SymbolId,
    pub ignorant_about: SymbolId,
    pub accessible_world: SymbolId,      // (agent, accessible_world, microtheory)
    pub common_knowledge: SymbolId,      // both/all agents know
    pub information_asymmetry: SymbolId, // one agent knows something another doesn't
}

/// An agent's complete epistemic profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicProfile {
    /// The agent this profile describes.
    pub agent_id: SymbolId,
    /// Microtheory serving as this agent's "possible world".
    pub world_mt: Option<SymbolId>,
    /// Propositions this agent is known to know.
    pub known_propositions: Vec<PropositionRef>,
    /// Propositions this agent is known to believe.
    pub believed_propositions: Vec<PropositionRef>,
    /// Propositions this agent is ignorant about.
    pub ignorance: Vec<PropositionRef>,
    /// Last updated timestamp.
    pub updated_at: u64,
}

/// The result of an epistemic query: "who knows what?"
#[derive(Debug, Clone)]
pub struct EpistemicQueryResult {
    pub agent: SymbolId,
    pub proposition: PropositionRef,
    pub modality: EpistemicModality,
    pub supporting_evidence: Vec<SymbolId>,
    pub confidence: f32,
}
```

**EpistemicStateManager** methods:
- `new(engine)` — init predicates
- `register_agent(agent_id, microtheory_id)` — associate agent with their "world"
- `assert_knows(agent, proposition, engine)` — record that agent knows prop (add to their microtheory)
- `assert_believes(agent, proposition, engine)` — record belief (lower confidence than knows)
- `mark_ignorant(agent, proposition)` — explicitly record ignorance
- `query_modality(agent, proposition, engine) -> EpistemicModality` — determine what modality applies
- `build_profile(agent, engine) -> EpistemicProfile` — collect all epistemic facts about an agent
- `common_knowledge(agents, engine) -> Vec<PropositionRef>` — intersection of knowledge across agents
- `information_asymmetry(agent_a, agent_b, engine) -> Vec<EpistemicProposition>` — what A knows that B doesn't (and vice versa)
- `sync_from_interlocutor(interlocutor_profile, engine)` — populate epistemic state from existing Phase 12d data

**Provenance**: `DerivationKind::EpistemicAssessment { agent_id_raw, modality, proposition_count }` (tag 68)

**Tests (~12)**:
1. epistemic_predicates_namespace
2. assert_knows_adds_to_world
3. assert_believes_lower_confidence
4. query_modality_knows
5. query_modality_ignorant
6. build_profile_collects_all
7. common_knowledge_intersection
8. information_asymmetry_finds_gaps
9. proposition_ref_triple
10. proposition_ref_pattern
11. sync_from_interlocutor
12. serialization_roundtrip

### 19b — Dynamic Epistemic Logic Operations (~400 lines)

**Input**: Epistemic state from 19a + communication channel events

**Output**: Formal epistemic update operations + e-graph rules for epistemic inference

**Types**:
```rust
/// An epistemic event that changes agents' knowledge/belief states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpistemicEvent {
    /// Public announcement: all present agents learn that phi is true.
    /// Post-condition: for all agents a in audience, Knows(a, phi).
    PublicAnnouncement {
        announcer: SymbolId,
        proposition: PropositionRef,
        audience: Vec<SymbolId>,
    },
    /// Private message: only the recipient learns phi.
    /// Post-condition: Knows(recipient, phi) AND NOT Knows(others, phi).
    PrivateMessage {
        sender: SymbolId,
        recipient: SymbolId,
        proposition: PropositionRef,
    },
    /// Observation: agent directly observes a fact.
    /// Post-condition: Knows(observer, phi) with high confidence.
    Observation {
        observer: SymbolId,
        proposition: PropositionRef,
    },
    /// Deceptive announcement: announcer claims phi, but phi may be false.
    /// Post-condition: Believes(audience, phi) at confidence = source_reliability.
    DeceptiveAnnouncement {
        announcer: SymbolId,
        claimed: PropositionRef,
        audience: Vec<SymbolId>,
        source_reliability: f32,
    },
    /// Question: agent asks about phi, revealing ignorance.
    /// Post-condition: Knows(all, IgnorantAbout(asker, phi)).
    Question {
        asker: SymbolId,
        about: PropositionRef,
        audience: Vec<SymbolId>,
    },
    /// Retraction: announcer retracts a previous claim.
    /// Post-condition: move from Knows/Believes to IgnorantAbout.
    Retraction {
        announcer: SymbolId,
        retracted: PropositionRef,
        audience: Vec<SymbolId>,
    },
}

/// Result of applying an epistemic event.
#[derive(Debug, Clone)]
pub struct EpistemicUpdateResult {
    /// Agents whose epistemic state changed.
    pub updated_agents: Vec<SymbolId>,
    /// New epistemic propositions created.
    pub new_propositions: Vec<EpistemicProposition>,
    /// Propositions that were revised (changed modality).
    pub revised_propositions: Vec<(EpistemicProposition, EpistemicModality)>,
    /// Any information asymmetries created by this event.
    pub new_asymmetries: Vec<(SymbolId, SymbolId, PropositionRef)>,
}
```

**Methods**:
- `apply_event(event, epistemic_mgr, engine) -> EpistemicUpdateResult` — dispatch to specific handler
- `apply_public_announcement(announcer, prop, audience, engine)` — add Knows for all in audience
- `apply_private_message(sender, recipient, prop, engine)` — add Knows for recipient only, create asymmetry record
- `apply_observation(observer, prop, engine)` — high-confidence Knows
- `apply_deceptive_announcement(announcer, prop, audience, reliability, engine)` — Believes at source reliability level (not Knows)
- `apply_question(asker, about, audience, engine)` — mark asker as IgnorantAbout, all learn this fact
- `apply_retraction(announcer, retracted, audience, engine)` — move to IgnorantAbout, log retraction
- `infer_knowledge_from_communication(channel_msg, engine)` — auto-detect epistemic event type from incoming messages

**E-graph rules** (added to `src/reason/mod.rs`):
```
// Knowledge axiom K: if agent knows (P → Q) and knows P, then knows Q
rewrite!("k-axiom"; "(knows ?a (implies ?p ?q))" "(knows ?a ?p)" => "(knows ?a ?q)")

// Positive introspection (S4): if agent knows P, then agent knows that they know P
rewrite!("positive-introspection"; "(knows ?a ?p)" => "(knows ?a (knows ?a ?p))")

// Knowledge implies belief
rewrite!("knows-implies-believes"; "(knows ?a ?p)" => "(believes ?a ?p)")

// Public announcement: if phi is announced and agent was present, agent knows phi
rewrite!("pub-announce"; "(announced ?p ?audience)" "(member ?a ?audience)" => "(knows ?a ?p)")

// Negative introspection: if agent doesn't know P, agent knows they don't know P
rewrite!("neg-introspection"; "(ignorant-about ?a ?p)" => "(knows ?a (ignorant-about ?a ?p))")
```

**Provenance**: `DerivationKind::EpistemicUpdate { event_kind, updated_agent_count, asymmetries_created }` (tag 69)

**Tests (~12)**:
1. public_announcement_updates_all
2. private_message_updates_recipient_only
3. private_message_creates_asymmetry
4. observation_high_confidence
5. deceptive_announcement_uses_reliability
6. question_reveals_ignorance
7. retraction_moves_to_ignorant
8. infer_from_text_message
9. egraph_k_axiom_fires
10. egraph_positive_introspection
11. egraph_knows_implies_believes
12. chain_of_events_accumulates

### 19c — Theory of Mind Engine (~450 lines)

**Input**: Epistemic states from 19a/19b + InterlocutorProfile (Phase 12d) + trust model (Phase 18a)

**Output**: Recursive belief modeling, behavior prediction, strategic communication

**Types**:
```rust
/// Level of theory of mind recursion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToMLevel {
    /// Level 0: Non-strategic. The other agent acts randomly or by fixed heuristic.
    Level0,
    /// Level 1: Model what the other agent knows/believes and predict behavior.
    Level1,
    /// Level 2: Model what the other agent thinks *I* know, and act accordingly.
    Level2,
    /// Level 3: Recursive — model their model of my model. Diminishing returns.
    Level3,
}

/// A theory-of-mind model for a specific agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToMModel {
    /// The agent being modeled.
    pub target_agent: SymbolId,
    /// Depth of recursive modeling.
    pub level: ToMLevel,
    /// What we believe they know.
    pub their_knowledge: Vec<PropositionRef>,
    /// What we believe they believe.
    pub their_beliefs: Vec<PropositionRef>,
    /// What we believe they want (inferred goals).
    pub their_goals: Vec<String>,
    /// What we believe they think *we* know (Level 2+).
    pub their_model_of_us: Option<Box<ToMModel>>,
    /// Predicted next action based on their modeled state.
    pub predicted_actions: Vec<PredictedAction>,
    /// Confidence in this model.
    pub confidence: f32,
}

/// A predicted action by another agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedAction {
    pub description: String,
    pub probability: f32,
    pub rationale: String,
}

/// Information advantage assessment.
#[derive(Debug, Clone)]
pub struct InformationAdvantage {
    /// What we know that they don't.
    pub our_advantages: Vec<PropositionRef>,
    /// What they know that we don't.
    pub their_advantages: Vec<PropositionRef>,
    /// Propositions where we have higher confidence.
    pub confidence_advantages: Vec<(PropositionRef, f32, f32)>,
    /// Overall strategic assessment.
    pub assessment: StrategicAssessment,
}

#[derive(Debug, Clone, Copy)]
pub enum StrategicAssessment {
    /// We have significant information advantage.
    InformationSuperior,
    /// Roughly symmetric information.
    InformationParity,
    /// They have significant information advantage.
    InformationInferior,
    /// Unable to assess (insufficient ToM model).
    Unknown,
}

/// Recommendation for communication strategy based on ToM.
#[derive(Debug, Clone)]
pub struct CommunicationStrategy {
    /// What to share (maximizes our goals).
    pub share: Vec<PropositionRef>,
    /// What to withhold (revealing would harm our position).
    pub withhold: Vec<PropositionRef>,
    /// What to ask about (fills our information gaps).
    pub inquire: Vec<PropositionRef>,
    /// Whether deception was considered and rejected (transparency).
    pub deception_rejected_reason: Option<String>,
    /// Confidence in strategy.
    pub confidence: f32,
}
```

**ToMEngine** methods:
- `new(max_level)` — default max ToMLevel::Level2
- `build_model(target, epistemic_mgr, engine) -> ToMModel` — construct ToM model from epistemic state + interaction history
- `predict_behavior(model) -> Vec<PredictedAction>` — given their beliefs and goals, what would they do?
- `information_advantage(us, them, epistemic_mgr, engine) -> InformationAdvantage` — compute asymmetry
- `plan_communication(our_goals, target, model, engine) -> CommunicationStrategy` — what to share/withhold/ask
- `update_model_from_observation(model, observed_action, engine)` — refine model after seeing what they actually did
- `model_accuracy(model, observed_actions) -> f32` — how well did our predictions match reality?
- `recursive_model(target, level, epistemic_mgr, engine) -> ToMModel` — build nested models up to max_level

**Key algorithm — Level-1 behavior prediction**:
1. From epistemic profile, collect their known/believed propositions
2. From interaction history, infer their goals (what topics they ask about, what outcomes they pursue)
3. Using the causal model (Phase 15), predict which actions would advance their goals given their beliefs
4. Rank predicted actions by expected utility from *their* perspective

**Key algorithm — Communication strategy**:
1. Identify our active goals
2. For each proposition we could share, simulate: if they learn this, how does it change their predicted behavior?
3. Share propositions where their changed behavior would help our goals
4. Withhold propositions where their changed behavior would hurt our goals
5. Ask about propositions in their_advantages that would help our goals if we knew them
6. Never recommend deception — but log that the option was considered and why it was rejected (transparency and ethical constraint)

**Provenance**: `DerivationKind::TheoryOfMind { target_agent_raw, tom_level, prediction_count }` (tag 70)

**Tests (~14)**:
1. tom_level_ordering
2. build_model_level0
3. build_model_level1
4. build_model_level2_recursive
5. predict_behavior_from_beliefs
6. predict_behavior_from_goals
7. information_advantage_symmetric
8. information_advantage_asymmetric
9. communication_strategy_share
10. communication_strategy_withhold
11. update_model_from_observation
12. model_accuracy_perfect
13. model_accuracy_poor
14. deception_always_rejected

## Cross-Cutting Changes

### Agent (`src/agent/agent.rs`)
- New fields: `epistemic_manager: EpistemicStateManager`, `tom_engine: ToMEngine`
- Init: register self as agent, sync existing interlocutors

### Module registry (`src/agent/mod.rs`)
- `pub mod epistemic;`

### Error (`src/agent/error.rs`)
- `Epistemic(#[from] super::epistemic::EpistemicError)` variant

### Provenance (`src/provenance.rs`)
- Tags 68–70

### Communication (`src/agent/channel.rs`)
- On message receive: auto-generate EpistemicEvent and apply

### Interlocutor (`src/agent/interlocutor.rs`)
- `EpistemicProfile` auto-created when interlocutor registered
- `InterlocutorProfile` gains `epistemic_profile: Option<EpistemicProfile>` field

### E-graph rules (`src/reason/mod.rs`)
- `epistemic_rules()` function returning 5 rewrite rules

### OODA (`src/agent/ooda.rs`)
- In `observe()`: apply epistemic events from incoming communication
- In `orient()`: build/update ToM models for active interlocutors
- In `decide()`: use CommunicationStrategy when selecting communicative actions

### NLP (`src/agent/nlp.rs`)
- `UserIntent::EpistemicQuery { agent, topic }` — "what does Bob know about...", "does Alice believe..."

### CLI (`src/main.rs`)
- `Commands::Epistemic { action: EpistemicAction }` with subcommands: WhoKnows { topic }, Asymmetry { agent_a, agent_b }, ToM { target }, Strategy { target }

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all existing + ~38 new tests pass
3. `cargo clippy` — no new warnings
4. Manual: `akh epistemic who-knows "rust"` shows who knows about Rust, `akh epistemic tom "bob"` shows ToM model for bob, `akh epistemic strategy "alice"` shows communication strategy
