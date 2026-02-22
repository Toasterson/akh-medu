# Phase 21 — Game-Theoretic Social Reasoning

> Date: 2026-02-22
> Research: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

- **Status**: Planned
- **Phase**: 21 (3 sub-phases: 21a–21c)
- **Depends on**: Phase 19 (epistemic logic + ToM), Phase 18 (source reliability), Phase 20 (active inference)
- **Provenance tags**: 73–75

## Goal

Give the agent the ability to reason about interactions as games: situations where multiple agents have their own goals, strategies, and information states. The agent should evaluate whether cooperation or caution is appropriate, detect when another agent's goals conflict with its own, model strategic communication (signaling games), and plan interactions that account for the rational responses of other agents.

## Architecture Overview

```
                 ┌───────────────────────┐
                 │  21a Interaction Game  │  Model interactions as games
                 │  Modeling             │  Utility functions per agent
                 │                       │  Payoff matrices from goals + ToM
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  21b Signaling Games   │  Communication under strategic incentives
                 │  + Level-k Reasoning  │  Cheap talk analysis
                 │                       │  Bayesian persuasion model
                 │                       │  Bounded rationality (Level-k)
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  21c Strategic Action  │  Best-response planning
                 │  Planning             │  Cooperative/competitive detection
                 │                       │  Coalition formation for multi-agent
                 │                       │  Integrated social decision-making
                 └───────────────────────┘
```

## Sub-phases

### 21a — Interaction Game Modeling (~400 lines)

**New file**: `src/agent/game_theory.rs`

**Input**: ToM models (Phase 19c) + goal system + epistemic states

**Output**: Game-theoretic model of agent interactions

**Types**:
```rust
/// Classification of a game (interaction type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameType {
    /// Both agents benefit from cooperation. Trust is safe.
    Cooperative,
    /// One agent's gain is the other's loss. Be cautious.
    ZeroSum,
    /// Partially aligned, partially conflicting goals.
    MixedMotive,
    /// Information exchange with possible deception. Signal carefully.
    Signaling,
    /// Leader commits to strategy first; follower responds.
    Stackelberg,
    /// Cannot classify (insufficient model of other agent).
    Unknown,
}

/// A game model for an interaction with another agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionGame {
    /// Our agent.
    pub self_agent: SymbolId,
    /// The other agent.
    pub other_agent: SymbolId,
    /// Classified game type.
    pub game_type: GameType,
    /// Our available strategies (actions).
    pub our_strategies: Vec<Strategy>,
    /// Their available strategies (predicted from ToM).
    pub their_strategies: Vec<Strategy>,
    /// Payoff matrix: our_strategy[i] x their_strategy[j] -> (our_payoff, their_payoff).
    pub payoff_matrix: Vec<Vec<(f32, f32)>>,
    /// Nash equilibria (if computable).
    pub equilibria: Vec<(usize, usize)>,
    /// Our recommended strategy.
    pub recommended_strategy: Option<usize>,
    /// Confidence in the game model.
    pub confidence: f32,
}

/// A strategy available to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    /// Human-readable name.
    pub name: String,
    /// The action(s) this strategy entails.
    pub actions: Vec<SymbolId>,
    /// Description of the strategy.
    pub description: String,
}

/// Payoff computation inputs.
#[derive(Debug, Clone)]
pub struct PayoffFactors {
    /// How much does this outcome advance our goals? (-1 to 1)
    pub goal_progress: f32,
    /// How much trust do we gain/lose? (-1 to 1)
    pub trust_change: f32,
    /// How much information do we gain? (0 to 1)
    pub information_gain: f32,
    /// Resource cost of this strategy. (0 to 1)
    pub resource_cost: f32,
}
```

**GameModeler** methods:
- `new()` — init
- `model_interaction(self_agent, other_agent, tom_model, engine) -> InteractionGame` — build game model from goals + ToM
- `classify_game(payoff_matrix) -> GameType` — analyze payoff structure to classify
- `compute_payoff(our_strategy, their_strategy, our_goals, their_model, engine) -> (f32, f32)` — estimate payoffs
- `find_nash_equilibria(game) -> Vec<(usize, usize)>` — for 2-player games, find pure-strategy Nash equilibria
- `find_dominant_strategies(game) -> (Option<usize>, Option<usize>)` — check for dominated strategies
- `recommend_strategy(game) -> usize` — select based on game type: cooperative → cooperate, zero-sum → minimax, mixed → maximin, signaling → use ToM
- `detect_goal_alignment(our_goals, their_goals, engine) -> f32` — cosine similarity between goal vectors
- `detect_prisoners_dilemma(game) -> bool` — check payoff structure for PD pattern

**Integration with InterlocutorProfile**: Each active interlocutor gains:
- `current_game: Option<InteractionGame>` — the modeled game for current interaction
- Auto-updated as conversation progresses

**Provenance**: `DerivationKind::GameTheoreticModel { game_type, strategy_count, equilibria_count }` (tag 73)

**Tests (~12)**:
1. classify_cooperative_game
2. classify_zero_sum_game
3. classify_mixed_motive
4. classify_signaling_game
5. find_nash_pure_strategy
6. find_dominant_strategy
7. recommend_cooperative_strategy
8. recommend_cautious_for_zero_sum
9. payoff_from_goal_progress
10. detect_prisoners_dilemma
11. goal_alignment_high
12. goal_alignment_low

### 21b — Signaling Games & Level-k Reasoning (~400 lines)

**Input**: GameModeler (21a) + epistemic states (Phase 19) + credibility analysis (Phase 18c)

**Output**: Strategic communication analysis, cheap talk evaluation, bounded-rational opponent modeling

**Types**:
```rust
/// Analysis of a communication as a signaling game.
#[derive(Debug, Clone)]
pub struct SignalingAnalysis {
    /// The sender and their message.
    pub sender: SymbolId,
    pub message: PropositionRef,
    /// Is this cheap talk (costless, non-binding) or costly signaling?
    pub signal_type: SignalType,
    /// What the sender would want us to believe.
    pub sender_desired_belief: Vec<PropositionRef>,
    /// What the sender actually knows (from our ToM).
    pub sender_actual_knowledge: Vec<PropositionRef>,
    /// Credibility of the signal given incentive analysis.
    pub incentive_credibility: f32,
    /// Whether the sender has an incentive to deceive on this topic.
    pub deception_incentive: bool,
    /// Recommended response.
    pub recommended_response: SignalingResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Costless communication (words are cheap).
    CheapTalk,
    /// Costly signaling (sender bears a cost, more credible).
    CostlySignal,
    /// Verifiable claim (can be checked against KG or external source).
    VerifiableClaim,
}

#[derive(Debug, Clone)]
pub enum SignalingResponse {
    /// Trust the signal (aligned incentives or verifiable).
    Trust,
    /// Partially trust (discount by incentive credibility).
    PartialTrust { discount: f32 },
    /// Verify before trusting.
    VerifyFirst { verification_method: String },
    /// Reject (strong deception incentive).
    Reject { reason: String },
}

/// Level-k reasoning about an opponent's strategy sophistication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelKModel {
    /// Our estimate of the opponent's strategic sophistication (0-3).
    pub estimated_opponent_level: ToMLevel,
    /// Evidence for each level.
    pub level_evidence: Vec<(ToMLevel, f32)>,
    /// Our best-response level (opponent_level + 1, capped at Level3).
    pub our_response_level: ToMLevel,
    /// How the opponent's past behavior matches each level's predictions.
    pub behavior_consistency: Vec<(ToMLevel, f32)>,
}

/// Cheap talk analysis: when communication is non-binding and costless.
#[derive(Debug, Clone)]
pub struct CheapTalkAnalysis {
    /// Can the sender credibly communicate in this context?
    pub credible: bool,
    /// Are interests sufficiently aligned for cheap talk to be informative?
    pub interests_aligned: f32,
    /// Babbling equilibrium risk: is the message meaningless?
    pub babbling_risk: f32,
    /// The informational content if we take the message at face value.
    pub face_value_content: Vec<PropositionRef>,
}
```

**SignalingAnalyzer** methods:
- `analyze_signal(sender, message, game, epistemic_mgr, engine) -> SignalingAnalysis`
- `classify_signal_type(message, sender, engine) -> SignalType` — verifiable claims are most credible
- `compute_deception_incentive(sender, message, game) -> bool` — does the sender benefit from us believing a false message?
- `compute_incentive_credibility(signal_type, deception_incentive, source_reliability) -> f32` — overall credibility given strategic context
- `analyze_cheap_talk(sender, game) -> CheapTalkAnalysis` — evaluate cheap talk informativeness
- `estimate_opponent_level(opponent, interaction_history) -> LevelKModel` — from past behavior, estimate their sophistication
- `best_response_at_level(level, game) -> usize` — compute best response assuming opponent plays at given level

**Key algorithm — deception incentive detection**:
1. From the game model, identify the sender's preferred outcomes
2. Check whether the message, if believed, would lead us to choose an action that produces the sender's preferred outcome
3. Check whether the sender's preferred outcome is bad for our goals
4. If yes to both → sender has a deception incentive
5. Cross-check with source reliability: a source with SourceReliability::A and deception incentive is concerning (high-capability deceiver)

**Provenance**: `DerivationKind::SignalingAnalysis { signal_type, incentive_credibility, deception_incentive }` (tag 74)

**Tests (~12)**:
1. classify_cheap_talk
2. classify_costly_signal
3. classify_verifiable_claim
4. deception_incentive_present
5. deception_incentive_absent
6. incentive_credibility_high_for_aligned
7. incentive_credibility_low_for_adversarial
8. signaling_response_trust_aligned
9. signaling_response_verify_costly
10. level_k_estimation_from_history
11. cheap_talk_informative_aligned
12. cheap_talk_babbling_adversarial

### 21c — Strategic Action Planning (~450 lines)

**Input**: GameModeler (21a) + SignalingAnalyzer (21b) + MCTS (Phase 16) + Active Inference (Phase 20)

**Output**: Socially-aware action planning that accounts for other agents' responses

**Types**:
```rust
/// A socially-aware action plan that considers other agents' responses.
#[derive(Debug, Clone)]
pub struct SocialPlan {
    /// Our planned action sequence.
    pub our_actions: Vec<SymbolId>,
    /// Predicted responses from other agents.
    pub predicted_responses: Vec<PredictedResponse>,
    /// Expected outcome considering responses.
    pub expected_outcome: SocialOutcome,
    /// Alternative plans considered.
    pub alternatives: Vec<(Vec<SymbolId>, SocialOutcome)>,
    /// Reasoning for the chosen plan.
    pub reasoning: String,
}

/// Predicted response from another agent to our action.
#[derive(Debug, Clone)]
pub struct PredictedResponse {
    pub agent: SymbolId,
    pub likely_action: String,
    pub probability: f32,
    pub based_on: ToMLevel,
    /// If they respond this way, how does it affect our plan?
    pub impact_on_our_plan: PlanImpact,
}

#[derive(Debug, Clone, Copy)]
pub enum PlanImpact {
    /// Their response helps our plan.
    Beneficial,
    /// Their response doesn't affect our plan.
    Neutral,
    /// Their response hinders our plan.
    Detrimental,
    /// Their response blocks our plan entirely.
    Blocking,
}

/// Expected social outcome (factoring in all agents' behavior).
#[derive(Debug, Clone)]
pub struct SocialOutcome {
    /// Our goal progress.
    pub goal_progress: f32,
    /// Trust changes with other agents.
    pub trust_changes: Vec<(SymbolId, f32)>,
    /// Information state changes.
    pub information_changes: Vec<(SymbolId, EpistemicModality)>,
    /// Overall expected utility.
    pub expected_utility: f32,
}

/// Detection of cooperation vs. competition dynamics.
#[derive(Debug, Clone)]
pub struct SocialDynamics {
    /// Agents we should cooperate with (aligned goals).
    pub cooperators: Vec<(SymbolId, f32)>,
    /// Agents we should be cautious of (conflicting goals).
    pub competitors: Vec<(SymbolId, f32)>,
    /// Agents whose alignment is unclear.
    pub unknown: Vec<SymbolId>,
    /// Overall social environment assessment.
    pub environment: SocialEnvironment,
}

#[derive(Debug, Clone, Copy)]
pub enum SocialEnvironment {
    /// Mostly cooperative, trust is generally safe.
    Cooperative,
    /// Mixed, need to assess per-interaction.
    Mixed,
    /// Mostly competitive, default to caution.
    Competitive,
    /// Unknown, insufficient information.
    Unknown,
}

/// Coalition: a group of agents with shared goals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coalition {
    /// Member agents.
    pub members: Vec<SymbolId>,
    /// Shared goals.
    pub shared_goals: Vec<String>,
    /// Coalition value (expected utility from cooperation).
    pub value: f32,
    /// Stability: would any member benefit from leaving?
    pub stable: bool,
}
```

**StrategicPlanner** methods:
- `new()` — init
- `plan_social_action(goals, interlocutors, game_models, mcts, engine) -> SocialPlan` — plan considering others' responses
- `predict_response(agent, our_action, tom_model, game) -> PredictedResponse` — from ToM + game model, predict their response
- `evaluate_social_outcome(our_actions, responses, engine) -> SocialOutcome` — compute combined utility
- `detect_dynamics(interlocutors, game_models) -> SocialDynamics` — classify cooperation/competition
- `form_coalition(agents, goals, game_models) -> Option<Coalition>` — identify beneficial coalitions
- `should_cooperate(other_agent, game, trust_model) -> bool` — decision framework for cooperation
- `minimax_plan(game) -> Vec<SymbolId>` — worst-case optimal plan for competitive games
- `cooperative_plan(game, coalition) -> Vec<SymbolId>` — joint optimum for cooperative games

**Integration with OODA (Phase 20b)**: The active inference Decide phase gains a social layer:
1. For each candidate policy, predict other agents' responses via StrategicPlanner
2. Adjust EFE computation to account for social outcomes
3. Select policy that maximizes EFE *including* social considerations

**Key algorithm — social-aware MCTS**:
In the MCTS rollout (Phase 16b), after each of our simulated actions, also simulate the predicted responses from other agents. This creates an adversarial/cooperative tree search:
```
for each our_action in rollout:
    apply our_action to simulated state
    for each active_interlocutor:
        their_response = predict_response(interlocutor, our_action, tom, game)
        apply their_response to simulated state
    evaluate simulated state
```

**Provenance**: `DerivationKind::StrategicPlanning { game_type, social_environment, coalition_size }` (tag 75)

**Tests (~14)**:
1. detect_cooperative_dynamics
2. detect_competitive_dynamics
3. detect_mixed_dynamics
4. predict_response_cooperative
5. predict_response_competitive
6. social_outcome_utility
7. should_cooperate_aligned_goals
8. should_cooperate_misaligned_goals
9. minimax_plan_safe
10. cooperative_plan_joint_optimum
11. form_coalition_beneficial
12. coalition_stability
13. social_plan_accounts_for_response
14. social_mcts_rollout_adversarial

## Cross-Cutting Changes

### Agent (`src/agent/agent.rs`)
- New fields: `game_modeler: GameModeler`, `signaling_analyzer: SignalingAnalyzer`, `strategic_planner: StrategicPlanner`

### Module registry (`src/agent/mod.rs`)
- `pub mod game_theory;`

### Error (`src/agent/error.rs`)
- `GameTheory(#[from] super::game_theory::GameTheoryError)` variant

### Provenance (`src/provenance.rs`)
- Tags 73–75

### OODA (`src/agent/ooda.rs`)
- In `orient()`: detect social dynamics from active interlocutors
- In `decide()`: invoke StrategicPlanner for social-aware action selection
- Social MCTS rollout: include predicted agent responses

### Communication (`src/agent/channel.rs`)
- On message receive: run signaling analysis
- On message send: evaluate signaling implications

### NLP (`src/agent/nlp.rs`)
- `UserIntent::StrategyQuery { agent }` — "what's the game with...", "should I cooperate with..."

### CLI (`src/main.rs`)
- `Commands::Strategy { action: StrategyAction }` with subcommands: Analyze { agent }, Dynamics, Coalition, Signal { message }

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all existing + ~38 new tests pass
3. `cargo clippy` — no new warnings
4. Manual: `akh strategy analyze "bob"` shows game model, `akh strategy dynamics` shows cooperation/competition landscape
