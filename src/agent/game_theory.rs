//! Game-theoretic social reasoning — Phase 21.
//!
//! Models multi-agent interactions as games, analyzes signaling credibility,
//! and plans socially-aware action sequences.
//!
//! ## Sub-phases
//!
//! - **21a**: Interaction game modeling (game types, payoff matrices, Nash equilibria)
//! - **21b**: Signaling games & Level-k reasoning (cheap talk, costly signals, deception incentives)
//! - **21c**: Strategic action planning (social MCTS, coalition formation, response prediction)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

use super::epistemic::{EpistemicModality, PropositionRef, ToMLevel};

// ═══════════════════════════════════════════════════════════════════════
// 21a — Interaction Game Modeling
// ═══════════════════════════════════════════════════════════════════════

/// Type of strategic interaction between agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameType {
    /// Both agents benefit from cooperating.
    Cooperative,
    /// One agent's gain is the other's loss.
    ZeroSum,
    /// Partially aligned interests.
    MixedMotive,
    /// Information exchange with deception risk.
    Signaling,
    /// One agent commits first (leader-follower).
    Stackelberg,
    /// Insufficient information to classify.
    Unknown,
}

impl GameType {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Cooperative => "cooperative",
            Self::ZeroSum => "zero-sum",
            Self::MixedMotive => "mixed-motive",
            Self::Signaling => "signaling",
            Self::Stackelberg => "stackelberg",
            Self::Unknown => "unknown",
        }
    }
}

/// A named strategy (action sequence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub name: String,
    pub actions: Vec<SymbolId>,
    pub description: String,
}

/// Factors contributing to a payoff computation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PayoffFactors {
    /// Goal progress (-1.0 to 1.0).
    pub goal_progress: f32,
    /// Trust change (-1.0 to 1.0).
    pub trust_change: f32,
    /// Information gained (0.0 to 1.0).
    pub information_gain: f32,
    /// Resource cost (0.0 to 1.0).
    pub resource_cost: f32,
}

impl PayoffFactors {
    /// Compute scalar payoff from factors.
    pub fn total(&self) -> f32 {
        self.goal_progress * 0.4
            + self.trust_change * 0.2
            + self.information_gain * 0.2
            - self.resource_cost * 0.2
    }
}

/// A game model between two agents.
#[derive(Debug, Clone)]
pub struct InteractionGame {
    pub self_agent: SymbolId,
    pub other_agent: SymbolId,
    pub game_type: GameType,
    pub our_strategies: Vec<Strategy>,
    pub their_strategies: Vec<Strategy>,
    /// Payoff matrix: [our_strategy][their_strategy] → (our_payoff, their_payoff).
    pub payoff_matrix: Vec<Vec<(f32, f32)>>,
    /// Nash equilibria as (our_strategy_idx, their_strategy_idx).
    pub equilibria: Vec<(usize, usize)>,
    /// Recommended strategy index for us.
    pub recommended_strategy: Option<usize>,
    /// Confidence in the model.
    pub confidence: f32,
}

/// Classify a game from its payoff matrix.
pub fn classify_game(payoff_matrix: &[Vec<(f32, f32)>]) -> GameType {
    if payoff_matrix.is_empty() || payoff_matrix[0].is_empty() {
        return GameType::Unknown;
    }

    let mut all_zero_sum = true;
    let mut has_conflict = false;

    // Check for asymmetric incentives (PD-like structure).
    let rows = payoff_matrix.len();
    let cols = payoff_matrix[0].len();

    for row in payoff_matrix {
        for &(our, their) in row {
            let sum = our + their;
            if sum.abs() > 0.1 {
                all_zero_sum = false;
            }
            if (our > 0.0 && their < 0.0) || (our < 0.0 && their > 0.0) {
                has_conflict = true;
            }
        }
    }

    if all_zero_sum {
        return GameType::ZeroSum;
    }

    // Check for temptation to defect: can either player improve by
    // unilaterally deviating from the mutually best outcome?
    if rows >= 2 && cols >= 2 {
        // Find the cell with highest joint payoff.
        let mut best_joint = f32::NEG_INFINITY;
        let mut best_cell = (0, 0);
        for i in 0..rows {
            for j in 0..cols {
                let joint = payoff_matrix[i][j].0 + payoff_matrix[i][j].1;
                if joint > best_joint {
                    best_joint = joint;
                    best_cell = (i, j);
                }
            }
        }
        // Can either deviate and improve?
        let (bi, bj) = best_cell;
        let our_at_best = payoff_matrix[bi][bj].0;
        let their_at_best = payoff_matrix[bi][bj].1;
        let our_tempted = (0..rows).any(|k| k != bi && payoff_matrix[k][bj].0 > our_at_best + 0.01);
        let their_tempted = (0..cols).any(|k| k != bj && payoff_matrix[bi][k].1 > their_at_best + 0.01);

        if our_tempted || their_tempted {
            return GameType::MixedMotive;
        }
    }

    if has_conflict {
        GameType::MixedMotive
    } else {
        GameType::Cooperative
    }
}

/// Find pure-strategy Nash equilibria in a 2-player game.
///
/// A cell (i, j) is a Nash equilibrium if:
/// - Our payoff at (i, j) >= our payoff at (k, j) for all k (best response to j)
/// - Their payoff at (i, j) >= their payoff at (i, k) for all k (best response to i)
pub fn find_nash_equilibria(payoff_matrix: &[Vec<(f32, f32)>]) -> Vec<(usize, usize)> {
    let rows = payoff_matrix.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = payoff_matrix[0].len();
    if cols == 0 {
        return Vec::new();
    }

    let mut equilibria = Vec::new();

    for i in 0..rows {
        for j in 0..cols {
            let (our_payoff, their_payoff) = payoff_matrix[i][j];

            // Check: is i our best response to their j?
            let our_best = (0..rows)
                .map(|k| payoff_matrix[k][j].0)
                .fold(f32::NEG_INFINITY, f32::max);
            if (our_payoff - our_best).abs() > 0.001 {
                continue;
            }

            // Check: is j their best response to our i?
            let their_best = (0..cols)
                .map(|k| payoff_matrix[i][k].1)
                .fold(f32::NEG_INFINITY, f32::max);
            if (their_payoff - their_best).abs() > 0.001 {
                continue;
            }

            equilibria.push((i, j));
        }
    }

    equilibria
}

/// Find dominant strategies (if any).
///
/// A strategy is dominant if it's at least as good as all alternatives
/// against every opponent strategy.
pub fn find_dominant_strategy(
    payoff_matrix: &[Vec<(f32, f32)>],
    for_us: bool,
) -> Option<usize> {
    let rows = payoff_matrix.len();
    if rows == 0 {
        return None;
    }
    let cols = payoff_matrix[0].len();

    if for_us {
        // Check each of our strategies (rows).
        'outer: for i in 0..rows {
            for k in 0..rows {
                if k == i {
                    continue;
                }
                // i must be >= k for all opponent strategies.
                for j in 0..cols {
                    if payoff_matrix[i][j].0 < payoff_matrix[k][j].0 - 0.001 {
                        continue 'outer;
                    }
                }
            }
            return Some(i);
        }
    } else {
        // Check each of their strategies (cols).
        'outer2: for j in 0..cols {
            for k in 0..cols {
                if k == j {
                    continue;
                }
                for i in 0..rows {
                    if payoff_matrix[i][j].1 < payoff_matrix[i][k].1 - 0.001 {
                        continue 'outer2;
                    }
                }
            }
            return Some(j);
        }
    }

    None
}

/// Recommend a strategy based on game type.
pub fn recommend_strategy(game: &InteractionGame) -> Option<usize> {
    // If there's a dominant strategy, use it.
    if let Some(dom) = find_dominant_strategy(&game.payoff_matrix, true) {
        return Some(dom);
    }

    // If there's a Nash equilibrium, use it.
    if let Some(&(our_idx, _)) = game.equilibria.first() {
        return Some(our_idx);
    }

    // Fallback: maximin (maximize minimum payoff).
    if game.payoff_matrix.is_empty() {
        return None;
    }
    let maximin = game
        .payoff_matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let min_payoff = row
                .iter()
                .map(|(our, _)| *our)
                .fold(f32::INFINITY, f32::min);
            (i, min_payoff)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);

    maximin
}

/// Detect if a game has Prisoner's Dilemma structure.
///
/// PD requires: T > R > P > S and 2R > T + S
/// where T=temptation, R=reward, P=punishment, S=sucker
pub fn is_prisoners_dilemma(payoff_matrix: &[Vec<(f32, f32)>]) -> bool {
    if payoff_matrix.len() != 2 || payoff_matrix[0].len() != 2 {
        return false;
    }
    // Convention: row 0 = cooperate, row 1 = defect
    let r = payoff_matrix[0][0].0; // cooperate-cooperate
    let s = payoff_matrix[0][1].0; // cooperate-defect (sucker)
    let t = payoff_matrix[1][0].0; // defect-cooperate (temptation)
    let p = payoff_matrix[1][1].0; // defect-defect (punishment)

    t > r && r > p && p > s && 2.0 * r > t + s
}

// ═══════════════════════════════════════════════════════════════════════
// 21b — Signaling Games & Level-k
// ═══════════════════════════════════════════════════════════════════════

/// Type of signal in a communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Costless, non-binding communication.
    CheapTalk,
    /// Sender bears cost to signal (credible).
    CostlySignal,
    /// Claim can be checked against KG.
    VerifiableClaim,
}

/// Recommended response to a signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalingResponse {
    /// Accept at face value.
    Trust,
    /// Accept with reduced confidence.
    PartialTrust { discount: f32 },
    /// Verify before accepting.
    VerifyFirst { method: String },
    /// Reject the signal.
    Reject { reason: String },
}

/// Analysis of a signaling interaction.
#[derive(Debug, Clone)]
pub struct SignalingAnalysis {
    pub sender: SymbolId,
    pub signal_type: SignalType,
    /// Does the sender have an incentive to deceive?
    pub deception_incentive: bool,
    /// Credibility of the signal given incentives.
    pub incentive_credibility: f32,
    /// Recommended response.
    pub recommended_response: SignalingResponse,
}

/// Cheap talk analysis: is the communication informative?
#[derive(Debug, Clone)]
pub struct CheapTalkAnalysis {
    /// Are interests sufficiently aligned for truthful communication?
    pub credible: bool,
    /// Degree of interest alignment (-1.0 to 1.0).
    pub interests_aligned: f32,
    /// Risk of "babbling" (uninformative equilibrium).
    pub babbling_risk: f32,
}

/// Level-k reasoning model for an opponent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelKModel {
    /// Estimated opponent sophistication level.
    pub estimated_level: ToMLevel,
    /// Our best-response level (opponent_level + 1, capped).
    pub our_response_level: ToMLevel,
    /// Confidence in the level estimate.
    pub confidence: f32,
}

impl LevelKModel {
    /// Estimate opponent level from interaction history.
    ///
    /// Simple heuristic: more observations → higher estimated level.
    pub fn estimate(observation_count: u32) -> Self {
        let (estimated, response) = if observation_count < 3 {
            (ToMLevel::Level0, ToMLevel::Level1)
        } else if observation_count < 10 {
            (ToMLevel::Level1, ToMLevel::Level2)
        } else {
            (ToMLevel::Level2, ToMLevel::Level2) // cap at Level2
        };

        LevelKModel {
            estimated_level: estimated,
            our_response_level: response,
            confidence: (observation_count as f32 / 20.0).min(0.9),
        }
    }
}

/// Analyze a signal's credibility given the game context.
pub fn analyze_signal(
    sender: SymbolId,
    signal_type: SignalType,
    goal_alignment: f32,
    source_reliability: f32,
) -> SignalingAnalysis {
    // Deception incentive: misaligned goals + cheap talk
    let deception_incentive =
        goal_alignment < 0.0 && signal_type == SignalType::CheapTalk;

    let incentive_credibility = if deception_incentive {
        source_reliability * 0.5 // heavily discount
    } else {
        match signal_type {
            SignalType::VerifiableClaim => 0.95,
            SignalType::CostlySignal => source_reliability * 0.9,
            SignalType::CheapTalk => {
                if goal_alignment > 0.5 {
                    source_reliability * 0.8
                } else {
                    source_reliability * 0.5
                }
            }
        }
    };

    let recommended_response = if incentive_credibility > 0.8 {
        SignalingResponse::Trust
    } else if incentive_credibility > 0.5 {
        SignalingResponse::PartialTrust {
            discount: 1.0 - incentive_credibility,
        }
    } else if signal_type == SignalType::VerifiableClaim {
        SignalingResponse::VerifyFirst {
            method: "check against KG".into(),
        }
    } else {
        SignalingResponse::Reject {
            reason: format!(
                "low credibility ({incentive_credibility:.2}) with {}deception incentive",
                if deception_incentive { "" } else { "no " }
            ),
        }
    };

    SignalingAnalysis {
        sender,
        signal_type,
        deception_incentive,
        incentive_credibility,
        recommended_response,
    }
}

/// Analyze cheap talk credibility.
pub fn analyze_cheap_talk(goal_alignment: f32) -> CheapTalkAnalysis {
    let credible = goal_alignment > 0.3;
    let babbling_risk = if goal_alignment < -0.3 {
        0.8
    } else if goal_alignment < 0.3 {
        0.4
    } else {
        0.1
    };

    CheapTalkAnalysis {
        credible,
        interests_aligned: goal_alignment,
        babbling_risk,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 21c — Strategic Action Planning
// ═══════════════════════════════════════════════════════════════════════

/// Impact of another agent's response on our plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanImpact {
    Beneficial,
    Neutral,
    Detrimental,
    Blocking,
}

/// Predicted response from another agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedResponse {
    pub agent: SymbolId,
    pub likely_action: String,
    pub probability: f32,
    pub based_on: ToMLevel,
    pub impact: PlanImpact,
}

/// Outcome of a social plan.
#[derive(Debug, Clone)]
pub struct SocialOutcome {
    /// Expected goal progress.
    pub goal_progress: f32,
    /// Trust changes per agent.
    pub trust_changes: Vec<(SymbolId, f32)>,
    /// Expected utility of the outcome.
    pub expected_utility: f32,
}

/// Social dynamics assessment.
#[derive(Debug, Clone)]
pub struct SocialDynamics {
    pub cooperators: Vec<(SymbolId, f32)>,
    pub competitors: Vec<(SymbolId, f32)>,
    pub unknown: Vec<SymbolId>,
    pub environment: SocialEnvironment,
}

/// Overall social environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SocialEnvironment {
    Cooperative,
    Mixed,
    Competitive,
    Unknown,
}

impl SocialEnvironment {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Cooperative => "cooperative",
            Self::Mixed => "mixed",
            Self::Competitive => "competitive",
            Self::Unknown => "unknown",
        }
    }
}

/// A coalition of cooperating agents.
#[derive(Debug, Clone)]
pub struct Coalition {
    pub members: Vec<SymbolId>,
    pub shared_goals: Vec<String>,
    pub value: f32,
    pub stable: bool,
}

/// Detect social dynamics from a set of game models.
pub fn detect_dynamics(games: &[(SymbolId, GameType, f32)]) -> SocialDynamics {
    let mut cooperators = Vec::new();
    let mut competitors = Vec::new();
    let mut unknown = Vec::new();

    for &(agent, game_type, alignment) in games {
        match game_type {
            GameType::Cooperative => cooperators.push((agent, alignment)),
            GameType::ZeroSum => competitors.push((agent, alignment)),
            GameType::MixedMotive => {
                if alignment > 0.3 {
                    cooperators.push((agent, alignment));
                } else if alignment < -0.3 {
                    competitors.push((agent, alignment));
                } else {
                    unknown.push(agent);
                }
            }
            _ => unknown.push(agent),
        }
    }

    let environment = if competitors.is_empty() && !cooperators.is_empty() {
        SocialEnvironment::Cooperative
    } else if cooperators.is_empty() && !competitors.is_empty() {
        SocialEnvironment::Competitive
    } else if !cooperators.is_empty() && !competitors.is_empty() {
        SocialEnvironment::Mixed
    } else {
        SocialEnvironment::Unknown
    };

    SocialDynamics {
        cooperators,
        competitors,
        unknown,
        environment,
    }
}

/// Should we cooperate with another agent?
///
/// Cooperate if: cooperative game + trust > threshold, or mixed-motive + high alignment.
pub fn should_cooperate(game_type: GameType, trust: f32, alignment: f32) -> bool {
    match game_type {
        GameType::Cooperative => trust > 0.3,
        GameType::MixedMotive => trust > 0.5 && alignment > 0.3,
        GameType::ZeroSum => false,
        _ => trust > 0.6,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    // ── 21a: Game classification ──────────────────────────────────

    #[test]
    fn classify_zero_sum() {
        let matrix = vec![
            vec![(1.0, -1.0), (-1.0, 1.0)],
            vec![(-1.0, 1.0), (1.0, -1.0)],
        ];
        assert_eq!(classify_game(&matrix), GameType::ZeroSum);
    }

    #[test]
    fn classify_cooperative() {
        let matrix = vec![
            vec![(3.0, 3.0), (0.0, 0.0)],
            vec![(0.0, 0.0), (1.0, 1.0)],
        ];
        assert_eq!(classify_game(&matrix), GameType::Cooperative);
    }

    #[test]
    fn classify_mixed_motive() {
        // Prisoner's dilemma structure.
        let matrix = vec![
            vec![(3.0, 3.0), (0.0, 5.0)],
            vec![(5.0, 0.0), (1.0, 1.0)],
        ];
        assert_eq!(classify_game(&matrix), GameType::MixedMotive);
    }

    #[test]
    fn classify_empty() {
        assert_eq!(classify_game(&[]), GameType::Unknown);
    }

    // ── Nash equilibria ───────────────────────────────────────────

    #[test]
    fn nash_equilibrium_prisoners_dilemma() {
        // PD: (C,C)=3,3  (C,D)=0,5  (D,C)=5,0  (D,D)=1,1
        let matrix = vec![
            vec![(3.0, 3.0), (0.0, 5.0)],
            vec![(5.0, 0.0), (1.0, 1.0)],
        ];
        let eq = find_nash_equilibria(&matrix);
        assert_eq!(eq, vec![(1, 1)]); // (Defect, Defect) is the Nash equilibrium
    }

    #[test]
    fn nash_equilibrium_coordination() {
        // Pure coordination: both prefer same choices.
        let matrix = vec![
            vec![(2.0, 2.0), (0.0, 0.0)],
            vec![(0.0, 0.0), (1.0, 1.0)],
        ];
        let eq = find_nash_equilibria(&matrix);
        assert_eq!(eq.len(), 2); // (0,0) and (1,1) are both NE
    }

    // ── Dominant strategies ───────────────────────────────────────

    #[test]
    fn dominant_strategy_pd() {
        let matrix = vec![
            vec![(3.0, 3.0), (0.0, 5.0)],
            vec![(5.0, 0.0), (1.0, 1.0)],
        ];
        // Defect dominates cooperate in PD.
        assert_eq!(find_dominant_strategy(&matrix, true), Some(1));
    }

    // ── Prisoners dilemma detection ───────────────────────────────

    #[test]
    fn prisoners_dilemma_detection() {
        let matrix = vec![
            vec![(3.0, 3.0), (0.0, 5.0)],
            vec![(5.0, 0.0), (1.0, 1.0)],
        ];
        assert!(is_prisoners_dilemma(&matrix));

        let not_pd = vec![
            vec![(2.0, 2.0), (0.0, 0.0)],
            vec![(0.0, 0.0), (1.0, 1.0)],
        ];
        assert!(!is_prisoners_dilemma(&not_pd));
    }

    // ── Payoff factors ────────────────────────────────────────────

    #[test]
    fn payoff_factors_total() {
        let f = PayoffFactors {
            goal_progress: 0.8,
            trust_change: 0.5,
            information_gain: 0.3,
            resource_cost: 0.1,
        };
        let total = f.total();
        // 0.8*0.4 + 0.5*0.2 + 0.3*0.2 - 0.1*0.2 = 0.32 + 0.10 + 0.06 - 0.02 = 0.46
        assert!((total - 0.46).abs() < 0.01);
    }

    // ── 21b: Signaling ────────────────────────────────────────────

    #[test]
    fn signal_verifiable_high_credibility() {
        let analysis = analyze_signal(
            sym(1),
            SignalType::VerifiableClaim,
            0.5,
            0.8,
        );
        assert!(analysis.incentive_credibility > 0.9);
        assert!(!analysis.deception_incentive);
    }

    #[test]
    fn signal_cheap_talk_with_deception_incentive() {
        let analysis = analyze_signal(
            sym(1),
            SignalType::CheapTalk,
            -0.5, // misaligned goals
            0.7,
        );
        assert!(analysis.deception_incentive);
        assert!(analysis.incentive_credibility < 0.5);
    }

    #[test]
    fn cheap_talk_aligned_credible() {
        let ct = analyze_cheap_talk(0.8);
        assert!(ct.credible);
        assert!(ct.babbling_risk < 0.2);
    }

    #[test]
    fn cheap_talk_misaligned_not_credible() {
        let ct = analyze_cheap_talk(-0.5);
        assert!(!ct.credible);
        assert!(ct.babbling_risk > 0.5);
    }

    #[test]
    fn level_k_estimation() {
        let low = LevelKModel::estimate(1);
        assert_eq!(low.estimated_level, ToMLevel::Level0);

        let mid = LevelKModel::estimate(5);
        assert_eq!(mid.estimated_level, ToMLevel::Level1);

        let high = LevelKModel::estimate(15);
        assert_eq!(high.estimated_level, ToMLevel::Level2);
    }

    // ── 21c: Strategic planning ───────────────────────────────────

    #[test]
    fn detect_cooperative_dynamics() {
        let games = vec![
            (sym(1), GameType::Cooperative, 0.8),
            (sym(2), GameType::Cooperative, 0.6),
        ];
        let dynamics = detect_dynamics(&games);
        assert_eq!(dynamics.environment, SocialEnvironment::Cooperative);
        assert_eq!(dynamics.cooperators.len(), 2);
    }

    #[test]
    fn detect_mixed_dynamics() {
        let games = vec![
            (sym(1), GameType::Cooperative, 0.8),
            (sym(2), GameType::ZeroSum, -0.5),
        ];
        let dynamics = detect_dynamics(&games);
        assert_eq!(dynamics.environment, SocialEnvironment::Mixed);
    }

    #[test]
    fn should_cooperate_cooperative_game() {
        assert!(should_cooperate(GameType::Cooperative, 0.5, 0.8));
        assert!(!should_cooperate(GameType::ZeroSum, 0.9, 0.9));
    }

    #[test]
    fn game_type_labels() {
        assert_eq!(GameType::Cooperative.as_label(), "cooperative");
        assert_eq!(GameType::ZeroSum.as_label(), "zero-sum");
    }

    #[test]
    fn social_environment_labels() {
        assert_eq!(SocialEnvironment::Mixed.as_label(), "mixed");
    }
}
