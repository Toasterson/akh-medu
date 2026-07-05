//! Phase 34a/34c/34e — Response strategy taxonomy, selector, content filter,
//! and shadow filter.
//!
//! ELIZA's 1966 insight: the CATEGORY of a response matters more than the
//! specific words. This module gives the agent the ability to CHOOSE how to
//! communicate — explain, hedge, challenge, empathise — driven by answer
//! quality (confidence, evidence conflict), psyche (OCEAN + archetypes), and
//! conversation trajectory (Phase 34d). Selection is a symbolic scoring
//! function: fast, deterministic, debuggable — never a neural classifier.
//!
//! Three stages (ADR 044):
//! - Stage 1 (this module): WHAT TYPE of response — [`StrategySelector`]
//! - Stage 2 (this module): WHAT CONTENT — [`select_content`]
//! - Stage 3 (`grammar::style_transform`): HOW to say it
//!
//! The shadow filter (34e) sits between stages 1 and 2: shadow bias patterns
//! can soften an aggressive strategy and journal the inhibition.
//!
//! Plan: docs/ai/plans/2026-03-26-phase34-communicative-intelligence.md
//! ADR:  docs/ai/decisions/044-communicative-intelligence.md

use serde::{Deserialize, Serialize};

use crate::compartment::psyche::{OceanProfile, Psyche, ShadowPattern};
use crate::symbol::SymbolId;

use super::conversation::GroundedTriple;
use super::trajectory::DiscourseMode;

// ── Confidence level ─────────────────────────────────────────────────────

/// Discretised confidence bands, for explicit calibration statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    VeryLow,
    Low,
    Moderate,
    High,
    VeryHigh,
}

impl ConfidenceLevel {
    /// Discretise a scalar confidence into a band.
    pub fn from_confidence(conf: f32) -> Self {
        match conf {
            c if c < 0.2 => Self::VeryLow,
            c if c < 0.4 => Self::Low,
            c if c < 0.7 => Self::Moderate,
            c if c < 0.9 => Self::High,
            _ => Self::VeryHigh,
        }
    }

    /// Human-readable label.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::VeryLow => "very low",
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::High => "high",
            Self::VeryHigh => "very high",
        }
    }
}

// ── Query type & sentiment ───────────────────────────────────────────────

/// Coarse classification of what kind of question is being answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QueryType {
    /// "What is X?" — a factual lookup.
    #[default]
    Factual,
    /// "Why does X…?" — a causal question.
    Causal,
    /// "How do I…?" — a procedural question.
    Procedural,
    /// "Is there…?" / "Does X exist?" — existential.
    Existential,
    /// "Who are you?" / "How confident are you?" — about the agent itself.
    Meta,
    /// Greeting / small talk.
    Social,
}

/// Coarse user sentiment, derived from the affective system (Phase 23).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sentiment {
    Positive,
    Neutral,
    Negative,
}

impl Sentiment {
    /// Map an affective valence in [-1.0, 1.0] to a sentiment band.
    pub fn from_valence(valence: f32) -> Self {
        if valence < -0.25 {
            Self::Negative
        } else if valence > 0.25 {
            Self::Positive
        } else {
            Self::Neutral
        }
    }
}

// ── Strategy taxonomy ────────────────────────────────────────────────────

/// The broad category a strategy belongs to (ADR 044 taxonomy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyCategory {
    /// State/organise known content.
    Knowledge,
    /// Communicate confidence / evidence state.
    Epistemic,
    /// Manage the interaction itself.
    Conversational,
    /// Communicate about the agent's own reasoning.
    MetaCognitive,
}

/// A response strategy — the TYPE of response the agent chooses to give.
///
/// Payloads carry the symbolic content the composer needs to frame the
/// response. Topic references are label strings (already resolved from the
/// KG by the grounding pipeline) — the composition layer is label-based.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseStrategy {
    // ── Knowledge strategies ────────────────────────────────────────
    /// State facts concisely.
    DirectAnswer,
    /// Provide a reasoning chain / context / "why".
    Explanation { depth: u8 },
    /// Place the answer in a broader framework.
    Contextualization { broader_topic: String },
    /// Relate the answer to something the user already knows.
    Comparison { compare_to: String },
    /// Express through a mapping from a different domain.
    ///
    /// NOTE: binary-VSA analogy is degenerate (known limitation), so the
    /// selector scores this conservatively — it only surfaces when a topic
    /// is being re-explained and openness is high.
    Analogy { source_domain: String },

    // ── Epistemic strategies ────────────────────────────────────────
    /// High belief, multiple sources — assert without hedging.
    ConfidentAssertion,
    /// Moderate belief, limited evidence — hedge.
    HedgedStatement { reason: String },
    /// Low confidence — be honest about limits.
    UncertaintyAdmission { confidence: f32, what_is_missing: String },
    /// Conflicting sources — present both sides.
    EvidenceWeighing { conflict: f32 },
    /// No triples found — optionally offer to learn.
    KnowledgeGap { topic: String, can_learn: bool },

    // ── Conversational strategies ───────────────────────────────────
    /// Ambiguous query — ask for more information.
    ClarificationRequest { ambiguity: String, options: Vec<String> },
    /// A related interesting topic is available — invite exploration.
    ExplorationPrompt { related_topic: String, connection: String },
    /// Counter-evidence exists — push back.
    Challenge { reasoning: String },
    /// Can't answer X but know about adjacent Y.
    Redirect { from_topic: String, to_topic: String, reason: String },
    /// Recognised frustration/confusion — acknowledge it.
    Empathy { recognized_state: String },

    // ── Meta-cognitive strategies ───────────────────────────────────
    /// Discovered a new connection while answering.
    SelfReflection { new_connection: String },
    /// Be explicit about the certainty level.
    ConfidenceCalibration { level: ConfidenceLevel, summary: String },
    /// Show the derivation chain (provenance-to-prose).
    ReasoningTrace { steps: Vec<SymbolId> },
}

impl ResponseStrategy {
    /// Stable short name for journaling, trajectory history, and tests.
    pub fn name(&self) -> &'static str {
        match self {
            Self::DirectAnswer => "DirectAnswer",
            Self::Explanation { .. } => "Explanation",
            Self::Contextualization { .. } => "Contextualization",
            Self::Comparison { .. } => "Comparison",
            Self::Analogy { .. } => "Analogy",
            Self::ConfidentAssertion => "ConfidentAssertion",
            Self::HedgedStatement { .. } => "HedgedStatement",
            Self::UncertaintyAdmission { .. } => "UncertaintyAdmission",
            Self::EvidenceWeighing { .. } => "EvidenceWeighing",
            Self::KnowledgeGap { .. } => "KnowledgeGap",
            Self::ClarificationRequest { .. } => "ClarificationRequest",
            Self::ExplorationPrompt { .. } => "ExplorationPrompt",
            Self::Challenge { .. } => "Challenge",
            Self::Redirect { .. } => "Redirect",
            Self::Empathy { .. } => "Empathy",
            Self::SelfReflection { .. } => "SelfReflection",
            Self::ConfidenceCalibration { .. } => "ConfidenceCalibration",
            Self::ReasoningTrace { .. } => "ReasoningTrace",
        }
    }

    /// Which taxonomy category this strategy belongs to.
    pub fn category(&self) -> StrategyCategory {
        match self {
            Self::DirectAnswer
            | Self::Explanation { .. }
            | Self::Contextualization { .. }
            | Self::Comparison { .. }
            | Self::Analogy { .. } => StrategyCategory::Knowledge,
            Self::ConfidentAssertion
            | Self::HedgedStatement { .. }
            | Self::UncertaintyAdmission { .. }
            | Self::EvidenceWeighing { .. }
            | Self::KnowledgeGap { .. } => StrategyCategory::Epistemic,
            Self::ClarificationRequest { .. }
            | Self::ExplorationPrompt { .. }
            | Self::Challenge { .. }
            | Self::Redirect { .. }
            | Self::Empathy { .. } => StrategyCategory::Conversational,
            Self::SelfReflection { .. }
            | Self::ConfidenceCalibration { .. }
            | Self::ReasoningTrace { .. } => StrategyCategory::MetaCognitive,
        }
    }
}

// ── Strategy context ─────────────────────────────────────────────────────

/// Everything the selector evaluates for one response.
///
/// Built once per query by the communicative pipeline (Phase 34f) from the
/// grounded response, the discourse trajectory (34d), and the affective
/// system (Phase 23).
#[derive(Debug, Clone)]
pub struct StrategyContext {
    /// The subject the query is about (resolved label).
    pub subject: String,
    /// What kind of question this is.
    pub query_type: QueryType,
    /// How many grounded triples back the answer.
    pub answer_triples: usize,
    /// Aggregate confidence across supporting triples [0.0, 1.0].
    pub confidence: f32,
    /// Dempster-Shafer belief interval `[bel, pl]` for the answer, if computed.
    pub belief_interval: Option<(f32, f32)>,
    /// Evidence conflict: 0.0 = none, 1.0 = total contradiction.
    pub evidence_conflict: f32,
    /// Average source reliability (Admiralty-derived), if known.
    pub source_reliability: Option<f32>,
    /// How many response turns this conversation has had.
    pub conversation_turns: usize,
    /// Whether this topic was already answered before (trajectory 34d).
    pub topic_repeated: bool,
    /// User sentiment from the affective system (Phase 23).
    pub user_sentiment: Option<Sentiment>,
    /// Current discourse mode (trajectory 34d).
    pub discourse_mode: DiscourseMode,
    /// Estimated user engagement [0.0, 1.0] (trajectory 34d).
    pub engagement: f32,
    /// A related topic worth exploring: `(topic_label, connection_label)`.
    pub related_topic: Option<(String, String)>,
    /// A comparison target the user likely knows, if one was found.
    pub comparison_target: Option<String>,
    /// Detected ambiguity: `(description, candidate options)`.
    pub ambiguity: Option<(String, Vec<String>)>,
    /// Knowledge gaps surfaced during grounding.
    pub gaps: Vec<String>,
    /// Whether provenance records back the answer (enables ReasoningTrace).
    pub provenance: Vec<SymbolId>,
    /// A new connection discovered while answering (enables SelfReflection).
    pub new_connection: Option<String>,
    /// The strategy used on the previous turn (variety pressure).
    pub last_strategy: Option<String>,
}

impl StrategyContext {
    /// A neutral context for an answered factual query. Callers override
    /// the fields they have real signals for.
    pub fn for_answer(subject: impl Into<String>, answer_triples: usize, confidence: f32) -> Self {
        Self {
            subject: subject.into(),
            query_type: QueryType::Factual,
            answer_triples,
            confidence: confidence.clamp(0.0, 1.0),
            belief_interval: None,
            evidence_conflict: 0.0,
            source_reliability: None,
            conversation_turns: 0,
            topic_repeated: false,
            user_sentiment: None,
            discourse_mode: DiscourseMode::Interrogative,
            engagement: 0.6,
            related_topic: None,
            comparison_target: None,
            ambiguity: None,
            gaps: Vec::new(),
            provenance: Vec::new(),
            new_connection: None,
            last_strategy: None,
        }
    }
}

/// Compute an evidence-conflict estimate from raw triple confidences.
///
/// Approximation of Dempster-Shafer pairwise conflict: confidences ≥ 0.5 are
/// treated as evidence FOR, confidences < 0.5 as (weak) evidence AGAINST.
/// The result is the maximum pairwise conflict factor K = m1(T)·m2(F) +
/// m1(F)·m2(T). Full per-claim assessment via `EvidenceIndex` (Phase 17) is
/// the upgrade path once claims carry evidence items.
pub fn evidence_conflict(confidences: &[f32]) -> f32 {
    use super::evidence::MassFunction;
    if confidences.len() < 2 {
        return 0.0;
    }
    let masses: Vec<MassFunction> = confidences
        .iter()
        .map(|&c| {
            let c = c.clamp(0.0, 1.0);
            if c >= 0.5 {
                MassFunction::from_confidence(c)
            } else {
                // Weak evidence against: mass on FALSE proportional to how
                // far below 0.5 the confidence sits.
                MassFunction::new(0.0, 1.0 - c)
            }
        })
        .collect();
    let mut max_k: f32 = 0.0;
    for i in 0..masses.len() {
        for j in (i + 1)..masses.len() {
            max_k = max_k.max(masses[i].conflict_with(&masses[j]));
        }
    }
    max_k.clamp(0.0, 1.0)
}

// ── Selector ─────────────────────────────────────────────────────────────

/// Configuration for the strategy selector.
#[derive(Debug, Clone)]
pub struct StrategySelectorConfig {
    /// Minimum score for a conversational strategy to ride along as a
    /// companion move (e.g. `DirectAnswer` + `Empathy`).
    pub companion_threshold: f32,
    /// Penalty applied when a candidate equals the previous turn's strategy.
    pub repetition_penalty: f32,
}

impl Default for StrategySelectorConfig {
    fn default() -> Self {
        Self {
            companion_threshold: 0.30,
            repetition_penalty: 0.15,
        }
    }
}

/// The outcome of strategy selection: a primary strategy, an optional
/// companion conversational move, and the full deliberation for
/// introspection (Phase 29c psyche exposure).
#[derive(Debug, Clone)]
pub struct StrategyChoice {
    /// The winning strategy.
    pub primary: ResponseStrategy,
    /// Score of the winner.
    pub score: f32,
    /// A conversational companion move (e.g. Empathy, ExplorationPrompt)
    /// that scored above the companion threshold, from a different category.
    pub companion: Option<ResponseStrategy>,
    /// All scored candidates `(name, score)`, sorted descending — the
    /// selector's reasoning, exposed for debugging and psyche introspection.
    pub deliberation: Vec<(&'static str, f32)>,
}

/// Symbolic response-strategy selector.
///
/// `score(strategy, ctx) = Σ signal_i × psyche_weight_i × affinity_i` — a
/// plain scoring function over the [`StrategyContext`], modulated by the
/// psyche's OCEAN profile and archetype weights. Deterministic: ties break
/// on strategy name.
pub struct StrategySelector {
    ocean: OceanProfile,
    sage: f32,
    healer: f32,
    guardian: f32,
    explorer: f32,
    config: StrategySelectorConfig,
}

impl StrategySelector {
    /// Build a selector from a psyche (uses the effective OCEAN profile).
    pub fn from_psyche(psyche: &Psyche) -> Self {
        Self::with_config(psyche, StrategySelectorConfig::default())
    }

    /// Build a selector with explicit configuration.
    pub fn with_config(psyche: &Psyche, config: StrategySelectorConfig) -> Self {
        Self {
            ocean: psyche.effective_ocean(),
            sage: psyche.archetypes.sage,
            healer: psyche.archetypes.healer,
            guardian: psyche.archetypes.guardian,
            explorer: psyche.archetypes.explorer,
            config,
        }
    }

    /// Select the best response strategy for this context.
    pub fn select(&self, ctx: &StrategyContext) -> StrategyChoice {
        let mut scored: Vec<(ResponseStrategy, f32)> = self
            .candidates(ctx)
            .into_iter()
            .map(|s| {
                let mut score = self.score(&s, ctx);
                if ctx.last_strategy.as_deref() == Some(s.name()) {
                    score -= self.config.repetition_penalty;
                }
                (s, score)
            })
            .collect();

        // Deterministic order: score descending, then name ascending.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.name().cmp(b.0.name()))
        });

        let deliberation: Vec<(&'static str, f32)> =
            scored.iter().map(|(s, sc)| (s.name(), *sc)).collect();

        let (primary, score) = scored
            .first()
            .cloned()
            .unwrap_or((ResponseStrategy::DirectAnswer, 0.0));

        // Companion: best conversational strategy above threshold, from a
        // different category than the primary.
        let companion = scored
            .iter()
            .find(|(s, sc)| {
                *sc >= self.config.companion_threshold
                    && s.category() == StrategyCategory::Conversational
                    && s.category() != primary.category()
                    && s.name() != primary.name()
            })
            .map(|(s, _)| s.clone());

        StrategyChoice {
            primary,
            score,
            companion,
            deliberation,
        }
    }

    /// Enumerate candidate strategies for this context (with payloads).
    fn candidates(&self, ctx: &StrategyContext) -> Vec<ResponseStrategy> {
        let mut out = vec![
            ResponseStrategy::DirectAnswer,
            ResponseStrategy::Explanation { depth: 2 },
            ResponseStrategy::ConfidentAssertion,
            ResponseStrategy::HedgedStatement {
                reason: "the evidence is limited".to_string(),
            },
            ResponseStrategy::UncertaintyAdmission {
                confidence: ctx.confidence,
                what_is_missing: ctx
                    .gaps
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "corroborating evidence".to_string()),
            },
            ResponseStrategy::EvidenceWeighing {
                conflict: ctx.evidence_conflict,
            },
            ResponseStrategy::ConfidenceCalibration {
                level: ConfidenceLevel::from_confidence(ctx.confidence),
                summary: format!("{} supporting fact(s)", ctx.answer_triples),
            },
            ResponseStrategy::ReasoningTrace {
                steps: ctx.provenance.clone(),
            },
        ];

        if ctx.answer_triples == 0 {
            out.push(ResponseStrategy::KnowledgeGap {
                topic: ctx.subject.clone(),
                can_learn: true,
            });
        }
        if let Some((topic, connection)) = &ctx.related_topic {
            out.push(ResponseStrategy::Contextualization {
                broader_topic: topic.clone(),
            });
            out.push(ResponseStrategy::ExplorationPrompt {
                related_topic: topic.clone(),
                connection: connection.clone(),
            });
            if ctx.answer_triples < 2 {
                out.push(ResponseStrategy::Redirect {
                    from_topic: ctx.subject.clone(),
                    to_topic: topic.clone(),
                    reason: format!("I know more about {topic}"),
                });
            }
        }
        if let Some(target) = &ctx.comparison_target {
            out.push(ResponseStrategy::Comparison {
                compare_to: target.clone(),
            });
        }
        if ctx.topic_repeated {
            out.push(ResponseStrategy::Analogy {
                source_domain: ctx
                    .related_topic
                    .as_ref()
                    .map(|(t, _)| t.clone())
                    .unwrap_or_else(|| "a familiar domain".to_string()),
            });
        }
        if let Some((ambiguity, options)) = &ctx.ambiguity {
            out.push(ResponseStrategy::ClarificationRequest {
                ambiguity: ambiguity.clone(),
                options: options.clone(),
            });
        }
        if ctx.evidence_conflict > 0.3 {
            out.push(ResponseStrategy::Challenge {
                reasoning: "the knowledge graph holds counter-evidence".to_string(),
            });
        }
        out.push(ResponseStrategy::Empathy {
            recognized_state: match ctx.user_sentiment {
                Some(Sentiment::Negative) => "some frustration".to_string(),
                _ => "your interest".to_string(),
            },
        });
        if let Some(conn) = &ctx.new_connection {
            out.push(ResponseStrategy::SelfReflection {
                new_connection: conn.clone(),
            });
        }
        out
    }

    /// Score one candidate strategy against the context.
    fn score(&self, strategy: &ResponseStrategy, ctx: &StrategyContext) -> f32 {
        let o = &self.ocean;
        match strategy {
            ResponseStrategy::DirectAnswer => {
                let mut s = 0.25 + ctx.confidence * 0.25 + (1.0 - o.openness) * 0.1;
                if ctx.discourse_mode == DiscourseMode::Interrogative {
                    s += 0.05;
                }
                if ctx.topic_repeated {
                    s -= 0.5; // same answer twice helps nobody
                }
                s
            }
            ResponseStrategy::Explanation { .. } => {
                let mut s = self.sage * 0.3
                    + o.conscientiousness * 0.2
                    + ctx.confidence * 0.15;
                if ctx.answer_triples > 3 {
                    s += 0.2;
                }
                if ctx.topic_repeated {
                    s -= 0.4;
                }
                s
            }
            ResponseStrategy::Contextualization { .. } => {
                let mut s = o.openness * 0.2 + self.explorer * 0.15;
                if ctx.related_topic.is_some() {
                    s += 0.25;
                }
                if ctx.topic_repeated {
                    s += 0.25; // a fresh frame for a repeated question
                }
                s
            }
            ResponseStrategy::Comparison { .. } => {
                if ctx.comparison_target.is_some() {
                    0.2 + o.openness * 0.15 + self.sage * 0.1
                } else {
                    0.0
                }
            }
            ResponseStrategy::Analogy { .. } => {
                // Conservative: binary-VSA analogy is degenerate, so analogy
                // only surfaces as a re-explanation device.
                let mut s = o.openness * 0.25 - 0.2;
                if ctx.topic_repeated {
                    s += 0.3;
                }
                s
            }
            ResponseStrategy::ConfidentAssertion => {
                let conf_excess = ((ctx.confidence - 0.7).max(0.0) / 0.3).min(1.0);
                conf_excess * 0.35
                    + (1.0 - o.neuroticism) * 0.1
                    + ctx.source_reliability.unwrap_or(0.5) * 0.1
                    - ctx.evidence_conflict * 0.3
            }
            ResponseStrategy::HedgedStatement { .. } => {
                // Peaks for mid-band confidence (~0.6).
                let band = (1.0 - (ctx.confidence - 0.6).abs() * 2.5).max(0.0);
                band * 0.3 + o.neuroticism * 0.15 + o.agreeableness * 0.1
            }
            ResponseStrategy::UncertaintyAdmission { .. } => {
                (1.0 - ctx.confidence) * 0.5
                    + o.neuroticism * 0.2
                    + self.healer * 0.15
                    + ctx.evidence_conflict * 0.3
            }
            ResponseStrategy::EvidenceWeighing { .. } => {
                let mut s = ctx.evidence_conflict * 0.55 + o.conscientiousness * 0.1;
                if ctx.discourse_mode == DiscourseMode::Argumentative {
                    s += 0.25;
                }
                s
            }
            ResponseStrategy::KnowledgeGap { .. } => {
                if ctx.answer_triples == 0 {
                    0.95
                } else {
                    0.0
                }
            }
            ResponseStrategy::ClarificationRequest { .. } => {
                if ctx.ambiguity.is_some() {
                    0.5 + o.conscientiousness * 0.1
                } else {
                    0.0
                }
            }
            ResponseStrategy::ExplorationPrompt { .. } => {
                let mut s = self.explorer * 0.3 + o.extraversion * 0.1;
                if ctx.discourse_mode == DiscourseMode::Exploratory {
                    s += 0.25;
                }
                if ctx.engagement < 0.4 {
                    s += 0.2; // re-engage a drifting user
                }
                if ctx.related_topic.is_some() {
                    s += 0.1;
                }
                s
            }
            ResponseStrategy::Challenge { .. } => {
                if ctx.evidence_conflict > 0.3
                    && ctx.discourse_mode == DiscourseMode::Argumentative
                {
                    0.25 + (1.0 - o.agreeableness) * 0.35 + ctx.evidence_conflict * 0.25
                } else if ctx.evidence_conflict > 0.5 {
                    (1.0 - o.agreeableness) * 0.3 + ctx.evidence_conflict * 0.2
                } else {
                    0.0
                }
            }
            ResponseStrategy::Redirect { .. } => {
                if ctx.answer_triples < 2 && ctx.related_topic.is_some() {
                    0.4 + self.guardian * 0.05
                } else {
                    0.0
                }
            }
            ResponseStrategy::Empathy { .. } => {
                if ctx.user_sentiment == Some(Sentiment::Negative) {
                    0.35 + self.healer * 0.3 + o.agreeableness * 0.2
                } else {
                    // Warm psyches still check in gently.
                    self.healer * 0.2 + o.agreeableness * 0.15
                }
            }
            ResponseStrategy::SelfReflection { .. } => {
                if ctx.new_connection.is_some() {
                    0.2 + o.openness * 0.15
                } else {
                    0.0
                }
            }
            ResponseStrategy::ConfidenceCalibration { .. } => {
                let mut s = o.conscientiousness * 0.05;
                if ctx.query_type == QueryType::Meta {
                    s += 0.35;
                }
                if let Some((bel, pl)) = ctx.belief_interval {
                    if pl - bel > 0.4 {
                        s += 0.15; // wide interval — worth calibrating aloud
                    }
                }
                s
            }
            ResponseStrategy::ReasoningTrace { .. } => {
                let mut s = o.conscientiousness * 0.1;
                if ctx.query_type == QueryType::Causal {
                    s += 0.35;
                }
                if !ctx.provenance.is_empty() {
                    s += 0.15;
                }
                s
            }
        }
    }
}

// ── 34c: Content filter per strategy ─────────────────────────────────────

/// Content selected for a strategy: the primary triples to verbalize, plus
/// a contrast set for two-sided strategies (EvidenceWeighing, Challenge).
#[derive(Debug, Clone, Default)]
pub struct SelectedContent {
    /// Triples to present as the answer.
    pub primary: Vec<GroundedTriple>,
    /// Counter-evidence / lower-confidence side (EvidenceWeighing, Challenge).
    pub contrast: Vec<GroundedTriple>,
}

/// Select the content that fits the chosen strategy (Stage 2, ADR 044).
///
/// Each strategy implies a different slice of the grounded triples:
/// concise answers take the strongest facts; explanations keep derived
/// facts; evidence weighing partitions into pro/con; interaction-only
/// strategies (clarification, knowledge gap, empathy) carry no triples.
pub fn select_content(
    strategy: &ResponseStrategy,
    all: &[GroundedTriple],
) -> SelectedContent {
    let mut sorted: Vec<GroundedTriple> = all.to_vec();
    // Deterministic: confidence descending, then label ascending.
    sorted.sort_by(|a, b| {
        b.confidence
            .unwrap_or(1.0)
            .partial_cmp(&a.confidence.unwrap_or(1.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.predicate_label.cmp(&b.predicate_label))
            .then_with(|| a.object_label.cmp(&b.object_label))
    });

    match strategy {
        ResponseStrategy::DirectAnswer | ResponseStrategy::ConfidentAssertion => SelectedContent {
            primary: sorted.into_iter().take(3).collect(),
            contrast: Vec::new(),
        },
        ResponseStrategy::Explanation { depth } => {
            // Core facts plus derived facts (provenance-tagged), scaled by depth.
            let take = 5 + (*depth as usize) * 3;
            SelectedContent {
                primary: sorted.into_iter().take(take).collect(),
                contrast: Vec::new(),
            }
        }
        ResponseStrategy::Contextualization { .. } | ResponseStrategy::Comparison { .. } => {
            // Taxonomic relations first — they carry the broader frame.
            let is_taxonomic = |t: &GroundedTriple| {
                matches!(
                    t.predicate_label.as_str(),
                    "is-a" | "subclass-of" | "part-of" | "member-of" | "instance-of" | "similar-to"
                )
            };
            let (mut taxo, rest): (Vec<_>, Vec<_>) =
                sorted.into_iter().partition(|t| is_taxonomic(t));
            taxo.extend(rest.into_iter().take(3));
            SelectedContent {
                primary: taxo.into_iter().take(6).collect(),
                contrast: Vec::new(),
            }
        }
        ResponseStrategy::EvidenceWeighing { .. } | ResponseStrategy::Challenge { .. } => {
            // Partition by confidence: strong evidence vs weak/conflicting.
            let (pro, con): (Vec<_>, Vec<_>) = sorted
                .into_iter()
                .partition(|t| t.confidence.unwrap_or(1.0) >= 0.5);
            SelectedContent {
                primary: pro.into_iter().take(4).collect(),
                contrast: con.into_iter().take(4).collect(),
            }
        }
        ResponseStrategy::ReasoningTrace { .. } => SelectedContent {
            // Derived facts only — those with a derivation tag carry the trace.
            primary: sorted
                .into_iter()
                .filter(|t| t.derivation_tag.is_some())
                .take(6)
                .collect(),
            contrast: Vec::new(),
        },
        ResponseStrategy::KnowledgeGap { .. }
        | ResponseStrategy::ClarificationRequest { .. }
        | ResponseStrategy::Empathy { .. } => SelectedContent::default(),
        ResponseStrategy::Redirect { .. } => SelectedContent {
            primary: sorted.into_iter().take(2).collect(),
            contrast: Vec::new(),
        },
        // HedgedStatement, UncertaintyAdmission, Analogy, ExplorationPrompt,
        // SelfReflection, ConfidenceCalibration: a moderate slice.
        _ => SelectedContent {
            primary: sorted.into_iter().take(5).collect(),
            contrast: Vec::new(),
        },
    }
}

// ── 34e: Shadow filter ───────────────────────────────────────────────────

/// Result of running the shadow filter over a chosen strategy.
#[derive(Debug, Clone)]
pub enum ShadowFilterResult {
    /// No shadow pattern triggered — proceed with the strategy.
    Pass,
    /// A shadow pattern inhibited the strategy — use the replacement and
    /// journal the encounter (visible via Phase 29c psyche exposure).
    Inhibited {
        /// The softer strategy to use instead.
        replacement: ResponseStrategy,
        /// Journal entry: "considered X, chose Y instead, because Z".
        journal_entry: String,
    },
}

/// Shadow integration for the response pipeline (Phase 34e).
///
/// Shadow bias patterns — the psyche's constrained anti-patterns — can veto
/// or soften aggressive communication strategies, mirroring how they bias
/// tool selection. Triggering conditions:
/// 1. An assertive strategy (`Challenge`, `ConfidentAssertion`,
///    `DirectAnswer`, `Redirect`) whose content mentions one of the
///    pattern's `action_triggers` (substring match on triple labels).
/// 2. A `Challenge` in the presence of any high-severity (≥ 0.7) pattern —
///    confrontation is the first thing a strong shadow inhibits.
pub struct ShadowFilter {
    patterns: Vec<ShadowPattern>,
}

impl ShadowFilter {
    /// Build the filter from a psyche's shadow (veto + bias patterns).
    pub fn from_psyche(psyche: &Psyche) -> Self {
        let mut patterns = psyche.shadow.veto_patterns.clone();
        patterns.extend(psyche.shadow.bias_patterns.iter().cloned());
        Self { patterns }
    }

    /// Check whether the strategy + content triggers a shadow pattern.
    pub fn filter(
        &self,
        strategy: &ResponseStrategy,
        content: &[GroundedTriple],
    ) -> ShadowFilterResult {
        if !Self::is_assertive(strategy) {
            return ShadowFilterResult::Pass;
        }

        for pattern in &self.patterns {
            let triggered = self.content_triggers(pattern, content)
                || (matches!(strategy, ResponseStrategy::Challenge { .. })
                    && pattern.severity >= 0.7);
            if triggered {
                let replacement = Self::softer_alternative(strategy);
                let journal_entry = format!(
                    "Shadow: considered {} but pattern '{}' triggered ({}). Softened to {}.",
                    strategy.name(),
                    pattern.name,
                    pattern.explanation,
                    replacement.name(),
                );
                return ShadowFilterResult::Inhibited {
                    replacement,
                    journal_entry,
                };
            }
        }
        ShadowFilterResult::Pass
    }

    /// Strategies the shadow can inhibit.
    fn is_assertive(strategy: &ResponseStrategy) -> bool {
        matches!(
            strategy,
            ResponseStrategy::Challenge { .. }
                | ResponseStrategy::ConfidentAssertion
                | ResponseStrategy::DirectAnswer
                | ResponseStrategy::Redirect { .. }
        )
    }

    /// Substring match of the pattern's action triggers against content labels.
    fn content_triggers(&self, pattern: &ShadowPattern, content: &[GroundedTriple]) -> bool {
        if pattern.action_triggers.is_empty() {
            return false;
        }
        content.iter().any(|t| {
            let text = format!(
                "{} {} {}",
                t.subject_label.to_lowercase(),
                t.predicate_label.to_lowercase(),
                t.object_label.to_lowercase()
            );
            pattern
                .action_triggers
                .iter()
                .any(|trig| text.contains(&trig.to_lowercase()))
        })
    }

    /// The softer alternative for an inhibited strategy.
    fn softer_alternative(strategy: &ResponseStrategy) -> ResponseStrategy {
        match strategy {
            ResponseStrategy::Challenge { .. } => ResponseStrategy::EvidenceWeighing {
                conflict: 0.5,
            },
            _ => ResponseStrategy::HedgedStatement {
                reason: "a shadow pattern advised caution".to_string(),
            },
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compartment::psyche::{ArchetypeWeights, OceanProfile};

    fn psyche_with(
        sage: f32,
        healer: f32,
        guardian: f32,
        explorer: f32,
        ocean: [f32; 5],
    ) -> Psyche {
        let mut p = Psyche::default();
        p.archetypes = ArchetypeWeights {
            sage,
            healer,
            guardian,
            explorer,
        };
        p.ocean = Some(OceanProfile {
            openness: ocean[0],
            conscientiousness: ocean[1],
            extraversion: ocean[2],
            agreeableness: ocean[3],
            neuroticism: ocean[4],
        });
        p
    }

    fn sage_psyche() -> Psyche {
        psyche_with(0.9, 0.4, 0.4, 0.4, [0.7, 0.8, 0.4, 0.6, 0.3])
    }

    fn explorer_psyche() -> Psyche {
        psyche_with(0.4, 0.4, 0.3, 0.9, [0.9, 0.4, 0.7, 0.5, 0.2])
    }

    fn healer_psyche() -> Psyche {
        psyche_with(0.4, 0.9, 0.4, 0.4, [0.5, 0.6, 0.5, 0.9, 0.5])
    }

    fn gt(pred: &str, obj: &str, conf: f32) -> GroundedTriple {
        GroundedTriple {
            subject_label: "dog".into(),
            predicate_label: pred.into(),
            object_label: obj.into(),
            subject_id: None,
            predicate_id: None,
            object_id: None,
            confidence: Some(conf),
            derivation_tag: None,
        }
    }

    fn rich_ctx() -> StrategyContext {
        let mut ctx = StrategyContext::for_answer("dog", 12, 0.92);
        ctx.related_topic = Some(("mammal".into(), "is-a".into()));
        ctx
    }

    #[test]
    fn sage_picks_explanation() {
        let sel = StrategySelector::from_psyche(&sage_psyche());
        let choice = sel.select(&rich_ctx());
        assert_eq!(choice.primary.name(), "Explanation");
    }

    #[test]
    fn explorer_picks_contextual_frame_with_exploration_companion() {
        let sel = StrategySelector::from_psyche(&explorer_psyche());
        let choice = sel.select(&rich_ctx());
        assert_eq!(choice.primary.name(), "Contextualization");
        assert_eq!(
            choice.companion.as_ref().map(|c| c.name()),
            Some("ExplorationPrompt")
        );
    }

    #[test]
    fn healer_gets_empathy_companion() {
        let sel = StrategySelector::from_psyche(&healer_psyche());
        let choice = sel.select(&rich_ctx());
        assert_eq!(
            choice.companion.as_ref().map(|c| c.name()),
            Some("Empathy"),
            "warm psyche should check in even without negative sentiment"
        );
    }

    #[test]
    fn sage_has_no_companion() {
        let sel = StrategySelector::from_psyche(&sage_psyche());
        let choice = sel.select(&rich_ctx());
        assert!(choice.companion.is_none());
    }

    #[test]
    fn low_confidence_admits_uncertainty() {
        let sel = StrategySelector::from_psyche(&healer_psyche());
        let mut ctx = StrategyContext::for_answer("dog", 2, 0.25);
        ctx.gaps = vec!["diet".into()];
        let choice = sel.select(&ctx);
        assert_eq!(choice.primary.name(), "UncertaintyAdmission");
    }

    #[test]
    fn conflict_in_argument_weighs_evidence() {
        let sel = StrategySelector::from_psyche(&sage_psyche());
        let mut ctx = StrategyContext::for_answer("dog", 6, 0.6);
        ctx.evidence_conflict = 0.8;
        ctx.discourse_mode = DiscourseMode::Argumentative;
        let choice = sel.select(&ctx);
        assert_eq!(choice.primary.name(), "EvidenceWeighing");
    }

    #[test]
    fn disagreeable_psyche_challenges() {
        let p = psyche_with(0.5, 0.2, 0.4, 0.5, [0.6, 0.6, 0.6, 0.1, 0.2]);
        let sel = StrategySelector::from_psyche(&p);
        let mut ctx = StrategyContext::for_answer("dog", 6, 0.7);
        ctx.evidence_conflict = 0.6;
        ctx.discourse_mode = DiscourseMode::Argumentative;
        let choice = sel.select(&ctx);
        assert_eq!(choice.primary.name(), "Challenge");
    }

    #[test]
    fn repeated_topic_switches_strategy() {
        let sel = StrategySelector::from_psyche(&sage_psyche());
        let first = sel.select(&rich_ctx());
        let mut again = rich_ctx();
        again.topic_repeated = true;
        again.last_strategy = Some(first.primary.name().to_string());
        let second = sel.select(&again);
        assert_ne!(
            first.primary.name(),
            second.primary.name(),
            "asking the same question twice must not yield the same strategy"
        );
    }

    #[test]
    fn no_triples_yields_knowledge_gap() {
        let sel = StrategySelector::from_psyche(&sage_psyche());
        let ctx = StrategyContext::for_answer("qux", 0, 0.0);
        let choice = sel.select(&ctx);
        assert_eq!(choice.primary.name(), "KnowledgeGap");
    }

    #[test]
    fn negative_sentiment_boosts_empathy() {
        let sel = StrategySelector::from_psyche(&healer_psyche());
        let mut ctx = rich_ctx();
        ctx.user_sentiment = Some(Sentiment::Negative);
        let choice = sel.select(&ctx);
        // Empathy should at least ride along as the companion.
        let has_empathy = choice.primary.name() == "Empathy"
            || choice.companion.as_ref().map(|c| c.name()) == Some("Empathy");
        assert!(has_empathy);
    }

    #[test]
    fn deliberation_is_sorted_and_exposed() {
        let sel = StrategySelector::from_psyche(&sage_psyche());
        let choice = sel.select(&rich_ctx());
        assert!(choice.deliberation.len() >= 8);
        for w in choice.deliberation.windows(2) {
            assert!(w[0].1 >= w[1].1, "deliberation must be sorted descending");
        }
        assert_eq!(choice.deliberation[0].0, choice.primary.name());
    }

    // ── Content filter (34c) ────────────────────────────────────────

    #[test]
    fn direct_answer_takes_top_three() {
        let triples = vec![
            gt("is-a", "mammal", 0.95),
            gt("has-a", "fur", 0.9),
            gt("similar-to", "wolf", 0.85),
            gt("has-property", "loyal", 0.6),
            gt("part-of", "pack", 0.5),
        ];
        let content = select_content(&ResponseStrategy::DirectAnswer, &triples);
        assert_eq!(content.primary.len(), 3);
        assert_eq!(content.primary[0].object_label, "mammal");
        assert!(content.contrast.is_empty());
    }

    #[test]
    fn evidence_weighing_partitions_pro_con() {
        let triples = vec![
            gt("is-a", "mammal", 0.95),
            gt("is-a", "reptile", 0.3),
            gt("has-a", "fur", 0.9),
        ];
        let content =
            select_content(&ResponseStrategy::EvidenceWeighing { conflict: 0.6 }, &triples);
        assert_eq!(content.primary.len(), 2);
        assert_eq!(content.contrast.len(), 1);
        assert_eq!(content.contrast[0].object_label, "reptile");
    }

    #[test]
    fn knowledge_gap_carries_no_content() {
        let triples = vec![gt("is-a", "mammal", 0.95)];
        let content = select_content(
            &ResponseStrategy::KnowledgeGap {
                topic: "dog".into(),
                can_learn: true,
            },
            &triples,
        );
        assert!(content.primary.is_empty());
    }

    #[test]
    fn contextualization_prefers_taxonomic_triples() {
        let triples = vec![
            gt("has-property", "loyal", 0.99),
            gt("is-a", "mammal", 0.8),
            gt("subclass-of", "canid", 0.75),
        ];
        let content = select_content(
            &ResponseStrategy::Contextualization {
                broader_topic: "mammal".into(),
            },
            &triples,
        );
        assert_eq!(content.primary[0].predicate_label, "is-a");
        assert_eq!(content.primary[1].predicate_label, "subclass-of");
    }

    // ── Shadow filter (34e) ─────────────────────────────────────────

    #[test]
    fn shadow_softens_challenge_under_high_severity() {
        // Default psyche has a severity-1.0 veto pattern.
        let filter = ShadowFilter::from_psyche(&Psyche::default());
        let result = filter.filter(
            &ResponseStrategy::Challenge {
                reasoning: "counter-evidence".into(),
            },
            &[],
        );
        match result {
            ShadowFilterResult::Inhibited {
                replacement,
                journal_entry,
            } => {
                assert_eq!(replacement.name(), "EvidenceWeighing");
                assert!(journal_entry.contains("considered Challenge"));
            }
            ShadowFilterResult::Pass => panic!("high-severity shadow must inhibit Challenge"),
        }
    }

    #[test]
    fn shadow_passes_gentle_strategies() {
        let filter = ShadowFilter::from_psyche(&Psyche::default());
        let result = filter.filter(
            &ResponseStrategy::HedgedStatement {
                reason: "limited evidence".into(),
            },
            &[gt("is-a", "rm -rf", 0.9)],
        );
        assert!(matches!(result, ShadowFilterResult::Pass));
    }

    #[test]
    fn shadow_triggers_on_content_labels() {
        let filter = ShadowFilter::from_psyche(&Psyche::default());
        // Default veto pattern has action_triggers ["delete all", "drop table", "rm -rf"].
        let result = filter.filter(
            &ResponseStrategy::ConfidentAssertion,
            &[gt("describes", "rm -rf usage", 0.9)],
        );
        assert!(matches!(result, ShadowFilterResult::Inhibited { .. }));
    }

    // ── Evidence conflict helper ────────────────────────────────────

    #[test]
    fn conflict_zero_for_agreeing_evidence() {
        assert!(evidence_conflict(&[0.9, 0.85, 0.95]) < 0.01);
    }

    #[test]
    fn conflict_high_for_contradicting_evidence() {
        let k = evidence_conflict(&[0.95, 0.1]);
        assert!(k > 0.5, "strong pro + strong con must conflict, got {k}");
    }

    #[test]
    fn confidence_level_bands() {
        assert_eq!(ConfidenceLevel::from_confidence(0.1), ConfidenceLevel::VeryLow);
        assert_eq!(ConfidenceLevel::from_confidence(0.5), ConfidenceLevel::Moderate);
        assert_eq!(ConfidenceLevel::from_confidence(0.95), ConfidenceLevel::VeryHigh);
        assert_eq!(ConfidenceLevel::Moderate.as_label(), "moderate");
    }

    #[test]
    fn sentiment_from_valence() {
        assert_eq!(Sentiment::from_valence(-0.6), Sentiment::Negative);
        assert_eq!(Sentiment::from_valence(0.0), Sentiment::Neutral);
        assert_eq!(Sentiment::from_valence(0.7), Sentiment::Positive);
    }
}
