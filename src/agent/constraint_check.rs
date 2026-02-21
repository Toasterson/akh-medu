//! Pre-communication constraint checking (Phase 12c).
//!
//! Before any outbound message is emitted, the constraint pipeline runs six
//! checks: consistency, confidence, rate, relevance, sensitivity, provenance.
//! Results are recorded on the `OutboundMessage` and behavior differs by
//! channel kind (operator: annotate; trusted: suppress violations; social/
//! public: suppress entirely).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::channel::ChannelKind;
use super::conversation::GroundedResponse;

// ── Error types ─────────────────────────────────────────────────────────

/// Errors from the constraint checking subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum ConstraintCheckError {
    #[error("constraint pipeline failed: {reason}")]
    #[diagnostic(
        code(akh::constraint::pipeline_failure),
        help("The constraint checking pipeline encountered an internal error. The message was not emitted.")
    )]
    PipelineFailure { reason: String },
}

pub type ConstraintCheckResult<T> = Result<T, ConstraintCheckError>;

// ── Violation & Warning types ───────────────────────────────────────────

/// A hard constraint violation that may block message emission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintViolation {
    /// A claim in the response contradicts existing KG knowledge.
    Contradiction {
        /// Label of the claim that contradicts.
        claim_label: String,
        /// SymbolId of the conflicting entity.
        contradicts: SymbolId,
        /// Description of the contradiction.
        description: String,
    },
    /// A claim's confidence is below the channel's minimum threshold.
    BelowConfidence {
        /// Label of the low-confidence claim.
        claim_label: String,
        /// Actual confidence.
        confidence: f32,
        /// Required minimum.
        threshold: f32,
    },
    /// The channel's message rate limit has been exceeded.
    RateLimitExceeded {
        /// Channel that exceeded its limit.
        channel_id: String,
        /// Max allowed messages per window.
        limit: u32,
        /// Current count in this window.
        current: u32,
    },
    /// Response includes entities with sensitivity levels above what the
    /// channel kind permits.
    SensitivityBreach {
        /// Entity with the sensitivity marking.
        entity: SymbolId,
        /// Label of the entity.
        entity_label: String,
        /// Minimum channel kind required.
        required_kind: ChannelKind,
        /// Actual channel kind.
        actual_kind: ChannelKind,
    },
}

/// A soft warning that does not block emission but annotates the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintWarning {
    /// A claim has no provenance record backing it.
    Ungrounded {
        /// Label of the ungrounded claim.
        claim_label: String,
    },
    /// The response's relevance to the query is low.
    LowRelevance {
        /// Measured similarity score.
        similarity: f32,
        /// Required minimum.
        threshold: f32,
    },
    /// A claim's confidence is above threshold but still low.
    LowConfidence {
        /// Label of the claim.
        claim_label: String,
        /// The confidence value.
        confidence: f32,
    },
}

// ── CheckOutcome ────────────────────────────────────────────────────────

/// The result of running the constraint check pipeline on an outbound message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckOutcome {
    /// Whether the message passed all hard constraints.
    pub passed: bool,
    /// Hard violations (any → passed = false).
    pub violations: Vec<ConstraintViolation>,
    /// Soft warnings.
    pub warnings: Vec<ConstraintWarning>,
}

impl CheckOutcome {
    /// A clean pass with no violations or warnings.
    pub fn clean() -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Total number of issues (violations + warnings).
    pub fn issue_count(&self) -> usize {
        self.violations.len() + self.warnings.len()
    }
}

// ── CommunicationBudget ─────────────────────────────────────────────────

/// Per-channel rate tracking for communication budget enforcement.
///
/// Tracks messages in a sliding time window. Generalizes Winter's
/// "am I posting too much?" Datalog rule.
#[derive(Debug, Clone)]
pub struct CommunicationBudget {
    /// Max messages per window.
    pub max_messages: u32,
    /// Time window duration.
    pub window: Duration,
    /// Cooldown after a burst (post-violation cool-off).
    pub cooldown_after_burst: Duration,
    /// Timestamps of recent messages.
    timestamps: Vec<Instant>,
    /// Whether we're in cooldown.
    in_cooldown_until: Option<Instant>,
}

impl CommunicationBudget {
    /// Create a new budget with the given limits.
    pub fn new(max_messages: u32, window: Duration, cooldown: Duration) -> Self {
        Self {
            max_messages,
            window,
            cooldown_after_burst: cooldown,
            timestamps: Vec::new(),
            in_cooldown_until: None,
        }
    }

    /// Create a budget from a channel's rate limit (messages/minute).
    pub fn from_rate_limit(rate: u32) -> Self {
        Self::new(rate, Duration::from_secs(60), Duration::from_secs(10))
    }

    /// Create an unlimited budget (no rate limiting).
    pub fn unlimited() -> Self {
        Self::new(u32::MAX, Duration::from_secs(60), Duration::ZERO)
    }

    /// Record a message emission.
    pub fn record_message(&mut self) {
        self.timestamps.push(Instant::now());
    }

    /// Prune old timestamps outside the window.
    fn prune(&mut self) {
        let cutoff = Instant::now() - self.window;
        self.timestamps.retain(|t| *t > cutoff);
    }

    /// Check if the budget allows another message.
    ///
    /// Returns `None` if allowed, `Some(violation)` if rate exceeded.
    pub fn check(&mut self, channel_id: &str) -> Option<ConstraintViolation> {
        // Check cooldown first.
        if let Some(until) = self.in_cooldown_until {
            if Instant::now() < until {
                return Some(ConstraintViolation::RateLimitExceeded {
                    channel_id: channel_id.to_string(),
                    limit: self.max_messages,
                    current: self.timestamps.len() as u32,
                });
            }
            self.in_cooldown_until = None;
        }

        self.prune();
        let current = self.timestamps.len() as u32;

        if current >= self.max_messages {
            // Enter cooldown.
            self.in_cooldown_until = Some(Instant::now() + self.cooldown_after_burst);
            Some(ConstraintViolation::RateLimitExceeded {
                channel_id: channel_id.to_string(),
                limit: self.max_messages,
                current,
            })
        } else {
            None
        }
    }
}

// ── ConstraintConfig ────────────────────────────────────────────────────

/// Configuration for the constraint checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintConfig {
    /// Minimum confidence threshold per channel kind.
    pub confidence_thresholds: ConfidenceThresholds,
    /// Below this confidence, emit a warning (but don't block).
    pub low_confidence_warning: f32,
    /// Minimum VSA similarity for relevance check.
    pub relevance_threshold: f32,
    /// Whether to enforce provenance checks.
    pub enforce_provenance: bool,
    /// Whether to enforce sensitivity checks.
    pub enforce_sensitivity: bool,
}

/// Minimum confidence thresholds per channel kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholds {
    /// Operator channel: tolerant (default 0.0 — no blocking).
    pub operator: f32,
    /// Trusted channel (default 0.3).
    pub trusted: f32,
    /// Social channel (default 0.5).
    pub social: f32,
    /// Public channel (default 0.7).
    pub public: f32,
}

impl Default for ConfidenceThresholds {
    fn default() -> Self {
        Self {
            operator: 0.0,
            trusted: 0.3,
            social: 0.5,
            public: 0.7,
        }
    }
}

impl ConfidenceThresholds {
    /// Get the threshold for a given channel kind.
    pub fn for_kind(&self, kind: ChannelKind) -> f32 {
        match kind {
            ChannelKind::Operator => self.operator,
            ChannelKind::Trusted => self.trusted,
            ChannelKind::Social => self.social,
            ChannelKind::Public => self.public,
        }
    }
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            confidence_thresholds: ConfidenceThresholds::default(),
            low_confidence_warning: 0.4,
            relevance_threshold: 0.1,
            enforce_provenance: true,
            enforce_sensitivity: true,
        }
    }
}

// ── Sensitivity levels ──────────────────────────────────────────────────

/// Sensitivity level of an entity.
///
/// Higher sensitivity requires higher-trust channel kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SensitivityLevel {
    /// No sensitivity — public information.
    Public,
    /// Low sensitivity — social channels and above.
    Low,
    /// Medium — trusted channels and above.
    Medium,
    /// High — operator only.
    High,
    /// Private — suppressed on all non-operator channels.
    Private,
}

impl SensitivityLevel {
    /// Parse from a string label (case-insensitive).
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_lowercase().as_str() {
            "public" | "none" => Some(Self::Public),
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            "private" | "confidential" => Some(Self::Private),
            _ => None,
        }
    }

    /// Minimum channel kind required for this sensitivity level.
    pub fn required_channel_kind(&self) -> ChannelKind {
        match self {
            Self::Public => ChannelKind::Public,
            Self::Low => ChannelKind::Social,
            Self::Medium => ChannelKind::Trusted,
            Self::High | Self::Private => ChannelKind::Operator,
        }
    }
}

// ── ConstraintChecker ───────────────────────────────────────────────────

/// Pre-communication constraint checker.
///
/// Runs a pipeline of checks on outbound messages before emission.
/// Integrates with the engine's KG, provenance ledger, and contradiction
/// detection infrastructure.
#[derive(Debug)]
pub struct ConstraintChecker {
    /// Configuration.
    pub config: ConstraintConfig,
    /// Per-channel communication budgets.
    budgets: HashMap<String, CommunicationBudget>,
}

impl ConstraintChecker {
    /// Create a new constraint checker with default config.
    pub fn new() -> Self {
        Self {
            config: ConstraintConfig::default(),
            budgets: HashMap::new(),
        }
    }

    /// Create with a specific configuration.
    pub fn with_config(config: ConstraintConfig) -> Self {
        Self {
            config,
            budgets: HashMap::new(),
        }
    }

    /// Register a channel's communication budget.
    pub fn register_budget(&mut self, channel_id: &str, budget: CommunicationBudget) {
        self.budgets.insert(channel_id.to_string(), budget);
    }

    /// Register a budget from a channel's rate limit.
    pub fn register_rate_limit(&mut self, channel_id: &str, rate_limit: Option<u32>) {
        let budget = match rate_limit {
            Some(rate) => CommunicationBudget::from_rate_limit(rate),
            None => CommunicationBudget::unlimited(),
        };
        self.budgets.insert(channel_id.to_string(), budget);
    }

    /// Record that a message was sent on a channel (updates the budget).
    pub fn record_emission(&mut self, channel_id: &str) {
        if let Some(budget) = self.budgets.get_mut(channel_id) {
            budget.record_message();
        }
    }

    /// Run the full constraint check pipeline on a grounded response.
    ///
    /// The six checks run in order:
    /// 1. Consistency (contradiction check)
    /// 2. Confidence threshold
    /// 3. Rate limit
    /// 4. Relevance (placeholder — requires VSA query vector)
    /// 5. Sensitivity
    /// 6. Provenance
    pub fn check_grounded(
        &mut self,
        response: &GroundedResponse,
        channel_id: &str,
        channel_kind: ChannelKind,
        engine: &Engine,
    ) -> CheckOutcome {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();

        // 1. Consistency check: do any supporting triples contradict existing KG?
        self.check_consistency(response, engine, &mut violations);

        // 2. Confidence check: are all triples above the channel's threshold?
        let threshold = self.config.confidence_thresholds.for_kind(channel_kind);
        self.check_confidence(response, threshold, &mut violations, &mut warnings);

        // 3. Rate check: has the channel exceeded its communication budget?
        if let Some(violation) = self.check_rate(channel_id) {
            violations.push(violation);
        }

        // 4. Relevance check: placeholder — full implementation requires query
        //    VSA vector which we don't carry yet. Skipped for Phase 12c.
        //    Will be wired in Phase 12f when explanation generation adds
        //    query context to the grounding pipeline.

        // 5. Sensitivity check: do any entities have sensitivity markings?
        if self.config.enforce_sensitivity {
            self.check_sensitivity(response, channel_kind, engine, &mut violations);
        }

        // 6. Provenance check: can every claim be traced to a source?
        if self.config.enforce_provenance {
            self.check_provenance(response, engine, &mut warnings);
        }

        let passed = violations.is_empty();
        CheckOutcome {
            passed,
            violations,
            warnings,
        }
    }

    /// Run a simplified check for non-grounded (legacy) messages.
    ///
    /// Only rate limiting applies — we have no structured claims to check.
    pub fn check_ungrounded(
        &mut self,
        channel_id: &str,
    ) -> CheckOutcome {
        let mut violations = Vec::new();

        if let Some(violation) = self.check_rate(channel_id) {
            violations.push(violation);
        }

        let passed = violations.is_empty();
        CheckOutcome {
            passed,
            violations,
            warnings: Vec::new(),
        }
    }

    // ── Individual checks ───────────────────────────────────────────────

    /// Check 1: Consistency — do supporting triples contradict the KG?
    fn check_consistency(
        &self,
        response: &GroundedResponse,
        engine: &Engine,
        violations: &mut Vec<ConstraintViolation>,
    ) {
        let func_preds = engine.functional_preds();
        let disjointness = engine.disjointness();

        for gt in &response.supporting_triples {
            // Reconstruct a Triple from the grounded triple's IDs.
            let (Some(subj), Some(pred), Some(obj)) = (gt.subject_id, gt.predicate_id, gt.object_id)
            else {
                continue;
            };

            let triple = crate::graph::Triple {
                subject: subj,
                predicate: pred,
                object: obj,
                confidence: gt.confidence.unwrap_or(1.0),
                timestamp: 0,
                provenance_id: None,
                compartment_id: None,
            };

            let contradictions = crate::graph::contradiction::check_contradictions(
                engine, &triple, &func_preds, &disjointness, None,
            );

            for c in contradictions {
                violations.push(ConstraintViolation::Contradiction {
                    claim_label: format!("{} {} {}", gt.subject_label, gt.predicate_label, gt.object_label),
                    contradicts: c.existing.subject,
                    description: format!("{:?}", c.kind),
                });
            }
        }
    }

    /// Check 2: Confidence — are all claims above the channel's threshold?
    fn check_confidence(
        &self,
        response: &GroundedResponse,
        threshold: f32,
        violations: &mut Vec<ConstraintViolation>,
        warnings: &mut Vec<ConstraintWarning>,
    ) {
        for gt in &response.supporting_triples {
            let conf = gt.confidence.unwrap_or(1.0);

            if conf < threshold {
                violations.push(ConstraintViolation::BelowConfidence {
                    claim_label: format!("{} {} {}", gt.subject_label, gt.predicate_label, gt.object_label),
                    confidence: conf,
                    threshold,
                });
            } else if conf < self.config.low_confidence_warning {
                warnings.push(ConstraintWarning::LowConfidence {
                    claim_label: format!("{} {} {}", gt.subject_label, gt.predicate_label, gt.object_label),
                    confidence: conf,
                });
            }
        }
    }

    /// Check 3: Rate limit — has the channel exceeded its budget?
    fn check_rate(&mut self, channel_id: &str) -> Option<ConstraintViolation> {
        self.budgets
            .get_mut(channel_id)
            .and_then(|budget| budget.check(channel_id))
    }

    /// Check 5: Sensitivity — do response entities have sensitivity markings?
    fn check_sensitivity(
        &self,
        response: &GroundedResponse,
        channel_kind: ChannelKind,
        engine: &Engine,
        violations: &mut Vec<ConstraintViolation>,
    ) {
        // Look for sensitivity predicates on entities in the response.
        let sensitivity_pred = engine.lookup_symbol("onto:sensitivity-level").ok();

        if let Some(pred_id) = sensitivity_pred {
            for sym_id in &response.provenance_ids {
                let triples = engine.triples_from(*sym_id);
                for t in triples {
                    if t.predicate == pred_id {
                        let level_label = engine.resolve_label(t.object);
                        if let Some(level) = SensitivityLevel::from_label(&level_label) {
                            let required = level.required_channel_kind();
                            if !channel_kind_permits(channel_kind, required) {
                                let entity_label = engine.resolve_label(*sym_id);
                                violations.push(ConstraintViolation::SensitivityBreach {
                                    entity: *sym_id,
                                    entity_label,
                                    required_kind: required,
                                    actual_kind: channel_kind,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check 6: Provenance — can claims be traced to source records?
    fn check_provenance(
        &self,
        response: &GroundedResponse,
        engine: &Engine,
        warnings: &mut Vec<ConstraintWarning>,
    ) {
        for gt in &response.supporting_triples {
            // Check each entity in the triple for provenance.
            let ids = [gt.subject_id, gt.predicate_id, gt.object_id];
            let has_provenance = ids.iter().filter_map(|id| *id).any(|id| {
                engine.provenance_of(id)
                    .map(|records| !records.is_empty())
                    .unwrap_or(false)
            });

            if !has_provenance && gt.derivation_tag.is_none() {
                warnings.push(ConstraintWarning::Ungrounded {
                    claim_label: format!("{} {} {}", gt.subject_label, gt.predicate_label, gt.object_label),
                });
            }
        }
    }
}

impl Default for ConstraintChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Emission behavior per channel kind ──────────────────────────────────

/// What to do with a checked message based on channel kind and outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmissionDecision {
    /// Emit the message (possibly with annotations).
    Emit,
    /// Suppress the message entirely.
    Suppress,
}

/// Decide whether to emit or suppress based on channel kind and check outcome.
pub fn emission_decision(kind: ChannelKind, outcome: &CheckOutcome) -> EmissionDecision {
    if outcome.passed {
        return EmissionDecision::Emit;
    }

    match kind {
        // Operator always sees everything (annotated with warnings).
        ChannelKind::Operator => EmissionDecision::Emit,
        // Trusted: suppress only on hard violations.
        ChannelKind::Trusted => {
            if outcome.violations.is_empty() {
                EmissionDecision::Emit
            } else {
                EmissionDecision::Suppress
            }
        }
        // Social & Public: suppress entirely on any violation.
        ChannelKind::Social | ChannelKind::Public => EmissionDecision::Suppress,
    }
}

/// Check if the actual channel kind is at least as trusted as the required kind.
fn channel_kind_permits(actual: ChannelKind, required: ChannelKind) -> bool {
    trust_level(actual) >= trust_level(required)
}

/// Map channel kinds to numeric trust levels for comparison.
fn trust_level(kind: ChannelKind) -> u8 {
    match kind {
        ChannelKind::Public => 0,
        ChannelKind::Social => 1,
        ChannelKind::Trusted => 2,
        ChannelKind::Operator => 3,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_outcome_clean() {
        let outcome = CheckOutcome::clean();
        assert!(outcome.passed);
        assert!(outcome.violations.is_empty());
        assert!(outcome.warnings.is_empty());
        assert_eq!(outcome.issue_count(), 0);
    }

    #[test]
    fn check_outcome_with_violations() {
        let outcome = CheckOutcome {
            passed: false,
            violations: vec![ConstraintViolation::RateLimitExceeded {
                channel_id: "test".to_string(),
                limit: 10,
                current: 15,
            }],
            warnings: vec![ConstraintWarning::LowConfidence {
                claim_label: "test claim".to_string(),
                confidence: 0.3,
            }],
        };
        assert!(!outcome.passed);
        assert_eq!(outcome.issue_count(), 2);
    }

    #[test]
    fn confidence_thresholds_per_kind() {
        let thresholds = ConfidenceThresholds::default();
        assert_eq!(thresholds.for_kind(ChannelKind::Operator), 0.0);
        assert_eq!(thresholds.for_kind(ChannelKind::Trusted), 0.3);
        assert_eq!(thresholds.for_kind(ChannelKind::Social), 0.5);
        assert_eq!(thresholds.for_kind(ChannelKind::Public), 0.7);
    }

    #[test]
    fn sensitivity_level_parsing() {
        assert_eq!(SensitivityLevel::from_label("public"), Some(SensitivityLevel::Public));
        assert_eq!(SensitivityLevel::from_label("Private"), Some(SensitivityLevel::Private));
        assert_eq!(SensitivityLevel::from_label("HIGH"), Some(SensitivityLevel::High));
        assert_eq!(SensitivityLevel::from_label("medium"), Some(SensitivityLevel::Medium));
        assert_eq!(SensitivityLevel::from_label("confidential"), Some(SensitivityLevel::Private));
        assert_eq!(SensitivityLevel::from_label("xyz"), None);
    }

    #[test]
    fn sensitivity_required_channel() {
        assert_eq!(SensitivityLevel::Public.required_channel_kind(), ChannelKind::Public);
        assert_eq!(SensitivityLevel::Low.required_channel_kind(), ChannelKind::Social);
        assert_eq!(SensitivityLevel::Medium.required_channel_kind(), ChannelKind::Trusted);
        assert_eq!(SensitivityLevel::High.required_channel_kind(), ChannelKind::Operator);
        assert_eq!(SensitivityLevel::Private.required_channel_kind(), ChannelKind::Operator);
    }

    #[test]
    fn channel_kind_trust_ordering() {
        assert!(channel_kind_permits(ChannelKind::Operator, ChannelKind::Public));
        assert!(channel_kind_permits(ChannelKind::Operator, ChannelKind::Operator));
        assert!(channel_kind_permits(ChannelKind::Trusted, ChannelKind::Social));
        assert!(!channel_kind_permits(ChannelKind::Social, ChannelKind::Trusted));
        assert!(!channel_kind_permits(ChannelKind::Public, ChannelKind::Operator));
    }

    #[test]
    fn emission_decision_operator_always_emits() {
        let outcome = CheckOutcome {
            passed: false,
            violations: vec![ConstraintViolation::RateLimitExceeded {
                channel_id: "op".to_string(),
                limit: 1,
                current: 5,
            }],
            warnings: Vec::new(),
        };
        assert_eq!(emission_decision(ChannelKind::Operator, &outcome), EmissionDecision::Emit);
    }

    #[test]
    fn emission_decision_social_suppresses_on_violation() {
        let outcome = CheckOutcome {
            passed: false,
            violations: vec![ConstraintViolation::BelowConfidence {
                claim_label: "x".to_string(),
                confidence: 0.1,
                threshold: 0.5,
            }],
            warnings: Vec::new(),
        };
        assert_eq!(emission_decision(ChannelKind::Social, &outcome), EmissionDecision::Suppress);
    }

    #[test]
    fn emission_decision_clean_always_emits() {
        let outcome = CheckOutcome::clean();
        assert_eq!(emission_decision(ChannelKind::Public, &outcome), EmissionDecision::Emit);
        assert_eq!(emission_decision(ChannelKind::Social, &outcome), EmissionDecision::Emit);
        assert_eq!(emission_decision(ChannelKind::Trusted, &outcome), EmissionDecision::Emit);
        assert_eq!(emission_decision(ChannelKind::Operator, &outcome), EmissionDecision::Emit);
    }

    #[test]
    fn budget_unlimited() {
        let mut budget = CommunicationBudget::unlimited();
        assert!(budget.check("ch-1").is_none());
        for _ in 0..1000 {
            budget.record_message();
        }
        // Still allowed (u32::MAX threshold).
        assert!(budget.check("ch-1").is_none());
    }

    #[test]
    fn budget_from_rate_limit() {
        let mut budget = CommunicationBudget::from_rate_limit(5);
        assert_eq!(budget.max_messages, 5);
        assert_eq!(budget.window, Duration::from_secs(60));

        // Should be allowed for first 5 messages.
        for _ in 0..5 {
            assert!(budget.check("ch-1").is_none());
            budget.record_message();
        }

        // 6th should be blocked.
        let violation = budget.check("ch-1");
        assert!(violation.is_some());
        if let Some(ConstraintViolation::RateLimitExceeded { limit, current, .. }) = violation {
            assert_eq!(limit, 5);
            assert_eq!(current, 5);
        }
    }

    #[test]
    fn checker_default_config() {
        let checker = ConstraintChecker::new();
        assert_eq!(checker.config.relevance_threshold, 0.1);
        assert!(checker.config.enforce_provenance);
        assert!(checker.config.enforce_sensitivity);
    }

    #[test]
    fn checker_ungrounded_no_rate_limit() {
        let mut checker = ConstraintChecker::new();
        let outcome = checker.check_ungrounded("ch-1");
        assert!(outcome.passed);
    }

    #[test]
    fn checker_ungrounded_with_rate_limit() {
        let mut checker = ConstraintChecker::new();
        checker.register_rate_limit("ch-1", Some(2));

        // First two pass.
        checker.record_emission("ch-1");
        checker.record_emission("ch-1");

        // Third blocked.
        let outcome = checker.check_ungrounded("ch-1");
        assert!(!outcome.passed);
        assert_eq!(outcome.violations.len(), 1);
    }

    #[test]
    fn constraint_config_defaults() {
        let config = ConstraintConfig::default();
        assert_eq!(config.low_confidence_warning, 0.4);
        assert_eq!(config.confidence_thresholds.operator, 0.0);
        assert_eq!(config.confidence_thresholds.public, 0.7);
    }
}
