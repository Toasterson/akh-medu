//! Email triage & priority routing (Phase 13c).
//!
//! Sender reputation tracking, four-feature importance scoring (social / content /
//! thread / label), VSA priority prototypes, HEY-style screening queue, and
//! routing logic.
//!
//! # Triage pipeline
//!
//! 1. Look up or create `SenderStats`, call `record_message()`
//! 2. Check screening: `needs_screening()` → return `EmailRoute::ScreeningQueue`
//! 3. Compute four scores: social + content + thread + label
//! 4. Combine with weights: `social*0.35 + content*0.25 + thread*0.20 + label*0.20`
//! 5. Compute VSA similarity to important/low-priority prototypes (if trained)
//! 6. Route decision: ≥0.65 → Important, ≥0.35 → Feed, else PaperTrail
//!
//! # OnlineHD adaptive update
//!
//! Training uses majority-vote bundling: `bundle([existing_prototype, new_example])`.
//! The existing prototype encodes accumulated history; new examples have diminishing
//! contribution as more training data accumulates.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;
use crate::vsa::encode::{encode_label, encode_token};
use crate::vsa::ops::VsaOps;

use super::error::{EmailError, EmailResult};
use super::parser::ParsedEmail;

// ── Constants ──────────────────────────────────────────────────────────────

/// Redb meta key for persisting the triage engine.
const META_KEY_TRIAGE: &[u8] = b"email:triage_engine";

/// Maximum number of tracked senders before eviction.
const MAX_SENDER_STATS: usize = 10_000;

/// Recency window in seconds (7 days).
const RECENCY_WINDOW_SECS: u64 = 604_800;

/// EMA decay factor for reply rate updates.
const REPLY_RATE_DECAY: f32 = 0.1;

/// Maximum body text bytes used as VSA features.
const BODY_PREVIEW_LEN: usize = 512;

/// Route threshold: combined score ≥ this → Important.
const THRESHOLD_IMPORTANT: f32 = 0.65;

/// Route threshold: combined score ≥ this → Feed.
const THRESHOLD_FEED: f32 = 0.35;

/// Route threshold: combined score ≥ this → PaperTrail (floor).
const THRESHOLD_PAPER_TRAIL: f32 = 0.15;

// ── EmailRoute ─────────────────────────────────────────────────────────────

/// Routing destination for a triaged email.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmailRoute {
    /// High-priority: needs attention.
    Important,
    /// Medium-priority: informational, digest-worthy.
    Feed,
    /// Low-priority: receipts, notifications, automated messages.
    PaperTrail,
    /// First-time sender not yet approved by operator.
    ScreeningQueue,
    /// Classified as spam (not routed further).
    Spam,
}

impl std::fmt::Display for EmailRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Important => write!(f, "Important"),
            Self::Feed => write!(f, "Feed"),
            Self::PaperTrail => write!(f, "PaperTrail"),
            Self::ScreeningQueue => write!(f, "ScreeningQueue"),
            Self::Spam => write!(f, "Spam"),
        }
    }
}

// ── SenderRelationship ─────────────────────────────────────────────────────

/// Relationship category for a known sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SenderRelationship {
    Colleague,
    Friend,
    Service,
    Newsletter,
    Unknown,
}

impl std::fmt::Display for SenderRelationship {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Colleague => write!(f, "Colleague"),
            Self::Friend => write!(f, "Friend"),
            Self::Service => write!(f, "Service"),
            Self::Newsletter => write!(f, "Newsletter"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl SenderRelationship {
    /// Numeric weight for social scoring (0.0–1.0).
    fn weight(self) -> f32 {
        match self {
            Self::Colleague => 0.9,
            Self::Friend => 0.8,
            Self::Service => 0.3,
            Self::Newsletter => 0.1,
            Self::Unknown => 0.5,
        }
    }
}

// ── SenderStats ────────────────────────────────────────────────────────────

/// Per-sender reputation profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenderStats {
    /// Sender email address (canonical lowercase).
    pub address: String,
    /// Total messages received from this sender.
    pub message_count: u64,
    /// Total replies we sent to this sender.
    pub reply_count: u64,
    /// Exponential moving average of reply rate (0.0–1.0).
    pub reply_rate: f32,
    /// EMA of reply time in seconds.
    pub avg_reply_time_secs: f64,
    /// Timestamp of last received message.
    pub last_message_ts: u64,
    /// Timestamp of last reply we sent.
    pub last_reply_ts: u64,
    /// Classified relationship.
    pub relationship: SenderRelationship,
    /// Operator-assigned routing (None = needs screening on first contact).
    pub routing: Option<EmailRoute>,
    /// Optional KG symbol for this sender.
    pub symbol_id: Option<SymbolId>,
}

impl SenderStats {
    /// Create a new sender profile for the given address.
    fn new(address: String) -> Self {
        Self {
            address,
            message_count: 0,
            reply_count: 0,
            reply_rate: 0.0,
            avg_reply_time_secs: 0.0,
            last_message_ts: 0,
            last_reply_ts: 0,
            relationship: SenderRelationship::Unknown,
            routing: None,
            symbol_id: None,
        }
    }

    /// Whether this sender needs operator screening (first contact, no routing set).
    pub fn needs_screening(&self) -> bool {
        self.routing.is_none() && self.message_count <= 1
    }

    /// Record an incoming message from this sender.
    pub fn record_message(&mut self, timestamp: u64) {
        self.message_count += 1;
        self.last_message_ts = timestamp;
    }

    /// Record a reply we sent to this sender, updating reply rate and avg time.
    pub fn record_reply(&mut self, reply_time_secs: u64, timestamp: u64) {
        self.reply_count += 1;
        self.last_reply_ts = timestamp;

        // Update reply rate EMA: rate = (1-α)*old + α*1.0
        self.reply_rate = (1.0 - REPLY_RATE_DECAY) * self.reply_rate + REPLY_RATE_DECAY;

        // Update average reply time EMA
        let alpha = REPLY_RATE_DECAY as f64;
        self.avg_reply_time_secs =
            (1.0 - alpha) * self.avg_reply_time_secs + alpha * reply_time_secs as f64;
    }
}

// ── TriageRoleVectors ──────────────────────────────────────────────────────

/// Pre-generated deterministic role hypervectors for triage feature encoding.
///
/// Each role vector is produced via `encode_token(ops, "triage-role:X")`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageRoleVectors {
    pub sender_reputation: HyperVec,
    pub reply_rate: HyperVec,
    pub recency: HyperVec,
    pub subject: HyperVec,
    pub body: HyperVec,
    pub thread_participation: HyperVec,
    pub has_attachments: HyperVec,
    pub relationship: HyperVec,
}

impl TriageRoleVectors {
    /// Generate all 8 role vectors deterministically from the VsaOps.
    fn generate(ops: &VsaOps) -> Self {
        Self {
            sender_reputation: encode_token(ops, "triage-role:sender-reputation"),
            reply_rate: encode_token(ops, "triage-role:reply-rate"),
            recency: encode_token(ops, "triage-role:recency"),
            subject: encode_token(ops, "triage-role:subject"),
            body: encode_token(ops, "triage-role:body"),
            thread_participation: encode_token(ops, "triage-role:thread-participation"),
            has_attachments: encode_token(ops, "triage-role:has-attachments"),
            relationship: encode_token(ops, "triage-role:relationship"),
        }
    }
}

// ── ImportanceWeights ──────────────────────────────────────────────────────

/// Configurable weights for the four scoring dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceWeights {
    pub social: f32,
    pub content: f32,
    pub thread: f32,
    pub label: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            social: 0.35,
            content: 0.25,
            thread: 0.20,
            label: 0.20,
        }
    }
}

// ── ImportanceBreakdown ────────────────────────────────────────────────────

/// Detailed breakdown of the four importance scores.
#[derive(Debug, Clone)]
pub struct ImportanceBreakdown {
    pub social_score: f32,
    pub content_score: f32,
    pub thread_score: f32,
    pub label_score: f32,
    pub combined: f32,
    pub reasoning: String,
}

// ── TriageResult ───────────────────────────────────────────────────────────

/// Full result of triaging an email.
#[derive(Debug, Clone)]
pub struct TriageResult {
    /// Routing destination.
    pub route: EmailRoute,
    /// Importance score breakdown.
    pub importance: ImportanceBreakdown,
    /// Whether the sender needs operator screening.
    pub needs_screening: bool,
    /// VSA similarity to the important prototype (if trained).
    pub vsa_important_similarity: f32,
    /// VSA similarity to the low-priority prototype (if trained).
    pub vsa_low_priority_similarity: f32,
}

// ── TriagePredicates ───────────────────────────────────────────────────────

/// Well-known KG predicates for sender reputation modeling.
#[derive(Debug, Clone)]
pub struct TriagePredicates {
    pub reply_rate: SymbolId,
    pub message_count: SymbolId,
    pub avg_reply_time: SymbolId,
    pub last_interaction: SymbolId,
    pub relationship: SymbolId,
    pub routing: SymbolId,
    pub importance_score: SymbolId,
}

impl TriagePredicates {
    /// Resolve or create all triage predicates in the engine.
    pub fn init(engine: &Engine) -> EmailResult<Self> {
        Ok(Self {
            reply_rate: engine
                .resolve_or_create_relation("sender:reply-rate")
                .map_err(EmailError::from)?,
            message_count: engine
                .resolve_or_create_relation("sender:message-count")
                .map_err(EmailError::from)?,
            avg_reply_time: engine
                .resolve_or_create_relation("sender:avg-reply-time")
                .map_err(EmailError::from)?,
            last_interaction: engine
                .resolve_or_create_relation("sender:last-interaction")
                .map_err(EmailError::from)?,
            relationship: engine
                .resolve_or_create_relation("sender:relationship")
                .map_err(EmailError::from)?,
            routing: engine
                .resolve_or_create_relation("sender:routing")
                .map_err(EmailError::from)?,
            importance_score: engine
                .resolve_or_create_relation("sender:importance-score")
                .map_err(EmailError::from)?,
        })
    }
}

// ── TriageEngine ───────────────────────────────────────────────────────────

/// Main triage engine: sender reputation, importance scoring, VSA prototypes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageEngine {
    /// Per-sender reputation stats (keyed by lowercase email address).
    sender_stats: HashMap<String, SenderStats>,
    /// OnlineHD important-email prototype.
    important_prototype: Option<HyperVec>,
    /// OnlineHD low-priority-email prototype.
    low_priority_prototype: Option<HyperVec>,
    /// Training count for important prototype.
    important_count: u64,
    /// Training count for low-priority prototype.
    low_priority_count: u64,
    /// Role vectors for triage feature encoding.
    role_vectors: TriageRoleVectors,
    /// Configurable importance weights.
    weights: ImportanceWeights,
}

impl TriageEngine {
    /// Create a new untrained triage engine with role vectors.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            sender_stats: HashMap::new(),
            important_prototype: None,
            low_priority_prototype: None,
            important_count: 0,
            low_priority_count: 0,
            role_vectors: TriageRoleVectors::generate(ops),
            weights: ImportanceWeights::default(),
        }
    }

    /// Full triage pipeline for an incoming email.
    ///
    /// `now` is the current unix timestamp. `user_email` is the operator's email
    /// address (used for thread participation detection).
    pub fn triage(
        &mut self,
        ops: &VsaOps,
        email: &ParsedEmail,
        now: u64,
        _user_email: &str,
    ) -> EmailResult<TriageResult> {
        let sender_key = email.from.to_ascii_lowercase();

        // 1. Look up or create sender stats, record message.
        let stats = self
            .sender_stats
            .entry(sender_key.clone())
            .or_insert_with(|| SenderStats::new(sender_key.clone()));
        stats.record_message(now);

        // Evict least-recently-seen senders if over limit.
        if self.sender_stats.len() > MAX_SENDER_STATS {
            self.evict_oldest_senders();
        }

        // 2. Check screening.
        let needs_screening = self.sender_stats[&sender_key].needs_screening();
        if needs_screening {
            let importance = ImportanceBreakdown {
                social_score: 0.0,
                content_score: 0.0,
                thread_score: 0.0,
                label_score: 0.5,
                combined: 0.0,
                reasoning: "first-time sender — awaiting operator screening".to_string(),
            };
            return Ok(TriageResult {
                route: EmailRoute::ScreeningQueue,
                importance,
                needs_screening: true,
                vsa_important_similarity: 0.0,
                vsa_low_priority_similarity: 0.0,
            });
        }

        let stats = &self.sender_stats[&sender_key];

        // 3. Compute four scores.
        let social_score = self.score_social(stats, now);
        let content_score = self.score_content(ops, email)?;
        let thread_score = Self::score_thread(email);
        let label_score = Self::score_label(stats.routing);

        // 4. Combine with weights.
        let combined = self.weights.social * social_score
            + self.weights.content * content_score
            + self.weights.thread * thread_score
            + self.weights.label * label_score;

        // 5. VSA similarity to prototypes.
        let (vsa_imp_sim, vsa_low_sim) = self.compute_vsa_similarity(ops, email)?;

        // 6. Route decision.
        let route = if combined >= THRESHOLD_IMPORTANT {
            EmailRoute::Important
        } else if combined >= THRESHOLD_FEED {
            EmailRoute::Feed
        } else {
            EmailRoute::PaperTrail
        };

        let reasoning = format!(
            "social={social_score:.3}, content={content_score:.3}, \
             thread={thread_score:.3}, label={label_score:.3}; \
             combined={combined:.3} (thresholds: {THRESHOLD_IMPORTANT}/{THRESHOLD_FEED}/{THRESHOLD_PAPER_TRAIL}); \
             VSA: imp_sim={vsa_imp_sim:.3}, low_sim={vsa_low_sim:.3}"
        );

        Ok(TriageResult {
            route,
            importance: ImportanceBreakdown {
                social_score,
                content_score,
                thread_score,
                label_score,
                combined,
                reasoning,
            },
            needs_screening: false,
            vsa_important_similarity: vsa_imp_sim,
            vsa_low_priority_similarity: vsa_low_sim,
        })
    }

    /// Encode an email into a triage-oriented composite hypervector.
    ///
    /// Features: sender reputation token, reply rate token, recency token,
    /// subject keywords, body preview, thread participation, attachments,
    /// relationship token.
    pub fn encode_email(&self, ops: &VsaOps, email: &ParsedEmail) -> EmailResult<HyperVec> {
        let mut features: Vec<HyperVec> = Vec::with_capacity(8);

        // 1. Sender reputation (encode address as token)
        let sender_filler = encode_token(ops, &email.from.to_ascii_lowercase());
        let sender_bound = ops
            .bind(&self.role_vectors.sender_reputation, &sender_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for sender reputation: {e}"),
            })?;
        features.push(sender_bound);

        // 2. Reply rate bucket
        let sender_key = email.from.to_ascii_lowercase();
        let rate_bucket = self
            .sender_stats
            .get(&sender_key)
            .map(|s| format!("reply-rate-{}", (s.reply_rate * 10.0) as u32))
            .unwrap_or_else(|| "reply-rate-0".to_string());
        let rate_filler = encode_token(ops, &rate_bucket);
        let rate_bound = ops
            .bind(&self.role_vectors.reply_rate, &rate_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for reply rate: {e}"),
            })?;
        features.push(rate_bound);

        // 3. Subject keywords
        if !email.subject.is_empty() {
            let subj_filler = encode_label(ops, &email.subject).map_err(|e| EmailError::Parse {
                message: format!("VSA encode_label failed for subject: {e}"),
            })?;
            let subj_bound = ops
                .bind(&self.role_vectors.subject, &subj_filler)
                .map_err(|e| EmailError::Parse {
                    message: format!("VSA bind failed for subject: {e}"),
                })?;
            features.push(subj_bound);
        }

        // 4. Body preview
        if let Some(body) = email.best_body() {
            let preview = if body.len() > BODY_PREVIEW_LEN {
                &body[..body.floor_char_boundary(BODY_PREVIEW_LEN)]
            } else {
                body
            };
            if !preview.trim().is_empty() {
                let body_filler =
                    encode_label(ops, preview).map_err(|e| EmailError::Parse {
                        message: format!("VSA encode_label failed for body: {e}"),
                    })?;
                let body_bound = ops
                    .bind(&self.role_vectors.body, &body_filler)
                    .map_err(|e| EmailError::Parse {
                        message: format!("VSA bind failed for body: {e}"),
                    })?;
                features.push(body_bound);
            }
        }

        // 5. Thread participation (in_reply_to presence)
        let thread_token = if email.in_reply_to.is_some() {
            "in-thread"
        } else {
            "no-thread"
        };
        let thread_filler = encode_token(ops, thread_token);
        let thread_bound = ops
            .bind(&self.role_vectors.thread_participation, &thread_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for thread participation: {e}"),
            })?;
        features.push(thread_bound);

        // 6. Attachments
        let attach_token = if email.has_attachments { "true" } else { "false" };
        let attach_filler = encode_token(ops, attach_token);
        let attach_bound = ops
            .bind(&self.role_vectors.has_attachments, &attach_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for attachments: {e}"),
            })?;
        features.push(attach_bound);

        // 7. Relationship
        let rel_token = self
            .sender_stats
            .get(&sender_key)
            .map(|s| format!("rel:{}", s.relationship))
            .unwrap_or_else(|| "rel:Unknown".to_string());
        let rel_filler = encode_token(ops, &rel_token);
        let rel_bound = ops
            .bind(&self.role_vectors.relationship, &rel_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for relationship: {e}"),
            })?;
        features.push(rel_bound);

        // Bundle all features
        let refs: Vec<&HyperVec> = features.iter().collect();
        ops.bundle(&refs).map_err(|e| EmailError::Parse {
            message: format!("VSA bundle failed for triage encoding: {e}"),
        })
    }

    /// OnlineHD update for the important-email prototype.
    pub fn train_important(&mut self, ops: &VsaOps, email: &ParsedEmail) -> EmailResult<()> {
        let email_vec = self.encode_email(ops, email)?;

        self.important_prototype = Some(match &self.important_prototype {
            Some(existing) => {
                let refs = [existing, &email_vec];
                ops.bundle(&refs).map_err(|e| EmailError::Parse {
                    message: format!("VSA bundle failed during important training: {e}"),
                })?
            }
            None => email_vec,
        });
        self.important_count += 1;
        Ok(())
    }

    /// OnlineHD update for the low-priority-email prototype.
    pub fn train_low_priority(&mut self, ops: &VsaOps, email: &ParsedEmail) -> EmailResult<()> {
        let email_vec = self.encode_email(ops, email)?;

        self.low_priority_prototype = Some(match &self.low_priority_prototype {
            Some(existing) => {
                let refs = [existing, &email_vec];
                ops.bundle(&refs).map_err(|e| EmailError::Parse {
                    message: format!("VSA bundle failed during low-priority training: {e}"),
                })?
            }
            None => email_vec,
        });
        self.low_priority_count += 1;
        Ok(())
    }

    /// Update sender reply stats.
    pub fn record_reply(&mut self, sender: &str, reply_time_secs: u64, timestamp: u64) {
        let key = sender.to_ascii_lowercase();
        if let Some(stats) = self.sender_stats.get_mut(&key) {
            stats.record_reply(reply_time_secs, timestamp);
        }
    }

    /// Set operator-approved routing for a sender (HEY screening result).
    pub fn set_sender_routing(&mut self, sender: &str, route: EmailRoute) {
        let key = sender.to_ascii_lowercase();
        let stats = self
            .sender_stats
            .entry(key.clone())
            .or_insert_with(|| SenderStats::new(key));
        stats.routing = Some(route);
    }

    /// Classify sender relationship.
    pub fn set_sender_relationship(&mut self, sender: &str, rel: SenderRelationship) {
        let key = sender.to_ascii_lowercase();
        let stats = self
            .sender_stats
            .entry(key.clone())
            .or_insert_with(|| SenderStats::new(key));
        stats.relationship = rel;
    }

    /// Read-only sender stats lookup.
    pub fn sender_stats(&self, address: &str) -> Option<&SenderStats> {
        self.sender_stats.get(&address.to_ascii_lowercase())
    }

    /// Whether both prototypes are present (at least one important + one low-priority).
    pub fn is_trained(&self) -> bool {
        self.important_prototype.is_some() && self.low_priority_prototype.is_some()
    }

    /// Persist the triage engine to the engine's durable store via bincode.
    pub fn persist(&self, engine: &Engine) -> EmailResult<()> {
        let encoded = bincode::serialize(self).map_err(|e| EmailError::Parse {
            message: format!("failed to serialize triage engine: {e}"),
        })?;
        engine
            .store()
            .put_meta(META_KEY_TRIAGE, &encoded)
            .map_err(|e| EmailError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    /// Restore a previously persisted triage engine.
    ///
    /// Returns `Ok(None)` if no engine has been persisted yet.
    pub fn restore(engine: &Engine, ops: &VsaOps) -> EmailResult<Option<Self>> {
        let data = match engine.store().get_meta(META_KEY_TRIAGE) {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };

        match data {
            Some(bytes) => match bincode::deserialize::<Self>(&bytes) {
                Ok(eng) => Ok(Some(eng)),
                Err(_) => Ok(Some(Self::new(ops))),
            },
            None => Ok(None),
        }
    }

    // ── Private scoring methods ────────────────────────────────────────

    /// Social score (0.0–1.0): reply rate, frequency, recency, relationship.
    fn score_social(&self, stats: &SenderStats, now: u64) -> f32 {
        // 40% reply rate (EMA, already [0,1])
        let reply_component = stats.reply_rate;

        // 30% frequency: min(log2(msg_count), 10) / 10
        let freq = if stats.message_count > 0 {
            ((stats.message_count as f32).log2().min(10.0)) / 10.0
        } else {
            0.0
        };

        // 20% recency: max(0, 1 - elapsed/RECENCY_WINDOW)
        let elapsed = now.saturating_sub(stats.last_message_ts);
        let recency = (1.0 - elapsed as f32 / RECENCY_WINDOW_SECS as f32).max(0.0);

        // 10% relationship weight
        let rel_weight = stats.relationship.weight();

        0.4 * reply_component + 0.3 * freq + 0.2 * recency + 0.1 * rel_weight
    }

    /// Content score (0.0–1.0): VSA similarity to important/low-priority prototypes.
    fn score_content(&self, ops: &VsaOps, email: &ParsedEmail) -> EmailResult<f32> {
        match (&self.important_prototype, &self.low_priority_prototype) {
            (Some(imp_proto), Some(low_proto)) => {
                let email_vec = self.encode_email(ops, email)?;
                let imp_sim = ops.similarity(&email_vec, imp_proto).unwrap_or(0.5);
                let low_sim = ops.similarity(&email_vec, low_proto).unwrap_or(0.5);
                let range = imp_sim + low_sim;
                if range > 0.0 {
                    Ok(imp_sim / range)
                } else {
                    Ok(0.5)
                }
            }
            _ => Ok(0.5), // Untrained — neutral
        }
    }

    /// Thread score (0.0–1.0): reply indicator, thread depth, has references.
    fn score_thread(email: &ParsedEmail) -> f32 {
        // 40% user_replied: has in_reply_to (ongoing conversation)
        let user_replied = if email.in_reply_to.is_some() { 1.0 } else { 0.0 };

        // 40% thread depth: min(references.len(), 10) / 10
        let depth = (email.references.len().min(10) as f32) / 10.0;

        // 20% has references: binary
        let has_refs = if email.references.is_empty() {
            0.0
        } else {
            1.0
        };

        0.4 * user_replied + 0.4 * depth + 0.2 * has_refs
    }

    /// Label score (0.0–1.0): based on operator-assigned route.
    fn score_label(routing: Option<EmailRoute>) -> f32 {
        match routing {
            Some(EmailRoute::Important) => 1.0,
            Some(EmailRoute::Feed) => 0.6,
            Some(EmailRoute::PaperTrail) => 0.3,
            Some(EmailRoute::Spam) => 0.0,
            Some(EmailRoute::ScreeningQueue) | None => 0.5,
        }
    }

    /// Compute VSA similarity to important/low-priority prototypes.
    fn compute_vsa_similarity(
        &self,
        ops: &VsaOps,
        email: &ParsedEmail,
    ) -> EmailResult<(f32, f32)> {
        match (&self.important_prototype, &self.low_priority_prototype) {
            (Some(imp_proto), Some(low_proto)) => {
                let email_vec = self.encode_email(ops, email)?;
                let imp_sim = ops.similarity(&email_vec, imp_proto).unwrap_or(0.0);
                let low_sim = ops.similarity(&email_vec, low_proto).unwrap_or(0.0);
                Ok((imp_sim, low_sim))
            }
            _ => Ok((0.0, 0.0)),
        }
    }

    /// Evict the oldest senders (by last_message_ts) down to 80% capacity.
    fn evict_oldest_senders(&mut self) {
        let target = MAX_SENDER_STATS * 4 / 5;
        let mut entries: Vec<(String, u64)> = self
            .sender_stats
            .iter()
            .map(|(k, v)| (k.clone(), v.last_message_ts))
            .collect();
        entries.sort_by_key(|e| e.1);

        let to_remove = self.sender_stats.len().saturating_sub(target);
        for (key, _) in entries.iter().take(to_remove) {
            self.sender_stats.remove(key);
        }
    }
}

// ── Free functions: provenance & KG sync ───────────────────────────────────

/// Record provenance for an email triage decision.
///
/// Stores a `DerivationKind::EmailTriaged` record linked to the email's
/// symbol in the KG.
pub fn record_triage_provenance(
    engine: &Engine,
    email_symbol: SymbolId,
    result: &TriageResult,
    email: &ParsedEmail,
) -> EmailResult<()> {
    let mut record = ProvenanceRecord::new(
        email_symbol,
        DerivationKind::EmailTriaged {
            email_message_id: email.message_id.clone(),
            route: result.route.to_string(),
            importance_score: result.importance.combined,
        },
    );
    record.confidence = result.importance.combined;

    engine
        .store_provenance(&mut record)
        .map_err(|e| EmailError::Engine(Box::new(e)))?;
    Ok(())
}

/// Sync sender reputation stats to KG triples for SPARQL queryability.
///
/// Creates or updates triples for the sender's reply rate, message count,
/// average reply time, last interaction, relationship, and routing.
pub fn sync_sender_to_kg(
    sender: &SenderStats,
    engine: &Engine,
    predicates: &TriagePredicates,
) -> EmailResult<()> {
    // Resolve or create sender entity.
    let sender_sym = engine
        .resolve_or_create_entity(&sender.address)
        .map_err(EmailError::from)?;

    // Helper: add triple sender → predicate → value.
    let add_triple = |pred: SymbolId, value: &str| -> EmailResult<()> {
        let val_sym = engine
            .resolve_or_create_entity(value)
            .map_err(EmailError::from)?;
        let triple = Triple::new(sender_sym, pred, val_sym);
        engine.add_triple(&triple).map_err(EmailError::from)?;
        Ok(())
    };

    add_triple(
        predicates.reply_rate,
        &format!("{:.3}", sender.reply_rate),
    )?;
    add_triple(
        predicates.message_count,
        &sender.message_count.to_string(),
    )?;
    add_triple(
        predicates.avg_reply_time,
        &format!("{:.0}", sender.avg_reply_time_secs),
    )?;
    add_triple(
        predicates.last_interaction,
        &sender.last_message_ts.to_string(),
    )?;
    add_triple(predicates.relationship, &sender.relationship.to_string())?;
    if let Some(route) = &sender.routing {
        add_triple(predicates.routing, &route.to_string())?;
    }

    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::simd;
    use crate::vsa::{Dimension, Encoding};

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    fn test_engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    fn test_engine_durable() -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let config = EngineConfig {
            data_dir: Some(dir.path().to_path_buf()),
            ..EngineConfig::default()
        };
        let engine = Engine::new(config).unwrap();
        (engine, dir)
    }

    fn important_email() -> ParsedEmail {
        ParsedEmail {
            message_id: "<imp-001@company.com>".to_string(),
            from: "boss@company.com".to_string(),
            from_display: Some("Boss Person".to_string()),
            to: vec!["me@company.com".to_string()],
            cc: Vec::new(),
            subject: "Urgent: Q4 review meeting".to_string(),
            date: Some(1700000000),
            in_reply_to: Some("<prev-001@company.com>".to_string()),
            references: vec![
                "<root@company.com>".to_string(),
                "<prev-001@company.com>".to_string(),
            ],
            body_text: Some(
                "Please prepare the quarterly review slides for tomorrow's meeting. \
                 This is critical for the board presentation."
                    .to_string(),
            ),
            body_html: None,
            has_attachments: true,
            list_id: None,
            content_type: "multipart/mixed".to_string(),
        }
    }

    fn low_priority_email() -> ParsedEmail {
        ParsedEmail {
            message_id: "<low-001@notifications.example.com>".to_string(),
            from: "noreply@notifications.example.com".to_string(),
            from_display: Some("Notifications".to_string()),
            to: vec!["me@company.com".to_string()],
            cc: Vec::new(),
            subject: "Your monthly statement is ready".to_string(),
            date: Some(1700000000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some("Your statement for December is now available online.".to_string()),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        }
    }

    fn first_time_email() -> ParsedEmail {
        ParsedEmail {
            message_id: "<new-001@stranger.com>".to_string(),
            from: "stranger@unknown.com".to_string(),
            from_display: Some("Stranger".to_string()),
            to: vec!["me@company.com".to_string()],
            cc: Vec::new(),
            subject: "Hello there".to_string(),
            date: Some(1700000000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some("I'd like to introduce myself.".to_string()),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        }
    }

    // ── Untrained state ────────────────────────────────────────────────

    #[test]
    fn untrained_state() {
        let ops = test_ops();
        let eng = TriageEngine::new(&ops);

        assert!(!eng.is_trained());
        assert!(eng.sender_stats.is_empty());
        assert!(eng.important_prototype.is_none());
        assert!(eng.low_priority_prototype.is_none());
        assert_eq!(eng.important_count, 0);
        assert_eq!(eng.low_priority_count, 0);
    }

    // ── SenderStats lifecycle ──────────────────────────────────────────

    #[test]
    fn sender_stats_new() {
        let stats = SenderStats::new("test@example.com".to_string());
        assert_eq!(stats.address, "test@example.com");
        assert_eq!(stats.message_count, 0);
        assert_eq!(stats.reply_count, 0);
        assert_eq!(stats.reply_rate, 0.0);
        assert_eq!(stats.relationship, SenderRelationship::Unknown);
        assert!(stats.routing.is_none());
        assert!(stats.needs_screening());
    }

    #[test]
    fn sender_stats_record_message() {
        let mut stats = SenderStats::new("test@example.com".to_string());
        stats.record_message(1000);
        assert_eq!(stats.message_count, 1);
        assert_eq!(stats.last_message_ts, 1000);

        stats.record_message(2000);
        assert_eq!(stats.message_count, 2);
        assert_eq!(stats.last_message_ts, 2000);
    }

    #[test]
    fn sender_stats_record_reply() {
        let mut stats = SenderStats::new("test@example.com".to_string());
        stats.record_reply(300, 1000);
        assert_eq!(stats.reply_count, 1);
        assert_eq!(stats.last_reply_ts, 1000);
        assert!(stats.reply_rate > 0.0);
        assert!(stats.avg_reply_time_secs > 0.0);
    }

    #[test]
    fn sender_stats_needs_screening() {
        let mut stats = SenderStats::new("test@example.com".to_string());

        // No routing, no messages → needs screening
        assert!(stats.needs_screening());

        // Record one message → still needs screening (message_count <= 1)
        stats.record_message(1000);
        assert!(stats.needs_screening());

        // Set routing → no longer needs screening
        stats.routing = Some(EmailRoute::Feed);
        assert!(!stats.needs_screening());

        // Clear routing but message_count > 1 → no screening
        stats.routing = None;
        stats.record_message(2000);
        assert!(!stats.needs_screening());
    }

    // ── Role vectors ───────────────────────────────────────────────────

    #[test]
    fn role_vectors_deterministic() {
        let ops = test_ops();
        let rv1 = TriageRoleVectors::generate(&ops);
        let rv2 = TriageRoleVectors::generate(&ops);

        assert_eq!(rv1.sender_reputation, rv2.sender_reputation);
        assert_eq!(rv1.reply_rate, rv2.reply_rate);
        assert_eq!(rv1.subject, rv2.subject);
        assert_eq!(rv1.body, rv2.body);
    }

    #[test]
    fn role_vectors_distinct() {
        let ops = test_ops();
        let rv = TriageRoleVectors::generate(&ops);

        let sim = ops
            .similarity(&rv.sender_reputation, &rv.reply_rate)
            .unwrap();
        assert!(sim < 0.6, "role vectors should be dissimilar: sim={sim}");

        let sim2 = ops.similarity(&rv.subject, &rv.body).unwrap();
        assert!(
            sim2 < 0.6,
            "role vectors should be dissimilar: sim={sim2}"
        );
    }

    // ── Encoding ───────────────────────────────────────────────────────

    #[test]
    fn encode_email_valid() {
        let ops = test_ops();
        let eng = TriageEngine::new(&ops);
        let email = important_email();

        let vec = eng.encode_email(&ops, &email).unwrap();
        assert_eq!(vec.dim(), ops.dim());
    }

    #[test]
    fn encode_email_deterministic() {
        let ops = test_ops();
        let eng = TriageEngine::new(&ops);
        let email = important_email();

        let v1 = eng.encode_email(&ops, &email).unwrap();
        let v2 = eng.encode_email(&ops, &email).unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn encode_email_different() {
        let ops = test_ops();
        let eng = TriageEngine::new(&ops);

        let v_imp = eng.encode_email(&ops, &important_email()).unwrap();
        let v_low = eng.encode_email(&ops, &low_priority_email()).unwrap();

        let sim = ops.similarity(&v_imp, &v_low).unwrap();
        assert!(
            sim < 0.9,
            "different emails should produce different vectors: sim={sim}"
        );
    }

    // ── Scoring ────────────────────────────────────────────────────────

    #[test]
    fn social_scoring() {
        let ops = test_ops();
        let eng = TriageEngine::new(&ops);
        let now = 1700000000_u64;

        // New sender with no history
        let stats = SenderStats::new("test@example.com".to_string());
        let score = eng.score_social(&stats, now);
        assert!(score >= 0.0 && score <= 1.0);

        // Sender with some history
        let mut active_stats = SenderStats::new("active@example.com".to_string());
        active_stats.message_count = 50;
        active_stats.reply_rate = 0.8;
        active_stats.last_message_ts = now;
        active_stats.relationship = SenderRelationship::Colleague;

        let active_score = eng.score_social(&active_stats, now);
        assert!(
            active_score > score,
            "active sender should score higher: {active_score} vs {score}"
        );
    }

    #[test]
    fn content_scoring_untrained() {
        let ops = test_ops();
        let eng = TriageEngine::new(&ops);

        let score = eng.score_content(&ops, &important_email()).unwrap();
        assert!(
            (score - 0.5).abs() < 0.01,
            "untrained content should be neutral: {score}"
        );
    }

    #[test]
    fn thread_scoring() {
        // Email in a thread
        let threaded = important_email(); // has in_reply_to and references
        let thread_score = TriageEngine::score_thread(&threaded);
        assert!(thread_score > 0.0);

        // Email not in a thread
        let standalone = low_priority_email(); // no in_reply_to, no references
        let standalone_score = TriageEngine::score_thread(&standalone);
        assert_eq!(standalone_score, 0.0);

        assert!(
            thread_score > standalone_score,
            "threaded email should score higher: {thread_score} vs {standalone_score}"
        );
    }

    #[test]
    fn label_scoring() {
        assert_eq!(TriageEngine::score_label(Some(EmailRoute::Important)), 1.0);
        assert_eq!(TriageEngine::score_label(Some(EmailRoute::Feed)), 0.6);
        assert_eq!(TriageEngine::score_label(Some(EmailRoute::PaperTrail)), 0.3);
        assert_eq!(TriageEngine::score_label(Some(EmailRoute::Spam)), 0.0);
        assert_eq!(
            TriageEngine::score_label(Some(EmailRoute::ScreeningQueue)),
            0.5
        );
        assert_eq!(TriageEngine::score_label(None), 0.5);
    }

    // ── Triage pipeline ────────────────────────────────────────────────

    #[test]
    fn triage_first_time_screening() {
        let ops = test_ops();
        let mut eng = TriageEngine::new(&ops);

        let result = eng
            .triage(&ops, &first_time_email(), 1700000000, "me@company.com")
            .unwrap();

        assert_eq!(result.route, EmailRoute::ScreeningQueue);
        assert!(result.needs_screening);
    }

    #[test]
    fn triage_known_sender_routing() {
        let ops = test_ops();
        let mut eng = TriageEngine::new(&ops);

        // Pre-register sender
        eng.set_sender_routing("boss@company.com", EmailRoute::Important);

        // First triage creates the record (message_count becomes 1, but routing is set)
        let result = eng
            .triage(&ops, &important_email(), 1700000000, "me@company.com")
            .unwrap();

        // Should not be screening since routing is set
        assert!(!result.needs_screening);
        assert_ne!(result.route, EmailRoute::ScreeningQueue);
    }

    // ── set_sender_routing / set_sender_relationship ───────────────────

    #[test]
    fn set_sender_routing() {
        let ops = test_ops();
        let mut eng = TriageEngine::new(&ops);

        eng.set_sender_routing("alice@example.com", EmailRoute::Feed);
        let stats = eng.sender_stats("alice@example.com").unwrap();
        assert_eq!(stats.routing, Some(EmailRoute::Feed));
    }

    #[test]
    fn set_sender_relationship() {
        let ops = test_ops();
        let mut eng = TriageEngine::new(&ops);

        eng.set_sender_relationship("bob@example.com", SenderRelationship::Colleague);
        let stats = eng.sender_stats("bob@example.com").unwrap();
        assert_eq!(stats.relationship, SenderRelationship::Colleague);
    }

    // ── Training ───────────────────────────────────────────────────────

    #[test]
    fn train_important() {
        let ops = test_ops();
        let mut eng = TriageEngine::new(&ops);

        assert!(!eng.is_trained());
        eng.train_important(&ops, &important_email()).unwrap();
        assert!(!eng.is_trained()); // Need both

        eng.train_low_priority(&ops, &low_priority_email()).unwrap();
        assert!(eng.is_trained());
        assert_eq!(eng.important_count, 1);
        assert_eq!(eng.low_priority_count, 1);
    }

    #[test]
    fn train_low_priority() {
        let ops = test_ops();
        let mut eng = TriageEngine::new(&ops);

        eng.train_low_priority(&ops, &low_priority_email()).unwrap();
        assert_eq!(eng.low_priority_count, 1);
        assert!(eng.low_priority_prototype.is_some());
    }

    // ── Route thresholds ───────────────────────────────────────────────

    #[test]
    fn route_thresholds() {
        // Verify the threshold constants make sense
        assert!(THRESHOLD_IMPORTANT > THRESHOLD_FEED);
        assert!(THRESHOLD_FEED > THRESHOLD_PAPER_TRAIL);
        assert!(THRESHOLD_PAPER_TRAIL > 0.0);
    }

    // ── Persistence ────────────────────────────────────────────────────

    #[test]
    fn persistence_roundtrip() {
        let ops = test_ops();
        let (engine, _dir) = test_engine_durable();
        let mut eng = TriageEngine::new(&ops);

        eng.train_important(&ops, &important_email()).unwrap();
        eng.train_low_priority(&ops, &low_priority_email()).unwrap();
        eng.set_sender_routing("alice@example.com", EmailRoute::Feed);

        eng.persist(&engine).unwrap();

        let restored = TriageEngine::restore(&engine, &ops).unwrap().unwrap();
        assert!(restored.is_trained());
        assert_eq!(restored.important_count, 1);
        assert_eq!(restored.low_priority_count, 1);
        assert!(restored.sender_stats("alice@example.com").is_some());
    }

    #[test]
    fn restore_none() {
        let ops = test_ops();
        let (engine, _dir) = test_engine_durable();

        let result = TriageEngine::restore(&engine, &ops).unwrap();
        assert!(result.is_none());
    }

    // ── Eviction ───────────────────────────────────────────────────────

    #[test]
    fn eviction() {
        let ops = test_ops();
        let mut eng = TriageEngine::new(&ops);

        // Add MAX_SENDER_STATS + 10 senders
        for i in 0..MAX_SENDER_STATS + 10 {
            let addr = format!("user{i}@example.com");
            let mut stats = SenderStats::new(addr.clone());
            stats.last_message_ts = i as u64;
            stats.message_count = 2; // avoid screening
            stats.routing = Some(EmailRoute::Feed);
            eng.sender_stats.insert(addr, stats);
        }

        eng.evict_oldest_senders();

        assert!(
            eng.sender_stats.len() <= MAX_SENDER_STATS,
            "should have evicted: {} entries",
            eng.sender_stats.len()
        );
    }

    // ── Display traits ─────────────────────────────────────────────────

    #[test]
    fn display_traits() {
        assert_eq!(EmailRoute::Important.to_string(), "Important");
        assert_eq!(EmailRoute::Feed.to_string(), "Feed");
        assert_eq!(EmailRoute::PaperTrail.to_string(), "PaperTrail");
        assert_eq!(EmailRoute::ScreeningQueue.to_string(), "ScreeningQueue");
        assert_eq!(EmailRoute::Spam.to_string(), "Spam");

        assert_eq!(SenderRelationship::Colleague.to_string(), "Colleague");
        assert_eq!(SenderRelationship::Friend.to_string(), "Friend");
        assert_eq!(SenderRelationship::Service.to_string(), "Service");
        assert_eq!(SenderRelationship::Newsletter.to_string(), "Newsletter");
        assert_eq!(SenderRelationship::Unknown.to_string(), "Unknown");
    }

    // ── ImportanceWeights default ──────────────────────────────────────

    #[test]
    fn importance_weights_default() {
        let w = ImportanceWeights::default();
        let sum = w.social + w.content + w.thread + w.label;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "weights should sum to 1.0: {sum}"
        );
    }
}
