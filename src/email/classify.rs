//! OnlineHD spam & relevance classification (Phase 13b).
//!
//! VSA-native spam/ham classification using:
//! - **OnlineHD** prototype vectors for single-pass incremental learning
//! - **Robinson chi-square** Bayesian token probability supplement
//! - **Deterministic rule overrides** (whitelist/blacklist/mailing-list)
//! - **User feedback training** via `train()`
//! - **Persistence** via redb (`put_meta`/`get_meta` + bincode)
//! - **Full provenance** via `DerivationKind::SpamClassification`

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;
use crate::vsa::encode::{encode_label, encode_token};
use crate::vsa::ops::VsaOps;

use super::error::{EmailError, EmailResult};
use super::parser::{ParsedEmail, extract_domain};

// ── Constants ──────────────────────────────────────────────────────────────

/// Redb meta key for persisting the spam classifier.
const META_KEY_CLASSIFIER: &[u8] = b"email:spam_classifier";

/// Default combined-score threshold above which email is classified as spam.
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.55;

/// Maximum number of tokens in the probability table before eviction.
const MAX_TOKEN_TABLE_SIZE: usize = 50_000;

/// Number of most-informative tokens to use for Robinson chi-square.
const ROBINSON_TOP_N: usize = 15;

/// Maximum body text bytes to use as features.
const BODY_PREVIEW_LEN: usize = 512;

/// Number of four-hour time buckets in a day.
const TIME_BUCKETS: u64 = 6;

/// Weight of VSA similarity in the combined score.
const VSA_WEIGHT: f32 = 0.7;

/// Weight of Bayesian score in the combined score.
const BAYESIAN_WEIGHT: f32 = 0.3;

/// Threshold above which the combined score is spam.
const SPAM_THRESHOLD: f32 = 0.55;

/// Threshold below which the combined score is ham.
const HAM_THRESHOLD: f32 = 0.45;

// ── SpamDecision ───────────────────────────────────────────────────────────

/// Classification decision for an email.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpamDecision {
    Spam,
    Ham,
    Uncertain,
}

impl std::fmt::Display for SpamDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spam => write!(f, "spam"),
            Self::Ham => write!(f, "ham"),
            Self::Uncertain => write!(f, "uncertain"),
        }
    }
}

// ── ClassificationResult ───────────────────────────────────────────────────

/// Full result of classifying an email, including all scoring signals.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Final decision.
    pub decision: SpamDecision,
    /// VSA cosine similarity to the spam prototype (0.0–1.0).
    pub vsa_spam_similarity: f32,
    /// VSA cosine similarity to the ham prototype (0.0–1.0).
    pub vsa_ham_similarity: f32,
    /// Robinson chi-square Bayesian score (0.0 = ham, 1.0 = spam).
    pub bayesian_score: f32,
    /// Combined confidence score (0.0 = ham, 1.0 = spam).
    pub confidence: f32,
    /// Whether a deterministic rule override was applied.
    pub rule_override: Option<String>,
    /// Human-readable reasoning string.
    pub reasoning: String,
}

// ── SpamRoleVectors ────────────────────────────────────────────────────────

/// Pre-generated deterministic role hypervectors for email feature encoding.
///
/// Each role vector is produced via `encode_token(ops, "email-role:X")`, giving
/// deterministic, well-separated role vectors for structured VSA binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpamRoleVectors {
    pub sender: HyperVec,
    pub domain: HyperVec,
    pub subject: HyperVec,
    pub body: HyperVec,
    pub has_attachments: HyperVec,
    pub has_list_id: HyperVec,
    pub time_bucket: HyperVec,
}

impl SpamRoleVectors {
    /// Generate all 7 role vectors deterministically from the VsaOps.
    fn generate(ops: &VsaOps) -> Self {
        Self {
            sender: encode_token(ops, "email-role:sender"),
            domain: encode_token(ops, "email-role:domain"),
            subject: encode_token(ops, "email-role:subject"),
            body: encode_token(ops, "email-role:body"),
            has_attachments: encode_token(ops, "email-role:has-attachments"),
            has_list_id: encode_token(ops, "email-role:has-list-id"),
            time_bucket: encode_token(ops, "email-role:time-bucket"),
        }
    }
}

// ── TokenStats ─────────────────────────────────────────────────────────────

/// Per-token spam/ham occurrence counts for Bayesian scoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenStats {
    pub spam_count: u32,
    pub ham_count: u32,
}

// ── TokenProbabilityTable ──────────────────────────────────────────────────

/// Bayesian token probability table (Robinson chi-square supplement).
///
/// Tracks per-token spam/ham counts and computes Robinson's chi-square
/// combination of the top-N most informative token probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenProbabilityTable {
    tokens: HashMap<String, TokenStats>,
    total_spam: u64,
    total_ham: u64,
}

impl TokenProbabilityTable {
    fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            total_spam: 0,
            total_ham: 0,
        }
    }

    /// Record a training example's tokens.
    fn train(&mut self, tokens: &[String], is_spam: bool) {
        if is_spam {
            self.total_spam += 1;
        } else {
            self.total_ham += 1;
        }

        for token in tokens {
            let stats = self.tokens.entry(token.clone()).or_default();
            if is_spam {
                stats.spam_count += 1;
            } else {
                stats.ham_count += 1;
            }
        }

        // Evict least-seen tokens if table exceeds max size.
        if self.tokens.len() > MAX_TOKEN_TABLE_SIZE {
            self.evict_least_seen();
        }
    }

    /// Compute Robinson chi-square combined score for the given tokens.
    ///
    /// Returns a score in \[0.0, 1.0\] where 1.0 = definitely spam, 0.0 = definitely ham.
    /// Returns 0.5 (neutral) if there are no informative tokens.
    fn robinson_chi_square(&self, tokens: &[String]) -> f32 {
        if self.total_spam == 0 && self.total_ham == 0 {
            return 0.5;
        }

        // Compute per-token spam probability using Robinson's formula:
        //   p(token) = spam_count / (spam_count + ham_count)
        // with a strength parameter s=1.0 and assumed prob x=0.5.
        let s = 1.0_f64;
        let x = 0.5_f64;

        let mut scored: Vec<(f64, &str)> = tokens
            .iter()
            .filter_map(|t| {
                let stats = self.tokens.get(t.as_str())?;
                let n = stats.spam_count as f64 + stats.ham_count as f64;
                if n == 0.0 {
                    return None;
                }
                let raw_p = stats.spam_count as f64 / n;
                // Robinson's formula with strength prior
                let p = (s * x + n * raw_p) / (s + n);
                Some((p, t.as_str()))
            })
            .collect();

        if scored.is_empty() {
            return 0.5;
        }

        // Select the top-N most informative tokens (highest |p - 0.5|).
        scored.sort_by(|a, b| {
            let dev_a = (a.0 - 0.5).abs();
            let dev_b = (b.0 - 0.5).abs();
            dev_b.partial_cmp(&dev_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(ROBINSON_TOP_N);

        let n = scored.len() as f64;

        // H = -2 * ln(∏(1 - p_i)) — evidence for ham
        // S = -2 * ln(∏ p_i)       — evidence for spam
        let ln_prod_1mp: f64 = scored.iter().map(|(p, _)| (1.0 - p).max(1e-200).ln()).sum();
        let ln_prod_p: f64 = scored.iter().map(|(p, _)| p.max(1e-200).ln()).sum();

        let h = -2.0 * ln_prod_1mp;
        let s_stat = -2.0 * ln_prod_p;

        // Simplified Fisher approximation using chi-square CDF:
        // chi2_cdf(x, 2n) ≈ 1 - exp(-x/2) * Σ_{k=0}^{n-1} (x/2)^k / k!
        let chi2_cdf = |x: f64, df: f64| -> f64 {
            let half_df = (df / 2.0) as usize;
            let half_x = x / 2.0;
            let mut sum = 0.0_f64;
            let mut term = 1.0_f64;
            for k in 0..half_df {
                if k > 0 {
                    term *= half_x / k as f64;
                }
                sum += term;
            }
            1.0 - (-half_x).exp() * sum
        };

        let df = 2.0 * n;
        let h_p = chi2_cdf(h, df); // probability email is spam (ham evidence is low)
        let s_p = chi2_cdf(s_stat, df); // probability email is ham (spam evidence is low)

        // Combined: (h_p - s_p + 1) / 2 — maps to [0, 1] where 1 = spam
        let combined = ((h_p - s_p + 1.0) / 2.0).clamp(0.0, 1.0);
        combined as f32
    }

    /// Evict the least-seen tokens to bring table back to 80% capacity.
    fn evict_least_seen(&mut self) {
        let target = MAX_TOKEN_TABLE_SIZE * 4 / 5;
        let mut entries: Vec<(String, u32)> = self
            .tokens
            .iter()
            .map(|(k, v)| (k.clone(), v.spam_count + v.ham_count))
            .collect();
        entries.sort_by_key(|e| e.1);

        let to_remove = self.tokens.len().saturating_sub(target);
        for (key, _) in entries.iter().take(to_remove) {
            self.tokens.remove(key);
        }
    }
}

// ── SpamClassifier ─────────────────────────────────────────────────────────

/// OnlineHD spam classifier with Bayesian supplement and deterministic overrides.
///
/// The classifier starts untrained (no prototypes). After at least one spam
/// and one ham training example, it produces meaningful VSA similarity scores.
/// The Bayesian token table also requires training data.
///
/// # Classification pipeline
///
/// 1. Deterministic rule check (whitelist → Ham, blacklist → Spam, list-id → Ham)
/// 2. VSA encoding of email features via role-filler binding
/// 3. Hamming similarity to spam/ham prototypes
/// 4. Robinson chi-square Bayesian score from token probabilities
/// 5. Combined score: `0.7 * VSA_normalized + 0.3 * Bayesian`
/// 6. Threshold decision (>0.55 → Spam, <0.45 → Ham, else Uncertain)
///
/// # OnlineHD adaptive update
///
/// Training uses majority-vote bundling: `bundle([existing_prototype, new_example])`.
/// The existing prototype encodes accumulated history; new examples have diminishing
/// contribution as more training data accumulates (natural property of majority vote).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpamClassifier {
    spam_prototype: Option<HyperVec>,
    ham_prototype: Option<HyperVec>,
    spam_count: u64,
    ham_count: u64,
    token_probs: TokenProbabilityTable,
    role_vectors: SpamRoleVectors,
    whitelist: Vec<String>,
    blacklist: Vec<String>,
    confidence_threshold: f32,
}

impl SpamClassifier {
    /// Create a new untrained classifier with role vectors.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            spam_prototype: None,
            ham_prototype: None,
            spam_count: 0,
            ham_count: 0,
            token_probs: TokenProbabilityTable::new(),
            role_vectors: SpamRoleVectors::generate(ops),
            whitelist: Vec::new(),
            blacklist: Vec::new(),
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        }
    }

    /// Encode an email into a composite hypervector using role-filler binding.
    ///
    /// Features:
    /// 1. `bind(role_domain, encode_token(domain))` — sender domain
    /// 2. `bind(role_subject, encode_label(subject))` — subject keywords
    /// 3. `bind(role_body, encode_label(body_preview))` — body preview
    /// 4. `bind(role_has_attachments, encode_token("true"/"false"))` — boolean
    /// 5. `bind(role_has_list_id, encode_token("true"/"false"))` — boolean
    /// 6. `bind(role_time_bucket, encode_token("bucket-N"))` — 6 four-hour windows
    pub fn encode_email(&self, ops: &VsaOps, email: &ParsedEmail) -> EmailResult<HyperVec> {
        let mut feature_vecs: Vec<HyperVec> = Vec::with_capacity(7);

        // 1. Sender domain
        let domain = extract_domain(&email.from);
        let domain_filler = encode_token(ops, domain);
        let domain_bound = ops
            .bind(&self.role_vectors.domain, &domain_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for domain: {e}"),
            })?;
        feature_vecs.push(domain_bound);

        // 2. Subject keywords
        if !email.subject.is_empty() {
            let subj_filler = encode_label(ops, &email.subject).map_err(|e| EmailError::Parse {
                message: format!("VSA encode_label failed for subject: {e}"),
            })?;
            let subj_bound = ops
                .bind(&self.role_vectors.subject, &subj_filler)
                .map_err(|e| EmailError::Parse {
                    message: format!("VSA bind failed for subject: {e}"),
                })?;
            feature_vecs.push(subj_bound);
        }

        // 3. Body preview
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
                feature_vecs.push(body_bound);
            }
        }

        // 4. Has attachments
        let attach_token = if email.has_attachments { "true" } else { "false" };
        let attach_filler = encode_token(ops, attach_token);
        let attach_bound = ops
            .bind(&self.role_vectors.has_attachments, &attach_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for attachments: {e}"),
            })?;
        feature_vecs.push(attach_bound);

        // 5. Has list-id
        let list_token = if email.list_id.is_some() {
            "true"
        } else {
            "false"
        };
        let list_filler = encode_token(ops, list_token);
        let list_bound = ops
            .bind(&self.role_vectors.has_list_id, &list_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for list-id: {e}"),
            })?;
        feature_vecs.push(list_bound);

        // 6. Time bucket (6 four-hour windows)
        let bucket = email
            .date
            .map(|ts| {
                let hour = (ts / 3600) % 24;
                hour / (24 / TIME_BUCKETS)
            })
            .unwrap_or(0);
        let bucket_token = format!("bucket-{bucket}");
        let bucket_filler = encode_token(ops, &bucket_token);
        let bucket_bound = ops
            .bind(&self.role_vectors.time_bucket, &bucket_filler)
            .map_err(|e| EmailError::Parse {
                message: format!("VSA bind failed for time bucket: {e}"),
            })?;
        feature_vecs.push(bucket_bound);

        // Bundle all features
        let refs: Vec<&HyperVec> = feature_vecs.iter().collect();
        ops.bundle(&refs).map_err(|e| EmailError::Parse {
            message: format!("VSA bundle failed for email encoding: {e}"),
        })
    }

    /// Classify an email through the full pipeline.
    ///
    /// Returns a [`ClassificationResult`] with the decision, scores, and reasoning.
    pub fn classify(
        &self,
        ops: &VsaOps,
        email: &ParsedEmail,
    ) -> EmailResult<ClassificationResult> {
        // 1. Deterministic rule overrides
        let domain = extract_domain(&email.from);

        if self.whitelist.iter().any(|w| domain.eq_ignore_ascii_case(w)) {
            return Ok(ClassificationResult {
                decision: SpamDecision::Ham,
                vsa_spam_similarity: 0.0,
                vsa_ham_similarity: 0.0,
                bayesian_score: 0.0,
                confidence: 0.0,
                rule_override: Some(format!("whitelisted domain: {domain}")),
                reasoning: format!("Domain {domain} is whitelisted — classified as ham"),
            });
        }

        if self.blacklist.iter().any(|b| domain.eq_ignore_ascii_case(b)) {
            return Ok(ClassificationResult {
                decision: SpamDecision::Spam,
                vsa_spam_similarity: 0.0,
                vsa_ham_similarity: 0.0,
                bayesian_score: 0.0,
                confidence: 1.0,
                rule_override: Some(format!("blacklisted domain: {domain}")),
                reasoning: format!("Domain {domain} is blacklisted — classified as spam"),
            });
        }

        if email.list_id.is_some() {
            return Ok(ClassificationResult {
                decision: SpamDecision::Ham,
                vsa_spam_similarity: 0.0,
                vsa_ham_similarity: 0.0,
                bayesian_score: 0.0,
                confidence: 0.0,
                rule_override: Some("mailing list (List-Id present)".to_string()),
                reasoning: "Email has List-Id header — classified as ham (mailing list)"
                    .to_string(),
            });
        }

        // 2. Encode email
        let email_vec = self.encode_email(ops, email)?;

        // 3. VSA similarity to prototypes
        let (vsa_spam_sim, vsa_ham_sim) = match (&self.spam_prototype, &self.ham_prototype) {
            (Some(spam_proto), Some(ham_proto)) => {
                let spam_sim = ops.similarity(&email_vec, spam_proto).unwrap_or(0.5);
                let ham_sim = ops.similarity(&email_vec, ham_proto).unwrap_or(0.5);
                (spam_sim, ham_sim)
            }
            _ => (0.5, 0.5), // Untrained — neutral
        };

        // Normalize VSA to [0,1] where 1.0 = spam
        let vsa_range = vsa_spam_sim + vsa_ham_sim;
        let vsa_normalized = if vsa_range > 0.0 {
            vsa_spam_sim / vsa_range
        } else {
            0.5
        };

        // 4. Bayesian supplement
        let tokens = extract_tokens(email);
        let bayesian_score = self.token_probs.robinson_chi_square(&tokens);

        // 5. Combined score
        let combined = VSA_WEIGHT * vsa_normalized + BAYESIAN_WEIGHT * bayesian_score;

        // 6. Threshold decision
        let decision = if combined > SPAM_THRESHOLD {
            SpamDecision::Spam
        } else if combined < HAM_THRESHOLD {
            SpamDecision::Ham
        } else {
            SpamDecision::Uncertain
        };

        let reasoning = format!(
            "VSA: spam_sim={vsa_spam_sim:.3}, ham_sim={vsa_ham_sim:.3}, \
             normalized={vsa_normalized:.3}; \
             Bayesian: {bayesian_score:.3}; \
             Combined: {combined:.3} (threshold: {SPAM_THRESHOLD}/{HAM_THRESHOLD})"
        );

        Ok(ClassificationResult {
            decision,
            vsa_spam_similarity: vsa_spam_sim,
            vsa_ham_similarity: vsa_ham_sim,
            bayesian_score,
            confidence: combined,
            rule_override: None,
            reasoning,
        })
    }

    /// Train the classifier with a labeled email example.
    ///
    /// Uses OnlineHD adaptive update: bundles the new email vector with the
    /// existing prototype. The existing prototype encodes the accumulated history
    /// of all prior examples; majority vote naturally gives diminishing weight
    /// to each new example as more data accumulates.
    pub fn train(
        &mut self,
        ops: &VsaOps,
        email: &ParsedEmail,
        is_spam: bool,
    ) -> EmailResult<()> {
        let email_vec = self.encode_email(ops, email)?;

        if is_spam {
            self.spam_prototype = Some(match &self.spam_prototype {
                Some(existing) => {
                    let refs = [existing, &email_vec];
                    ops.bundle(&refs).map_err(|e| EmailError::Parse {
                        message: format!("VSA bundle failed during spam training: {e}"),
                    })?
                }
                None => email_vec,
            });
            self.spam_count += 1;
        } else {
            self.ham_prototype = Some(match &self.ham_prototype {
                Some(existing) => {
                    let refs = [existing, &email_vec];
                    ops.bundle(&refs).map_err(|e| EmailError::Parse {
                        message: format!("VSA bundle failed during ham training: {e}"),
                    })?
                }
                None => email_vec,
            });
            self.ham_count += 1;
        }

        // Train Bayesian token table
        let tokens = extract_tokens(email);
        self.token_probs.train(&tokens, is_spam);

        Ok(())
    }

    /// Add a domain to the operator whitelist (emails from this domain → Ham).
    pub fn whitelist_domain(&mut self, domain: String) {
        if !self.whitelist.iter().any(|w| w.eq_ignore_ascii_case(&domain)) {
            self.whitelist.push(domain);
        }
    }

    /// Add a domain to the operator blacklist (emails from this domain → Spam).
    pub fn blacklist_domain(&mut self, domain: String) {
        if !self.blacklist.iter().any(|b| b.eq_ignore_ascii_case(&domain)) {
            self.blacklist.push(domain);
        }
    }

    /// Remove a domain from the whitelist. Returns `true` if it was present.
    pub fn remove_whitelist(&mut self, domain: &str) -> bool {
        let before = self.whitelist.len();
        self.whitelist
            .retain(|w| !w.eq_ignore_ascii_case(domain));
        self.whitelist.len() < before
    }

    /// Remove a domain from the blacklist. Returns `true` if it was present.
    pub fn remove_blacklist(&mut self, domain: &str) -> bool {
        let before = self.blacklist.len();
        self.blacklist
            .retain(|b| !b.eq_ignore_ascii_case(domain));
        self.blacklist.len() < before
    }

    /// Whether both prototypes are present (at least one spam + one ham example).
    pub fn is_trained(&self) -> bool {
        self.spam_prototype.is_some() && self.ham_prototype.is_some()
    }

    /// Total number of training examples seen.
    pub fn training_count(&self) -> u64 {
        self.spam_count + self.ham_count
    }

    /// Persist the classifier to the engine's durable store via bincode.
    pub fn persist(&self, engine: &Engine) -> EmailResult<()> {
        let encoded = bincode::serialize(self).map_err(|e| EmailError::Parse {
            message: format!("failed to serialize spam classifier: {e}"),
        })?;
        engine
            .store()
            .put_meta(META_KEY_CLASSIFIER, &encoded)
            .map_err(|e| EmailError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    /// Restore a previously persisted classifier from the engine's durable store.
    ///
    /// Returns `Ok(None)` if no classifier has been persisted yet (including
    /// when the meta table hasn't been created).
    /// If the stored data is corrupted, returns a fresh classifier via `new(ops)`.
    pub fn restore(engine: &Engine, ops: &VsaOps) -> EmailResult<Option<Self>> {
        let data = match engine.store().get_meta(META_KEY_CLASSIFIER) {
            Ok(d) => d,
            Err(_) => {
                // Meta table may not exist yet — treat as "not persisted".
                return Ok(None);
            }
        };

        match data {
            Some(bytes) => match bincode::deserialize::<Self>(&bytes) {
                Ok(classifier) => Ok(Some(classifier)),
                Err(_) => {
                    // Corrupted data — return a fresh classifier.
                    Ok(Some(Self::new(ops)))
                }
            },
            None => Ok(None),
        }
    }
}

// ── Provenance ─────────────────────────────────────────────────────────────

/// Record provenance for an email classification decision.
///
/// Stores a `DerivationKind::SpamClassification` record linked to the email's
/// symbol in the KG.
pub fn record_classification_provenance(
    engine: &Engine,
    email_symbol: SymbolId,
    result: &ClassificationResult,
    email: &ParsedEmail,
) -> EmailResult<()> {
    let mut record = ProvenanceRecord::new(
        email_symbol,
        DerivationKind::SpamClassification {
            email_message_id: email.message_id.clone(),
            decision: result.decision.to_string(),
            vsa_confidence: result.confidence,
            bayesian_score: result.bayesian_score,
        },
    );
    record.confidence = 1.0 - result.confidence.clamp(0.0, 1.0);
    // Confidence here represents how confident we are in the classification.
    // For spam: confidence is the combined score itself.
    // For ham: confidence is 1.0 - combined score.
    record.confidence = match result.decision {
        SpamDecision::Spam => result.confidence,
        SpamDecision::Ham => 1.0 - result.confidence,
        SpamDecision::Uncertain => 0.5,
    };

    engine
        .store_provenance(&mut record)
        .map_err(|e| EmailError::Engine(Box::new(e)))?;
    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Extract classification-relevant tokens from a parsed email.
///
/// Tokenizes sender domain, subject, and body preview into lowercase words.
fn extract_tokens(email: &ParsedEmail) -> Vec<String> {
    let mut tokens = Vec::new();

    // Domain token
    let domain = extract_domain(&email.from);
    tokens.push(format!("domain:{}", domain.to_ascii_lowercase()));

    // Subject tokens
    for word in email.subject.split_whitespace() {
        let lower = word.to_ascii_lowercase();
        if lower.len() >= 2 {
            tokens.push(lower);
        }
    }

    // Body tokens (first BODY_PREVIEW_LEN bytes)
    if let Some(body) = email.best_body() {
        let preview = if body.len() > BODY_PREVIEW_LEN {
            &body[..body.floor_char_boundary(BODY_PREVIEW_LEN)]
        } else {
            body
        };
        for word in preview.split_whitespace() {
            let lower = word.to_ascii_lowercase();
            if lower.len() >= 2 {
                tokens.push(lower);
            }
        }
    }

    tokens
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

    fn spam_email() -> ParsedEmail {
        ParsedEmail {
            message_id: "<spam-001@spammer.com>".to_string(),
            from: "scammer@spammer.com".to_string(),
            from_display: Some("Nigerian Prince".to_string()),
            to: vec!["victim@example.com".to_string()],
            cc: Vec::new(),
            subject: "URGENT: You Won Million Dollars Free Money".to_string(),
            date: Some(1700000000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some(
                "Congratulations! You have been selected to receive one million dollars. \
                 Click here immediately to claim your prize. Send bank details now."
                    .to_string(),
            ),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        }
    }

    fn ham_email() -> ParsedEmail {
        ParsedEmail {
            message_id: "<work-001@company.com>".to_string(),
            from: "alice@company.com".to_string(),
            from_display: Some("Alice Smith".to_string()),
            to: vec!["bob@company.com".to_string()],
            cc: Vec::new(),
            subject: "Meeting tomorrow at 10am".to_string(),
            date: Some(1700050000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some(
                "Hi Bob, just a reminder about our meeting tomorrow morning. \
                 Please bring the quarterly report."
                    .to_string(),
            ),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        }
    }

    fn list_email() -> ParsedEmail {
        ParsedEmail {
            message_id: "<list-001@lists.rust-lang.org>".to_string(),
            from: "noreply@rust-lang.org".to_string(),
            from_display: Some("Rust Announce".to_string()),
            to: vec!["user@example.com".to_string()],
            cc: Vec::new(),
            subject: "Rust 1.80 released".to_string(),
            date: Some(1700100000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some("Rust 1.80 is now available.".to_string()),
            body_html: None,
            has_attachments: false,
            list_id: Some("<announce.rust-lang.org>".to_string()),
            content_type: "text/plain".to_string(),
        }
    }

    // ── Untrained state ────────────────────────────────────────────────

    #[test]
    fn untrained_classifier_state() {
        let ops = test_ops();
        let clf = SpamClassifier::new(&ops);

        assert!(!clf.is_trained());
        assert_eq!(clf.training_count(), 0);
        assert_eq!(clf.spam_count, 0);
        assert_eq!(clf.ham_count, 0);
        assert!(clf.spam_prototype.is_none());
        assert!(clf.ham_prototype.is_none());
    }

    // ── Role vectors ───────────────────────────────────────────────────

    #[test]
    fn role_vectors_are_deterministic() {
        let ops = test_ops();
        let rv1 = SpamRoleVectors::generate(&ops);
        let rv2 = SpamRoleVectors::generate(&ops);

        assert_eq!(rv1.sender, rv2.sender);
        assert_eq!(rv1.domain, rv2.domain);
        assert_eq!(rv1.subject, rv2.subject);
        assert_eq!(rv1.body, rv2.body);
    }

    #[test]
    fn role_vectors_are_distinct() {
        let ops = test_ops();
        let rv = SpamRoleVectors::generate(&ops);

        let sim = ops.similarity(&rv.sender, &rv.domain).unwrap();
        assert!(sim < 0.6, "role vectors should be dissimilar: sim={sim}");

        let sim2 = ops.similarity(&rv.subject, &rv.body).unwrap();
        assert!(
            sim2 < 0.6,
            "role vectors should be dissimilar: sim={sim2}"
        );
    }

    // ── Encoding ───────────────────────────────────────────────────────

    #[test]
    fn encode_email_produces_valid_vector() {
        let ops = test_ops();
        let clf = SpamClassifier::new(&ops);
        let email = ham_email();

        let vec = clf.encode_email(&ops, &email).unwrap();
        assert_eq!(vec.dim(), ops.dim());
    }

    #[test]
    fn encode_email_is_deterministic() {
        let ops = test_ops();
        let clf = SpamClassifier::new(&ops);
        let email = ham_email();

        let v1 = clf.encode_email(&ops, &email).unwrap();
        let v2 = clf.encode_email(&ops, &email).unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_emails_produce_different_vectors() {
        let ops = test_ops();
        let clf = SpamClassifier::new(&ops);

        let v_spam = clf.encode_email(&ops, &spam_email()).unwrap();
        let v_ham = clf.encode_email(&ops, &ham_email()).unwrap();

        let sim = ops.similarity(&v_spam, &v_ham).unwrap();
        assert!(
            sim < 0.9,
            "different emails should produce different vectors: sim={sim}"
        );
    }

    // ── Training ───────────────────────────────────────────────────────

    #[test]
    fn training_creates_prototypes() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);

        assert!(!clf.is_trained());

        clf.train(&ops, &spam_email(), true).unwrap();
        assert!(!clf.is_trained()); // Need both

        clf.train(&ops, &ham_email(), false).unwrap();
        assert!(clf.is_trained());
        assert_eq!(clf.training_count(), 2);
    }

    #[test]
    fn training_updates_token_table() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);

        clf.train(&ops, &spam_email(), true).unwrap();
        assert_eq!(clf.token_probs.total_spam, 1);
        assert_eq!(clf.token_probs.total_ham, 0);
        assert!(!clf.token_probs.tokens.is_empty());

        clf.train(&ops, &ham_email(), false).unwrap();
        assert_eq!(clf.token_probs.total_spam, 1);
        assert_eq!(clf.token_probs.total_ham, 1);
    }

    // ── Classification ─────────────────────────────────────────────────

    #[test]
    fn untrained_classifier_returns_uncertain() {
        let ops = test_ops();
        let clf = SpamClassifier::new(&ops);

        let result = clf.classify(&ops, &ham_email()).unwrap();
        // Without prototypes, VSA is neutral (0.5) and Bayesian is neutral (0.5).
        // Combined = 0.5, which falls between thresholds → Uncertain.
        assert_eq!(result.decision, SpamDecision::Uncertain);
        assert!(result.rule_override.is_none());
    }

    #[test]
    fn trained_classifier_differentiates() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);

        // Train with multiple examples
        for _ in 0..5 {
            clf.train(&ops, &spam_email(), true).unwrap();
            clf.train(&ops, &ham_email(), false).unwrap();
        }

        let spam_result = clf.classify(&ops, &spam_email()).unwrap();
        let ham_result = clf.classify(&ops, &ham_email()).unwrap();

        // Spam email should score higher (more spam-like) than ham email
        assert!(
            spam_result.confidence > ham_result.confidence,
            "spam confidence ({}) should exceed ham confidence ({})",
            spam_result.confidence,
            ham_result.confidence
        );
    }

    // ── Whitelist / Blacklist ──────────────────────────────────────────

    #[test]
    fn whitelist_overrides_to_ham() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);
        clf.whitelist_domain("spammer.com".to_string());

        let result = clf.classify(&ops, &spam_email()).unwrap();
        assert_eq!(result.decision, SpamDecision::Ham);
        assert!(result.rule_override.is_some());
        assert!(result.rule_override.unwrap().contains("whitelisted"));
    }

    #[test]
    fn blacklist_overrides_to_spam() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);
        clf.blacklist_domain("company.com".to_string());

        let result = clf.classify(&ops, &ham_email()).unwrap();
        assert_eq!(result.decision, SpamDecision::Spam);
        assert!(result.rule_override.is_some());
        assert!(result.rule_override.unwrap().contains("blacklisted"));
    }

    #[test]
    fn list_id_overrides_to_ham() {
        let ops = test_ops();
        let clf = SpamClassifier::new(&ops);

        let result = clf.classify(&ops, &list_email()).unwrap();
        assert_eq!(result.decision, SpamDecision::Ham);
        assert!(result.rule_override.is_some());
        assert!(result.rule_override.unwrap().contains("mailing list"));
    }

    #[test]
    fn whitelist_dedup_and_remove() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);

        clf.whitelist_domain("example.com".to_string());
        clf.whitelist_domain("example.com".to_string()); // duplicate
        assert_eq!(clf.whitelist.len(), 1);

        assert!(clf.remove_whitelist("example.com"));
        assert!(!clf.remove_whitelist("example.com")); // already removed
        assert!(clf.whitelist.is_empty());
    }

    #[test]
    fn blacklist_dedup_and_remove() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);

        clf.blacklist_domain("evil.com".to_string());
        clf.blacklist_domain("evil.com".to_string());
        assert_eq!(clf.blacklist.len(), 1);

        assert!(clf.remove_blacklist("evil.com"));
        assert!(!clf.remove_blacklist("evil.com"));
    }

    // ── Token probability table ────────────────────────────────────────

    #[test]
    fn token_table_robinson_neutral_when_empty() {
        let table = TokenProbabilityTable::new();
        let score = table.robinson_chi_square(&["hello".to_string()]);
        assert!((score - 0.5).abs() < 0.01, "empty table should be neutral: {score}");
    }

    #[test]
    fn token_table_scoring_after_training() {
        let mut table = TokenProbabilityTable::new();

        // Train spam tokens
        let spam_tokens: Vec<String> = vec!["free", "money", "urgent", "click"]
            .into_iter()
            .map(String::from)
            .collect();
        table.train(&spam_tokens, true);
        table.train(&spam_tokens, true);

        // Train ham tokens
        let ham_tokens: Vec<String> = vec!["meeting", "report", "quarterly", "reminder"]
            .into_iter()
            .map(String::from)
            .collect();
        table.train(&ham_tokens, false);
        table.train(&ham_tokens, false);

        // Spam-like query should score high
        let spam_query: Vec<String> = vec!["free", "money"].into_iter().map(String::from).collect();
        let spam_score = table.robinson_chi_square(&spam_query);

        // Ham-like query should score low
        let ham_query: Vec<String> = vec!["meeting", "report"]
            .into_iter()
            .map(String::from)
            .collect();
        let ham_score = table.robinson_chi_square(&ham_query);

        assert!(
            spam_score > ham_score,
            "spam tokens ({spam_score}) should score higher than ham tokens ({ham_score})"
        );
    }

    #[test]
    fn token_table_eviction() {
        let mut table = TokenProbabilityTable::new();

        // Fill beyond MAX_TOKEN_TABLE_SIZE
        let tokens: Vec<String> = (0..MAX_TOKEN_TABLE_SIZE + 100)
            .map(|i| format!("token-{i}"))
            .collect();
        table.train(&tokens, true);

        // Should have evicted down to ~80%
        assert!(
            table.tokens.len() <= MAX_TOKEN_TABLE_SIZE,
            "table should be evicted: {} entries",
            table.tokens.len()
        );
    }

    // ── Persistence ────────────────────────────────────────────────────

    #[test]
    fn persistence_roundtrip() {
        let ops = test_ops();
        let (engine, _dir) = test_engine_durable();
        let mut clf = SpamClassifier::new(&ops);

        // Train some data
        clf.train(&ops, &spam_email(), true).unwrap();
        clf.train(&ops, &ham_email(), false).unwrap();
        clf.whitelist_domain("trusted.com".to_string());

        // Persist
        clf.persist(&engine).unwrap();

        // Restore
        let restored = SpamClassifier::restore(&engine, &ops).unwrap().unwrap();
        assert!(restored.is_trained());
        assert_eq!(restored.training_count(), 2);
        assert_eq!(restored.whitelist.len(), 1);
        assert_eq!(restored.whitelist[0], "trusted.com");
    }

    #[test]
    fn restore_returns_none_when_not_persisted() {
        let ops = test_ops();
        let (engine, _dir) = test_engine_durable();

        let result = SpamClassifier::restore(&engine, &ops).unwrap();
        assert!(result.is_none());
    }

    // ── OnlineHD adaptive update ───────────────────────────────────────

    #[test]
    fn adaptive_update_converges() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);

        // Train with the same spam email multiple times
        let email = spam_email();
        clf.train(&ops, &email, true).unwrap();
        let proto1 = clf.spam_prototype.clone().unwrap();

        clf.train(&ops, &email, true).unwrap();
        let proto2 = clf.spam_prototype.clone().unwrap();

        // Prototypes should be very similar (converging)
        let sim = ops.similarity(&proto1, &proto2).unwrap();
        assert!(
            sim > 0.8,
            "repeated training should converge prototype: sim={sim}"
        );
    }

    // ── SpamDecision Display ───────────────────────────────────────────

    #[test]
    fn spam_decision_display() {
        assert_eq!(SpamDecision::Spam.to_string(), "spam");
        assert_eq!(SpamDecision::Ham.to_string(), "ham");
        assert_eq!(SpamDecision::Uncertain.to_string(), "uncertain");
    }

    // ── Token extraction ───────────────────────────────────────────────

    #[test]
    fn extract_tokens_includes_domain_and_words() {
        let email = ham_email();
        let tokens = extract_tokens(&email);

        assert!(tokens.contains(&"domain:company.com".to_string()));
        assert!(tokens.contains(&"meeting".to_string()));
        assert!(tokens.contains(&"tomorrow".to_string()));
    }

    // ── Classification result reasoning ────────────────────────────────

    #[test]
    fn classification_result_has_reasoning() {
        let ops = test_ops();
        let mut clf = SpamClassifier::new(&ops);

        clf.train(&ops, &spam_email(), true).unwrap();
        clf.train(&ops, &ham_email(), false).unwrap();

        let result = clf.classify(&ops, &spam_email()).unwrap();
        assert!(!result.reasoning.is_empty());
        assert!(result.reasoning.contains("VSA"));
        assert!(result.reasoning.contains("Bayesian"));
    }
}
