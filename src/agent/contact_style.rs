//! Communication Style Profiles — Phase 25d.
//!
//! Learns per-contact communication style preferences via rule-based
//! heuristics and VSA prototype vectors. Tracks formality, verbosity,
//! message length, and response timing via exponential moving averages.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;
use crate::vsa::HyperVec;

use super::contact::{ContactError, ContactResult};

// ═══════════════════════════════════════════════════════════════════════
// StylePredicates
// ═══════════════════════════════════════════════════════════════════════

/// Well-known predicates for style metadata in the KG.
pub struct StylePredicates {
    pub formality: SymbolId,
    pub verbosity: SymbolId,
    pub expertise_area: SymbolId,
    pub preferred_channel: SymbolId,
    pub response_pattern: SymbolId,
    pub style_observation: SymbolId,
}

impl StylePredicates {
    pub fn init(engine: &Engine) -> ContactResult<Self> {
        Ok(Self {
            formality: engine.resolve_or_create_relation("contact:formality")?,
            verbosity: engine.resolve_or_create_relation("contact:verbosity")?,
            expertise_area: engine.resolve_or_create_relation("contact:expertise-area")?,
            preferred_channel: engine.resolve_or_create_relation("contact:preferred-channel")?,
            response_pattern: engine.resolve_or_create_relation("contact:response-pattern")?,
            style_observation: engine.resolve_or_create_relation("contact:style-observation")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// StyleRoleVectors
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic role hypervectors for encoding style patterns.
pub struct StyleRoleVectors {
    pub formality: HyperVec,
    pub verbosity: HyperVec,
    pub expertise: HyperVec,
    pub channel: HyperVec,
    pub timing: HyperVec,
}

impl StyleRoleVectors {
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            formality: encode_token(ops, "style-role:formality"),
            verbosity: encode_token(ops, "style-role:verbosity"),
            expertise: encode_token(ops, "style-role:expertise"),
            channel: encode_token(ops, "style-role:channel"),
            timing: encode_token(ops, "style-role:timing"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CommunicationStyle
// ═══════════════════════════════════════════════════════════════════════

/// Learned communication style profile for a contact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    /// Contact this profile belongs to.
    pub contact_id: String,
    /// Formality score: 0.0 (casual) to 1.0 (formal), EMA.
    pub formality: f32,
    /// Verbosity score: 0.0 (terse) to 1.0 (verbose), EMA.
    pub verbosity: f32,
    /// Expertise areas (KG SymbolIds).
    pub expertise_areas: Vec<SymbolId>,
    /// Average message length in characters (EMA).
    pub avg_message_length: f32,
    /// Average response time in seconds (EMA).
    pub avg_response_time: f64,
    /// Number of observations used to build this profile.
    pub observation_count: u64,
    /// Last observation timestamp (UNIX seconds).
    pub last_observed: u64,
    /// VSA style prototype (OnlineHD bundled).
    #[serde(skip)]
    pub style_prototype: Option<HyperVec>,
}

/// An observation of a contact's communication in a single message.
pub struct StyleObservation {
    /// The raw message text.
    pub text: String,
    /// Message length in characters.
    pub message_length: usize,
    /// Response time in seconds (None if first message in conversation).
    pub response_time: Option<f64>,
    /// Timestamp of the observation (UNIX seconds).
    pub timestamp: u64,
}

// ═══════════════════════════════════════════════════════════════════════
// Formality heuristic (no LLM)
// ═══════════════════════════════════════════════════════════════════════

/// Rule-based formality scoring: returns a value in [0.0, 1.0].
///
/// Formal markers push the score up, casual markers push it down.
/// The result is the proportion of formal signals found.
pub fn score_formality(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let mut formal_signals = 0i32;
    let mut total_signals = 0i32;

    // Formal markers (positive).
    let formal_markers = [
        "dear ", "sincerely", "regards", "respectfully", "please",
        "thank you", "would you", "could you", "i would appreciate",
        "kindly", "hereby", "furthermore", "therefore", "accordingly",
    ];
    for marker in &formal_markers {
        if lower.contains(marker) {
            formal_signals += 1;
            total_signals += 1;
        }
    }

    // Casual markers (negative).
    let casual_markers = [
        "hey", "lol", "haha", "gonna", "wanna", "gotta", "btw",
        "omg", "nah", "yeah", "yep", "nope", "sup", "yo ",
    ];
    for marker in &casual_markers {
        if lower.contains(marker) {
            total_signals += 1;
            // Don't increment formal_signals.
        }
    }

    // Contractions decrease formality.
    let contractions = ["n't", "'re", "'ve", "'ll", "'d ", "i'm", "it's"];
    for contraction in &contractions {
        if lower.contains(contraction) {
            total_signals += 1;
        }
    }

    // Emoji/emoticon signals casual.
    if text.contains(":)") || text.contains(":(") || text.contains(":D") {
        total_signals += 1;
    }

    // Sentence length as a signal: longer sentences → more formal.
    let word_count = text.split_whitespace().count();
    if word_count > 20 {
        formal_signals += 1;
        total_signals += 1;
    }

    if total_signals == 0 {
        return 0.5; // neutral
    }

    (formal_signals as f32 / total_signals as f32).clamp(0.0, 1.0)
}

/// Score verbosity based on message length relative to typical ranges.
pub fn score_verbosity(message_length: usize) -> f32 {
    // Under 20 chars → terse (0.0), over 500 chars → verbose (1.0).
    let clamped = (message_length as f32).clamp(20.0, 500.0);
    (clamped - 20.0) / 480.0
}

// ═══════════════════════════════════════════════════════════════════════
// StyleManager
// ═══════════════════════════════════════════════════════════════════════

/// EMA smoothing factor (α). Higher = more weight on recent observations.
const EMA_ALPHA: f32 = 0.3;

/// Manages per-contact communication style profiles.
#[derive(Default)]
pub struct StyleManager {
    styles: HashMap<String, CommunicationStyle>,
    predicates: Option<StylePredicates>,
    role_vectors: Option<StyleRoleVectors>,
}

impl Serialize for StyleManager {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.styles.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StyleManager {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let styles = HashMap::<String, CommunicationStyle>::deserialize(deserializer)?;
        Ok(Self {
            styles,
            predicates: None,
            role_vectors: None,
        })
    }
}

impl fmt::Debug for StyleManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StyleManager")
            .field("profile_count", &self.styles.len())
            .finish()
    }
}

impl StyleManager {
    /// Create a new manager, initializing predicates and role vectors.
    pub fn new(engine: &Engine) -> ContactResult<Self> {
        let predicates = StylePredicates::init(engine)?;
        let role_vectors = StyleRoleVectors::new(engine.ops());
        Ok(Self {
            styles: HashMap::new(),
            predicates: Some(predicates),
            role_vectors: Some(role_vectors),
        })
    }

    /// Ensure predicates and role vectors are initialized (post-deserialization).
    pub fn ensure_init(&mut self, engine: &Engine) -> ContactResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(StylePredicates::init(engine)?);
        }
        if self.role_vectors.is_none() {
            self.role_vectors = Some(StyleRoleVectors::new(engine.ops()));
        }
        Ok(())
    }

    /// Number of style profiles.
    pub fn profile_count(&self) -> usize {
        self.styles.len()
    }

    /// Observe a message from a contact, updating their style profile.
    pub fn observe(
        &mut self,
        engine: &Engine,
        contact_id: &str,
        observation: &StyleObservation,
    ) -> ContactResult<()> {
        let formality = score_formality(&observation.text);
        let verbosity = score_verbosity(observation.message_length);

        let style = self
            .styles
            .entry(contact_id.to_string())
            .or_insert_with(|| CommunicationStyle {
                contact_id: contact_id.to_string(),
                formality,
                verbosity,
                expertise_areas: Vec::new(),
                avg_message_length: observation.message_length as f32,
                avg_response_time: observation.response_time.unwrap_or(0.0),
                observation_count: 0,
                last_observed: observation.timestamp,
                style_prototype: None,
            });

        // EMA update.
        style.formality = style.formality * (1.0 - EMA_ALPHA) + formality * EMA_ALPHA;
        style.verbosity = style.verbosity * (1.0 - EMA_ALPHA) + verbosity * EMA_ALPHA;
        style.avg_message_length = style.avg_message_length * (1.0 - EMA_ALPHA)
            + observation.message_length as f32 * EMA_ALPHA;

        if let Some(rt) = observation.response_time {
            style.avg_response_time =
                style.avg_response_time * (1.0 - EMA_ALPHA as f64) + rt * EMA_ALPHA as f64;
        }

        style.observation_count += 1;
        style.last_observed = observation.timestamp;

        // Update VSA prototype.
        if let Some(roles) = &self.role_vectors {
            let ops = engine.ops();
            let formality_vec = encode_token(ops, &format!("formality:{:.1}", style.formality));
            let verbosity_vec = encode_token(ops, &format!("verbosity:{:.1}", style.verbosity));

            let bound_f = ops.bind(&roles.formality, &formality_vec);
            let bound_v = ops.bind(&roles.verbosity, &verbosity_vec);

            if let (Ok(bf), Ok(bv)) = (bound_f, bound_v) {
                if let Ok(bundled) = ops.bundle(&[&bf, &bv]) {
                    style.style_prototype = Some(bundled);
                }
            }
        }

        // Record provenance.
        let contact_sym = engine.resolve_or_create_entity(&format!("contact:{contact_id}"))?;
        let mut record = ProvenanceRecord::new(
            contact_sym,
            DerivationKind::StyleObserved {
                contact_id: contact_id.to_string(),
                formality: style.formality,
                verbosity: style.verbosity,
            },
        );
        let _ = engine.store_provenance(&mut record);

        Ok(())
    }

    /// Get the learned style profile for a contact.
    pub fn style_for(&self, contact_id: &str) -> Option<&CommunicationStyle> {
        self.styles.get(contact_id)
    }

    /// Suggest formality level for communicating with a contact.
    pub fn suggest_formality(&self, contact_id: &str) -> f32 {
        self.styles
            .get(contact_id)
            .map(|s| s.formality)
            .unwrap_or(0.5)
    }

    /// Suggest verbosity level for communicating with a contact.
    pub fn suggest_verbosity(&self, contact_id: &str) -> f32 {
        self.styles
            .get(contact_id)
            .map(|s| s.verbosity)
            .unwrap_or(0.5)
    }

    /// Find contacts with similar communication style (by VSA prototype similarity).
    pub fn similar_style(&self, contact_id: &str, k: usize) -> Vec<(String, f64)> {
        let target = match self.styles.get(contact_id).and_then(|s| s.style_prototype.as_ref()) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut similarities: Vec<(String, f64)> = self
            .styles
            .iter()
            .filter(|(cid, _)| *cid != contact_id)
            .filter_map(|(cid, style)| {
                style
                    .style_prototype
                    .as_ref()
                    .map(|v| (cid.clone(), hamming_similarity(target, v)))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }

    /// Apply observation decay: reduce the weight of old observations.
    pub fn decay_all(&mut self, now: u64, lambda: f64) {
        for style in self.styles.values_mut() {
            let days = (now.saturating_sub(style.last_observed) as f64) / 86400.0;
            let factor = (-lambda * days).exp() as f32;
            // Move formality and verbosity toward neutral (0.5) based on decay.
            style.formality = 0.5 + (style.formality - 0.5) * factor;
            style.verbosity = 0.5 + (style.verbosity - 0.5) * factor;
        }
    }

    // ── Persistence ───────────────────────────────────────────────────

    /// Persist to the engine's durable store.
    pub fn persist(&self, engine: &Engine) -> ContactResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            ContactError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to serialize style manager: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:style_manager", &bytes)
            .map_err(|e| ContactError::Engine(Box::new(crate::error::AkhError::Store(e))))?;
        Ok(())
    }

    /// Restore from the engine's durable store.
    pub fn restore(engine: &Engine) -> ContactResult<Self> {
        let bytes = engine
            .store()
            .get_meta(b"agent:style_manager")
            .map_err(|e| ContactError::Engine(Box::new(crate::error::AkhError::Store(e))))?
            .ok_or(ContactError::NotFound {
                contact_id: "<store>".to_string(),
            })?;
        let mut manager: Self = bincode::deserialize(&bytes).map_err(|e| {
            ContactError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to deserialize style manager: {e}"),
                },
            )))
        })?;
        manager.ensure_init(engine)?;
        Ok(manager)
    }
}

/// Simple Hamming-similarity between two binary hypervectors.
fn hamming_similarity(a: &HyperVec, b: &HyperVec) -> f64 {
    let a_data = a.data();
    let b_data = b.data();
    let total_bits = a_data.len() * 8;
    if total_bits == 0 {
        return 0.0;
    }
    let differing: u32 = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    1.0 - (differing as f64 / total_bits as f64)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    fn test_engine() -> Engine {
        use crate::engine::EngineConfig;
        use crate::vsa::Dimension;
        Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .expect("in-memory engine")
    }

    #[test]
    fn observe_and_retrieve() {
        let engine = test_engine();
        let mut mgr = StyleManager::new(&engine).unwrap();

        mgr.observe(
            &engine,
            "alice",
            &StyleObservation {
                text: "Dear Sir, I would appreciate your assistance.".to_string(),
                message_length: 46,
                response_time: Some(300.0),
                timestamp: 1000,
            },
        )
        .unwrap();

        let style = mgr.style_for("alice").unwrap();
        assert!(style.formality > 0.5); // formal message
        assert_eq!(style.observation_count, 1);
    }

    #[test]
    fn ema_convergence() {
        let engine = test_engine();
        let mut mgr = StyleManager::new(&engine).unwrap();

        // Send many casual messages.
        for i in 0..10 {
            mgr.observe(
                &engine,
                "bob",
                &StyleObservation {
                    text: "hey lol gonna do it".to_string(),
                    message_length: 19,
                    response_time: Some(5.0),
                    timestamp: 1000 + i,
                },
            )
            .unwrap();
        }

        let style = mgr.style_for("bob").unwrap();
        assert!(style.formality < 0.3); // converged toward casual
        assert!(style.verbosity < 0.2); // short messages → terse
    }

    #[test]
    fn formality_heuristic() {
        let formal = score_formality("Dear Professor, I would appreciate your feedback on my thesis.");
        let casual = score_formality("hey lol what's up gonna grab coffee?");
        assert!(formal > casual);
    }

    #[test]
    fn verbosity_scoring() {
        assert!(score_verbosity(10) < score_verbosity(200));
        assert!(score_verbosity(200) < score_verbosity(500));
    }

    #[test]
    fn decay_toward_neutral() {
        let engine = test_engine();
        let mut mgr = StyleManager::new(&engine).unwrap();

        mgr.observe(
            &engine,
            "alice",
            &StyleObservation {
                text: "Dear Sir, please find attached.".to_string(),
                message_length: 30,
                response_time: None,
                timestamp: 1000,
            },
        )
        .unwrap();

        let initial_formality = mgr.style_for("alice").unwrap().formality;

        // Decay 365 days later.
        mgr.decay_all(1000 + 365 * 86400, 0.01);

        let decayed_formality = mgr.style_for("alice").unwrap().formality;
        assert!((decayed_formality - 0.5).abs() < (initial_formality - 0.5).abs());
    }

    #[test]
    fn persist_and_restore() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(crate::engine::EngineConfig {
            data_dir: Some(dir.path().to_path_buf()),
            dimension: crate::vsa::Dimension(1000),
            ..crate::engine::EngineConfig::default()
        })
        .unwrap();
        let mut mgr = StyleManager::new(&engine).unwrap();

        mgr.observe(
            &engine,
            "alice",
            &StyleObservation {
                text: "Hello!".to_string(),
                message_length: 6,
                response_time: None,
                timestamp: 1000,
            },
        )
        .unwrap();

        mgr.persist(&engine).unwrap();

        let restored = StyleManager::restore(&engine).unwrap();
        assert_eq!(restored.profile_count(), 1);
        assert!(restored.style_for("alice").is_some());
    }
}
