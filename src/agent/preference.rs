//! Preference Learning & Proactive Assistance — Phase 13g.
//!
//! VSA-native preference profiles with temporal decay, Just-in-Time Information
//! Retrieval (JITIR), and a serendipity engine for non-obvious connections.
//! Always-on, no feature gate, no new crate dependencies.

use std::fmt;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;
use crate::vsa::HyperVec;

use super::goal::Goal;
use super::memory::WorkingMemory;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the preference learning subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum PreferenceError {
    #[error("preference profile not found")]
    #[diagnostic(
        code(akh::agent::preference::profile_not_found),
        help("No preference profile exists yet. Record feedback to create one.")
    )]
    ProfileNotFound,

    #[error("empty context: no active goals or working memory entries to encode")]
    #[diagnostic(
        code(akh::agent::preference::empty_context),
        help("The JITIR query requires at least one active goal or WM entry for context.")
    )]
    EmptyContext,

    #[error("serendipity search failed: {message}")]
    #[diagnostic(
        code(akh::agent::preference::serendipity_failed),
        help("The near-miss HNSW search could not find candidates in the [0.3, 0.6] similarity window.")
    )]
    SerendipityFailed { message: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::preference::engine),
        help("Engine-level error during preference operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for PreferenceError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type PreferenceResult<T> = std::result::Result<T, PreferenceError>;

// ═══════════════════════════════════════════════════════════════════════
// ProactivityLevel
// ═══════════════════════════════════════════════════════════════════════

/// How aggressively the agent surfaces suggestions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProactivityLevel {
    /// Silent background learning only.
    #[default]
    Ambient,
    /// Subtle hints appended to existing responses.
    Nudge,
    /// Offer suggestions when confidence is high.
    Offer,
    /// Surface suggestions on a schedule.
    Scheduled,
    /// Act on preferences autonomously.
    Autonomous,
}

impl ProactivityLevel {
    /// Stable label for serialization / KG predicates.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Ambient => "ambient",
            Self::Nudge => "nudge",
            Self::Offer => "offer",
            Self::Scheduled => "scheduled",
            Self::Autonomous => "autonomous",
        }
    }

    /// Parse from a label (case-insensitive).
    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ambient" => Some(Self::Ambient),
            "nudge" => Some(Self::Nudge),
            "offer" => Some(Self::Offer),
            "scheduled" => Some(Self::Scheduled),
            "autonomous" => Some(Self::Autonomous),
            _ => None,
        }
    }

    /// Whether this level interrupts the user (rather than being passive).
    pub fn is_interrupting(&self) -> bool {
        matches!(self, Self::Offer | Self::Scheduled | Self::Autonomous)
    }
}

impl fmt::Display for ProactivityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FeedbackSignal
// ═══════════════════════════════════════════════════════════════════════

/// An implicit or explicit feedback signal about an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackSignal {
    /// Operator replied to a message about this entity.
    Replied {
        entity: SymbolId,
        response_time_secs: u64,
    },
    /// Operator spent time reading content about this entity.
    ReadTime { entity: SymbolId, seconds: u64 },
    /// Operator archived without reading.
    ArchivedUnread { entity: SymbolId },
    /// Operator starred / bookmarked.
    Starred { entity: SymbolId },
    /// A goal touching these topics was completed.
    GoalCompleted {
        goal_id: SymbolId,
        topic_symbols: Vec<SymbolId>,
    },
    /// Explicit preference statement ("I'm interested in X").
    ExplicitPreference { topic: SymbolId, weight: f32 },
    /// Operator dismissed a suggestion.
    SuggestionDismissed { entity: SymbolId },
}

impl FeedbackSignal {
    /// The primary entity this signal relates to.
    pub fn entity(&self) -> SymbolId {
        match self {
            Self::Replied { entity, .. } => *entity,
            Self::ReadTime { entity, .. } => *entity,
            Self::ArchivedUnread { entity } => *entity,
            Self::Starred { entity } => *entity,
            Self::GoalCompleted { goal_id, .. } => *goal_id,
            Self::ExplicitPreference { topic, .. } => *topic,
            Self::SuggestionDismissed { entity } => *entity,
        }
    }

    /// Feedback strength in [-1.0, 1.0]. Positive = interest, negative = disinterest.
    pub fn strength(&self) -> f32 {
        match self {
            Self::Replied {
                response_time_secs, ..
            } => {
                // Fast reply → strong interest; slow reply → mild interest.
                if *response_time_secs < 60 {
                    0.9
                } else if *response_time_secs < 300 {
                    0.8
                } else {
                    0.6
                }
            }
            Self::ReadTime { seconds, .. } => {
                // Longer read time → more interest, capped at 0.7.
                ((*seconds as f32) / 120.0).min(0.7)
            }
            Self::ArchivedUnread { .. } => -0.5,
            Self::Starred { .. } => 1.0,
            Self::GoalCompleted { .. } => 0.7,
            Self::ExplicitPreference { weight, .. } => weight.clamp(-1.0, 1.0),
            Self::SuggestionDismissed { .. } => -0.3,
        }
    }

    /// Short label describing the signal kind.
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::Replied { .. } => "replied",
            Self::ReadTime { .. } => "read_time",
            Self::ArchivedUnread { .. } => "archived_unread",
            Self::Starred { .. } => "starred",
            Self::GoalCompleted { .. } => "goal_completed",
            Self::ExplicitPreference { .. } => "explicit",
            Self::SuggestionDismissed { .. } => "dismissed",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PreferencePredicates
// ═══════════════════════════════════════════════════════════════════════

/// Well-known KG relations in the `pref:` namespace.
#[derive(Debug, Clone)]
pub struct PreferencePredicates {
    pub interest_topic: SymbolId,
    pub interest_weight: SymbolId,
    pub feedback_signal: SymbolId,
    pub feedback_time: SymbolId,
    pub proactivity_level: SymbolId,
    pub suggestion_count: SymbolId,
    pub suggestion_accepted: SymbolId,
    pub last_jitir_cycle: SymbolId,
}

impl PreferencePredicates {
    /// Resolve or create all well-known predicates in the engine.
    pub fn init(engine: &Engine) -> PreferenceResult<Self> {
        Ok(Self {
            interest_topic: engine.resolve_or_create_relation("pref:interest-topic")?,
            interest_weight: engine.resolve_or_create_relation("pref:interest-weight")?,
            feedback_signal: engine.resolve_or_create_relation("pref:feedback-signal")?,
            feedback_time: engine.resolve_or_create_relation("pref:feedback-time")?,
            proactivity_level: engine.resolve_or_create_relation("pref:proactivity-level")?,
            suggestion_count: engine.resolve_or_create_relation("pref:suggestion-count")?,
            suggestion_accepted: engine.resolve_or_create_relation("pref:suggestion-accepted")?,
            last_jitir_cycle: engine.resolve_or_create_relation("pref:last-jitir-cycle")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PreferenceRoleVectors
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic VSA role vectors for preference encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceRoleVectors {
    pub topic: HyperVec,
    pub interaction_type: HyperVec,
    pub recency: HyperVec,
    pub frequency: HyperVec,
    pub source_channel: HyperVec,
    pub entity_kind: HyperVec,
}

impl PreferenceRoleVectors {
    /// Create role vectors using the given VSA ops.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            topic: encode_token(ops, "pref-role:topic"),
            interaction_type: encode_token(ops, "pref-role:interaction_type"),
            recency: encode_token(ops, "pref-role:recency"),
            frequency: encode_token(ops, "pref-role:frequency"),
            source_channel: encode_token(ops, "pref-role:source_channel"),
            entity_kind: encode_token(ops, "pref-role:entity_kind"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PreferenceProfile
// ═══════════════════════════════════════════════════════════════════════

/// Learned preference profile for the operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceProfile {
    /// OnlineHD interest prototype — bundled from feedback signals.
    pub interest_prototype: Option<HyperVec>,
    /// Temporal decay rate (exponential lambda, default 0.01).
    pub decay_rate: f32,
    /// Total number of feedback interactions recorded.
    pub interaction_count: u64,
    /// Recent interaction history: (entity_id, timestamp). Bounded by `max_history`.
    pub interaction_history: Vec<(SymbolId, u64)>,
    /// Maximum history entries to retain.
    pub max_history: usize,
    /// Current proactivity level.
    pub proactivity_level: ProactivityLevel,
    /// Total suggestions offered.
    pub suggestions_offered: u64,
    /// Total suggestions accepted by the operator.
    pub suggestions_accepted: u64,
}

impl Default for PreferenceProfile {
    fn default() -> Self {
        Self {
            interest_prototype: None,
            decay_rate: 0.01,
            interaction_count: 0,
            interaction_history: Vec::new(),
            max_history: 1000,
            proactivity_level: ProactivityLevel::default(),
            suggestions_offered: 0,
            suggestions_accepted: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Suggestion, JitirResult, PreferenceReview
// ═══════════════════════════════════════════════════════════════════════

/// A proactive suggestion surfaced by JITIR or serendipity.
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// The entity being suggested.
    pub entity: SymbolId,
    /// Human-readable label.
    pub label: String,
    /// Relevance score [0.0, 1.0].
    pub relevance: f32,
    /// Whether this came from the serendipity engine (near-miss).
    pub serendipitous: bool,
    /// Why this was suggested.
    pub reasoning: String,
    /// Minimum proactivity level required to surface this.
    pub proactivity_level: ProactivityLevel,
}

/// Result of a JITIR (Just-in-Time Information Retrieval) query.
#[derive(Debug, Clone, Default)]
pub struct JitirResult {
    /// Directly relevant matches from HNSW / KG.
    pub direct_matches: Vec<Suggestion>,
    /// Near-miss serendipity matches.
    pub serendipity_matches: Vec<Suggestion>,
    /// Summary of the context used for retrieval.
    pub context_summary: String,
}

/// Summary of preference state for reflection.
#[derive(Debug, Clone)]
pub struct PreferenceReview {
    /// Total interactions recorded.
    pub interaction_count: u64,
    /// Top interest topics (label, weight) pairs.
    pub top_interests: Vec<(String, f32)>,
    /// Suggestion acceptance rate [0.0, 1.0].
    pub acceptance_rate: f32,
    /// Current proactivity level.
    pub proactivity_level: ProactivityLevel,
}

// ═══════════════════════════════════════════════════════════════════════
// PreferenceManager
// ═══════════════════════════════════════════════════════════════════════

/// Manages preference learning, JITIR retrieval, and proactive assistance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreferenceManager {
    /// Learned profile.
    pub profile: PreferenceProfile,
    /// Lazily initialized predicates (not serialized).
    #[serde(skip)]
    predicates: Option<PreferencePredicates>,
    /// Lazily initialized role vectors (not serialized).
    #[serde(skip)]
    role_vectors: Option<PreferenceRoleVectors>,
}

impl PreferenceManager {
    /// Create a new preference manager initialized with the engine.
    pub fn new(engine: &Engine) -> PreferenceResult<Self> {
        let mut mgr = Self::default();
        mgr.ensure_init(engine)?;
        Ok(mgr)
    }

    /// Ensure predicates and role vectors are initialized.
    pub fn ensure_init(&mut self, engine: &Engine) -> PreferenceResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(PreferencePredicates::init(engine)?);
        }
        if self.role_vectors.is_none() {
            self.role_vectors = Some(PreferenceRoleVectors::new(engine.ops()));
        }
        Ok(())
    }

    /// Restore a preference manager from the durable store.
    pub fn restore(engine: &Engine) -> PreferenceResult<Self> {
        let store = engine.store();
        match store.get_meta(b"agent:preference_manager").ok().flatten() {
            Some(bytes) => {
                let mut mgr: Self = bincode::deserialize(&bytes).map_err(|e| {
                    PreferenceError::Engine(Box::new(crate::error::AkhError::Store(
                        crate::error::StoreError::Serialization {
                            message: format!("preference manager deserialize: {e}"),
                        },
                    )))
                })?;
                mgr.ensure_init(engine)?;
                Ok(mgr)
            }
            None => Self::new(engine),
        }
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &Engine) -> PreferenceResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            PreferenceError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("preference manager serialize: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:preference_manager", &bytes)
            .map_err(|e| PreferenceError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    // ─── Feedback ────────────────────────────────────────────────────

    /// Record a feedback signal, updating the interest prototype via OnlineHD.
    pub fn record_feedback(
        &mut self,
        signal: &FeedbackSignal,
        engine: &Engine,
    ) -> PreferenceResult<()> {
        self.ensure_init(engine)?;
        let ops = engine.ops();
        let roles = self.role_vectors.as_ref().unwrap();

        // Encode the entity as a VSA vector.
        let entity_vec = self.encode_interaction(ops, roles, signal.entity(), engine);

        let strength = signal.strength();
        let count = self.profile.interaction_count;
        // Adaptive learning rate: starts high, decreases over time.
        let lr = 1.0 / (1.0 + count as f32 * 0.01);

        if let Some(ref proto) = self.profile.interest_prototype {
            // OnlineHD adaptive update: blend prototype toward/away from entity.
            let scaled_lr = lr * strength;
            if scaled_lr > 0.0 {
                // Positive feedback: bundle toward entity (majority vote).
                if let Ok(bundled) = ops.bundle(&[proto, &entity_vec]) {
                    self.profile.interest_prototype = Some(bundled);
                }
            }
            // Negative feedback: we don't update the prototype (no complement op).
            // The signal is still recorded in interaction history and provenance.
        } else {
            // First feedback — seed the prototype.
            self.profile.interest_prototype = Some(entity_vec);
        }

        // Record in history.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.profile
            .interaction_history
            .push((signal.entity(), now));
        if self.profile.interaction_history.len() > self.profile.max_history {
            self.profile.interaction_history.remove(0);
        }
        self.profile.interaction_count += 1;

        // Record provenance.
        self.record_feedback_provenance(engine, signal)?;

        Ok(())
    }

    /// Encode a single entity interaction as a role-filler VSA vector.
    fn encode_interaction(
        &self,
        ops: &VsaOps,
        roles: &PreferenceRoleVectors,
        entity: SymbolId,
        engine: &Engine,
    ) -> HyperVec {
        let entity_vec = engine
            .item_memory()
            .get(entity)
            .unwrap_or_else(|| encode_token(ops, &format!("entity:{}", entity.get())));

        // Bind entity with topic role.
        let bound = match ops.bind(&roles.topic, &entity_vec) {
            Ok(b) => b,
            Err(_) => return entity_vec,
        };

        // Also encode the entity kind if we can resolve it.
        let label = engine.resolve_label(entity);
        let kind_filler = encode_token(ops, &format!("kind:{label}"));
        let kind_bound = match ops.bind(&roles.entity_kind, &kind_filler) {
            Ok(b) => b,
            Err(_) => return bound,
        };

        ops.bundle(&[&bound, &kind_bound]).unwrap_or(bound)
    }

    /// Record provenance for a feedback signal.
    fn record_feedback_provenance(
        &self,
        engine: &Engine,
        signal: &FeedbackSignal,
    ) -> PreferenceResult<()> {
        let mut record = ProvenanceRecord::new(
            signal.entity(),
            DerivationKind::PreferenceLearned {
                signal_kind: signal.kind_label().to_string(),
                entity_id_raw: signal.entity().get(),
                weight: signal.strength(),
            },
        )
        .with_confidence(signal.strength().abs());

        let _ = engine.store_provenance(&mut record);
        Ok(())
    }

    // ─── Temporal Decay ──────────────────────────────────────────────

    /// Apply exponential temporal decay: `w * exp(-λ * age_days)`.
    pub fn apply_temporal_decay(weight: f32, age_secs: u64, decay_rate: f32) -> f32 {
        let age_days = age_secs as f64 / 86400.0;
        let decay = (-(decay_rate as f64) * age_days).exp() as f32;
        weight * decay
    }

    // ─── JITIR ───────────────────────────────────────────────────────

    /// Run a Just-in-Time Information Retrieval query.
    ///
    /// Encodes the current context (active goals + recent WM entries),
    /// then searches for directly relevant and serendipitous matches.
    pub fn jitir_query(
        &self,
        wm: &WorkingMemory,
        goals: &[Goal],
        engine: &Engine,
    ) -> PreferenceResult<JitirResult> {
        let ops = engine.ops();
        let roles = match &self.role_vectors {
            Some(r) => r,
            None => {
                return Ok(JitirResult::default());
            }
        };

        // Gather context symbols: active goals + recent 5 WM entries.
        let mut context_symbols = Vec::new();
        for goal in goals {
            if goal.status == super::goal::GoalStatus::Active {
                context_symbols.push(goal.symbol_id);
            }
        }
        for entry in wm.recent(5) {
            context_symbols.extend_from_slice(&entry.symbols);
        }
        context_symbols.dedup();

        if context_symbols.is_empty() {
            return Err(PreferenceError::EmptyContext);
        }

        // Encode context: bind each symbol with topic role, then bundle.
        let context_vec = self.encode_context(ops, roles, &context_symbols, engine);

        let context_summary = format!(
            "{} symbols from {} goals and recent WM",
            context_symbols.len(),
            goals
                .iter()
                .filter(|g| g.status == super::goal::GoalStatus::Active)
                .count()
        );

        // Direct HNSW search (k=5).
        let direct_matches = self.search_direct(engine, &context_vec, 5);

        // Serendipity search: k=50, filter to [0.3, 0.6] Hamming similarity window.
        let serendipity_matches = self.search_serendipity(engine, &context_vec);

        // KG multi-hop serendipity: BFS from context entities, depth 3.
        let kg_serendipity = self.kg_multi_hop(engine, &context_symbols, 3);

        // Merge serendipity sources, take top 3.
        let mut all_serendipity = serendipity_matches;
        all_serendipity.extend(kg_serendipity);
        all_serendipity.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
        all_serendipity.truncate(3);

        // Record provenance for suggestions.
        for suggestion in direct_matches.iter().chain(all_serendipity.iter()) {
            let mut record = ProvenanceRecord::new(
                suggestion.entity,
                DerivationKind::JitirSuggestion {
                    entity_id_raw: suggestion.entity.get(),
                    relevance: suggestion.relevance,
                    serendipitous: suggestion.serendipitous,
                },
            )
            .with_confidence(suggestion.relevance);
            let _ = engine.store_provenance(&mut record);
        }

        Ok(JitirResult {
            direct_matches,
            serendipity_matches: all_serendipity,
            context_summary,
        })
    }

    /// Encode a context vector from a set of symbols.
    fn encode_context(
        &self,
        ops: &VsaOps,
        roles: &PreferenceRoleVectors,
        symbols: &[SymbolId],
        engine: &Engine,
    ) -> HyperVec {
        let mut vecs = Vec::new();
        for &sym in symbols {
            let entity_vec = engine
                .item_memory()
                .get(sym)
                .unwrap_or_else(|| encode_token(ops, &format!("entity:{}", sym.get())));
            if let Ok(bound) = ops.bind(&roles.topic, &entity_vec) {
                vecs.push(bound);
            }
        }
        if vecs.is_empty() {
            return encode_token(ops, "pref:empty-context");
        }
        let refs: Vec<&HyperVec> = vecs.iter().collect();
        ops.bundle(&refs).unwrap_or_else(|_| encode_token(ops, "pref:empty-context"))
    }

    /// Direct HNSW search for relevant entities.
    fn search_direct(
        &self,
        engine: &Engine,
        context_vec: &HyperVec,
        k: usize,
    ) -> Vec<Suggestion> {
        let results = match engine.item_memory().search(context_vec, k) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        results
            .into_iter()
            .filter_map(|sr| {
                if sr.similarity < 0.1 {
                    return None;
                }
                let label = engine.resolve_label(sr.symbol_id);
                Some(Suggestion {
                    entity: sr.symbol_id,
                    label,
                    relevance: sr.similarity,
                    serendipitous: false,
                    reasoning: format!("direct HNSW match (similarity: {:.2})", sr.similarity),
                    proactivity_level: ProactivityLevel::Nudge,
                })
            })
            .collect()
    }

    /// Serendipity search: HNSW with k=50, filter to similarity [0.3, 0.6].
    fn search_serendipity(
        &self,
        engine: &Engine,
        context_vec: &HyperVec,
    ) -> Vec<Suggestion> {
        let results = match engine.item_memory().search(context_vec, 50) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let mut serendipitous: Vec<Suggestion> = results
            .into_iter()
            .filter_map(|sr| {
                // Serendipity zone: related but not obvious.
                if !serendipity_zone_filter(sr.similarity, 0.3, 0.6) {
                    return None;
                }
                let label = engine.resolve_label(sr.symbol_id);
                Some(Suggestion {
                    entity: sr.symbol_id,
                    label,
                    relevance: sr.similarity,
                    serendipitous: true,
                    reasoning: format!(
                        "serendipity: near-miss connection (similarity: {:.2})",
                        sr.similarity
                    ),
                    proactivity_level: ProactivityLevel::Offer,
                })
            })
            .collect();

        serendipitous.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        serendipitous.truncate(3);
        serendipitous
    }

    /// KG multi-hop serendipity: BFS from context entities up to `max_depth`.
    fn kg_multi_hop(
        &self,
        engine: &Engine,
        context_symbols: &[SymbolId],
        max_depth: usize,
    ) -> Vec<Suggestion> {
        use std::collections::{HashSet, VecDeque};

        let context_set: HashSet<SymbolId> = context_symbols.iter().copied().collect();
        let mut visited: HashSet<SymbolId> = context_set.clone();
        let mut queue: VecDeque<(SymbolId, usize)> = VecDeque::new();
        let mut discoveries: Vec<Suggestion> = Vec::new();

        for &sym in context_symbols {
            queue.push_back((sym, 0));
        }

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Follow outgoing KG edges.
            let triples = engine.triples_from(current);
            for triple in triples {
                if visited.contains(&triple.object) {
                    continue;
                }
                visited.insert(triple.object);

                let is_serendipitous = !context_set.contains(&triple.object);
                if is_serendipitous && depth >= 1 {
                    let label = engine.resolve_label(triple.object);
                    // Relevance decreases with depth.
                    let relevance = 0.5 / (depth as f32 + 1.0);
                    discoveries.push(Suggestion {
                        entity: triple.object,
                        label,
                        relevance,
                        serendipitous: true,
                        reasoning: format!(
                            "KG discovery at depth {} from context",
                            depth + 1
                        ),
                        proactivity_level: ProactivityLevel::Offer,
                    });
                }

                if depth + 1 < max_depth {
                    queue.push_back((triple.object, depth + 1));
                }
            }
        }

        discoveries.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        discoveries.truncate(3);
        discoveries
    }

    // ─── Proactivity ─────────────────────────────────────────────────

    /// Current proactivity level.
    pub fn proactivity_level(&self) -> ProactivityLevel {
        self.profile.proactivity_level
    }

    /// Set the proactivity level.
    pub fn set_proactivity_level(&mut self, level: ProactivityLevel) {
        self.profile.proactivity_level = level;
    }

    /// Suggestion acceptance rate [0.0, 1.0].
    pub fn suggestion_acceptance_rate(&self) -> f32 {
        if self.profile.suggestions_offered == 0 {
            return 0.0;
        }
        self.profile.suggestions_accepted as f32 / self.profile.suggestions_offered as f32
    }

    /// Record whether a suggestion was accepted or dismissed.
    pub fn record_suggestion_outcome(&mut self, accepted: bool) {
        self.profile.suggestions_offered += 1;
        if accepted {
            self.profile.suggestions_accepted += 1;
        }
    }

    // ─── Query ───────────────────────────────────────────────────────

    /// Compute similarity between the interest prototype and an entity.
    pub fn interest_similarity(
        &self,
        entity: SymbolId,
        engine: &Engine,
    ) -> f32 {
        let proto = match &self.profile.interest_prototype {
            Some(p) => p,
            None => return 0.0,
        };
        let ops = engine.ops();
        let entity_vec = engine
            .item_memory()
            .get(entity)
            .unwrap_or_else(|| encode_token(ops, &format!("entity:{}", entity.get())));
        ops.similarity(proto, &entity_vec).unwrap_or(0.0)
    }

    /// Return top-k interests by similarity to the prototype.
    pub fn top_interests(
        &self,
        engine: &Engine,
        k: usize,
    ) -> Vec<(String, f32)> {
        let proto = match &self.profile.interest_prototype {
            Some(p) => p,
            None => return Vec::new(),
        };

        let results = match engine.item_memory().search(proto, k) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        results
            .into_iter()
            .filter_map(|sr| {
                if sr.similarity < 0.05 {
                    return None;
                }
                let label = engine.resolve_label(sr.symbol_id);
                Some((label, sr.similarity))
            })
            .collect()
    }

    /// Build a preference review for reflection.
    pub fn review(&self, engine: &Engine) -> PreferenceReview {
        let top = self.top_interests(engine, 5);
        PreferenceReview {
            interaction_count: self.profile.interaction_count,
            top_interests: top,
            acceptance_rate: self.suggestion_acceptance_rate(),
            proactivity_level: self.profile.proactivity_level,
        }
    }

    /// Record proactive assistance provenance.
    pub fn record_assistance_provenance(
        &self,
        engine: &Engine,
        suggestion_count: usize,
    ) -> PreferenceResult<()> {
        // Only record if there are actual suggestions.
        if suggestion_count == 0 {
            return Ok(());
        }

        // Use a synthetic symbol for the provenance record.
        let level_sym = engine
            .resolve_or_create_relation("pref:assistance-event")
            .unwrap_or_else(|_| {
                SymbolId::new(1).unwrap()
            });

        let mut record = ProvenanceRecord::new(
            level_sym,
            DerivationKind::ProactiveAssistance {
                level: self.profile.proactivity_level.as_label().to_string(),
                suggestion_count: suggestion_count as u32,
                acceptance_rate: self.suggestion_acceptance_rate(),
            },
        )
        .with_confidence(self.suggestion_acceptance_rate());

        let _ = engine.store_provenance(&mut record);
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Serendipity zone filter (standalone for testability)
// ═══════════════════════════════════════════════════════════════════════

/// Filter similarities to the serendipity zone [low, high].
pub fn serendipity_zone_filter(similarity: f32, low: f32, high: f32) -> bool {
    similarity >= low && similarity <= high
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

    // ── ProactivityLevel ─────────────────────────────────────────────

    #[test]
    fn proactivity_level_labels() {
        for level in &[
            ProactivityLevel::Ambient,
            ProactivityLevel::Nudge,
            ProactivityLevel::Offer,
            ProactivityLevel::Scheduled,
            ProactivityLevel::Autonomous,
        ] {
            let label = level.as_label();
            let parsed = ProactivityLevel::from_label(label);
            assert_eq!(parsed, Some(*level), "roundtrip failed for {label}");
        }
    }

    #[test]
    fn proactivity_default_is_ambient() {
        assert_eq!(ProactivityLevel::default(), ProactivityLevel::Ambient);
    }

    #[test]
    fn proactivity_is_interrupting() {
        assert!(!ProactivityLevel::Ambient.is_interrupting());
        assert!(!ProactivityLevel::Nudge.is_interrupting());
        assert!(ProactivityLevel::Offer.is_interrupting());
        assert!(ProactivityLevel::Scheduled.is_interrupting());
        assert!(ProactivityLevel::Autonomous.is_interrupting());
    }

    #[test]
    fn proactivity_display() {
        assert_eq!(format!("{}", ProactivityLevel::Ambient), "ambient");
        assert_eq!(format!("{}", ProactivityLevel::Nudge), "nudge");
        assert_eq!(format!("{}", ProactivityLevel::Offer), "offer");
        assert_eq!(format!("{}", ProactivityLevel::Scheduled), "scheduled");
        assert_eq!(format!("{}", ProactivityLevel::Autonomous), "autonomous");
    }

    // ── FeedbackSignal ───────────────────────────────────────────────

    #[test]
    fn feedback_signal_strength_replied() {
        let sig = FeedbackSignal::Replied {
            entity: sym(1),
            response_time_secs: 30,
        };
        let s = sig.strength();
        assert!(s > 0.7 && s <= 1.0, "replied strength should be ~0.9, got {s}");
    }

    #[test]
    fn feedback_signal_strength_archived() {
        let sig = FeedbackSignal::ArchivedUnread { entity: sym(1) };
        let s = sig.strength();
        assert!(
            (s - (-0.5)).abs() < 0.01,
            "archived strength should be -0.5, got {s}"
        );
    }

    #[test]
    fn feedback_signal_strength_starred() {
        let sig = FeedbackSignal::Starred { entity: sym(1) };
        let s = sig.strength();
        assert!(
            (s - 1.0).abs() < 0.01,
            "starred strength should be 1.0, got {s}"
        );
    }

    #[test]
    fn feedback_signal_entity() {
        let sig = FeedbackSignal::Replied {
            entity: sym(42),
            response_time_secs: 10,
        };
        assert_eq!(sig.entity(), sym(42));

        let sig = FeedbackSignal::GoalCompleted {
            goal_id: sym(99),
            topic_symbols: vec![sym(1), sym(2)],
        };
        assert_eq!(sig.entity(), sym(99));
    }

    #[test]
    fn feedback_signal_goal_completed() {
        let sig = FeedbackSignal::GoalCompleted {
            goal_id: sym(10),
            topic_symbols: vec![sym(1), sym(2), sym(3)],
        };
        let s = sig.strength();
        assert!(s > 0.5, "goal completed should be positive, got {s}");
        assert_eq!(sig.entity(), sym(10));
    }

    // ── PreferenceProfile ────────────────────────────────────────────

    #[test]
    fn preference_profile_default() {
        let profile = PreferenceProfile::default();
        assert_eq!(profile.interaction_count, 0);
        assert!(profile.interest_prototype.is_none());
        assert_eq!(profile.proactivity_level, ProactivityLevel::Ambient);
        assert_eq!(profile.suggestions_offered, 0);
        assert_eq!(profile.suggestions_accepted, 0);
    }

    // ── Temporal Decay ───────────────────────────────────────────────

    #[test]
    fn temporal_decay_recent() {
        // 1 hour old → barely decays with λ=0.01
        let decayed = PreferenceManager::apply_temporal_decay(1.0, 3600, 0.01);
        assert!(
            decayed > 0.99,
            "1 hour should barely decay: got {decayed}"
        );
    }

    #[test]
    fn temporal_decay_old() {
        // 90 days old → should be significantly decayed
        let age_secs = 90 * 86400;
        let decayed = PreferenceManager::apply_temporal_decay(1.0, age_secs, 0.01);
        assert!(
            decayed < 0.5,
            "90 days should heavily decay: got {decayed}"
        );
    }

    #[test]
    fn temporal_decay_zero_age() {
        let decayed = PreferenceManager::apply_temporal_decay(1.0, 0, 0.01);
        assert!(
            (decayed - 1.0).abs() < 0.001,
            "zero age should not decay: got {decayed}"
        );
    }

    // ── Serialization ────────────────────────────────────────────────

    #[test]
    fn preference_manager_serialization_roundtrip() {
        let mgr = PreferenceManager::default();
        let bytes = bincode::serialize(&mgr).unwrap();
        let restored: PreferenceManager = bincode::deserialize(&bytes).unwrap();
        assert_eq!(
            restored.profile.interaction_count,
            mgr.profile.interaction_count
        );
        assert_eq!(
            restored.profile.decay_rate,
            mgr.profile.decay_rate
        );
        assert_eq!(
            restored.profile.proactivity_level,
            mgr.profile.proactivity_level
        );
    }

    // ── Suggestion Acceptance ────────────────────────────────────────

    #[test]
    fn suggestion_acceptance_rate_empty() {
        let mgr = PreferenceManager::default();
        assert_eq!(mgr.suggestion_acceptance_rate(), 0.0);
    }

    #[test]
    fn suggestion_acceptance_rate_mixed() {
        let mut mgr = PreferenceManager::default();
        mgr.record_suggestion_outcome(true);
        mgr.record_suggestion_outcome(true);
        mgr.record_suggestion_outcome(true);
        mgr.record_suggestion_outcome(false);
        mgr.record_suggestion_outcome(false);
        let rate = mgr.suggestion_acceptance_rate();
        assert!(
            (rate - 0.6).abs() < 0.01,
            "3/5 should be 0.6, got {rate}"
        );
    }

    #[test]
    fn suggestion_dismissed_decreases() {
        let sig = FeedbackSignal::SuggestionDismissed { entity: sym(1) };
        assert!(sig.strength() < 0.0, "dismissed should be negative");
    }

    // ── Context / JITIR ──────────────────────────────────────────────

    #[test]
    fn jitir_result_default() {
        let result = JitirResult::default();
        assert!(result.direct_matches.is_empty());
        assert!(result.serendipity_matches.is_empty());
        assert!(result.context_summary.is_empty());
    }

    // ── Serendipity Zone ─────────────────────────────────────────────

    #[test]
    fn serendipity_zone_filter_test() {
        assert!(!serendipity_zone_filter(0.2, 0.3, 0.6));
        assert!(serendipity_zone_filter(0.3, 0.3, 0.6));
        assert!(serendipity_zone_filter(0.45, 0.3, 0.6));
        assert!(serendipity_zone_filter(0.6, 0.3, 0.6));
        assert!(!serendipity_zone_filter(0.7, 0.3, 0.6));
    }

    // ── PreferencePredicates ─────────────────────────────────────────

    #[test]
    fn preference_predicates_namespace() {
        // Verify the relation names all start with "pref:" (by construction).
        // We can't test init without an engine, so test the literals directly.
        let names = [
            "pref:interest-topic",
            "pref:interest-weight",
            "pref:feedback-signal",
            "pref:feedback-time",
            "pref:proactivity-level",
            "pref:suggestion-count",
            "pref:suggestion-accepted",
            "pref:last-jitir-cycle",
        ];
        for name in &names {
            assert!(name.starts_with("pref:"), "{name} should start with pref:");
        }
    }

    // ── Role Vectors ─────────────────────────────────────────────────

    #[test]
    fn role_vectors_distinct() {
        let ops = VsaOps::new(
            crate::simd::best_kernel(),
            crate::vsa::Dimension::TEST,
            crate::vsa::Encoding::Bipolar,
        );
        let roles = PreferenceRoleVectors::new(&ops);

        // All role vectors should be pairwise dissimilar.
        let vecs = [
            &roles.topic,
            &roles.interaction_type,
            &roles.recency,
            &roles.frequency,
            &roles.source_channel,
            &roles.entity_kind,
        ];
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let sim = ops.similarity(vecs[i], vecs[j]).unwrap_or(0.0);
                assert!(
                    sim < 0.7,
                    "role vectors {i} and {j} too similar: {sim}"
                );
            }
        }
    }

    // ── PreferenceReview ─────────────────────────────────────────────

    #[test]
    fn preference_review_from_manager() {
        let mut mgr = PreferenceManager::default();
        mgr.profile.interaction_count = 42;
        mgr.profile.suggestions_offered = 10;
        mgr.profile.suggestions_accepted = 7;
        mgr.profile.proactivity_level = ProactivityLevel::Nudge;

        // Without engine, we can test the non-engine parts.
        let review = PreferenceReview {
            interaction_count: mgr.profile.interaction_count,
            top_interests: Vec::new(),
            acceptance_rate: mgr.suggestion_acceptance_rate(),
            proactivity_level: mgr.profile.proactivity_level,
        };
        assert_eq!(review.interaction_count, 42);
        assert!((review.acceptance_rate - 0.7).abs() < 0.01);
        assert_eq!(review.proactivity_level, ProactivityLevel::Nudge);
    }
}
