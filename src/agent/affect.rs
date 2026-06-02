//! Affective system — Phase 23.
//!
//! Dimensional emotion model (Russell's circumplex: valence × arousal × dominance)
//! with Damasio's somatic marker hypothesis: learned emotional associations that
//! bias decision-making as computational shortcuts.
//!
//! ## Sub-phases
//!
//! - **23a**: Appraisal engine + mood state (dimensional emotion from events)
//! - **23b**: Somatic markers (learned concept-valence associations from outcomes)
//! - **23c**: Affective memory salience (emotion-prioritized consolidation)

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

// ═══════════════════════════════════════════════════════════════════════
// 23a — Affective State & Appraisal
// ═══════════════════════════════════════════════════════════════════════

/// Dimensional emotion state (Russell's circumplex + dominance).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AffectiveState {
    /// Valence: -1.0 (unpleasant) to 1.0 (pleasant).
    pub valence: f32,
    /// Arousal: 0.0 (calm) to 1.0 (activated).
    pub arousal: f32,
    /// Dominance: 0.0 (submissive) to 1.0 (dominant/in-control).
    pub dominance: f32,
}

impl Default for AffectiveState {
    fn default() -> Self {
        Self {
            valence: 0.0,   // neutral
            arousal: 0.3,   // mild alertness
            dominance: 0.5, // balanced
        }
    }
}

impl AffectiveState {
    /// Create a specific affective state.
    pub fn new(valence: f32, arousal: f32, dominance: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            dominance: dominance.clamp(0.0, 1.0),
        }
    }

    /// Qualitative label for the current state.
    pub fn label(&self) -> &'static str {
        if self.valence > 0.3 && self.arousal > 0.5 {
            "excited"
        } else if self.valence > 0.3 && self.arousal <= 0.5 {
            "content"
        } else if self.valence < -0.3 && self.arousal > 0.5 {
            "stressed"
        } else if self.valence < -0.3 && self.arousal <= 0.5 {
            "melancholic"
        } else if self.arousal > 0.7 {
            "alert"
        } else {
            "neutral"
        }
    }

    /// Blend toward another state by a factor (0.0–1.0).
    pub fn blend(&self, target: &AffectiveState, factor: f32) -> Self {
        let f = factor.clamp(0.0, 1.0);
        Self::new(
            self.valence + (target.valence - self.valence) * f,
            self.arousal + (target.arousal - self.arousal) * f,
            self.dominance + (target.dominance - self.dominance) * f,
        )
    }
}

/// An appraisal of an event (Scherer's model, simplified).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Appraisal {
    /// What triggered this appraisal.
    pub trigger: AppraisalTrigger,
    /// How relevant is this to current goals (0.0–1.0).
    pub relevance: f32,
    /// How unexpected (0.0 = expected, 1.0 = very surprising).
    pub novelty: f32,
    /// Outcome valence: positive (goal progress) or negative (setback).
    pub valence: f32,
    /// Agent's perceived ability to cope (0.0–1.0).
    pub coping: f32,
    /// Timestamp.
    pub timestamp: u64,
}

/// What triggered an appraisal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AppraisalTrigger {
    /// A goal was completed.
    GoalCompleted { goal_id: SymbolId },
    /// A goal failed.
    GoalFailed { goal_id: SymbolId },
    /// A contradiction was detected.
    Contradiction { claim: SymbolId },
    /// Novel information arrived.
    Novelty { entity: SymbolId },
    /// User gave positive feedback.
    PositiveFeedback,
    /// User gave negative feedback.
    NegativeFeedback,
    /// Resource pressure (memory, storage, etc.).
    ResourcePressure { severity: f32 },
    /// Temporal signal (from Phase 15d).
    TemporalSignal { valence: f32, arousal: f32 },
}

impl Appraisal {
    /// Convert this appraisal to an affective state delta.
    pub fn to_affect_delta(&self) -> AffectiveState {
        let valence = self.valence * self.relevance;
        let arousal = (self.novelty * 0.5 + self.relevance * 0.3 + (1.0 - self.coping) * 0.2)
            .clamp(0.0, 1.0);
        let dominance = self.coping;
        AffectiveState::new(valence, arousal, dominance)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Mood State
// ═══════════════════════════════════════════════════════════════════════

/// Slow-moving mood baseline that integrates recent appraisals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodState {
    /// Current baseline affective state.
    pub baseline: AffectiveState,
    /// Recent appraisals (bounded window).
    pub recent_appraisals: VecDeque<(Appraisal, u64)>,
    /// Exponential moving average of valence.
    pub ema_valence: f32,
    /// Exponential moving average of arousal.
    pub ema_arousal: f32,
    /// EMA smoothing factor (lower = slower mood changes).
    pub alpha: f32,
    /// Maximum recent appraisals to keep.
    pub window_size: usize,
}

impl Default for MoodState {
    fn default() -> Self {
        Self {
            baseline: AffectiveState::default(),
            recent_appraisals: VecDeque::new(),
            ema_valence: 0.0,
            ema_arousal: 0.3,
            alpha: 0.1,
            window_size: 50,
        }
    }
}

impl MoodState {
    /// Process an appraisal and update mood.
    pub fn process_appraisal(&mut self, appraisal: Appraisal) {
        let delta = appraisal.to_affect_delta();
        let ts = appraisal.timestamp;

        // Update EMA.
        self.ema_valence += self.alpha * (delta.valence - self.ema_valence);
        self.ema_arousal += self.alpha * (delta.arousal - self.ema_arousal);

        // Update baseline.
        self.baseline = AffectiveState::new(
            self.ema_valence,
            self.ema_arousal,
            self.baseline.dominance * 0.9 + delta.dominance * 0.1,
        );

        // Store in window.
        self.recent_appraisals.push_back((appraisal, ts));
        while self.recent_appraisals.len() > self.window_size {
            self.recent_appraisals.pop_front();
        }
    }

    /// Current affective state.
    pub fn current(&self) -> &AffectiveState {
        &self.baseline
    }

    /// Average valence over recent window.
    pub fn recent_valence(&self) -> f32 {
        if self.recent_appraisals.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .recent_appraisals
            .iter()
            .map(|(a, _)| a.valence * a.relevance)
            .sum();
        sum / self.recent_appraisals.len() as f32
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 23b — Somatic Markers
// ═══════════════════════════════════════════════════════════════════════

/// A learned emotional association between a concept and a valence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaticMarker {
    /// The concept this marker is about.
    pub concept: SymbolId,
    /// Learned valence (-1.0 to 1.0).
    pub valence: f32,
    /// Association strength (0.0–1.0).
    pub confidence: f32,
    /// How many outcomes contributed.
    pub observation_count: u32,
}

/// Registry of somatic markers — fast-path emotional associations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SomaticMarkerRegistry {
    pub markers: HashMap<u64, SomaticMarker>,
}

impl SomaticMarkerRegistry {
    /// Get the somatic marker for a concept (if any).
    pub fn get(&self, concept: SymbolId) -> Option<&SomaticMarker> {
        self.markers.get(&concept.get())
    }

    /// Update a somatic marker from an observed outcome.
    ///
    /// Uses EMA: marker_valence += α * (outcome_valence - marker_valence).
    pub fn update(&mut self, concept: SymbolId, outcome_valence: f32) {
        let marker = self.markers.entry(concept.get()).or_insert(SomaticMarker {
            concept,
            valence: 0.0,
            confidence: 0.0,
            observation_count: 0,
        });

        marker.observation_count += 1;
        let alpha = 1.0 / (1.0 + marker.observation_count as f32 * 0.2);
        marker.valence += alpha * (outcome_valence - marker.valence);
        marker.confidence = (marker.observation_count as f32 / (marker.observation_count as f32 + 5.0)).min(0.95);
    }

    /// Get the decision bias for a concept.
    ///
    /// Returns a valence × confidence value that can bias action selection.
    /// Positive = approach, negative = avoid.
    pub fn decision_bias(&self, concept: SymbolId) -> f32 {
        self.markers
            .get(&concept.get())
            .map(|m| m.valence * m.confidence)
            .unwrap_or(0.0)
    }

    /// Top-k most positive markers (approach signals).
    pub fn most_positive(&self, k: usize) -> Vec<&SomaticMarker> {
        let mut sorted: Vec<_> = self.markers.values().collect();
        sorted.sort_by(|a, b| b.valence.partial_cmp(&a.valence).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }

    /// Top-k most negative markers (avoidance signals).
    pub fn most_negative(&self, k: usize) -> Vec<&SomaticMarker> {
        let mut sorted: Vec<_> = self.markers.values().collect();
        sorted.sort_by(|a, b| a.valence.partial_cmp(&b.valence).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 23c — Affective Memory Salience
// ═══════════════════════════════════════════════════════════════════════

/// Compute memory salience boost from emotional significance.
///
/// Emotionally significant events (high |valence| × arousal) get a
/// consolidation priority boost.
pub fn emotional_salience(affect: &AffectiveState) -> f32 {
    let emotional_intensity = affect.valence.abs() * affect.arousal;
    // Salience boost: 1.0 (neutral) to 2.0 (highly emotional).
    1.0 + emotional_intensity
}

/// Mood-congruent retrieval bias.
///
/// When current mood is negative, memories with negative valence are
/// easier to recall (and vice versa). Returns a relevance multiplier.
pub fn mood_congruence_bias(current_mood: &AffectiveState, memory_valence: f32) -> f32 {
    // Same sign = congruent → boost; opposite sign = incongruent → suppress.
    let alignment = current_mood.valence * memory_valence;
    if alignment > 0.0 {
        1.0 + alignment * 0.3 // boost up to 1.3×
    } else {
        1.0 + alignment * 0.1 // slight suppression (down to 0.9×)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// AffectiveSystem (combined)
// ═══════════════════════════════════════════════════════════════════════

/// The complete affective system: mood + somatic markers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AffectiveSystem {
    /// Current mood state.
    pub mood: MoodState,
    /// Learned somatic markers.
    pub markers: SomaticMarkerRegistry,
}

impl AffectiveSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process an appraisal event.
    pub fn appraise(&mut self, appraisal: Appraisal) {
        self.mood.process_appraisal(appraisal);
    }

    /// Record an outcome for somatic marker learning.
    pub fn record_outcome(&mut self, concept: SymbolId, valence: f32) {
        self.markers.update(concept, valence);
    }

    /// Current affective state.
    pub fn current_affect(&self) -> &AffectiveState {
        self.mood.current()
    }

    /// Decision bias for a concept.
    pub fn bias_for(&self, concept: SymbolId) -> f32 {
        self.markers.decision_bias(concept)
    }

    /// Behavioral modulation: exploration factor.
    ///
    /// Positive mood → more exploration. Negative → conservative.
    pub fn exploration_modulation(&self) -> f32 {
        let v = self.mood.baseline.valence;
        if v > 0.2 {
            1.0 + v * 0.3 // up to 1.3×
        } else if v < -0.2 {
            1.0 + v * 0.2 // down to 0.8×
        } else {
            1.0
        }
    }

    /// Behavioral modulation: decision speed.
    ///
    /// High arousal → faster (fewer MCTS iterations). Low → more deliberate.
    pub fn decision_speed_factor(&self) -> f32 {
        let a = self.mood.baseline.arousal;
        if a > 0.7 {
            0.7 // faster (fewer iterations)
        } else if a < 0.3 {
            1.3 // more deliberate
        } else {
            1.0
        }
    }

    /// Summary for monitoring.
    pub fn summary(&self) -> String {
        let a = self.mood.current();
        format!(
            "affect: {} (v={:.2}, a={:.2}, d={:.2}), markers: {}, recent: {}",
            a.label(),
            a.valence,
            a.arousal,
            a.dominance,
            self.markers.markers.len(),
            self.mood.recent_appraisals.len(),
        )
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &crate::engine::Engine) -> Result<(), String> {
        let bytes = bincode::serialize(self)
            .map_err(|e| format!("serialize affective system: {e}"))?;
        engine
            .store()
            .put_meta(b"agent:affective_system", &bytes)
            .map_err(|e| format!("persist affective system: {e}"))?;
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &crate::engine::Engine) -> Self {
        engine
            .store()
            .get_meta(b"agent:affective_system")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default()
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

    fn now() -> u64 {
        100
    }

    // ── AffectiveState ────────────────────────────────────────────

    #[test]
    fn affective_state_default_neutral() {
        let s = AffectiveState::default();
        assert_eq!(s.label(), "neutral");
    }

    #[test]
    fn affective_state_labels() {
        assert_eq!(AffectiveState::new(0.5, 0.8, 0.5).label(), "excited");
        assert_eq!(AffectiveState::new(0.5, 0.3, 0.5).label(), "content");
        assert_eq!(AffectiveState::new(-0.5, 0.8, 0.5).label(), "stressed");
        assert_eq!(AffectiveState::new(-0.5, 0.3, 0.5).label(), "melancholic");
    }

    #[test]
    fn affective_state_blend() {
        let a = AffectiveState::new(0.0, 0.0, 0.5);
        let b = AffectiveState::new(1.0, 1.0, 1.0);
        let blended = a.blend(&b, 0.5);
        assert!((blended.valence - 0.5).abs() < 0.01);
        assert!((blended.arousal - 0.5).abs() < 0.01);
    }

    // ── Appraisal ─────────────────────────────────────────────────

    #[test]
    fn appraisal_goal_completed() {
        let appraisal = Appraisal {
            trigger: AppraisalTrigger::GoalCompleted { goal_id: sym(1) },
            relevance: 0.9,
            novelty: 0.2,
            valence: 0.8,
            coping: 0.9,
            timestamp: now(),
        };
        let delta = appraisal.to_affect_delta();
        assert!(delta.valence > 0.5, "goal completion → positive valence");
        assert!(delta.dominance > 0.5, "high coping → high dominance");
    }

    #[test]
    fn appraisal_goal_failed() {
        let appraisal = Appraisal {
            trigger: AppraisalTrigger::GoalFailed { goal_id: sym(1) },
            relevance: 0.9,
            novelty: 0.5,
            valence: -0.8,
            coping: 0.3,
            timestamp: now(),
        };
        let delta = appraisal.to_affect_delta();
        assert!(delta.valence < -0.5, "goal failure → negative valence");
        assert!(delta.arousal > 0.3, "high novelty → high arousal");
    }

    // ── MoodState ─────────────────────────────────────────────────

    #[test]
    fn mood_shifts_with_positive_appraisals() {
        let mut mood = MoodState::default();
        for _ in 0..10 {
            mood.process_appraisal(Appraisal {
                trigger: AppraisalTrigger::PositiveFeedback,
                relevance: 0.8,
                novelty: 0.3,
                valence: 0.7,
                coping: 0.8,
                timestamp: now(),
            });
        }
        assert!(mood.baseline.valence > 0.0, "mood should be positive: {}", mood.baseline.valence);
    }

    #[test]
    fn mood_shifts_with_negative_appraisals() {
        let mut mood = MoodState::default();
        for _ in 0..10 {
            mood.process_appraisal(Appraisal {
                trigger: AppraisalTrigger::NegativeFeedback,
                relevance: 0.9,
                novelty: 0.5,
                valence: -0.8,
                coping: 0.2,
                timestamp: now(),
            });
        }
        assert!(mood.baseline.valence < 0.0, "mood should be negative");
    }

    #[test]
    fn mood_window_bounded() {
        let mut mood = MoodState { window_size: 5, ..Default::default() };
        for i in 0..20 {
            mood.process_appraisal(Appraisal {
                trigger: AppraisalTrigger::Novelty { entity: sym(i + 1) },
                relevance: 0.5,
                novelty: 0.5,
                valence: 0.1,
                coping: 0.5,
                timestamp: i,
            });
        }
        assert_eq!(mood.recent_appraisals.len(), 5);
    }

    // ── Somatic Markers ───────────────────────────────────────────

    #[test]
    fn somatic_marker_learns_from_outcomes() {
        let mut registry = SomaticMarkerRegistry::default();
        let concept = sym(42);

        // Positive outcomes.
        for _ in 0..5 {
            registry.update(concept, 0.8);
        }

        let marker = registry.get(concept).unwrap();
        assert!(marker.valence > 0.3, "should learn positive: {}", marker.valence);
        assert!(marker.confidence > 0.3, "should gain confidence");
    }

    #[test]
    fn somatic_marker_negative() {
        let mut registry = SomaticMarkerRegistry::default();
        let concept = sym(42);

        for _ in 0..5 {
            registry.update(concept, -0.7);
        }

        let bias = registry.decision_bias(concept);
        assert!(bias < 0.0, "should produce avoidance bias: {bias}");
    }

    #[test]
    fn somatic_marker_unknown_concept() {
        let registry = SomaticMarkerRegistry::default();
        assert_eq!(registry.decision_bias(sym(999)), 0.0);
    }

    #[test]
    fn somatic_marker_ranking() {
        let mut registry = SomaticMarkerRegistry::default();
        registry.update(sym(1), 0.9);
        registry.update(sym(2), -0.8);
        registry.update(sym(3), 0.1);

        let positive = registry.most_positive(1);
        assert_eq!(positive[0].concept, sym(1));

        let negative = registry.most_negative(1);
        assert_eq!(negative[0].concept, sym(2));
    }

    // ── Memory salience ───────────────────────────────────────────

    #[test]
    fn emotional_salience_boost() {
        let neutral = AffectiveState::new(0.0, 0.3, 0.5);
        let emotional = AffectiveState::new(0.8, 0.9, 0.5);

        assert!((emotional_salience(&neutral) - 1.0).abs() < 0.1);
        assert!(emotional_salience(&emotional) > 1.5);
    }

    #[test]
    fn mood_congruent_retrieval() {
        let positive_mood = AffectiveState::new(0.7, 0.5, 0.5);

        // Positive memory + positive mood → boost.
        let boost = mood_congruence_bias(&positive_mood, 0.6);
        assert!(boost > 1.0);

        // Negative memory + positive mood → slight suppression.
        let suppress = mood_congruence_bias(&positive_mood, -0.5);
        assert!(suppress < 1.0);
    }

    // ── AffectiveSystem ───────────────────────────────────────────

    #[test]
    fn affective_system_exploration_modulation() {
        let mut sys = AffectiveSystem::new();
        // Positive mood → more exploration.
        for _ in 0..20 {
            sys.appraise(Appraisal {
                trigger: AppraisalTrigger::PositiveFeedback,
                relevance: 0.8,
                novelty: 0.2,
                valence: 0.7,
                coping: 0.9,
                timestamp: now(),
            });
        }
        assert!(sys.exploration_modulation() > 1.0);
    }

    #[test]
    fn affective_system_summary() {
        let sys = AffectiveSystem::new();
        let s = sys.summary();
        assert!(s.contains("affect:"));
        assert!(s.contains("markers:"));
    }

    #[test]
    fn serialization_roundtrip() {
        let mut sys = AffectiveSystem::new();
        sys.record_outcome(sym(1), 0.5);
        sys.appraise(Appraisal {
            trigger: AppraisalTrigger::PositiveFeedback,
            relevance: 0.5,
            novelty: 0.3,
            valence: 0.4,
            coping: 0.7,
            timestamp: 100,
        });

        let bytes = bincode::serialize(&sys).unwrap();
        let restored: AffectiveSystem = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.markers.markers.len(), 1);
        assert_eq!(restored.mood.recent_appraisals.len(), 1);
    }
}
