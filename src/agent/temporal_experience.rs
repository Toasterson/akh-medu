//! Temporal experience — Phase 15d (Lifeform Engine Extension).
//!
//! A sense of time passing. Computes subjective temporal signals from the
//! agent's event history: event density ("time flies"), boredom (idle too long),
//! deadline pressure (urgency), and a circadian-like activity cycle.
//!
//! These signals modulate behavior:
//! - Boredom amplifies curiosity drive
//! - Urgency suppresses exploration in favor of exploitation
//! - Circadian phase modulates consolidation (more during "night")
//! - Event density affects episodic memory subjective duration
//!
//! Designed to output signals consumable by Phase 23 (Affective System)
//! when implemented.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════
// TemporalExperience
// ═══════════════════════════════════════════════════════════════════════

/// The agent's subjective sense of time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalExperience {
    /// Ratio of meaningful events per unit time. High = "time flies".
    pub event_density: f32,
    /// Seconds since last meaningful event. Long → boredom.
    pub time_since_last_event: u64,
    /// Urgency signal from approaching deadlines (0.0–1.0).
    pub deadline_pressure: f32,
    /// Circadian-like phase (0.0–1.0 cycling over ~24h).
    /// 0.0 = "dawn" (high activity), 0.5 = "dusk" (consolidation), 1.0 = "dawn" again.
    pub circadian_phase: f32,
    /// Total uptime in seconds.
    pub uptime_secs: u64,
    /// Number of meaningful events recorded.
    pub event_count: u64,
    /// Timestamp of the last meaningful event.
    pub last_event_at: u64,
    /// Timestamp when the agent started.
    pub started_at: u64,
}

impl Default for TemporalExperience {
    fn default() -> Self {
        let now = now_secs();
        Self {
            event_density: 0.0,
            time_since_last_event: 0,
            deadline_pressure: 0.0,
            circadian_phase: 0.0,
            uptime_secs: 0,
            event_count: 0,
            last_event_at: now,
            started_at: now,
        }
    }
}

impl TemporalExperience {
    /// Update the temporal experience state.
    ///
    /// Call this periodically (e.g., each OODA cycle or idle tick).
    pub fn update(&mut self, now: u64) {
        self.uptime_secs = now.saturating_sub(self.started_at);
        self.time_since_last_event = now.saturating_sub(self.last_event_at);

        // Compute event density: events per hour (smoothed).
        let hours = (self.uptime_secs as f32 / 3600.0).max(0.01);
        self.event_density = self.event_count as f32 / hours;

        // Circadian phase: cycles over 24h (86400 seconds).
        let phase_secs = self.uptime_secs % 86400;
        self.circadian_phase = phase_secs as f32 / 86400.0;
    }

    /// Record that a meaningful event occurred.
    pub fn record_event(&mut self, now: u64) {
        self.event_count += 1;
        self.last_event_at = now;
        self.time_since_last_event = 0;
    }

    /// Set deadline pressure from approaching deadlines.
    ///
    /// `seconds_until_deadline`: time remaining. Pressure increases as it approaches 0.
    pub fn set_deadline_pressure(&mut self, seconds_until_deadline: Option<u64>) {
        self.deadline_pressure = match seconds_until_deadline {
            None => 0.0,
            Some(0) => 1.0,
            Some(secs) => {
                // Exponential urgency: pressure = 1 / (1 + secs/300)
                // At 5min: 0.5, at 1min: 0.83, at 30min: 0.14
                1.0 / (1.0 + secs as f32 / 300.0)
            }
        };
    }

    // ─── Derived Signals ──────────────────────────────────────────

    /// Boredom level (0.0–1.0). High when idle for a long time with low event density.
    pub fn boredom(&self) -> f32 {
        // Boredom increases with idle time, decreases with event density.
        let idle_factor = (self.time_since_last_event as f32 / 600.0).min(1.0); // maxes at 10min
        let density_factor = 1.0 - (self.event_density / 10.0).min(1.0); // low density → high
        (idle_factor * 0.6 + density_factor * 0.4).clamp(0.0, 1.0)
    }

    /// Arousal level (0.0–1.0). High during urgency, low during boredom.
    pub fn arousal(&self) -> f32 {
        let urgency_component = self.deadline_pressure;
        let density_component = (self.event_density / 20.0).min(0.5);
        (urgency_component * 0.7 + density_component * 0.3).clamp(0.0, 1.0)
    }

    /// Whether the agent is in "consolidation mode" (circadian night phase).
    ///
    /// Night phase: circadian_phase in [0.4, 0.8] → consolidation preferred.
    pub fn is_consolidation_phase(&self) -> bool {
        self.circadian_phase >= 0.4 && self.circadian_phase <= 0.8
    }

    /// Curiosity amplification factor based on boredom.
    ///
    /// When bored, curiosity drive is amplified (1.0–2.0×).
    pub fn curiosity_amplifier(&self) -> f32 {
        1.0 + self.boredom()
    }

    /// Exploration suppression factor based on urgency.
    ///
    /// Under deadline pressure, exploration is suppressed (0.0–1.0).
    /// 1.0 = full exploration, 0.0 = pure exploitation.
    pub fn exploration_factor(&self) -> f32 {
        (1.0 - self.deadline_pressure).max(0.1) // never fully suppress
    }

    /// Subjective duration of the last interval (for episodic memory).
    ///
    /// Dense event periods feel shorter; idle periods feel longer.
    /// Returns a multiplier: <1.0 = felt shorter, >1.0 = felt longer.
    pub fn subjective_duration_multiplier(&self) -> f32 {
        if self.event_density > 5.0 {
            0.7 // time flies
        } else if self.event_density < 0.5 {
            1.5 // time drags
        } else {
            1.0 // normal
        }
    }

    // ─── Affective Signals (for Phase 23) ─────────────────────────

    /// Generate temporal affective signals for the affective system.
    ///
    /// Returns (valence, arousal) where:
    /// - valence: -1.0 (negative) to 1.0 (positive)
    /// - arousal: 0.0 (calm) to 1.0 (activated)
    pub fn affective_signal(&self) -> (f32, f32) {
        let boredom = self.boredom();
        let arousal = self.arousal();

        // Valence:
        // - Boredom → mild negative
        // - Extreme urgency → anxiety (negative)
        // - Moderate activity → positive (flow state)
        let valence = if self.deadline_pressure > 0.7 {
            -0.5 * self.deadline_pressure // anxiety
        } else if boredom > 0.6 {
            -0.3 * boredom // restlessness
        } else if self.event_density > 2.0 && self.event_density < 15.0 {
            0.3 // flow state
        } else {
            0.0 // neutral
        };

        (valence.clamp(-1.0, 1.0), arousal)
    }

    /// Summary string for monitoring.
    pub fn summary(&self) -> String {
        let (valence, arousal) = self.affective_signal();
        format!(
            "density={:.1}/hr, idle={}s, boredom={:.2}, urgency={:.2}, \
             circadian={:.2}{}, arousal={:.2}, valence={:.2}",
            self.event_density,
            self.time_since_last_event,
            self.boredom(),
            self.deadline_pressure,
            self.circadian_phase,
            if self.is_consolidation_phase() { " [consolidation]" } else { "" },
            arousal,
            valence,
        )
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp(event_count: u64, last_event_at: u64, started_at: u64) -> TemporalExperience {
        let mut exp = TemporalExperience {
            started_at,
            last_event_at,
            event_count,
            ..Default::default()
        };
        exp.update(last_event_at + 60); // 60s after last event
        exp
    }

    // ── Event density ─────────────────────────────────────────────

    #[test]
    fn event_density_increases_with_events() {
        let exp = make_exp(100, 3600, 0); // 100 events in 1 hour
        assert!(exp.event_density > 50.0);
    }

    #[test]
    fn event_density_low_when_idle() {
        let exp = make_exp(1, 100, 0);
        assert!(exp.event_density < 50.0);
    }

    // ── Boredom ───────────────────────────────────────────────────

    #[test]
    fn boredom_high_when_idle() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;
        exp.last_event_at = 0;
        exp.event_count = 1;
        exp.update(1200); // 20 minutes idle
        assert!(exp.boredom() > 0.5, "boredom={}", exp.boredom());
    }

    #[test]
    fn boredom_low_when_active() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;
        exp.last_event_at = 3590;
        exp.event_count = 100;
        exp.update(3600); // just had an event
        assert!(exp.boredom() < 0.5, "boredom={}", exp.boredom());
    }

    // ── Deadline pressure ─────────────────────────────────────────

    #[test]
    fn deadline_pressure_none() {
        let mut exp = TemporalExperience::default();
        exp.set_deadline_pressure(None);
        assert_eq!(exp.deadline_pressure, 0.0);
    }

    #[test]
    fn deadline_pressure_imminent() {
        let mut exp = TemporalExperience::default();
        exp.set_deadline_pressure(Some(0));
        assert_eq!(exp.deadline_pressure, 1.0);
    }

    #[test]
    fn deadline_pressure_5min() {
        let mut exp = TemporalExperience::default();
        exp.set_deadline_pressure(Some(300)); // 5 minutes
        assert!((exp.deadline_pressure - 0.5).abs() < 0.01);
    }

    // ── Circadian phase ───────────────────────────────────────────

    #[test]
    fn circadian_phase_cycles() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;

        exp.update(0);
        assert!((exp.circadian_phase - 0.0).abs() < 0.01); // dawn

        exp.update(43200); // 12 hours
        assert!((exp.circadian_phase - 0.5).abs() < 0.01); // dusk

        exp.update(86400); // 24 hours — wraps to 0
        assert!((exp.circadian_phase - 0.0).abs() < 0.01); // dawn again
    }

    #[test]
    fn consolidation_phase() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;

        exp.update(43200); // 12h → phase 0.5
        assert!(exp.is_consolidation_phase());

        exp.update(7200); // 2h → phase ~0.08
        assert!(!exp.is_consolidation_phase());
    }

    // ── Behavioral modulation ─────────────────────────────────────

    #[test]
    fn curiosity_amplified_when_bored() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;
        exp.last_event_at = 0;
        exp.event_count = 0;
        exp.update(1200); // 20 min idle
        assert!(exp.curiosity_amplifier() > 1.0);
    }

    #[test]
    fn exploration_suppressed_under_pressure() {
        let mut exp = TemporalExperience::default();
        exp.set_deadline_pressure(Some(60)); // 1 minute
        assert!(exp.exploration_factor() < 0.5);
    }

    #[test]
    fn exploration_full_when_relaxed() {
        let exp = TemporalExperience::default();
        assert!((exp.exploration_factor() - 1.0).abs() < f32::EPSILON);
    }

    // ── Subjective duration ───────────────────────────────────────

    #[test]
    fn time_flies_when_busy() {
        let exp = make_exp(200, 3600, 0); // 200 events in ~1hr
        assert!(exp.subjective_duration_multiplier() < 1.0);
    }

    #[test]
    fn time_drags_when_idle() {
        let exp = make_exp(0, 3600, 0); // no events
        assert!(exp.subjective_duration_multiplier() > 1.0);
    }

    // ── Affective signals ─────────────────────────────────────────

    #[test]
    fn affective_anxiety_under_deadline() {
        let mut exp = TemporalExperience::default();
        exp.set_deadline_pressure(Some(30)); // 30 seconds
        let (valence, arousal) = exp.affective_signal();
        assert!(valence < 0.0, "should be anxious: valence={valence}");
        assert!(arousal > 0.3, "should be aroused: arousal={arousal}");
    }

    #[test]
    fn affective_boredom_when_idle() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;
        exp.last_event_at = 0;
        exp.event_count = 0;
        exp.update(1200);
        let (valence, _arousal) = exp.affective_signal();
        assert!(valence < 0.0, "should be restless: valence={valence}");
    }

    #[test]
    fn affective_flow_when_active() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;
        exp.last_event_at = 3590;
        exp.event_count = 20; // moderate activity
        exp.update(3600);
        let (valence, _) = exp.affective_signal();
        assert!(valence >= 0.0, "should be positive/neutral: valence={valence}");
    }

    // ── Summary ───────────────────────────────────────────────────

    #[test]
    fn summary_readable() {
        let exp = TemporalExperience::default();
        let s = exp.summary();
        assert!(s.contains("density="));
        assert!(s.contains("boredom="));
    }

    // ── Record event ──────────────────────────────────────────────

    #[test]
    fn record_event_updates() {
        let mut exp = TemporalExperience::default();
        exp.started_at = 0;
        exp.record_event(100);
        assert_eq!(exp.event_count, 1);
        assert_eq!(exp.last_event_at, 100);
        assert_eq!(exp.time_since_last_event, 0);
    }

    // ── Serialization ─────────────────────────────────────────────

    #[test]
    fn serialization_roundtrip() {
        let mut exp = TemporalExperience::default();
        exp.event_count = 42;
        exp.deadline_pressure = 0.7;

        let bytes = bincode::serialize(&exp).unwrap();
        let restored: TemporalExperience = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.event_count, 42);
        assert!((restored.deadline_pressure - 0.7).abs() < f32::EPSILON);
    }
}
