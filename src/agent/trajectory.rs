//! Phase 34d — Conversation trajectory analysis.
//!
//! Tracks the discourse trajectory of a conversation: which topics were
//! discussed (and how often), which response strategies the agent used,
//! the current discourse mode, and an estimate of user engagement.
//!
//! The trajectory feeds the response-strategy selector (Phase 34a): a
//! repeated topic penalises `DirectAnswer`/`Explanation` and favours
//! `Contextualization`/`Analogy`; an argumentative mode favours
//! `EvidenceWeighing`/`Challenge`; dropping engagement favours
//! re-engagement strategies (`ExplorationPrompt`).
//!
//! Plan: docs/ai/plans/2026-03-26-phase34-communicative-intelligence.md
//! ADR:  docs/ai/decisions/044-communicative-intelligence.md

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

// ── DiscourseMode ────────────────────────────────────────────────────────

/// The overall mode of the current discourse, inferred from user inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DiscourseMode {
    /// User is asking questions (Q&A mode).
    #[default]
    Interrogative,
    /// User is exploring a topic (teach mode).
    Exploratory,
    /// User is giving instructions (task mode).
    Directive,
    /// Greeting, small talk (chat mode).
    Social,
    /// User is debating / challenging (debate mode).
    Argumentative,
}

impl std::fmt::Display for DiscourseMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Interrogative => "interrogative",
            Self::Exploratory => "exploratory",
            Self::Directive => "directive",
            Self::Social => "social",
            Self::Argumentative => "argumentative",
        };
        write!(f, "{s}")
    }
}

/// Classify the discourse mode of a single user input, given the previous mode.
///
/// Pure heuristic — cheap substring / prefix checks, deterministic. When no
/// signal is found, the previous mode is retained (discourse modes are sticky).
pub fn classify_mode(input: &str, previous: DiscourseMode) -> DiscourseMode {
    let lower = input.trim().to_lowercase();
    if lower.is_empty() {
        return previous;
    }

    // Argumentative signals take priority — disagreement is a strong signal.
    const ARGUMENT_MARKERS: [&str; 8] = [
        "no,",
        "that's wrong",
        "thats wrong",
        "i disagree",
        "not true",
        "you're wrong",
        "youre wrong",
        "actually,",
    ];
    if ARGUMENT_MARKERS.iter().any(|m| lower.contains(m)) || lower.starts_with("but ") {
        return DiscourseMode::Argumentative;
    }

    // Social: greetings, thanks, farewells.
    const SOCIAL_MARKERS: [&str; 8] = [
        "hello",
        "hi ",
        "hey",
        "thanks",
        "thank you",
        "goodbye",
        "bye",
        "good morning",
    ];
    if SOCIAL_MARKERS.iter().any(|m| lower.starts_with(m)) || lower == "hi" {
        return DiscourseMode::Social;
    }

    // Exploratory: open-ended continuation prompts.
    const EXPLORE_MARKERS: [&str; 6] = [
        "tell me more",
        "what about",
        "what else",
        "interesting",
        "go on",
        "and then",
    ];
    if EXPLORE_MARKERS.iter().any(|m| lower.contains(m)) {
        return DiscourseMode::Exploratory;
    }

    // Interrogative: question word or trailing question mark.
    const QUESTION_STARTS: [&str; 9] = [
        "what", "who", "where", "when", "how", "why", "which", "is ", "are ",
    ];
    if lower.ends_with('?') || QUESTION_STARTS.iter().any(|q| lower.starts_with(q)) {
        return DiscourseMode::Interrogative;
    }

    // Directive: imperative verbs.
    const DIRECTIVE_STARTS: [&str; 8] = [
        "find ",
        "do ",
        "run ",
        "create ",
        "make ",
        "set ",
        "add ",
        "investigate ",
    ];
    if DIRECTIVE_STARTS.iter().any(|d| lower.starts_with(d)) {
        return DiscourseMode::Directive;
    }

    previous
}

// ── TurnRecord ───────────────────────────────────────────────────────────

/// A single recorded response turn in the trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    /// Normalised topic label the turn was about (lowercase).
    pub topic: Option<String>,
    /// Resolved topic symbol, when available.
    pub topic_id: Option<SymbolId>,
    /// Name of the response strategy the agent used for this turn.
    pub strategy: String,
    /// Unix timestamp (seconds).
    pub timestamp: u64,
}

// ── DiscourseTrajectory ──────────────────────────────────────────────────

/// Discourse trajectory tracking for one conversation channel.
///
/// Bounded: keeps at most `max_turns` turn records; the repetition index is
/// kept in sync with the retained window so old topics eventually "cool off".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseTrajectory {
    /// Recorded response turns, oldest first.
    turns: Vec<TurnRecord>,
    /// How many times each (normalised) topic label appears in `turns`.
    repetition_count: HashMap<String, usize>,
    /// Current discourse mode.
    discourse_mode: DiscourseMode,
    /// Estimated user engagement in [0.0, 1.0].
    user_engagement: f32,
    /// Recent user-input lengths (chars), for the engagement estimate.
    input_lengths: Vec<usize>,
    /// Maximum retained turn records.
    max_turns: usize,
}

impl Default for DiscourseTrajectory {
    fn default() -> Self {
        Self::new(50)
    }
}

impl DiscourseTrajectory {
    /// Create a new trajectory retaining at most `max_turns` records.
    pub fn new(max_turns: usize) -> Self {
        Self {
            turns: Vec::new(),
            repetition_count: HashMap::new(),
            discourse_mode: DiscourseMode::default(),
            user_engagement: 0.6,
            input_lengths: Vec::new(),
            max_turns: max_turns.max(1),
        }
    }

    /// Observe a raw user input: updates the discourse mode and the
    /// engagement estimate. Call once per user turn, before strategy selection.
    pub fn observe_input(&mut self, input: &str) {
        self.discourse_mode = classify_mode(input, self.discourse_mode);

        let len = input.trim().chars().count();
        self.input_lengths.push(len);
        if self.input_lengths.len() > 10 {
            self.input_lengths.remove(0);
        }
        self.user_engagement = self.estimate_engagement();
    }

    /// Record a response turn: which topic was answered and which strategy
    /// was used. Call after the response is composed.
    pub fn record_turn(&mut self, topic: Option<&str>, topic_id: Option<SymbolId>, strategy: &str) {
        let normalised = topic.map(|t| t.trim().to_lowercase());
        if let Some(ref t) = normalised {
            *self.repetition_count.entry(t.clone()).or_insert(0) += 1;
        }
        self.turns.push(TurnRecord {
            topic: normalised,
            topic_id,
            strategy: strategy.to_string(),
            timestamp: now_secs(),
        });

        // Evict oldest records and decay their repetition counts.
        while self.turns.len() > self.max_turns {
            let old = self.turns.remove(0);
            if let Some(topic) = old.topic {
                if let Some(count) = self.repetition_count.get_mut(&topic) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.repetition_count.remove(&topic);
                    }
                }
            }
        }
    }

    /// How many prior response turns covered this topic.
    pub fn times_seen(&self, topic: &str) -> usize {
        self.repetition_count
            .get(&topic.trim().to_lowercase())
            .copied()
            .unwrap_or(0)
    }

    /// Whether this topic was already answered at least once before.
    pub fn is_repeated(&self, topic: &str) -> bool {
        self.times_seen(topic) >= 1
    }

    /// The strategy used in the most recent response turn, if any.
    pub fn last_strategy(&self) -> Option<&str> {
        self.turns.last().map(|t| t.strategy.as_str())
    }

    /// Names of all strategies used, oldest first (bounded window).
    pub fn strategy_history(&self) -> Vec<&str> {
        self.turns.iter().map(|t| t.strategy.as_str()).collect()
    }

    /// Number of recorded response turns.
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Current discourse mode.
    pub fn discourse_mode(&self) -> DiscourseMode {
        self.discourse_mode
    }

    /// Estimated user engagement in [0.0, 1.0].
    pub fn user_engagement(&self) -> f32 {
        self.user_engagement
    }

    /// Engagement estimate: ratio of the recent input length to the running
    /// average. Steadily shrinking inputs ("ok", "hm") signal disengagement;
    /// long, elaborated inputs signal engagement.
    fn estimate_engagement(&self) -> f32 {
        if self.input_lengths.len() < 2 {
            return 0.6;
        }
        let total: usize = self.input_lengths.iter().sum();
        let avg = total as f32 / self.input_lengths.len() as f32;
        if avg < f32::EPSILON {
            return 0.3;
        }
        let recent_window = self.input_lengths.len().min(3);
        let recent: usize = self.input_lengths[self.input_lengths.len() - recent_window..]
            .iter()
            .sum();
        let recent_avg = recent as f32 / recent_window as f32;
        (0.5 * (recent_avg / avg)).clamp(0.05, 1.0)
    }
}

/// Current time as seconds since UNIX epoch.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_interrogative_from_question() {
        assert_eq!(
            classify_mode("What is a dog?", DiscourseMode::Social),
            DiscourseMode::Interrogative
        );
        assert_eq!(
            classify_mode("is rust fast", DiscourseMode::Social),
            DiscourseMode::Interrogative
        );
    }

    #[test]
    fn mode_argumentative_from_disagreement() {
        assert_eq!(
            classify_mode(
                "No, that's wrong — dogs are not reptiles",
                DiscourseMode::Interrogative
            ),
            DiscourseMode::Argumentative
        );
        assert_eq!(
            classify_mode("but you said otherwise", DiscourseMode::Interrogative),
            DiscourseMode::Argumentative
        );
    }

    #[test]
    fn mode_social_from_greeting() {
        assert_eq!(
            classify_mode("hello there", DiscourseMode::Interrogative),
            DiscourseMode::Social
        );
        assert_eq!(
            classify_mode("thanks!", DiscourseMode::Interrogative),
            DiscourseMode::Social
        );
    }

    #[test]
    fn mode_exploratory_from_continuation() {
        assert_eq!(
            classify_mode("tell me more about mammals", DiscourseMode::Interrogative),
            DiscourseMode::Exploratory
        );
    }

    #[test]
    fn mode_directive_from_imperative() {
        assert_eq!(
            classify_mode("find similar animals", DiscourseMode::Interrogative),
            DiscourseMode::Directive
        );
    }

    #[test]
    fn mode_sticky_without_signal() {
        assert_eq!(
            classify_mode("dogs mammals fur", DiscourseMode::Exploratory),
            DiscourseMode::Exploratory
        );
    }

    #[test]
    fn repetition_tracking() {
        let mut tr = DiscourseTrajectory::new(10);
        assert!(!tr.is_repeated("dog"));
        tr.record_turn(Some("Dog"), None, "DirectAnswer");
        assert!(tr.is_repeated("dog"));
        assert_eq!(tr.times_seen("DOG "), 1);
        tr.record_turn(Some("dog"), None, "Explanation");
        assert_eq!(tr.times_seen("dog"), 2);
        assert_eq!(tr.last_strategy(), Some("Explanation"));
    }

    #[test]
    fn eviction_decays_repetition() {
        let mut tr = DiscourseTrajectory::new(2);
        tr.record_turn(Some("a"), None, "DirectAnswer");
        tr.record_turn(Some("b"), None, "DirectAnswer");
        tr.record_turn(Some("c"), None, "DirectAnswer");
        // "a" evicted.
        assert_eq!(tr.times_seen("a"), 0);
        assert_eq!(tr.times_seen("b"), 1);
        assert_eq!(tr.turn_count(), 2);
    }

    #[test]
    fn engagement_drops_with_shrinking_inputs() {
        let mut tr = DiscourseTrajectory::new(10);
        tr.observe_input("Tell me everything you know about the taxonomy of mammals please");
        tr.observe_input("and what about their circulatory systems in detail?");
        let engaged = tr.user_engagement();
        tr.observe_input("ok");
        tr.observe_input("hm");
        tr.observe_input("k");
        assert!(
            tr.user_engagement() < engaged,
            "engagement should drop with terse inputs"
        );
    }

    #[test]
    fn observe_updates_mode() {
        let mut tr = DiscourseTrajectory::new(10);
        tr.observe_input("No, that's wrong.");
        assert_eq!(tr.discourse_mode(), DiscourseMode::Argumentative);
    }
}
