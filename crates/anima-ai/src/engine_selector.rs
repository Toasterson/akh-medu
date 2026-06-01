//! Heuristic engine selector: scores input and picks the best NL route.
//!
//! All scoring is regex + word-count based so routing overhead stays under 0.1 ms.
//! Thresholds are configurable; defaults are tuned for the three canonical cases:
//!   - Factual Q&A ("What is the Ennead?") → [`Route::Eliza`]
//!   - Creative generation ("Write a poem about …") → [`Route::Gemma`]
//!   - Advisory / reflection ("How should I approach this?") → [`Route::Hybrid`]

use std::sync::OnceLock;

use regex::Regex;

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------

/// The routing decision produced by [`EngineSelector::analyze`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// KG-grounded deterministic Q&A via ELIZA patterns. Expected latency < 1 ms.
    Eliza,
    /// Creative / novel / personality-rich generation via Gemma+Psyche. 1–3 s.
    Gemma,
    /// KG fact extraction (ELIZA) followed by Gemma personality enrichment. 1–3 s.
    Hybrid,
    /// All local engines unavailable; delegate to the configured remote LLM.
    RemoteFallback,
}

// ---------------------------------------------------------------------------
// SelectionContext
// ---------------------------------------------------------------------------

/// Scoring breakdown and final decision returned by [`EngineSelector::analyze`].
///
/// Useful for tracing, observability, and unit testing the routing heuristics.
#[derive(Debug, Clone)]
pub struct SelectionContext {
    /// Estimated probability that the KG holds a direct answer (0.0–1.0).
    pub kg_relevance: f32,
    /// Estimated probability that creative / novel generation is required (0.0–1.0).
    pub novelty_score: f32,
    /// Estimated probability that Psyche / reflection mode is appropriate (0.0–1.0).
    pub psyche_fit: f32,
    /// Final routing decision.
    pub route: Route,
}

// ---------------------------------------------------------------------------
// Compiled pattern helpers (OnceLock — compiled once, reused forever)
// ---------------------------------------------------------------------------

fn kg_patterns() -> &'static Regex {
    static PAT: OnceLock<Regex> = OnceLock::new();
    PAT.get_or_init(|| {
        Regex::new(
            r"(?i)\b(what\s+is|what\s+are|who\s+is|define|describe|list|tell\s+me\s+about|how\s+many|explain)\b",
        )
        .expect("kg_patterns regex is valid")
    })
}

fn novelty_patterns() -> &'static Regex {
    static PAT: OnceLock<Regex> = OnceLock::new();
    PAT.get_or_init(|| {
        Regex::new(
            r"(?i)\b(write|create|imagine|compose|invent|suggest|generate|poem|story|metaphor|analogy|dream|fantasy|creative|novel|design\s+a|craft)\b",
        )
        .expect("novelty_patterns regex is valid")
    })
}

fn psyche_patterns() -> &'static Regex {
    static PAT: OnceLock<Regex> = OnceLock::new();
    PAT.get_or_init(|| {
        Regex::new(
            r"(?i)\b(how\s+should\s+i|what\s+do\s+you\s+think|help\s+me\s+(think|decide|approach|plan)|feel\s+about|approach\s+(this|the)\s+(problem|issue|challenge)|design\s+problem|perspective|advice|guidance|reflect)\b",
        )
        .expect("psyche_patterns regex is valid")
    })
}

// ---------------------------------------------------------------------------
// EngineSelector
// ---------------------------------------------------------------------------

/// Analyzes input text and routes it to the best NL engine.
///
/// # Scoring
/// - **KG relevance**: factual-question keywords + trailing `?`
/// - **Novelty score**: creative-directive keywords (write, compose, imagine …)
/// - **Psyche fit**: reflective / advisory keywords + first-person modal phrases
///
/// # Decision logic
/// 1. `novelty ≥ novelty_threshold` → [`Route::Gemma`]
/// 2. `kg ≥ kg_threshold` AND `novelty < novelty_threshold` → [`Route::Eliza`]
/// 3. `psyche ≥ psyche_threshold` OR moderate KG + psyche signal → [`Route::Hybrid`]
/// 4. Otherwise → [`Route::Gemma`] (open-ended default)
pub struct EngineSelector {
    /// KG relevance score that triggers the ELIZA path (default 0.4).
    pub kg_threshold: f32,
    /// Novelty score that triggers the Gemma path (default 0.35).
    pub novelty_threshold: f32,
    /// Psyche fit score that triggers the Hybrid path (default 0.3).
    pub psyche_threshold: f32,
}

impl Default for EngineSelector {
    fn default() -> Self {
        Self {
            kg_threshold: 0.4,
            novelty_threshold: 0.35,
            psyche_threshold: 0.3,
        }
    }
}

impl EngineSelector {
    /// Create a selector with explicit threshold values.
    pub fn new(kg_threshold: f32, novelty_threshold: f32, psyche_threshold: f32) -> Self {
        Self {
            kg_threshold,
            novelty_threshold,
            psyche_threshold,
        }
    }

    /// Score `input` and return a [`SelectionContext`] with the routing decision.
    ///
    /// This is the hot path for routing; all operations are O(input_len).
    pub fn analyze(&self, input: &str) -> SelectionContext {
        let lower = input.to_lowercase();

        // KG relevance: factual keyword hits + trailing question mark bonus
        let kg_hits = kg_patterns().find_iter(input).count();
        let ends_with_q = input.trim_end().ends_with('?');
        let kg_relevance = ((kg_hits as f32) * 0.4 + if ends_with_q { 0.2 } else { 0.0 }).min(1.0);

        // Novelty score: creative directive keyword hits
        let novelty_hits = novelty_patterns().find_iter(input).count();
        let novelty_score = ((novelty_hits as f32) * 0.4).min(1.0);

        // Psyche fit: reflective keyword hits + soft first-person modal signals
        let psyche_hits = psyche_patterns().find_iter(input).count();
        let fp_modal =
            lower.contains("should i") || lower.contains("can you help") || lower.contains("how do i");
        let psyche_fit =
            ((psyche_hits as f32) * 0.5 + if fp_modal { 0.15 } else { 0.0 }).min(1.0);

        let route = self.decide(kg_relevance, novelty_score, psyche_fit);
        SelectionContext {
            kg_relevance,
            novelty_score,
            psyche_fit,
            route,
        }
    }

    fn decide(&self, kg: f32, novelty: f32, psyche: f32) -> Route {
        if novelty >= self.novelty_threshold {
            return Route::Gemma;
        }
        if kg >= self.kg_threshold {
            return Route::Eliza;
        }
        if psyche >= self.psyche_threshold || (kg > 0.15 && psyche > 0.1) {
            return Route::Hybrid;
        }
        Route::Gemma
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sel() -> EngineSelector {
        EngineSelector::default()
    }

    #[test]
    fn factual_question_routes_to_eliza() {
        let ctx = sel().analyze("What is the Ennead?");
        assert!(
            ctx.kg_relevance >= 0.4,
            "expected kg_relevance ≥ 0.4, got {}",
            ctx.kg_relevance
        );
        assert_eq!(ctx.route, Route::Eliza);
    }

    #[test]
    fn creative_request_routes_to_gemma() {
        let ctx = sel().analyze("Write a poem about the Ennead");
        assert!(
            ctx.novelty_score >= 0.35,
            "expected novelty_score ≥ 0.35, got {}",
            ctx.novelty_score
        );
        assert_eq!(ctx.route, Route::Gemma);
    }

    #[test]
    fn reflective_request_routes_to_hybrid() {
        let ctx = sel().analyze("How should I approach this design problem?");
        assert!(
            ctx.psyche_fit >= 0.3,
            "expected psyche_fit ≥ 0.3, got {}",
            ctx.psyche_fit
        );
        assert_eq!(ctx.route, Route::Hybrid);
    }

    #[test]
    fn who_is_routes_to_eliza() {
        let ctx = sel().analyze("Who is Ra in the Egyptian pantheon?");
        assert_eq!(ctx.route, Route::Eliza);
    }

    #[test]
    fn compose_routes_to_gemma() {
        let ctx = sel().analyze("Compose a story about Osiris");
        assert_eq!(ctx.route, Route::Gemma);
    }

    #[test]
    fn open_ended_defaults_to_gemma() {
        let ctx = sel().analyze("Tell me something interesting");
        // low kg, low novelty, low psyche → fallback Gemma
        assert_eq!(ctx.route, Route::Gemma);
    }
}
