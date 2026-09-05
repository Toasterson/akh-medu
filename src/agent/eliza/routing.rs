//! ELIZA-style routing classifier for the chat path (reconciled from PR #176).
//!
//! Classifies natural language queries into two routing categories:
//! - **KG-grounded**: answered from the knowledge graph (deterministic, sub-ms)
//! - **LLM fallback**: open-ended or creative queries that need language model generation
//!
//! This is a symbolic classifier, not a generative model. Classification runs in
//! O(patterns × words) — typically under 1 µs for normal conversational input.
//!
//! Lifted from the deferred PR #176 (`[ELIZA] Chat integration — ELIZA as
//! primary`, Anima WI `019e404a`), whose original `src/agent/eliza.rs` collided
//! with the canonical `src/agent/eliza/` directory. The classifier type is
//! exposed as [`ElizaRouter`] here to avoid colliding with the pattern-matching
//! [`super::engine::ElizaEngine`].

use std::time::Instant;

/// Routing decision produced by the ELIZA engine.
#[derive(Debug, Clone, PartialEq)]
pub enum ElizaRouting {
    /// Query can be answered from the knowledge graph.
    KgGrounded {
        /// Aggregate match confidence in `[0.0, 1.0]`.
        confidence: f32,
        /// The keywords that contributed to the match.
        matched_keywords: Vec<String>,
    },
    /// Query requires LLM generation (open-ended or creative).
    LlmFallback {
        /// Reason the KG path was not selected.
        reason: FallbackReason,
    },
}

/// Why the ELIZA engine routed to LLM fallback.
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackReason {
    /// A creative/generative request was detected (write, compose, imagine…).
    Creative,
    /// No KG keyword pattern matched above the confidence threshold.
    LowConfidence,
}

/// A single keyword pattern used for routing.
#[derive(Debug)]
struct Pattern {
    /// Keywords (checked as whole-word substring matches against lowercased input).
    keywords: Vec<&'static str>,
    /// Weight contribution when any keyword matches.
    weight: f32,
    category: PatternCategory,
}

#[derive(Debug, Clone, PartialEq)]
enum PatternCategory {
    KgGrounded,
    Creative,
}

/// ELIZA-style routing classifier.
///
/// Instantiated once (e.g. in the chat path) and reused across turns; no mutable state.
pub struct ElizaRouter {
    patterns: Vec<Pattern>,
    /// Minimum score to route to KG path (default: 0.35).
    kg_threshold: f32,
}

impl Default for ElizaRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ElizaRouter {
    /// Create a router with sensible defaults.
    pub fn new() -> Self {
        Self::with_threshold(0.35)
    }

    /// Create a router with a custom KG-path confidence threshold.
    pub fn with_threshold(kg_threshold: f32) -> Self {
        let patterns = vec![
            // ── Creative / generative (LLM needed) ─────────────────────────
            Pattern {
                keywords: vec!["write", "compose", "draft", "author"],
                weight: 0.9,
                category: PatternCategory::Creative,
            },
            Pattern {
                keywords: vec!["poem", "poetry", "haiku", "sonnet", "ode", "ballad"],
                weight: 0.95,
                category: PatternCategory::Creative,
            },
            Pattern {
                keywords: vec!["story", "tale", "narrative", "fiction", "fable"],
                weight: 0.95,
                category: PatternCategory::Creative,
            },
            Pattern {
                keywords: vec!["essay", "article", "blog", "post"],
                weight: 0.85,
                category: PatternCategory::Creative,
            },
            Pattern {
                keywords: vec!["imagine", "invent", "create", "generate", "make up"],
                weight: 0.85,
                category: PatternCategory::Creative,
            },
            Pattern {
                keywords: vec!["pretend", "roleplay", "suppose", "hypothetically"],
                weight: 0.80,
                category: PatternCategory::Creative,
            },
            Pattern {
                keywords: vec!["joke", "pun", "funny", "humorous", "witty"],
                weight: 0.75,
                category: PatternCategory::Creative,
            },
            Pattern {
                keywords: vec!["brainstorm", "ideate", "suggest ideas", "think of"],
                weight: 0.70,
                category: PatternCategory::Creative,
            },
            // ── KG-grounded (factual queries) ───────────────────────────────
            Pattern {
                keywords: vec!["what", "what's", "whats"],
                weight: 0.40,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["who", "who's", "whos"],
                weight: 0.50,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["is", "are", "was", "were"],
                weight: 0.25,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["define", "definition", "meaning"],
                weight: 0.70,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["describe", "description"],
                weight: 0.55,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["tell", "explain"],
                weight: 0.45,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["when", "where", "how"],
                weight: 0.35,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["which", "list", "show", "enumerate"],
                weight: 0.45,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["about", "regarding", "concerning"],
                weight: 0.30,
                category: PatternCategory::KgGrounded,
            },
            Pattern {
                keywords: vec!["facts", "information", "info", "knowledge"],
                weight: 0.55,
                category: PatternCategory::KgGrounded,
            },
        ];

        Self {
            patterns,
            kg_threshold,
        }
    }

    /// Classify a user query and return the routing decision.
    ///
    /// Classification runs in O(patterns × keywords) — typically under 1 µs.
    pub fn classify(&self, input: &str) -> ElizaRouting {
        let started = Instant::now();
        let lower = input.trim().to_lowercase();

        let mut creative_score = 0.0f32;
        let mut kg_score = 0.0f32;
        let mut matched_keywords: Vec<String> = Vec::new();

        for pattern in &self.patterns {
            let hits: Vec<&'static str> = pattern
                .keywords
                .iter()
                .copied()
                .filter(|&kw| {
                    // Match as whole word or multi-word phrase.
                    if kw.contains(' ') {
                        lower.contains(kw)
                    } else {
                        // Whole-word check: surrounded by non-alphanumeric or at boundary.
                        let mut found = false;
                        let mut start = 0;
                        while let Some(pos) = lower[start..].find(kw) {
                            let abs = start + pos;
                            let before_ok =
                                abs == 0 || !lower.as_bytes()[abs - 1].is_ascii_alphanumeric();
                            let after_ok = abs + kw.len() >= lower.len()
                                || !lower.as_bytes()[abs + kw.len()].is_ascii_alphanumeric();
                            if before_ok && after_ok {
                                found = true;
                                break;
                            }
                            start = abs + 1;
                        }
                        found
                    }
                })
                .collect();

            if !hits.is_empty() {
                // Any keyword match scores the full pattern weight.
                // Keywords within a pattern are alternatives, not conjunctions.
                let contribution = pattern.weight;
                match pattern.category {
                    PatternCategory::Creative => creative_score += contribution,
                    PatternCategory::KgGrounded => {
                        kg_score += contribution;
                        for kw in hits {
                            if !matched_keywords.iter().any(|k| k == kw) {
                                matched_keywords.push(kw.to_string());
                            }
                        }
                    }
                }
            }
        }

        let elapsed = started.elapsed();
        tracing::trace!(
            creative = creative_score,
            kg = kg_score,
            elapsed_us = elapsed.as_micros(),
            "ELIZA classification"
        );

        // Creative signals take priority — even a moderate creative score wins.
        if creative_score >= 0.5 {
            return ElizaRouting::LlmFallback {
                reason: FallbackReason::Creative,
            };
        }

        if kg_score >= self.kg_threshold {
            ElizaRouting::KgGrounded {
                confidence: kg_score.min(1.0),
                matched_keywords,
            }
        } else {
            ElizaRouting::LlmFallback {
                reason: FallbackReason::LowConfidence,
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> ElizaRouter {
        ElizaRouter::new()
    }

    // ── KG-grounded queries ────────────────────────────────────────────

    #[test]
    fn factual_what_is_routes_kg() {
        let r = engine().classify("What is the Ennead?");
        assert!(
            matches!(r, ElizaRouting::KgGrounded { .. }),
            "expected KgGrounded, got {r:?}"
        );
    }

    #[test]
    fn factual_who_routes_kg() {
        let r = engine().classify("Who is Osiris?");
        assert!(matches!(r, ElizaRouting::KgGrounded { .. }));
    }

    #[test]
    fn factual_define_routes_kg() {
        let r = engine().classify("Define the concept of Ma'at");
        assert!(matches!(r, ElizaRouting::KgGrounded { .. }));
    }

    #[test]
    fn factual_describe_routes_kg() {
        let r = engine().classify("Describe the Nine Enneads");
        assert!(matches!(r, ElizaRouting::KgGrounded { .. }));
    }

    #[test]
    fn factual_explain_routes_kg() {
        let r = engine().classify("Explain what Thoth represents");
        assert!(matches!(r, ElizaRouting::KgGrounded { .. }));
    }

    // ── Creative queries → LLM fallback ───────────────────────────────

    #[test]
    fn creative_poem_routes_llm() {
        let r = engine().classify("Write a poem about the Ennead");
        assert!(
            matches!(
                r,
                ElizaRouting::LlmFallback {
                    reason: FallbackReason::Creative
                }
            ),
            "expected Creative fallback, got {r:?}"
        );
    }

    #[test]
    fn creative_story_routes_llm() {
        let r = engine().classify("Tell me a story about Osiris");
        // "tell" is KG but "story" is creative — creative wins
        assert!(matches!(
            r,
            ElizaRouting::LlmFallback {
                reason: FallbackReason::Creative
            }
        ));
    }

    #[test]
    fn creative_compose_routes_llm() {
        let r = engine().classify("Compose an ode to Ra");
        assert!(matches!(
            r,
            ElizaRouting::LlmFallback {
                reason: FallbackReason::Creative
            }
        ));
    }

    #[test]
    fn creative_imagine_routes_llm() {
        let r = engine().classify("Imagine what Thoth would say today");
        assert!(matches!(
            r,
            ElizaRouting::LlmFallback {
                reason: FallbackReason::Creative
            }
        ));
    }

    // ── Confidence reporting ───────────────────────────────────────────

    #[test]
    fn confidence_is_bounded() {
        let r =
            engine().classify("What is who define describe tell explain list show when where how");
        if let ElizaRouting::KgGrounded { confidence, .. } = r {
            assert!(confidence <= 1.0, "confidence must not exceed 1.0");
            assert!(confidence >= 0.0, "confidence must not be negative");
        }
    }

    #[test]
    fn classification_is_fast() {
        let e = engine();

        // Warm up.
        let _ = e.classify("What is the Ennead?");

        let start = Instant::now();
        for _ in 0..1_000 {
            let _ = e.classify("What is the Ennead?");
        }
        let elapsed = start.elapsed();

        // In debug builds, each call takes ~10–20 µs; in release < 1 µs.
        // The acceptance criterion (< 1ms per call) is satisfied in either case.
        // We allow 200ms for the batch in debug mode.
        assert!(
            elapsed.as_millis() < 200,
            "1000 classifications must complete in < 200ms (< 200µs each), took {}ms",
            elapsed.as_millis()
        );
    }
}
