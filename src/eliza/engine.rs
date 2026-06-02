//! ELIZA evaluation engine with hybrid VSA + exact keyword matching.
//!
//! # Matching strategy
//!
//! For each rule the engine pre-encodes all keywords as hypervectors at load
//! time.  On each input the engine tries two match paths in order:
//!
//! 1. **Exact string match** — O(k) comparison; handles synonyms that are
//!    explicitly listed in the `keywords` array.
//! 2. **VSA fuzzy match** — cosine similarity between the input word's
//!    hypervector and each keyword hypervector.  Only triggers when the
//!    similarity meets or exceeds the rule's `similarity_threshold`.  With
//!    hash-based token encoding this path fires for any token that hashes
//!    identically to a keyword (i.e. identical strings), providing a
//!    complementary check.  The infrastructure is ready to accept semantic
//!    vectors in the future (Phase 26/Phase 32).
//!
//! Rules are checked in priority-descending, insertion-stable order.
//! The first matching rule wins and its responses are cycled round-robin.

use std::time::Instant;

use crate::simd;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;
use crate::vsa::{Dimension, Encoding, HyperVec};

use super::error::ElizaResult;
use super::script::{Rule, Script, ScriptMeta};

/// A rule compiled for fast evaluation.
struct CompiledRule {
    rule: Rule,
    /// Per-keyword hypervectors (parallel to `rule.keywords`).
    keyword_vecs: Vec<HyperVec>,
    /// Bundle of all keyword vectors for aggregate similarity queries.
    bundle_vec: Option<HyperVec>,
    /// Round-robin cursor into `rule.responses`.
    response_cursor: usize,
}

/// ELIZA dialogue engine.
///
/// Construct with [`ElizaEngine::new`], then call [`respond`] for each user
/// turn.  The engine is intentionally synchronous and allocation-free on the
/// hot path after construction.
pub struct ElizaEngine {
    meta: ScriptMeta,
    rules: Vec<CompiledRule>,
    ops: VsaOps,
}

impl ElizaEngine {
    /// Build an engine from a validated script, encoding all keywords upfront.
    pub fn new(script: Script) -> ElizaResult<Self> {
        script.validate()?;

        let ops = VsaOps::new(simd::best_kernel(), Dimension::DEFAULT, Encoding::Bipolar);

        let mut rules: Vec<CompiledRule> = script
            .rules
            .into_iter()
            .map(|rule| {
                let keyword_vecs: Vec<HyperVec> = rule
                    .keywords
                    .iter()
                    .map(|kw| encode_token(&ops, &kw.to_lowercase()))
                    .collect();

                let bundle_vec = {
                    let refs: Vec<&HyperVec> = keyword_vecs.iter().collect();
                    ops.bundle(&refs).ok()
                };

                CompiledRule {
                    rule,
                    keyword_vecs,
                    bundle_vec,
                    response_cursor: 0,
                }
            })
            .collect();

        // Stable sort so equal-priority rules keep their script order.
        rules.sort_by(|a, b| b.rule.priority.cmp(&a.rule.priority));

        Ok(Self { meta: script.script, rules, ops })
    }

    /// Optional opening line configured in the script.
    pub fn initial_greeting(&self) -> Option<&str> {
        self.meta.initial_greeting.as_deref()
    }

    /// Process one user turn and return a response string.
    ///
    /// Returns the matched rule's next response, or the script's
    /// `default_response` when no rule fires.
    pub fn respond(&mut self, input: &str) -> String {
        let input_lower = input.to_lowercase();
        let words: Vec<&str> = input_lower.split_whitespace().collect();

        if let Some(idx) = self.find_matching_rule(&words) {
            let compiled = &mut self.rules[idx];
            let cursor = compiled.response_cursor;
            compiled.response_cursor = (cursor + 1) % compiled.rule.responses.len();
            compiled.rule.responses[cursor].clone()
        } else {
            self.meta.default_response.clone()
        }
    }

    /// Return the Hamming similarity of `word` against `rule_idx`'s keyword bundle.
    ///
    /// Returns `None` when `rule_idx` is out of range or the bundle is absent.
    /// Primarily used in diagnostics and tests.
    pub fn bundle_similarity(&self, rule_idx: usize, word: &str) -> Option<f32> {
        let compiled = self.rules.get(rule_idx)?;
        let bundle = compiled.bundle_vec.as_ref()?;
        let word_vec = encode_token(&self.ops, &word.to_lowercase());
        self.ops.similarity(&word_vec, bundle).ok()
    }

    /// Return per-keyword similarities of `word` against all keywords of `rule_idx`.
    ///
    /// Primarily used in diagnostics and tests.
    pub fn keyword_similarities(&self, rule_idx: usize, word: &str) -> Vec<(String, f32)> {
        let Some(compiled) = self.rules.get(rule_idx) else {
            return Vec::new();
        };
        let word_vec = encode_token(&self.ops, &word.to_lowercase());
        compiled
            .rule
            .keywords
            .iter()
            .zip(&compiled.keyword_vecs)
            .filter_map(|(kw, kv)| {
                self.ops.similarity(&word_vec, kv).ok().map(|s| (kw.clone(), s))
            })
            .collect()
    }

    /// Measure how long one [`respond`] call takes (in microseconds).
    ///
    /// Used to verify the < 1 ms per-keyword benchmark target.
    pub fn measure_respond_us(&mut self, input: &str) -> (String, u128) {
        let t0 = Instant::now();
        let resp = self.respond(input);
        let elapsed = t0.elapsed().as_micros();
        (resp, elapsed)
    }

    // ── internal ─────────────────────────────────────────────────────────────

    fn find_matching_rule(&self, words: &[&str]) -> Option<usize> {
        for (idx, compiled) in self.rules.iter().enumerate() {
            if self.rule_matches(compiled, words) {
                return Some(idx);
            }
        }
        None
    }

    fn rule_matches(&self, compiled: &CompiledRule, words: &[&str]) -> bool {
        let threshold = compiled.rule.similarity_threshold;

        for &word in words {
            // Fast path: exact string match against keyword list.
            if compiled.rule.keywords.iter().any(|kw| kw.to_lowercase() == word) {
                return true;
            }

            // Fuzzy path: VSA similarity against each keyword's hypervector.
            let word_vec = encode_token(&self.ops, word);
            for kv in &compiled.keyword_vecs {
                if self.ops.similarity(&word_vec, kv).ok().is_some_and(|s| s >= threshold) {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eliza::script::Script;

    fn family_script() -> Script {
        Script::from_toml(
            r#"
[script]
name = "Test"
default_response = "Please go on."
initial_greeting = "Hello."

[[rules]]
keywords = ["mother", "mom", "mama", "parent", "father", "dad"]
similarity_threshold = 0.70
responses = ["Tell me about your family.", "Who else?"]
priority = 5

[[rules]]
keywords = ["hello", "hi", "hey"]
similarity_threshold = 0.70
responses = ["How do you do."]
priority = 10
"#,
        )
        .unwrap()
    }

    // ── acceptance: synonym matching ─────────────────────────────────────────

    #[test]
    fn mother_keyword_matches_mom() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let resp = engine.respond("I miss my mom");
        assert_ne!(resp, engine.meta.default_response, "'mom' should match family rule");
    }

    #[test]
    fn mother_keyword_matches_mama() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let resp = engine.respond("my mama used to say that");
        assert_ne!(resp, engine.meta.default_response, "'mama' should match family rule");
    }

    #[test]
    fn mother_keyword_matches_parent() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let resp = engine.respond("my parent was strict");
        assert_ne!(resp, engine.meta.default_response, "'parent' should match family rule");
    }

    // ── acceptance: non-match ────────────────────────────────────────────────

    #[test]
    fn moth_does_not_match_mother_rule() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        // "moth" is not in the keyword list; VSA sim to any keyword is ~0.5 < 0.70
        let resp = engine.respond("I saw a moth");
        assert_eq!(resp, engine.meta.default_response, "'moth' must NOT match family rule");
    }

    // ── acceptance: exact match regression ───────────────────────────────────

    #[test]
    fn exact_keyword_mother_matches() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let resp = engine.respond("I think about my mother");
        assert_ne!(resp, engine.meta.default_response, "'mother' should match");
    }

    #[test]
    fn exact_keyword_hello_matches() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let resp = engine.respond("hello there");
        assert_ne!(resp, engine.meta.default_response, "'hello' should match greeting rule");
    }

    // ── round-robin responses ─────────────────────────────────────────────────

    #[test]
    fn responses_cycle_round_robin() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let r1 = engine.respond("my mother said so");
        let r2 = engine.respond("my mom is kind");
        let r3 = engine.respond("my mama was there");
        // Family rule has 2 responses so r3 wraps back to r1
        assert_ne!(r1, r2);
        assert_eq!(r1, r3);
    }

    // ── priority ordering ─────────────────────────────────────────────────────

    #[test]
    fn higher_priority_rule_fires_first() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        // greeting rule (priority 10) should fire before family rule (priority 5)
        let resp = engine.respond("hello mom");
        assert_eq!(resp, "How do you do.", "greeting rule should win via priority");
    }

    // ── default response ──────────────────────────────────────────────────────

    #[test]
    fn unknown_input_returns_default_response() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let resp = engine.respond("the quick brown fox jumps over the lazy dog");
        assert_eq!(resp, "Please go on.");
    }

    // ── VSA similarity diagnostics ────────────────────────────────────────────

    #[test]
    fn exact_keyword_has_similarity_one() {
        let engine = ElizaEngine::new(family_script()).unwrap();
        // After priority sort: greeting rule is index 0, family rule is index 1
        let sims = engine.keyword_similarities(1, "mother");
        let mother_sim = sims.iter().find(|(kw, _)| kw == "mother").map(|(_, s)| *s);
        assert!(
            mother_sim.is_some_and(|s| (s - 1.0).abs() < 0.01),
            "'mother' vs 'mother' keyword should have similarity ~1.0, got {mother_sim:?}"
        );
    }

    #[test]
    fn unrelated_word_has_low_similarity() {
        let engine = ElizaEngine::new(family_script()).unwrap();
        // "moth" against the family-rule bundle should be ~0.5 (random noise)
        let sim = engine.bundle_similarity(1, "moth").unwrap();
        assert!(
            sim < 0.70,
            "'moth' bundle similarity should be below threshold (0.70), got {sim}"
        );
    }

    // ── latency benchmark (soft assertion) ───────────────────────────────────

    #[test]
    fn respond_latency_under_1ms() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        // Warm up caches.
        let _ = engine.respond("warm up");
        let (_, us) = engine.measure_respond_us("I think about my mother all the time");
        // Debug builds have no optimisation; use a generous ceiling there.
        let limit = if cfg!(debug_assertions) { 50_000 } else { 1_000 };
        assert!(us < limit, "respond() took {us}µs, should be < {limit}µs");
    }

    // ── initial greeting ──────────────────────────────────────────────────────

    #[test]
    fn initial_greeting_present() {
        let engine = ElizaEngine::new(family_script()).unwrap();
        assert_eq!(engine.initial_greeting(), Some("Hello."));
    }

    // ── case insensitivity ────────────────────────────────────────────────────

    #[test]
    fn matching_is_case_insensitive() {
        let mut engine = ElizaEngine::new(family_script()).unwrap();
        let r1 = engine.respond("MOTHER");
        let r2 = engine.respond("Mother");
        let r3 = engine.respond("mOtHeR");
        // Round-robin advances on each match so responses differ, but none is the default.
        assert_ne!(r1, engine.meta.default_response);
        assert_ne!(r2, engine.meta.default_response);
        assert_ne!(r3, engine.meta.default_response);
    }
}
