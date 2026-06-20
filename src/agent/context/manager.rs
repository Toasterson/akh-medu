//! The [`ContextManager`] seam, the [`TokenCounter`] abstraction, and the
//! byte-identical [`PassThroughManager`] (Phase 44a, ADR 065).

use std::collections::BTreeMap;

use serde_json::json;

use super::error::ContextResult;
use super::types::{
    AssembledContext, ContextSection, EpisodeState, StrategyId, TokenBudget, TurnOutcome,
};

/// Counts tokens for budgeting.
///
/// The agent path has no single tokenizer: Ollama reports token counts only
/// *after* a call (via the model manager's `UsageLedger`), and the Candle
/// tokenizer is feature-gated and model-specific. This abstracts the source so
/// the allocator (44b) stays backend-agnostic.
pub trait TokenCounter: Send + Sync {
    /// Estimate the token count of `text`.
    fn count(&self, text: &str) -> usize;
    /// Short label for telemetry (which counter produced the numbers).
    fn label(&self) -> &str;
}

/// `ceil(len/4)` — the universal default, matching
/// [`crate::compartment::prompt::estimate_tokens`]. Cheap and always available;
/// reconcile against the backend's post-hoc counts for calibration.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicTokenCounter;

impl TokenCounter for HeuristicTokenCounter {
    fn count(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
    fn label(&self) -> &str {
        "heuristic(len/4)"
    }
}

/// Exact token counts from a Candle model's own tokenizer (when the Candle
/// backing is in use). Falls back to the heuristic if encoding fails.
///
/// Wiring this into the loop requires knowing the live backend is Candle; 44a
/// ships it + tests the heuristic default. The richer sub-phases select it when
/// the akh's backing is Candle.
#[cfg(feature = "candle-backend")]
pub struct CandleTokenCounter {
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "candle-backend")]
impl CandleTokenCounter {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }
}

#[cfg(feature = "candle-backend")]
impl TokenCounter for CandleTokenCounter {
    fn count(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or_else(|_| text.len().div_ceil(4))
    }
    fn label(&self) -> &str {
        "candle-tokenizer"
    }
}

/// Assembles the working context for one agentic turn (ADR 065 D9).
///
/// `run_agentic_turn` calls [`assemble`](ContextManager::assemble) to build the
/// messages it sends, then [`commit`](ContextManager::commit) with the outcome so
/// stateful strategies can learn. Shipped managers are wrapped in
/// [`SafetyEnforced`](super::safety::SafetyEnforced) so eviction can never drop a
/// safety-pinned span.
pub trait ContextManager: Send + Sync {
    /// Identifies this strategy (telemetry + the A/B harness).
    fn id(&self) -> StrategyId;

    /// Build the conversation messages for this turn within `budget`.
    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext>;

    /// Feed back the turn outcome. Default: stateless no-op (the pass-through and
    /// the stateless curating managers need nothing here).
    fn commit(&mut self, _outcome: &TurnOutcome) {}
}

/// The default manager: reproduces the pre-Phase-44 behaviour exactly — a single
/// user turn, no history, no KG block. Landing the seam with this changes
/// nothing (ADR 065 — "pass-through first").
#[derive(Debug, Default, Clone, Copy)]
pub struct PassThroughManager;

impl ContextManager for PassThroughManager {
    fn id(&self) -> StrategyId {
        StrategyId::new("pass-through")
    }

    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        _budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext> {
        // Byte-identical to the pre-Phase-44 `run_agentic_turn` line:
        //   vec![json!({"role": "user", "content": user_prompt})]
        let messages = vec![json!({"role": "user", "content": episode.user_prompt})];
        let mut section_tokens = BTreeMap::new();
        section_tokens.insert(ContextSection::UserPrompt, counter.count(episode.user_prompt));
        Ok(AssembledContext {
            messages,
            section_tokens,
            dropped: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use serde_json::json;

    fn test_engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (engine, tmp)
    }

    #[test]
    fn heuristic_counts_len_over_four() {
        let c = HeuristicTokenCounter;
        assert_eq!(c.count(""), 0);
        assert_eq!(c.count("abcd"), 1);
        assert_eq!(c.count("abcde"), 2); // ceil(5/4)
        assert_eq!(c.label(), "heuristic(len/4)");
    }

    #[test]
    fn passthrough_is_byte_identical_to_legacy_prompt() {
        let (engine, _tmp) = test_engine();
        let episode = EpisodeState {
            agent_id: "akh",
            user_prompt: "what is the capital of France?",
            engine: &engine,
            conversation: None,
        };
        let out = PassThroughManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");

        // This must equal the literal the loop built before Phase 44.
        assert_eq!(
            out.messages,
            vec![json!({"role": "user", "content": "what is the capital of France?"})]
        );
        assert!(out.dropped.is_empty());
        assert_eq!(
            out.section_tokens.get(&ContextSection::UserPrompt).copied(),
            Some("what is the capital of France?".len().div_ceil(4))
        );
    }
}
