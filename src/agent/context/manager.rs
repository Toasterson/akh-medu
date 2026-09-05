//! The [`ContextManager`] seam, the [`TokenCounter`] abstraction, and the
//! byte-identical [`PassThroughManager`] (Phase 44a, ADR 065).

use std::collections::BTreeMap;

use serde_json::json;

use super::error::ContextResult;
use super::types::{
    AssembledContext, ContextSection, ContextWindowStats, DropReason, DroppedSpan, EpisodeState,
    SectionStats, StrategyId, TokenBudget, TurnOutcome,
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

    /// Normalized context-window statistics for `ctx` under `budget` (Phase 44m,
    /// ADR 065 D12). The default impl derives everything from
    /// [`AssembledContext`] alone — `PassThroughManager` and `PassiveManager`
    /// need no override. `CompactingManager` and `VirtualContextManager`
    /// override to fill their strategy extras (`compacted` / `vsa_hit_rate` /
    /// …) and to count `provenance_pinned_count` against the live engine, which
    /// the default cannot do (it has no `Engine` borrow).
    ///
    /// `safety_violations` is always 0 in the default impl; the
    /// [`SafetyEnforced`](super::safety::SafetyEnforced) decorator overrides
    /// `stats` to surface its own counter, so a caller reading stats off the
    /// *decorated* manager sees the real violation count.
    fn stats(&self, ctx: &AssembledContext, budget: &TokenBudget) -> ContextWindowStats {
        default_stats(self.id(), ctx, budget, 0)
    }
}

/// Build a [`ContextWindowStats`] purely from an [`AssembledContext`] — the
/// shared body of every default `stats()` impl and the strategy overrides
/// (which pass their own `safety_violations` count and let the extras default
/// to `None` before they patch them in).
///
/// Exposed to children so a manager's override can call it then mutate the
/// extras, without re-deriving the per-section accounting.
pub(crate) fn default_stats(
    strategy: StrategyId,
    ctx: &AssembledContext,
    budget: &TokenBudget,
    safety_violations: usize,
) -> ContextWindowStats {
    // Per-section: tokens_kept is what survived (section_tokens); tokens_in
    // is kept + the tokens of the dropped spans in that section; tokens_dropped
    // is the difference. Drop-reason histogram: pick the dominant reason in
    // declaration order (Budget > Retention > Age > Vsa) so two managers that
    // drop the same section for different reasons still diff cleanly.
    let reason_rank = |r: DropReason| match r {
        DropReason::Budget => 0,
        DropReason::Retention => 1,
        DropReason::Age => 2,
        DropReason::Vsa => 3,
    };

    // Accumulate dropped tokens + a reason histogram per section.
    let mut dropped_tokens: BTreeMap<ContextSection, usize> = BTreeMap::new();
    let mut reason_counts: BTreeMap<ContextSection, std::collections::HashMap<DropReason, usize>> =
        BTreeMap::new();
    for d in &ctx.dropped {
        *dropped_tokens.entry(d.section).or_insert(0) += estimate_dropped_tokens(d);
        let map = reason_counts.entry(d.section).or_default();
        *map.entry(d.reason).or_insert(0) += 1;
    }

    let mut sections: Vec<SectionStats> = Vec::new();
    // Iterate every section that either survived or was dropped, in
    // ContextSection's declaration order (BTreeMap order, which matches).
    let mut all_sections: std::collections::BTreeSet<ContextSection> =
        std::collections::BTreeSet::new();
    for &s in ctx.section_tokens.keys() {
        all_sections.insert(s);
    }
    for &s in dropped_tokens.keys() {
        all_sections.insert(s);
    }
    for s in all_sections {
        let kept = ctx.section_tokens.get(&s).copied().unwrap_or(0);
        let dropped = dropped_tokens.get(&s).copied().unwrap_or(0);
        let inp = kept + dropped;
        let drop_reason = reason_counts
            .get(&s)
            .and_then(|map| map.iter().max_by_key(|(_, c)| **c).map(|(r, _)| *r))
            .map(|dominant| {
                // Tie-break by declaration order: among reasons with the same
                // count, the lowest `reason_rank` wins.
                let max_count = reason_counts
                    .get(&s)
                    .and_then(|m| m.get(&dominant).copied())
                    .unwrap_or(0);
                reason_counts
                    .get(&s)
                    .into_iter()
                    .flat_map(|m| m.iter())
                    .filter(|(_, c)| **c == max_count)
                    .map(|(r, _)| *r)
                    .min_by_key(|r| reason_rank(*r))
                    .unwrap_or(dominant)
            });
        sections.push(SectionStats {
            section: s,
            tokens_in: inp,
            tokens_kept: kept,
            tokens_dropped: dropped,
            drop_reason,
        });
    }

    let kept_total: usize = sections.iter().map(|s| s.tokens_kept).sum();
    let available = budget.available();
    let pressure = if available == 0 {
        // Window reserved larger than the window itself: the assembler had no
        // room. Report 1.0 (full) rather than NaN.
        if kept_total == 0 { 0.0 } else { 1.0 }
    } else {
        let p = kept_total as f32 / available as f32;
        // Clamp >1.0 (an over-spent manager is itself a signal worth showing
        // as a full bar, not a misleading 1.2).
        if p > 1.0 { 1.0 } else { p }
    };

    ContextWindowStats {
        strategy,
        window_total: budget.window,
        reserved: budget.reserved_output,
        available,
        budget_pressure: pressure,
        sections,
        provenance_pinned_count: 0,
        safety_violations,
        kv_cache_tokens: None,
        compacted: None,
        compaction_ratio: None,
        virtual_referenced: None,
        physical_materialized: None,
        vsa_hit_rate: None,
    }
}

/// Best-effort per-span token estimate for the dropped-token accounting in
/// [`default_stats`]. `AssembledContext` records the dropped *text* (not its
/// pre-allocation token count), so we approximate with the same `ceil(len/4)`
/// the heuristic counter uses. This is for the breakdown bar, not for billing;
/// the kept-token side is exact (it comes from `section_tokens`).
fn estimate_dropped_tokens(d: &DroppedSpan) -> usize {
    // The heuristic counter is `ceil(len/4)`; this matches every shipped
    // manager's default counter. A real-tokenizer backend would diverge here,
    // but the kept side (the exact side) is what the panel is anchored on.
    d.text.len().div_ceil(4)
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
        section_tokens.insert(
            ContextSection::UserPrompt,
            counter.count(episode.user_prompt),
        );
        Ok(AssembledContext {
            messages,
            // Byte-identical to the legacy path: every tool passes through unchanged.
            tools: episode.tool_catalog.to_vec(),
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
        let episode = EpisodeState::new("akh", "what is the capital of France?", &engine);
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

    // ── Phase 44m: normalized stats tests (ADR 065 D12) ────────────────────

    #[test]
    fn passthrough_stats_equal_input_token_count() {
        // Acceptance #1: pass-through stats equal the input token count.
        let (engine, _tmp) = test_engine();
        let prompt = "what is the capital of France?";
        let episode = EpisodeState::new("akh", prompt, &engine);
        let budget = TokenBudget::default();
        let out = PassThroughManager
            .assemble(&episode, &budget, &HeuristicTokenCounter)
            .expect("assemble");
        let stats = PassThroughManager.stats(&out, &budget);

        let prompt_tokens = HeuristicTokenCounter.count(prompt);
        let kept: usize = stats.sections.iter().map(|s| s.tokens_kept).sum();
        assert_eq!(kept, prompt_tokens, "kept == input tokens");
        assert_eq!(stats.available, budget.available());
        // No drops ⇒ budget_pressure is kept/available, not 1.0.
        assert!(
            (stats.budget_pressure - (prompt_tokens as f32 / budget.available() as f32)).abs()
                < 0.001
        );
        assert_eq!(stats.safety_violations, 0);
        // PassThroughManager is the default impl (no extras).
        assert!(stats.compacted.is_none());
        assert!(stats.vsa_hit_rate.is_none());
    }

    #[test]
    fn stats_budget_pressure_is_zero_when_window_is_empty() {
        // A pass-through with an empty prompt: kept=0, pressure=0.
        let (engine, _tmp) = test_engine();
        let episode = EpisodeState::new("akh", "", &engine);
        let budget = TokenBudget::default();
        let out = PassThroughManager
            .assemble(&episode, &budget, &HeuristicTokenCounter)
            .expect("assemble");
        let stats = PassThroughManager.stats(&out, &budget);
        assert_eq!(stats.budget_pressure, 0.0);
    }

    #[test]
    fn dropped_span_records_reason() {
        // A dropped span carries a reason (default Budget) and round-trips
        // through the with_reason constructor.
        let d = DroppedSpan::new(ContextSection::History, "old turn", None);
        assert_eq!(d.reason, DropReason::Budget);
        let d2 =
            DroppedSpan::with_reason(ContextSection::History, "older turn", None, DropReason::Age);
        assert_eq!(d2.reason, DropReason::Age);
        assert_eq!(DropReason::Age.as_str(), "age");
    }
}
