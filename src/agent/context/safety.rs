//! The [`SafetyEnforced`] decorator (Phase 44f, ADR 065 D6).
//!
//! Wraps any [`ContextManager`] and rejects an assembly that evicted a
//! `PinnedSafety` span. Every shipped manager is wrapped in this before use, so a
//! curation bug can never *silently* drop a decision, a safety veto, or a
//! dangerous tool's record from the window — it fails closed with a diagnostic.

use std::sync::atomic::{AtomicU64, Ordering};

use super::error::{ContextError, ContextResult};
use super::manager::{ContextManager, TokenCounter};
use super::retention::{ProvenanceRetentionClassifier, Retention};
use super::types::{
    AssembledContext, ContextWindowStats, EpisodeState, StrategyId, TokenBudget, TurnOutcome,
};

/// Decorates a [`ContextManager`], asserting no `PinnedSafety` span was evicted.
///
/// Complements the ADR-060 *execution* gate: that prevents dangerous tool
/// *calls*; this prevents dangerous facts from being dropped out of *context*.
pub struct SafetyEnforced<M: ContextManager> {
    inner: M,
    classifier: ProvenanceRetentionClassifier,
    violations: AtomicU64,
}

impl<M: ContextManager> SafetyEnforced<M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            classifier: ProvenanceRetentionClassifier::new(),
            violations: AtomicU64::new(0),
        }
    }

    /// How many safety-pin violations this guard has caught (for the harness /
    /// telemetry — must be 0 in a healthy backend).
    pub fn violation_count(&self) -> u64 {
        self.violations.load(Ordering::Relaxed)
    }
}

impl<M: ContextManager> ContextManager for SafetyEnforced<M> {
    fn id(&self) -> StrategyId {
        StrategyId::new(format!("safety-enforced({})", self.inner.id().as_str()))
    }

    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext> {
        let assembled = self.inner.assemble(episode, budget, counter)?;
        for span in &assembled.dropped {
            if let Some(anchor) = span.anchor
                && self.classifier.classify(episode.engine, anchor) == Retention::PinnedSafety
            {
                self.violations.fetch_add(1, Ordering::Relaxed);
                let first: String = span.text.chars().take(80).collect();
                tracing::error!(
                    section = span.section.as_str(),
                    "phase44f: context manager tried to evict a safety-pinned span"
                );
                return Err(ContextError::SafetyPinViolation {
                    section: span.section.as_str().to_string(),
                    first,
                });
            }
        }
        Ok(assembled)
    }

    fn commit(&mut self, outcome: &TurnOutcome) {
        self.inner.commit(outcome);
    }

    fn stats(&self, ctx: &AssembledContext, budget: &TokenBudget) -> ContextWindowStats {
        // Surface the decorator's own violation counter so a caller reading
        // stats off the *decorated* manager sees the real number (the inner
        // manager's default stats() would report 0).
        let mut s = self.inner.stats(ctx, budget);
        s.safety_violations = self.violations.load(Ordering::Relaxed) as usize;
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::manager::{HeuristicTokenCounter, PassThroughManager};
    use super::super::types::{AssembledContext, ContextSection, DroppedSpan};
    use crate::engine::{Engine, EngineConfig};
    use crate::provenance::{DerivationKind, ProvenanceRecord};
    use crate::symbol::{SymbolId, SymbolKind};
    use serde_json::json;
    use std::collections::BTreeMap;

    fn test_engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (engine, tmp)
    }

    /// A manager that always drops one span with a configurable anchor — exercises
    /// the guard without needing a real curating manager.
    struct DroppingManager {
        anchor: Option<SymbolId>,
    }
    impl ContextManager for DroppingManager {
        fn id(&self) -> StrategyId {
            StrategyId::new("dropping")
        }
        fn assemble(
            &self,
            episode: &EpisodeState<'_>,
            _budget: &TokenBudget,
            _counter: &dyn TokenCounter,
        ) -> ContextResult<AssembledContext> {
            Ok(AssembledContext {
                messages: vec![json!({"role": "user", "content": episode.user_prompt})],
                tools: Vec::new(),
                section_tokens: BTreeMap::new(),
                dropped: vec![DroppedSpan::new(
                    ContextSection::KnowledgeGraph,
                    "an evicted fact",
                    self.anchor,
                )],
            })
        }
    }

    #[test]
    fn passthrough_wrapped_is_a_noop() {
        let (engine, _tmp) = test_engine();
        let mgr = SafetyEnforced::new(PassThroughManager);
        let episode = EpisodeState::new("a", "hi", &engine);
        let out = mgr
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("pass-through never violates");
        assert_eq!(out.messages, vec![json!({"role": "user", "content": "hi"})]);
        assert_eq!(mgr.violation_count(), 0);
    }

    #[test]
    fn rejects_dropping_a_pinned_span() {
        let (engine, _tmp) = test_engine();
        let pinned = engine.create_symbol(SymbolKind::Entity, "pinned").unwrap().id;
        let goal = engine.create_symbol(SymbolKind::Entity, "g").unwrap().id;
        let mut rec = ProvenanceRecord::new(pinned, DerivationKind::AgentDecision { goal, cycle: 1 });
        engine.store_provenance(&mut rec).unwrap();

        let mgr = SafetyEnforced::new(DroppingManager {
            anchor: Some(pinned),
        });
        let episode = EpisodeState::new("a", "hi", &engine);
        let err = mgr
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect_err("dropping a pinned span must fail");
        assert!(matches!(err, ContextError::SafetyPinViolation { .. }));
        assert_eq!(mgr.violation_count(), 1);
    }

    #[test]
    fn allows_dropping_a_volatile_span() {
        let (engine, _tmp) = test_engine();
        // No provenance ⇒ Volatile ⇒ free to drop.
        let plain = engine.create_symbol(SymbolKind::Entity, "plain").unwrap().id;
        let mgr = SafetyEnforced::new(DroppingManager {
            anchor: Some(plain),
        });
        let episode = EpisodeState::new("a", "hi", &engine);
        assert!(
            mgr.assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
                .is_ok()
        );
        assert_eq!(mgr.violation_count(), 0);
    }
}
