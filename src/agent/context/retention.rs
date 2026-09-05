//! Provenance-derived retention classes (Phase 44f, ADR 065 D6).
//!
//! Classifies a context span by the provenance of its anchoring symbol so the
//! [`SafetyEnforced`](super::safety::SafetyEnforced) guard can refuse to evict
//! safety-critical spans. This guards *retention* (eviction from the window);
//! the ADR-060 gate in [`ToolRegistry::execute`](crate::agent::tool::ToolRegistry)
//! guards *execution* (dangerous tool calls). They are complementary.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::agent::tool_manifest::DangerLevel;
use crate::engine::Engine;
use crate::provenance::DerivationKind;
use crate::symbol::SymbolId;

/// How aggressively a span may be evicted from the working context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// Free to drop (recency/relevance permitting): the default.
    Volatile,
    /// Reconstructable from episodic memory if dropped — safe to evict, recall
    /// can bring it back.
    Recoverable,
    /// Must never be silently evicted: anchors a decision, a safety veto, or a
    /// dangerous side-effecting action.
    PinnedSafety,
}

/// Classifies symbols into [`Retention`] classes by reading the provenance
/// ledger, with a per-instance cache to bound latency over a turn.
#[derive(Default)]
pub struct ProvenanceRetentionClassifier {
    cache: Mutex<HashMap<SymbolId, Retention>>,
}

impl ProvenanceRetentionClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Classify `symbol` by its provenance. Cached after the first lookup.
    pub fn classify(&self, engine: &Engine, symbol: SymbolId) -> Retention {
        if let Some(r) = self
            .cache
            .lock()
            .expect("retention cache poisoned")
            .get(&symbol)
        {
            return *r;
        }
        let r = self.classify_uncached(engine, symbol);
        self.cache
            .lock()
            .expect("retention cache poisoned")
            .insert(symbol, r);
        r
    }

    fn classify_uncached(&self, engine: &Engine, symbol: SymbolId) -> Retention {
        // No ledger / lookup error ⇒ nothing pins it; treat as Volatile.
        let records = match engine.provenance_of(symbol) {
            Ok(r) => r,
            Err(_) => return Retention::Volatile,
        };
        let mut retention = Retention::Volatile;
        for rec in &records {
            match &rec.kind {
                // A decision or a safety veto is load-bearing — never drop it.
                DerivationKind::AgentDecision { .. } | DerivationKind::ShadowVeto { .. } => {
                    return Retention::PinnedSafety;
                }
                // A dangerous-or-higher tool execution must stay visible.
                DerivationKind::WasmToolExecution { danger_level, .. }
                | DerivationKind::CliToolExecution { danger_level, .. } => {
                    if danger_at_least(danger_level, DangerLevel::Dangerous) {
                        return Retention::PinnedSafety;
                    }
                }
                // Consolidated to episodic memory ⇒ recoverable by recall.
                DerivationKind::AgentConsolidation { .. } => {
                    retention = Retention::Recoverable;
                }
                _ => {}
            }
        }
        retention
    }
}

/// Parse a provenance `danger_level` string (as written by the tool executors,
/// matching [`DangerLevel`]'s `Display`) case-insensitively.
fn parse_danger(s: &str) -> Option<DangerLevel> {
    match s.trim().to_ascii_lowercase().as_str() {
        "safe" => Some(DangerLevel::Safe),
        "cautious" => Some(DangerLevel::Cautious),
        "dangerous" => Some(DangerLevel::Dangerous),
        "critical" => Some(DangerLevel::Critical),
        _ => None,
    }
}

/// Whether the danger-level string is at least `floor`. Unparseable ⇒ false (do
/// not pin on a level we can't read; the execution gate already vetted it).
fn danger_at_least(s: &str, floor: DangerLevel) -> bool {
    parse_danger(s).map(|d| d >= floor).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::provenance::ProvenanceRecord;
    use crate::symbol::SymbolKind;

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
    fn classifies_each_provenance_kind() {
        let (engine, _tmp) = test_engine();
        let classifier = ProvenanceRetentionClassifier::new();

        // AgentDecision ⇒ PinnedSafety.
        let decided = engine
            .create_symbol(SymbolKind::Entity, "decided")
            .unwrap()
            .id;
        let goal = engine.create_symbol(SymbolKind::Entity, "goal").unwrap().id;
        let mut rec =
            ProvenanceRecord::new(decided, DerivationKind::AgentDecision { goal, cycle: 1 });
        engine.store_provenance(&mut rec).unwrap();
        assert_eq!(
            classifier.classify(&engine, decided),
            Retention::PinnedSafety
        );

        // ShadowVeto ⇒ PinnedSafety.
        let vetoed = engine
            .create_symbol(SymbolKind::Entity, "vetoed")
            .unwrap()
            .id;
        let mut rec = ProvenanceRecord::new(
            vetoed,
            DerivationKind::ShadowVeto {
                pattern_name: "destructive".into(),
                severity: 0.9,
            },
        );
        engine.store_provenance(&mut rec).unwrap();
        assert_eq!(
            classifier.classify(&engine, vetoed),
            Retention::PinnedSafety
        );

        // Critical CLI tool execution ⇒ PinnedSafety.
        let danger = engine
            .create_symbol(SymbolKind::Entity, "danger")
            .unwrap()
            .id;
        let mut rec = ProvenanceRecord::new(
            danger,
            DerivationKind::CliToolExecution {
                tool_name: "rm".into(),
                binary_path: "/bin/rm".into(),
                danger_level: "Critical".into(),
            },
        );
        engine.store_provenance(&mut rec).unwrap();
        assert_eq!(
            classifier.classify(&engine, danger),
            Retention::PinnedSafety
        );

        // A Cautious tool execution is NOT pinned.
        let mild = engine.create_symbol(SymbolKind::Entity, "mild").unwrap().id;
        let mut rec = ProvenanceRecord::new(
            mild,
            DerivationKind::WasmToolExecution {
                tool_name: "read".into(),
                skill_id: "s".into(),
                danger_level: "Cautious".into(),
            },
        );
        engine.store_provenance(&mut rec).unwrap();
        assert_eq!(classifier.classify(&engine, mild), Retention::Volatile);

        // AgentConsolidation ⇒ Recoverable.
        let recalled = engine
            .create_symbol(SymbolKind::Entity, "recalled")
            .unwrap()
            .id;
        let mut rec = ProvenanceRecord::new(
            recalled,
            DerivationKind::AgentConsolidation {
                reason: "salient".into(),
                relevance_score: 0.7,
            },
        );
        engine.store_provenance(&mut rec).unwrap();
        assert_eq!(
            classifier.classify(&engine, recalled),
            Retention::Recoverable
        );

        // No provenance ⇒ Volatile.
        let plain = engine
            .create_symbol(SymbolKind::Entity, "plain")
            .unwrap()
            .id;
        assert_eq!(classifier.classify(&engine, plain), Retention::Volatile);
    }

    #[test]
    fn danger_parsing_is_case_insensitive_and_ordered() {
        assert!(danger_at_least("dangerous", DangerLevel::Dangerous));
        assert!(danger_at_least("CRITICAL", DangerLevel::Dangerous));
        assert!(!danger_at_least("cautious", DangerLevel::Dangerous));
        assert!(!danger_at_least("safe", DangerLevel::Dangerous));
        assert!(!danger_at_least("nonsense", DangerLevel::Dangerous));
    }
}
