//! Subsystem integration: unified neuro-symbolic reasoning cycles.
//!
//! Closes the loop between forward-chaining rules, symbol grounding,
//! spreading activation with superposition, multi-path confidence fusion,
//! and commits high-confidence results to the KG.

use crate::autonomous::fusion::{self, FusedConfidence, FusionConfig, InferencePath};
use crate::autonomous::rule_engine::{RuleEngine, RuleEngineConfig, RuleEngineResult};
use crate::autonomous::rules::RuleSet;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::infer::superposition::{self, SuperpositionConfig, SuperpositionResult};
use crate::symbol::SymbolId;
use crate::vsa::grounding::{self, GroundingConfig};

use super::error::AutonomousResult;

/// Configuration for a full reasoning cycle.
#[derive(Debug, Clone)]
pub struct ReasoningCycleConfig {
    /// Forward-chaining rule engine config.
    pub rule_config: RuleEngineConfig,
    /// Symbol grounding config.
    pub grounding_config: GroundingConfig,
    /// Superposition inference config.
    pub superposition_config: SuperpositionConfig,
    /// Confidence fusion config.
    pub fusion_config: FusionConfig,
    /// Minimum quality score to commit a derived triple (default: 0.5).
    pub commit_threshold: f32,
}

impl Default for ReasoningCycleConfig {
    fn default() -> Self {
        Self {
            rule_config: RuleEngineConfig::default(),
            grounding_config: GroundingConfig::default(),
            superposition_config: SuperpositionConfig::default(),
            fusion_config: FusionConfig::default(),
            commit_threshold: 0.5,
        }
    }
}

/// Result of a full reasoning cycle.
#[derive(Debug)]
pub struct ReasoningCycleResult {
    /// Forward-chaining rule results.
    pub rule_result: Option<RuleEngineResult>,
    /// Number of symbols re-grounded.
    pub symbols_grounded: usize,
    /// Superposition inference result.
    pub superposition_result: Option<SuperpositionResult>,
    /// Fused inference paths.
    pub fused: Vec<FusedConfidence>,
    /// Number of new triples committed to the KG.
    pub triples_committed: usize,
}

/// Run a full neuro-symbolic reasoning cycle:
///
/// 1. Forward-chain rules → new triples
/// 2. Re-ground affected symbols (neighborhood changed)
/// 3. Run spreading activation with superposition
/// 4. Fuse multi-path confidence with interference signals
/// 5. Commit high-confidence results to KG
pub fn reasoning_cycle(
    engine: &Engine,
    seeds: &[SymbolId],
    config: &ReasoningCycleConfig,
) -> AutonomousResult<ReasoningCycleResult> {
    let ops = engine.ops();
    let im = engine.item_memory();

    // ── Step 1: Forward-chain rules ──
    let rule_result = {
        let rule_engine =
            RuleEngine::new(config.rule_config.clone()).with_rules(RuleSet::builtin());
        match rule_engine.run(engine) {
            Ok(result) => Some(result),
            Err(e) => {
                tracing::warn!(error = %e, "forward-chaining rules failed, continuing");
                None
            }
        }
    };

    // ── Step 2: Re-ground symbols ──
    let symbols_grounded = match grounding::ground_all(engine, ops, im, &config.grounding_config) {
        Ok(result) => result.symbols_updated,
        Err(e) => {
            tracing::warn!(error = %e, "symbol grounding failed, continuing");
            0
        }
    };

    // ── Step 3: Superposition inference ──
    let superposition_result = if !seeds.is_empty() {
        match superposition::infer_with_superposition(seeds, engine, &config.superposition_config) {
            Ok(result) => Some(result),
            Err(e) => {
                tracing::warn!(error = %e, "superposition inference failed, continuing");
                None
            }
        }
    } else {
        None
    };

    // ── Step 4: Build inference paths and fuse confidence ──
    let mut inference_paths: Vec<InferencePath> = Vec::new();

    // Add paths from rule-derived triples
    if let Some(ref rr) = rule_result {
        for derived in &rr.derived {
            inference_paths.push(InferencePath {
                subject: derived.triple.subject,
                predicate: derived.triple.predicate,
                object: derived.triple.object,
                path_confidence: derived.triple.confidence,
                chain: vec![(
                    derived.triple.subject,
                    derived.triple.predicate,
                    derived.triple.object,
                )],
                rule_name: derived.rule_name.clone(),
            });
        }
    }

    // Add paths from superposition hypotheses
    if let Some(ref sr) = superposition_result {
        for hyp in &sr.hypotheses {
            for (sym, conf) in &hyp.activated {
                // Create paths from seed to activated symbol
                for seed in seeds {
                    if seed != sym {
                        // Find a connecting predicate if one exists
                        let triples = engine.triples_from(*seed);
                        for t in &triples {
                            if t.object == *sym {
                                inference_paths.push(InferencePath {
                                    subject: *seed,
                                    predicate: t.predicate,
                                    object: *sym,
                                    path_confidence: *conf,
                                    chain: vec![(*seed, t.predicate, *sym)],
                                    rule_name: "superposition".into(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    let fused = if !inference_paths.is_empty() {
        match fusion::fuse_paths(&inference_paths, ops, im, &config.fusion_config) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(error = %e, "confidence fusion failed");
                vec![]
            }
        }
    } else {
        vec![]
    };

    // ── Step 5: Commit high-confidence results to KG ──
    let mut triples_committed = 0;

    for fc in &fused {
        if fc.quality_score >= config.commit_threshold && fc.is_constructive {
            let triple = Triple::new(fc.subject, fc.predicate, fc.object)
                .with_confidence(fc.fused_confidence);
            if engine.add_triple(&triple).is_ok() {
                triples_committed += 1;
            }
        }
    }

    Ok(ReasoningCycleResult {
        rule_result,
        symbols_grounded,
        superposition_result,
        fused,
        triples_committed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
    use crate::symbol::SymbolKind;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn reasoning_cycle_runs_without_error() {
        let engine = test_engine();

        let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap().id;
        let mammal = engine
            .create_symbol(SymbolKind::Entity, "Mammal")
            .unwrap()
            .id;
        let is_a = engine
            .create_symbol(SymbolKind::Relation, "is-a")
            .unwrap()
            .id;

        engine.add_triple(&Triple::new(dog, is_a, mammal)).unwrap();

        let config = ReasoningCycleConfig::default();
        let result = reasoning_cycle(&engine, &[dog], &config).unwrap();

        assert!(result.symbols_grounded > 0, "Should have grounded symbols");
    }

    #[test]
    fn reasoning_cycle_with_empty_seeds() {
        let engine = test_engine();
        let config = ReasoningCycleConfig::default();

        let result = reasoning_cycle(&engine, &[], &config).unwrap();
        assert!(result.superposition_result.is_none());
    }
}
