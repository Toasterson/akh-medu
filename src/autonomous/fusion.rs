//! Multi-path confidence fusion using Noisy-OR and VSA interference.
//!
//! When multiple inference paths support the same triple, their confidences
//! are fused using Noisy-OR and validated via VSA interference signals.

use std::collections::HashMap;

use crate::symbol::SymbolId;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

use super::error::{AutonomousError, AutonomousResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single inference path supporting a triple.
#[derive(Debug, Clone)]
pub struct InferencePath {
    pub subject: SymbolId,
    pub predicate: SymbolId,
    pub object: SymbolId,
    pub path_confidence: f32,
    pub chain: Vec<(SymbolId, SymbolId, SymbolId)>,
    pub rule_name: String,
}

/// Result of fusing multiple paths for a single triple.
#[derive(Debug, Clone)]
pub struct FusedConfidence {
    pub subject: SymbolId,
    pub predicate: SymbolId,
    pub object: SymbolId,
    /// Noisy-OR fused confidence.
    pub fused_confidence: f32,
    /// Number of supporting inference paths.
    pub path_count: usize,
    /// VSA interference signal: -1.0 (destructive) to +1.0 (constructive).
    pub interference_signal: f32,
    /// Combined quality score (0.0 to 1.0).
    pub quality_score: f32,
    /// Whether the interference is constructive (signal > 0).
    pub is_constructive: bool,
}

/// Configuration for confidence fusion.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Weight for Noisy-OR confidence in quality score (default: 0.6).
    pub confidence_weight: f32,
    /// Weight for interference signal in quality score (default: 0.4).
    pub interference_weight: f32,
    /// Threshold below which interference is considered contradictory (default: -0.3).
    pub contradiction_threshold: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            confidence_weight: 0.6,
            interference_weight: 0.4,
            contradiction_threshold: -0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

/// Fuse multiple inference paths into single confidence scores per triple.
///
/// Paths are grouped by (subject, predicate, object). For each group:
/// 1. **Noisy-OR**: `fused = 1 - (1-c1)(1-c2)...(1-cN)`
/// 2. **VSA interference**: `bind(S, P)` similarity to `O` averaged across paths
/// 3. **Quality**: weighted combination of both signals
pub fn fuse_paths(
    paths: &[InferencePath],
    ops: &VsaOps,
    item_memory: &ItemMemory,
    config: &FusionConfig,
) -> AutonomousResult<Vec<FusedConfidence>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }

    // Group paths by triple.
    let mut groups: HashMap<(u64, u64, u64), Vec<&InferencePath>> = HashMap::new();
    for path in paths {
        let key = (path.subject.get(), path.predicate.get(), path.object.get());
        groups.entry(key).or_default().push(path);
    }

    let mut results = Vec::new();

    for ((s, p, o), group) in &groups {
        let subject = SymbolId::new(*s).ok_or(AutonomousError::Fusion {
            message: "invalid subject symbol ID".into(),
        })?;
        let predicate = SymbolId::new(*p).ok_or(AutonomousError::Fusion {
            message: "invalid predicate symbol ID".into(),
        })?;
        let object = SymbolId::new(*o).ok_or(AutonomousError::Fusion {
            message: "invalid object symbol ID".into(),
        })?;

        // Noisy-OR fusion.
        let confs: Vec<f32> = group.iter().map(|p| p.path_confidence).collect();
        let fused = noisy_or(&confs);

        // VSA interference signal.
        let interference = compute_interference(subject, predicate, object, ops, item_memory);

        // Quality score.
        let normalized_interference = (interference + 1.0) / 2.0;
        let quality = config.confidence_weight * fused
            + config.interference_weight * normalized_interference;

        results.push(FusedConfidence {
            subject,
            predicate,
            object,
            fused_confidence: fused,
            path_count: group.len(),
            interference_signal: interference,
            quality_score: quality.clamp(0.0, 1.0),
            is_constructive: interference > 0.0,
        });
    }

    // Sort by quality score descending.
    results.sort_by(|a, b| {
        b.quality_score
            .partial_cmp(&a.quality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Noisy-OR: `1 - product(1 - ci)` for independent evidence sources.
pub fn noisy_or(confidences: &[f32]) -> f32 {
    let product: f32 = confidences.iter().map(|c| 1.0 - c.clamp(0.0, 1.0)).product();
    1.0 - product
}

/// Compute VSA interference signal for a triple (S, P, O).
///
/// 1. Compute `bind(S_vec, P_vec)` — the expected role-filler binding
/// 2. Compute `similarity(bind(S, P), O_vec)`
/// 3. Map to [-1, +1]: `(similarity - 0.5) * 2.0`
fn compute_interference(
    subject: SymbolId,
    predicate: SymbolId,
    object: SymbolId,
    ops: &VsaOps,
    item_memory: &ItemMemory,
) -> f32 {
    let s_vec = item_memory.get_or_create(ops, subject);
    let p_vec = item_memory.get_or_create(ops, predicate);
    let o_vec = item_memory.get_or_create(ops, object);

    let bound = match ops.bind(&s_vec, &p_vec) {
        Ok(b) => b,
        Err(_) => return 0.0,
    };

    let similarity = match ops.similarity(&bound, &o_vec) {
        Ok(sim) => sim,
        Err(_) => return 0.0,
    };

    // Map [0, 1] similarity to [-1, +1] interference signal.
    (similarity - 0.5) * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn noisy_or_single_path() {
        let result = noisy_or(&[0.7]);
        assert!((result - 0.7).abs() < 0.001);
    }

    #[test]
    fn noisy_or_two_paths() {
        // 1 - (1-0.6)(1-0.7) = 1 - 0.4*0.3 = 1 - 0.12 = 0.88
        let result = noisy_or(&[0.6, 0.7]);
        assert!((result - 0.88).abs() < 0.001);
    }

    #[test]
    fn noisy_or_three_paths() {
        // 1 - (1-0.5)(1-0.5)(1-0.5) = 1 - 0.125 = 0.875
        let result = noisy_or(&[0.5, 0.5, 0.5]);
        assert!((result - 0.875).abs() < 0.001);
    }

    #[test]
    fn fuse_empty_paths() {
        let engine = test_engine();
        let result =
            fuse_paths(&[], engine.ops(), engine.item_memory(), &FusionConfig::default())
                .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn fuse_single_path() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("A".into(), "rel".into(), "B".into(), 1.0)])
            .unwrap();

        let a = engine.lookup_symbol("A").unwrap();
        let rel = engine.lookup_symbol("rel").unwrap();
        let b = engine.lookup_symbol("B").unwrap();

        let paths = vec![InferencePath {
            subject: a,
            predicate: rel,
            object: b,
            path_confidence: 0.8,
            chain: vec![(a, rel, b)],
            rule_name: "test".into(),
        }];

        let result =
            fuse_paths(&paths, engine.ops(), engine.item_memory(), &FusionConfig::default())
                .unwrap();

        assert_eq!(result.len(), 1);
        assert!((result[0].fused_confidence - 0.8).abs() < 0.001);
        assert_eq!(result[0].path_count, 1);
    }

    #[test]
    fn fuse_multiple_paths_higher_confidence() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("A".into(), "rel".into(), "B".into(), 1.0)])
            .unwrap();

        let a = engine.lookup_symbol("A").unwrap();
        let rel = engine.lookup_symbol("rel").unwrap();
        let b = engine.lookup_symbol("B").unwrap();

        let paths = vec![
            InferencePath {
                subject: a,
                predicate: rel,
                object: b,
                path_confidence: 0.6,
                chain: vec![],
                rule_name: "rule1".into(),
            },
            InferencePath {
                subject: a,
                predicate: rel,
                object: b,
                path_confidence: 0.7,
                chain: vec![],
                rule_name: "rule2".into(),
            },
        ];

        let result =
            fuse_paths(&paths, engine.ops(), engine.item_memory(), &FusionConfig::default())
                .unwrap();

        assert_eq!(result.len(), 1);
        // Noisy-OR: 1 - (1-0.6)(1-0.7) = 0.88
        assert!((result[0].fused_confidence - 0.88).abs() < 0.01);
        assert!(result[0].fused_confidence > 0.7);
        assert_eq!(result[0].path_count, 2);
    }

    #[test]
    fn quality_score_in_bounds() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("X".into(), "pred".into(), "Y".into(), 1.0)])
            .unwrap();

        let x = engine.lookup_symbol("X").unwrap();
        let pred = engine.lookup_symbol("pred").unwrap();
        let y = engine.lookup_symbol("Y").unwrap();

        let paths = vec![InferencePath {
            subject: x,
            predicate: pred,
            object: y,
            path_confidence: 0.9,
            chain: vec![],
            rule_name: "test".into(),
        }];

        let result =
            fuse_paths(&paths, engine.ops(), engine.item_memory(), &FusionConfig::default())
                .unwrap();

        assert!(result[0].quality_score >= 0.0 && result[0].quality_score <= 1.0);
    }

    #[test]
    fn interference_signal_in_range() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("A".into(), "rel".into(), "B".into(), 1.0)])
            .unwrap();

        let a = engine.lookup_symbol("A").unwrap();
        let rel = engine.lookup_symbol("rel").unwrap();
        let b = engine.lookup_symbol("B").unwrap();

        let signal = compute_interference(a, rel, b, engine.ops(), engine.item_memory());
        assert!(signal >= -1.0 && signal <= 1.0);
    }
}
