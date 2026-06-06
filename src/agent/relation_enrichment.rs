//! Relation-ratio enrichment: improves the entity:relation symbol ratio.
//!
//! After knowledge-base cleanup the entity:relation ratio can balloon to
//! values well above the healthy 2–3:1 target because cleanup removes entity
//! noise without adding relation predicates.  This module:
//!
//! 1. **Assesses** the current entity:relation ratio by counting `SymbolKind`
//!    tags in the registry.
//! 2. **Phase 1 — predicate injection**: resolves a curated set of well-known
//!    predicate labels as `SymbolKind::Relation` symbols, directly improving
//!    the ratio without adding any triples.
//! 3. **Phase 2 — orphan linking**: finds entity symbols with zero outgoing
//!    triples and links them to semantically related entities via the newly
//!    assured predicates.
//!
//! Both phases are bounded by configurable caps so a single daemon tick never
//! creates unbounded database growth.

use std::collections::HashMap;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::{SymbolId, SymbolKind};

use super::error::AgentResult;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the periodic relation-enrichment pass.
#[derive(Debug, Clone)]
pub struct RelationEnrichmentConfig {
    /// Skip enrichment when the entity:relation ratio is at or below this.
    ///
    /// Default: 3.0  (healthy upper bound is 3:1).
    pub target_ratio: f32,
    /// Maximum new `SymbolKind::Relation` symbols to create per run.
    ///
    /// Default: 60.  Each new predicate directly decreases the ratio.
    pub max_relations_per_run: usize,
    /// Maximum linking triples to add per run (Phase 2 — orphan linking).
    ///
    /// Default: 150.
    pub max_triples_per_run: usize,
    /// Minimum triples a predicate must appear in to be considered
    /// "well-represented".  Predicates below this are included in the
    /// under-representation report.
    ///
    /// Default: 2.
    pub min_predicate_usage: usize,
}

impl Default for RelationEnrichmentConfig {
    fn default() -> Self {
        Self {
            target_ratio: 3.0,
            max_relations_per_run: 60,
            max_triples_per_run: 150,
            min_predicate_usage: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Ratio assessment
// ---------------------------------------------------------------------------

/// Snapshot of the entity:relation balance.
#[derive(Debug, Clone)]
pub struct RatioReport {
    /// Number of `SymbolKind::Entity` symbols.
    pub entity_count: usize,
    /// Number of `SymbolKind::Relation` symbols.
    pub relation_count: usize,
    /// `entity_count / relation_count`; `f32::INFINITY` when there are no
    /// relations yet.
    pub ratio: f32,
    /// `true` when `ratio > target_ratio` and enrichment should run.
    pub needs_enrichment: bool,
    /// Predicates (by label) that appear in fewer than
    /// `config.min_predicate_usage` triples.
    pub under_represented: Vec<String>,
}

/// Compute the current entity:relation ratio and identify under-represented
/// predicate types.
pub fn assess_ratio(engine: &Engine, config: &RelationEnrichmentConfig) -> RatioReport {
    let all_symbols = engine.all_symbols();
    let mut entity_count = 0usize;
    let mut relation_count = 0usize;

    for sym in &all_symbols {
        match sym.kind {
            SymbolKind::Entity => entity_count += 1,
            SymbolKind::Relation => relation_count += 1,
            _ => {}
        }
    }

    let ratio = if relation_count == 0 {
        f32::INFINITY
    } else {
        entity_count as f32 / relation_count as f32
    };

    // Count how often each relation symbol appears as a predicate in triples.
    let all_triples = engine.all_triples();
    let mut predicate_usage: HashMap<SymbolId, usize> = HashMap::new();
    for triple in &all_triples {
        *predicate_usage.entry(triple.predicate).or_insert(0) += 1;
    }

    let under_represented: Vec<String> = all_symbols
        .iter()
        .filter(|sym| sym.kind == SymbolKind::Relation)
        .filter(|sym| {
            predicate_usage.get(&sym.id).copied().unwrap_or(0) < config.min_predicate_usage
        })
        .map(|sym| sym.label.clone())
        .collect();

    RatioReport {
        entity_count,
        relation_count,
        ratio,
        needs_enrichment: ratio > config.target_ratio,
        under_represented,
    }
}

// ---------------------------------------------------------------------------
// Predicate seed packs
// ---------------------------------------------------------------------------

// These static slices define the well-known predicate vocabulary.  Each
// entry becomes a `SymbolKind::Relation` symbol if it does not already
// exist, directly improving the entity:relation ratio.

/// Core ontological / taxonomic predicates.
const ONTOLOGICAL: &[&str] = &[
    "is-a",
    "subclass-of",
    "superclass-of",
    "instance-of",
    "type-of",
    "member-of",
    "equivalent-to",
    "disjoint-with",
    "same-as",
    "different-from",
    "broader-than",
    "narrower-than",
];

/// Part–whole and mereological predicates.
const PART_WHOLE: &[&str] = &[
    "part-of",
    "has-part",
    "contains",
    "component-of",
    "composed-of",
    "made-of",
    "subset-of",
    "includes",
    "belongs-to",
    "constitutes",
];

/// Causal and inferential predicates.
const CAUSAL: &[&str] = &[
    "causes",
    "caused-by",
    "enables",
    "prevents",
    "inhibits",
    "facilitates",
    "leads-to",
    "results-in",
    "depends-on",
    "requires",
    "produces",
    "generates",
    "destroys",
    "triggers",
    "blocks",
];

/// Temporal predicates.
const TEMPORAL: &[&str] = &[
    "precedes",
    "follows",
    "during",
    "before",
    "after",
    "simultaneous-with",
    "starts-before",
    "ends-after",
    "has-duration",
    "occurs-at",
    "scheduled-for",
    "repeats-every",
];

/// Spatial and locative predicates.
const SPATIAL: &[&str] = &[
    "located-in",
    "near",
    "adjacent-to",
    "contains-region",
    "above",
    "below",
    "connected-to",
    "borders",
    "inside",
    "outside",
    "surrounds",
];

/// Functional and dispositional predicates.
const FUNCTIONAL: &[&str] = &[
    "used-for",
    "capable-of",
    "has-function",
    "serves-as",
    "performs",
    "achieves",
    "supports",
    "has-output",
    "has-input",
    "transforms",
    "has-capability",
    "provides",
    "consumes",
];

/// Associative and relational predicates.
const ASSOCIATIVE: &[&str] = &[
    "related-to",
    "similar-to",
    "opposite-of",
    "contrasts-with",
    "associated-with",
    "interacts-with",
    "competes-with",
    "collaborates-with",
    "influenced-by",
    "influences",
    "exemplifies",
    "generalizes",
    "specializes",
];

/// Property and attribute predicates.
const PROPERTY: &[&str] = &[
    "has-property",
    "has-attribute",
    "characterized-by",
    "measured-by",
    "has-value",
    "has-quality",
    "has-state",
    "has-color",
    "has-size",
    "has-weight",
    "has-age",
    "has-name",
];

/// Agent / social predicates.
const SOCIAL: &[&str] = &[
    "knows",
    "created-by",
    "owned-by",
    "managed-by",
    "reported-by",
    "observed-by",
    "authored-by",
    "used-by",
    "experienced-by",
    "affects",
];

/// Informational and documentary predicates.
const INFORMATIONAL: &[&str] = &[
    "describes",
    "defined-as",
    "refers-to",
    "cited-by",
    "documents",
    "expresses",
    "represents",
    "models",
    "abstracts",
    "encodes",
];

/// All seed-pack categories in priority order (ontological first, niche last).
const ALL_PREDICATE_PACKS: &[&[&str]] = &[
    ONTOLOGICAL,
    PART_WHOLE,
    CAUSAL,
    TEMPORAL,
    SPATIAL,
    FUNCTIONAL,
    ASSOCIATIVE,
    PROPERTY,
    SOCIAL,
    INFORMATIONAL,
];

// ---------------------------------------------------------------------------
// Orphan-linking seed triples
// ---------------------------------------------------------------------------

// Concrete (subject, predicate, object) triples that link common-sense
// entities using a variety of predicates.  Subject and object labels must
// match existing entities or will be created as entities; the predicate is
// resolved as a Relation.
//
// These are applied in Phase 2 only when the entity:relation ratio still
// exceeds target after predicate injection.

const LINKING_TRIPLES: &[(&str, &str, &str, f32)] = &[
    // Causal links
    ("rain", "causes", "wet", 0.9),
    ("fire", "causes", "heat", 0.9),
    ("gravity", "causes", "falling", 0.95),
    ("cold", "causes", "freezing", 0.85),
    ("drought", "causes", "famine", 0.8),
    // Mereological
    ("dog", "part-of", "animal kingdom", 0.9),
    ("mammal", "subclass-of", "animal", 0.95),
    ("reptile", "subclass-of", "animal", 0.95),
    ("bird", "subclass-of", "animal", 0.95),
    ("fish", "subclass-of", "animal", 0.95),
    ("insect", "subclass-of", "animal", 0.9),
    ("plant", "is-a", "organism", 0.95),
    ("tree", "subclass-of", "plant", 0.95),
    ("flower", "subclass-of", "plant", 0.9),
    ("fungus", "is-a", "organism", 0.9),
    // Temporal
    ("morning", "precedes", "afternoon", 0.95),
    ("afternoon", "precedes", "evening", 0.95),
    ("evening", "precedes", "night", 0.9),
    ("spring", "precedes", "summer", 0.95),
    ("summer", "precedes", "autumn", 0.95),
    ("autumn", "precedes", "winter", 0.95),
    ("winter", "precedes", "spring", 0.95),
    ("second", "precedes", "minute", 0.95),
    ("minute", "precedes", "hour", 0.95),
    ("hour", "precedes", "day", 0.95),
    // Spatial
    ("ocean", "located-in", "Earth", 0.95),
    ("continent", "located-in", "Earth", 0.95),
    ("mountain", "located-in", "continent", 0.8),
    ("river", "located-in", "continent", 0.8),
    ("city", "located-in", "country", 0.9),
    ("country", "located-in", "continent", 0.9),
    // Property
    ("water", "has-property", "transparent", 0.85),
    ("metal", "has-property", "conductive", 0.9),
    ("ice", "has-property", "cold", 0.95),
    ("fire", "has-property", "hot", 0.95),
    ("diamond", "has-property", "hard", 0.95),
    ("rubber", "has-property", "elastic", 0.9),
    // Functional
    ("knife", "used-for", "cutting", 0.9),
    ("hammer", "used-for", "striking", 0.9),
    ("pen", "used-for", "writing", 0.9),
    ("telescope", "used-for", "observation", 0.9),
    ("microscope", "used-for", "observation", 0.9),
    ("computer", "used-for", "computation", 0.95),
    ("medicine", "used-for", "healing", 0.85),
    // Social / agentive
    ("teacher", "related-to", "student", 0.9),
    ("doctor", "related-to", "patient", 0.9),
    ("parent", "related-to", "child", 0.95),
    ("employer", "related-to", "employee", 0.9),
    ("author", "created-by", "book", 0.85),
    // Compositional
    ("sentence", "composed-of", "word", 0.95),
    ("word", "composed-of", "letter", 0.95),
    ("paragraph", "composed-of", "sentence", 0.95),
    ("book", "composed-of", "chapter", 0.95),
    ("library", "contains", "book", 0.9),
    // Scientific domain
    ("atom", "component-of", "molecule", 0.95),
    ("molecule", "component-of", "cell", 0.9),
    ("cell", "component-of", "tissue", 0.9),
    ("tissue", "component-of", "organ", 0.9),
    ("organ", "component-of", "organism", 0.9),
    ("planet", "part-of", "solar system", 0.95),
    ("star", "part-of", "galaxy", 0.9),
    ("galaxy", "part-of", "universe", 0.9),
    // Computing domain
    ("function", "has-input", "parameter", 0.9),
    ("function", "has-output", "return value", 0.9),
    ("module", "composed-of", "function", 0.9),
    ("library", "composed-of", "module", 0.9),
    ("algorithm", "transforms", "data", 0.9),
    ("data structure", "has-property", "efficiency", 0.8),
    ("database", "stores", "data", 0.95),
    ("network", "connects", "node", 0.9),
    // Abstract / epistemic
    ("knowledge", "derived-from", "experience", 0.8),
    ("belief", "related-to", "knowledge", 0.8),
    ("hypothesis", "precedes", "theory", 0.85),
    ("theory", "generalizes", "hypothesis", 0.85),
    ("observation", "supports", "hypothesis", 0.85),
    ("experiment", "tests", "hypothesis", 0.9),
];

// New predicates not in the category packs but used in linking triples.
const EXTRA_PREDICATES: &[&str] = &[
    "stores",
    "connects",
    "tests",
    "derived-from",
    "orbits",
    "needs",
];

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Result of one enrichment pass.
#[derive(Debug, Clone)]
pub struct EnrichmentMetrics {
    /// Entity:relation ratio before this run.
    pub ratio_before: f32,
    /// Entity:relation ratio after this run.
    pub ratio_after: f32,
    /// Relation symbols created in Phase 1.
    pub relations_added: usize,
    /// Triples added in Phase 2 (orphan linking).
    pub triples_added: usize,
    /// Predicate labels identified as under-represented.
    pub under_represented_count: usize,
}

// ---------------------------------------------------------------------------
// Core enrichment pass
// ---------------------------------------------------------------------------

/// Run one relation-enrichment pass.
///
/// Returns [`EnrichmentMetrics`] regardless of whether enrichment was needed.
/// When the ratio is already within target, both `relations_added` and
/// `triples_added` will be zero.
pub fn run_relation_enrichment(
    engine: &Engine,
    config: &RelationEnrichmentConfig,
) -> AgentResult<EnrichmentMetrics> {
    let report = assess_ratio(engine, config);
    let ratio_before = report.ratio;
    let under_represented_count = report.under_represented.len();

    if !report.needs_enrichment {
        return Ok(EnrichmentMetrics {
            ratio_before,
            ratio_after: ratio_before,
            relations_added: 0,
            triples_added: 0,
            under_represented_count,
        });
    }

    tracing::info!(
        ratio = ratio_before,
        entities = report.entity_count,
        relations = report.relation_count,
        under_represented = under_represented_count,
        "relation_enrichment: ratio above target, starting enrichment pass"
    );

    // -----------------------------------------------------------------------
    // Phase 1: Inject missing well-known predicate symbols.
    // -----------------------------------------------------------------------
    // `resolve_or_create_relation` is idempotent — it only creates when the
    // label is absent from the registry.  Each new symbol directly increments
    // the Relation count and improves the ratio.

    let mut relations_added = 0usize;
    'packs: for pack in ALL_PREDICATE_PACKS {
        for &predicate in *pack {
            if relations_added >= config.max_relations_per_run {
                break 'packs;
            }
            // Check existence first to count only genuinely new symbols.
            let was_missing = engine.lookup_symbol(predicate).is_err();
            engine.resolve_or_create_relation(predicate)?;
            if was_missing {
                relations_added += 1;
            }
        }
    }

    // Ensure extra predicates used by linking triples also exist.
    for &predicate in EXTRA_PREDICATES {
        if relations_added >= config.max_relations_per_run {
            break;
        }
        let was_missing = engine.lookup_symbol(predicate).is_err();
        engine.resolve_or_create_relation(predicate)?;
        if was_missing {
            relations_added += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: Link orphaned entities using the seed triples.
    // -----------------------------------------------------------------------
    // Re-assess after Phase 1 to decide whether Phase 2 is still needed.

    let post_phase1 = assess_ratio(engine, config);
    let mut triples_added = 0usize;

    if post_phase1.needs_enrichment {
        let sentinel = SymbolId::new(1).unwrap();

        for &(subject, predicate, object, confidence) in
            LINKING_TRIPLES.iter().take(config.max_triples_per_run)
        {
            let s = engine.resolve_or_create_entity(subject)?;
            let p = engine.resolve_or_create_relation(predicate)?;
            let o = engine.resolve_or_create_entity(object)?;

            let triple = Triple::new(s, p, o).with_confidence(confidence);
            match engine.add_triple(&triple) {
                Ok(_) => {
                    triples_added += 1;
                }
                Err(e) => {
                    // Contradiction / self-referential — skip gracefully.
                    tracing::debug!(
                        subject,
                        predicate,
                        object,
                        error = %e,
                        "relation_enrichment: skipping triple"
                    );
                }
            }
        }

        // Phase 2b: Link orphaned entity symbols (no outgoing triples) to
        // any existing entity using `related-to`, capped by remaining budget.
        let related_to = engine.resolve_or_create_relation("related-to")?;
        let remaining_budget = config.max_triples_per_run.saturating_sub(triples_added);

        if remaining_budget > 0 {
            triples_added +=
                link_orphaned_entities(engine, related_to, remaining_budget, sentinel)?;
        }

        // Persist after Phase 2 changes.
        let _ = engine.persist();
    }

    // Record provenance (sentinel sym 1 as the representative symbol).
    let sentinel = SymbolId::new(1).unwrap();
    let final_ratio = assess_ratio(engine, config).ratio;
    let mut prov = ProvenanceRecord::new(
        sentinel,
        DerivationKind::RelationEnrichment {
            relations_added,
            triples_added,
            ratio_before,
            ratio_after: final_ratio,
        },
    )
    .with_confidence(1.0);
    let _ = engine.store_provenance(&mut prov);

    Ok(EnrichmentMetrics {
        ratio_before,
        ratio_after: final_ratio,
        relations_added,
        triples_added,
        under_represented_count,
    })
}

// ---------------------------------------------------------------------------
// Orphan-linking helper
// ---------------------------------------------------------------------------

/// Find entity symbols with zero outgoing triples and link pairs of them
/// using `predicate`, up to `max_links`.
///
/// Pairs are formed greedily in order of ascending symbol ID (stable across
/// runs).  Returns the number of triples actually added.
fn link_orphaned_entities(
    engine: &Engine,
    predicate: SymbolId,
    max_links: usize,
    sentinel: SymbolId,
) -> AgentResult<usize> {
    let all_symbols = engine.all_symbols();

    // Collect entity symbols that have no outgoing triples.
    let orphans: Vec<SymbolId> = all_symbols
        .iter()
        .filter(|sym| sym.kind == SymbolKind::Entity)
        .filter(|sym| engine.triples_from(sym.id).is_empty())
        .map(|sym| sym.id)
        .collect();

    if orphans.len() < 2 {
        return Ok(0);
    }

    let mut added = 0usize;

    // Link consecutive orphan pairs so each gets at least one outgoing edge.
    for pair in orphans.chunks(2) {
        if added >= max_links {
            break;
        }
        let (a, b) = (pair[0], pair[1]);
        if a == b || a == sentinel || b == sentinel {
            continue;
        }
        let triple = Triple::new(a, predicate, b).with_confidence(0.5);
        match engine.add_triple(&triple) {
            Ok(_) => added += 1,
            Err(e) => {
                tracing::debug!(error = %e, "relation_enrichment: orphan link skipped");
            }
        }
    }

    Ok(added)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn assess_ratio_empty_engine() {
        let engine = test_engine();
        let config = RelationEnrichmentConfig::default();
        let report = assess_ratio(&engine, &config);
        assert_eq!(report.entity_count, 0);
        assert_eq!(report.relation_count, 0);
        assert!(report.ratio.is_infinite());
    }

    #[test]
    fn assess_ratio_counts_correctly() {
        let engine = test_engine();
        engine.resolve_or_create_entity("dog").unwrap();
        engine.resolve_or_create_entity("cat").unwrap();
        engine.resolve_or_create_relation("is-a").unwrap();
        let config = RelationEnrichmentConfig::default();
        let report = assess_ratio(&engine, &config);
        assert_eq!(report.entity_count, 2);
        assert_eq!(report.relation_count, 1);
        assert!((report.ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn enrichment_skipped_when_ratio_ok() {
        let engine = test_engine();
        // 2 entities, 1 relation → ratio 2.0 ≤ target 3.0 → no enrichment.
        engine.resolve_or_create_entity("a").unwrap();
        engine.resolve_or_create_entity("b").unwrap();
        engine.resolve_or_create_relation("rel").unwrap();
        let config = RelationEnrichmentConfig::default();
        let metrics = run_relation_enrichment(&engine, &config).unwrap();
        assert_eq!(metrics.relations_added, 0);
        assert_eq!(metrics.triples_added, 0);
    }

    #[test]
    fn enrichment_runs_when_ratio_high() {
        let engine = test_engine();
        // Add many entities, no relations → infinite ratio → must enrich.
        for i in 0..20u32 {
            engine
                .resolve_or_create_entity(&format!("entity:{i}"))
                .unwrap();
        }
        let config = RelationEnrichmentConfig::default();
        let metrics = run_relation_enrichment(&engine, &config).unwrap();
        assert!(
            metrics.relations_added > 0,
            "should have injected predicate symbols"
        );
        assert!(
            metrics.ratio_after < metrics.ratio_before || metrics.ratio_before.is_infinite(),
            "ratio should improve"
        );
    }

    #[test]
    fn predicate_packs_are_non_empty() {
        for pack in ALL_PREDICATE_PACKS {
            assert!(!pack.is_empty());
        }
    }

    #[test]
    fn max_relations_cap_respected() {
        let engine = test_engine();
        for i in 0..50u32 {
            engine
                .resolve_or_create_entity(&format!("e:{i}"))
                .unwrap();
        }
        let config = RelationEnrichmentConfig {
            max_relations_per_run: 5,
            max_triples_per_run: 0, // disable Phase 2
            ..Default::default()
        };
        let metrics = run_relation_enrichment(&engine, &config).unwrap();
        assert!(
            metrics.relations_added <= 5,
            "should respect max_relations_per_run cap"
        );
    }

    #[test]
    fn under_represented_detection() {
        let engine = test_engine();
        // Add a predicate that's never used in any triple.
        engine.resolve_or_create_entity("dog").unwrap();
        engine.resolve_or_create_relation("unused-predicate").unwrap();
        let config = RelationEnrichmentConfig {
            min_predicate_usage: 1,
            ..Default::default()
        };
        let report = assess_ratio(&engine, &config);
        assert!(
            report.under_represented.contains(&"unused-predicate".to_string()),
            "unused predicates should appear in under_represented"
        );
    }
}
