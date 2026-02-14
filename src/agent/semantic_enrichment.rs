//! VSA-driven semantic enrichment: derives roles, importance, and data flow
//! from the knowledge graph and persists them as enrichment triples.
//!
//! Enrichment runs as a post-ingest pipeline step. The agent and synthesis
//! pipeline read enrichment triples from the KG alongside code structure triples.

use std::collections::{HashMap, HashSet};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;
use crate::vsa::grounding::bundle_symbols;

use super::error::AgentResult;
use super::tools::code_predicates::CodePredicates;

// ---------------------------------------------------------------------------
// Semantic predicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for semantic enrichment triples.
///
/// Follows the same pattern as `AgentPredicates` and `CodePredicates`.
#[derive(Debug, Clone)]
pub struct SemanticPredicates {
    /// "semantic:has-role" — module has a functional role (e.g., "storage", "transformation")
    pub has_role: SymbolId,
    /// "semantic:role-confidence" — confidence of the role classification
    pub role_confidence: SymbolId,
    /// "semantic:importance" — normalized PageRank importance score
    pub importance: SymbolId,
    /// "semantic:produces-type" — module produces (defines/returns) a type
    pub produces_type: SymbolId,
    /// "semantic:consumes-type" — module consumes (takes as parameter) a type
    pub consumes_type: SymbolId,
    /// "semantic:flows-to" — data flows from module A to module B
    pub flows_to: SymbolId,
    /// "semantic:flow-via-type" — the type that mediates a data flow edge
    pub flow_via_type: SymbolId,
}

impl SemanticPredicates {
    /// Resolve or create all well-known semantic predicates in the engine.
    pub fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            has_role: engine.resolve_or_create_relation("semantic:has-role")?,
            role_confidence: engine.resolve_or_create_relation("semantic:role-confidence")?,
            importance: engine.resolve_or_create_relation("semantic:importance")?,
            produces_type: engine.resolve_or_create_relation("semantic:produces-type")?,
            consumes_type: engine.resolve_or_create_relation("semantic:consumes-type")?,
            flows_to: engine.resolve_or_create_relation("semantic:flows-to")?,
            flow_via_type: engine.resolve_or_create_relation("semantic:flow-via-type")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Role archetypes
// ---------------------------------------------------------------------------

/// Role archetype definitions: (role_name, concept_keywords).
///
/// Each archetype is encoded as a VSA bundle of grounded concept words.
const ROLE_ARCHETYPES: &[(&str, &[&str])] = &[
    (
        "storage",
        &[
            "store", "memory", "index", "search", "insert", "get", "cache", "persist", "retrieve",
            "database",
        ],
    ),
    (
        "transformation",
        &[
            "encode",
            "transform",
            "convert",
            "map",
            "parse",
            "serialize",
            "format",
            "decode",
        ],
    ),
    (
        "computation",
        &[
            "calculate",
            "compute",
            "operate",
            "process",
            "evaluate",
            "execute",
            "run",
            "apply",
            "reduce",
        ],
    ),
    (
        "coordination",
        &[
            "dispatch",
            "registry",
            "route",
            "manage",
            "orchestrate",
            "coordinate",
            "pipeline",
            "schedule",
        ],
    ),
    (
        "analysis",
        &[
            "analyze", "inspect", "measure", "compare", "score", "rank", "detect", "diagnose",
            "gap",
        ],
    ),
    (
        "interface",
        &[
            "api", "client", "request", "response", "handler", "endpoint", "command", "cli", "repl",
        ],
    ),
];

/// A built archetype vector with its role name.
struct ArchetypeVec {
    role: String,
    vec: HyperVec,
}

/// Build archetype vectors from the KG. Returns None for archetypes that
/// fail to encode (e.g., no grounded symbols).
fn build_role_archetypes(engine: &Engine) -> Vec<ArchetypeVec> {
    let ops = engine.ops();
    let im = engine.item_memory();

    ROLE_ARCHETYPES
        .iter()
        .filter_map(|(role, concepts)| {
            bundle_symbols(engine, ops, im, concepts)
                .ok()
                .map(|vec| ArchetypeVec {
                    role: role.to_string(),
                    vec,
                })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Module profiles
// ---------------------------------------------------------------------------

/// A module's KG profile: its identity and a VSA vector of its children.
pub struct ModuleProfile {
    pub symbol: SymbolId,
    pub name: String,
    pub profile_vec: HyperVec,
    pub child_labels: Vec<String>,
}

/// Build module profiles from code structure triples in the KG.
///
/// For each entity with outgoing `code:contains-mod`, `code:defines-fn`,
/// `code:defines-struct`, etc. triples, bundle all child labels into a
/// profile vector.
fn build_module_profiles(engine: &Engine, code_preds: &CodePredicates) -> Vec<ModuleProfile> {
    let ops = engine.ops();
    let im = engine.item_memory();

    let child_predicates = [
        code_preds.contains_mod,
        code_preds.defines_fn,
        code_preds.defines_struct,
        code_preds.defines_enum,
        code_preds.defines_trait,
        code_preds.has_method,
    ];

    let all_symbols = engine.all_symbols();
    let mut profiles = Vec::new();

    for sym in &all_symbols {
        let triples = engine.triples_from(sym.id);
        let child_labels: Vec<String> = triples
            .iter()
            .filter(|t| child_predicates.contains(&t.predicate))
            .map(|t| engine.resolve_label(t.object))
            .filter(|label| !super::synthesize::is_metadata_label(label))
            .collect();

        if child_labels.is_empty() {
            continue;
        }

        // Bundle child labels into a profile vector.
        let label_refs: Vec<&str> = child_labels.iter().map(|s| s.as_str()).collect();
        let profile_vec = match bundle_symbols(engine, ops, im, &label_refs) {
            Ok(v) => v,
            Err(_) => continue,
        };

        profiles.push(ModuleProfile {
            symbol: sym.id,
            name: sym.label.clone(),
            profile_vec,
            child_labels,
        });
    }

    profiles
}

// ---------------------------------------------------------------------------
// Role classification
// ---------------------------------------------------------------------------

/// Result of classifying a module's role.
#[derive(Debug, Clone)]
pub struct RoleClassification {
    pub primary: String,
    pub primary_confidence: f32,
    pub secondary: Option<String>,
    pub secondary_confidence: Option<f32>,
}

/// Classify a module's role by comparing its profile vector against archetypes.
fn classify_role(
    profile: &ModuleProfile,
    archetypes: &[ArchetypeVec],
    ops: &crate::vsa::ops::VsaOps,
) -> Option<RoleClassification> {
    if archetypes.is_empty() {
        return None;
    }

    let mut scores: Vec<(String, f32)> = archetypes
        .iter()
        .filter_map(|a| {
            ops.similarity(&profile.profile_vec, &a.vec)
                .ok()
                .map(|s| (a.role.clone(), s))
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (primary, primary_confidence) = scores.first()?;

    // Runner-up within 0.03 of primary becomes secondary.
    let secondary = scores.get(1).and_then(|(role, score)| {
        if (primary_confidence - score).abs() < 0.03 {
            Some((role.clone(), *score))
        } else {
            None
        }
    });

    Some(RoleClassification {
        primary: primary.clone(),
        primary_confidence: *primary_confidence,
        secondary: secondary.as_ref().map(|(r, _)| r.clone()),
        secondary_confidence: secondary.map(|(_, s)| s),
    })
}

// ---------------------------------------------------------------------------
// Phase 1: Classify and persist roles
// ---------------------------------------------------------------------------

/// Classify all module entities and persist role triples to the KG.
///
/// Returns the number of modules enriched.
pub fn classify_and_persist_roles(
    engine: &Engine,
    predicates: &SemanticPredicates,
) -> AgentResult<usize> {
    let code_preds = CodePredicates::init(engine)?;
    let archetypes = build_role_archetypes(engine);

    if archetypes.is_empty() {
        return Ok(0);
    }

    let ops = engine.ops();
    let profiles = build_module_profiles(engine, &code_preds);
    let mut enriched = 0;

    for profile in &profiles {
        let classification = match classify_role(profile, &archetypes, ops) {
            Some(c) => c,
            None => continue,
        };

        // Create role entity and add has-role triple.
        let role_sym =
            engine.resolve_or_create_entity(&format!("role:{}", classification.primary))?;
        let triple = Triple::new(profile.symbol, predicates.has_role, role_sym)
            .with_confidence(classification.primary_confidence);
        engine.add_triple(&triple)?;

        // Provenance record.
        let mut prov = ProvenanceRecord::new(
            profile.symbol,
            DerivationKind::SemanticEnrichment {
                source: "role_archetype".into(),
            },
        )
        .with_confidence(classification.primary_confidence);
        let _ = engine.store_provenance(&mut prov);

        // Secondary role, if present.
        if let (Some(sec_role), Some(sec_conf)) = (
            &classification.secondary,
            classification.secondary_confidence,
        ) {
            let sec_sym = engine.resolve_or_create_entity(&format!("role:{sec_role}"))?;
            let sec_triple =
                Triple::new(profile.symbol, predicates.has_role, sec_sym).with_confidence(sec_conf);
            engine.add_triple(&sec_triple)?;
        }

        enriched += 1;
    }

    Ok(enriched)
}

// ---------------------------------------------------------------------------
// Phase 2: Importance ranking
// ---------------------------------------------------------------------------

/// Compute PageRank importance and persist as `semantic:importance` triples.
///
/// Returns the number of entities enriched.
pub fn compute_and_persist_importance(
    engine: &Engine,
    predicates: &SemanticPredicates,
) -> AgentResult<usize> {
    let scores = engine.pagerank(0.85, 20)?;

    if scores.is_empty() {
        return Ok(0);
    }

    let max_score = scores.first().map(|s| s.score).unwrap_or(1.0);
    if max_score <= 0.0 {
        return Ok(0);
    }

    let mut enriched = 0;

    for score in &scores {
        let normalized = (score.score / max_score) as f32;
        let imp_sym = engine.resolve_or_create_entity(&format!("importance:{:.2}", normalized))?;
        let triple = Triple::new(score.symbol, predicates.importance, imp_sym);
        engine.add_triple(&triple)?;

        let mut prov = ProvenanceRecord::new(
            score.symbol,
            DerivationKind::SemanticEnrichment {
                source: "pagerank".into(),
            },
        )
        .with_confidence(normalized);
        let _ = engine.store_provenance(&mut prov);

        enriched += 1;
    }

    Ok(enriched)
}

// ---------------------------------------------------------------------------
// Phase 3: Data flow detection
// ---------------------------------------------------------------------------

/// Collected I/O types for a module.
struct ModuleIO {
    symbol: SymbolId,
    #[allow(dead_code)]
    name: String,
    produces: HashSet<String>,
    consumes: HashSet<String>,
}

/// Detect data flows between modules via shared types and persist as triples.
///
/// Returns the number of flow edges persisted.
pub fn detect_and_persist_flows(
    engine: &Engine,
    predicates: &SemanticPredicates,
) -> AgentResult<usize> {
    let code_preds = CodePredicates::init(engine)?;

    let all_symbols = engine.all_symbols();
    let mut module_ios: Vec<ModuleIO> = Vec::new();

    // For each module, collect produce/consume types from its children.
    for sym in &all_symbols {
        let triples = engine.triples_from(sym.id);

        // Only process entities that have code:contains-mod or code:defines-fn
        let has_children = triples.iter().any(|t| {
            t.predicate == code_preds.contains_mod
                || t.predicate == code_preds.defines_fn
                || t.predicate == code_preds.defines_struct
        });
        if !has_children {
            continue;
        }

        let mut produces = HashSet::new();
        let mut consumes = HashSet::new();

        // Collect types from child entities.
        for child_triple in &triples {
            let child_triples = engine.triples_from(child_triple.object);
            for ct in &child_triples {
                let type_label = engine.resolve_label(ct.object);
                if ct.predicate == code_preds.returns_type {
                    produces.insert(type_label);
                } else if ct.predicate == code_preds.has_param {
                    // Parameters are often "name:Type" — extract type part.
                    let param_label = engine.resolve_label(ct.object);
                    if let Some((_name, type_part)) = param_label.rsplit_once(':') {
                        consumes.insert(type_part.trim().to_string());
                    } else {
                        consumes.insert(param_label);
                    }
                }
            }
        }

        // Also check types defined by this module (produces).
        for t in &triples {
            if t.predicate == code_preds.defines_struct || t.predicate == code_preds.defines_enum {
                produces.insert(engine.resolve_label(t.object));
            }
        }

        if !produces.is_empty() || !consumes.is_empty() {
            module_ios.push(ModuleIO {
                symbol: sym.id,
                name: sym.label.clone(),
                produces,
                consumes,
            });
        }
    }

    // Match producers to consumers via shared type names.
    let mut flow_count = 0;

    for i in 0..module_ios.len() {
        for j in 0..module_ios.len() {
            if i == j {
                continue;
            }

            let shared: HashSet<_> = module_ios[i]
                .produces
                .intersection(&module_ios[j].consumes)
                .cloned()
                .collect();

            if shared.is_empty() {
                continue;
            }

            // Persist flow edge: producer -> flows-to -> consumer.
            let confidence =
                (shared.len() as f32 / module_ios[i].produces.len().max(1) as f32).min(1.0);
            let flow_triple = Triple::new(
                module_ios[i].symbol,
                predicates.flows_to,
                module_ios[j].symbol,
            )
            .with_confidence(confidence);
            engine.add_triple(&flow_triple)?;

            // Persist the mediating types.
            for type_name in &shared {
                if let Ok(type_sym) = engine.resolve_or_create_entity(type_name) {
                    let via_triple =
                        Triple::new(module_ios[i].symbol, predicates.flow_via_type, type_sym);
                    engine.add_triple(&via_triple)?;
                }
            }

            let mut prov = ProvenanceRecord::new(
                module_ios[i].symbol,
                DerivationKind::SemanticEnrichment {
                    source: "data_flow".into(),
                },
            )
            .with_confidence(confidence);
            let _ = engine.store_provenance(&mut prov);

            flow_count += 1;
        }
    }

    Ok(flow_count)
}

// ---------------------------------------------------------------------------
// Top-level enrichment pipeline
// ---------------------------------------------------------------------------

/// Result of running the full enrichment pipeline.
#[derive(Debug)]
pub struct EnrichmentResult {
    pub roles_enriched: usize,
    pub importance_enriched: usize,
    pub flows_detected: usize,
}

/// Run the full semantic enrichment pipeline:
/// 1. Classify roles
/// 2. Compute importance
/// 3. Detect data flows
pub fn enrich(engine: &Engine) -> AgentResult<EnrichmentResult> {
    let predicates = SemanticPredicates::init(engine)?;

    let roles_enriched = classify_and_persist_roles(engine, &predicates)?;
    let importance_enriched = compute_and_persist_importance(engine, &predicates)?;
    let flows_detected = detect_and_persist_flows(engine, &predicates)?;

    Ok(EnrichmentResult {
        roles_enriched,
        importance_enriched,
        flows_detected,
    })
}

// ---------------------------------------------------------------------------
// Query helpers (used by synthesis and OODA)
// ---------------------------------------------------------------------------

/// Look up the role label for a symbol from `semantic:has-role` triples.
///
/// Returns the highest-confidence role label, or None if not enriched.
pub fn lookup_role(
    engine: &Engine,
    symbol: SymbolId,
    predicates: &SemanticPredicates,
) -> Option<String> {
    let triples = engine.triples_from(symbol);
    triples
        .iter()
        .filter(|t| t.predicate == predicates.has_role)
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|t| {
            engine
                .resolve_label(t.object)
                .trim_start_matches("role:")
                .to_string()
        })
}

/// Look up the normalized importance score for a symbol.
///
/// Returns the importance as f32 in [0.0, 1.0], or None if not enriched.
pub fn lookup_importance(
    engine: &Engine,
    symbol: SymbolId,
    predicates: &SemanticPredicates,
) -> Option<f32> {
    let triples = engine.triples_from(symbol);
    triples
        .iter()
        .find(|t| t.predicate == predicates.importance)
        .and_then(|t| {
            engine
                .resolve_label(t.object)
                .trim_start_matches("importance:")
                .parse::<f32>()
                .ok()
        })
}

/// Look up flow-to edges for a symbol.
///
/// Returns (target_symbol, confidence) pairs.
pub fn lookup_flows(
    engine: &Engine,
    symbol: SymbolId,
    predicates: &SemanticPredicates,
) -> Vec<(SymbolId, f32)> {
    let triples = engine.triples_from(symbol);
    triples
        .iter()
        .filter(|t| t.predicate == predicates.flows_to)
        .map(|t| (t.object, t.confidence))
        .collect()
}

/// Collect all flow-via-type types for a symbol.
pub fn lookup_flow_types(
    engine: &Engine,
    symbol: SymbolId,
    predicates: &SemanticPredicates,
) -> Vec<String> {
    let triples = engine.triples_from(symbol);
    triples
        .iter()
        .filter(|t| t.predicate == predicates.flow_via_type)
        .map(|t| engine.resolve_label(t.object))
        .collect()
}

/// Collect all enriched modules with their importance scores, sorted descending.
///
/// Used by OODA to prefer high-importance unexplored children.
pub fn importance_ranking(
    engine: &Engine,
    predicates: &SemanticPredicates,
) -> Vec<(SymbolId, f32)> {
    let all_symbols = engine.all_symbols();
    let mut ranked: Vec<(SymbolId, f32)> = all_symbols
        .iter()
        .filter_map(|sym| lookup_importance(engine, sym.id, predicates).map(|imp| (sym.id, imp)))
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// Build a data flow chain from flow-to triples for rendering.
///
/// Attempts a topological sort from sources (no incoming flows) to sinks.
/// Returns a sequence of (module_name, via_type) pairs. Falls back to empty
/// if no flow triples exist.
pub fn build_flow_chain(
    engine: &Engine,
    predicates: &SemanticPredicates,
    module_symbols: &[SymbolId],
) -> Vec<(String, Option<String>)> {
    // Build adjacency: source -> [(target, via_types)]
    let module_set: HashSet<SymbolId> = module_symbols.iter().cloned().collect();
    let mut edges: HashMap<SymbolId, Vec<(SymbolId, Vec<String>)>> = HashMap::new();
    let mut has_incoming: HashSet<SymbolId> = HashSet::new();

    for &sym in module_symbols {
        let flows = lookup_flows(engine, sym, predicates);
        let via_types = lookup_flow_types(engine, sym, predicates);
        for (target, _conf) in &flows {
            if module_set.contains(target) {
                edges
                    .entry(sym)
                    .or_default()
                    .push((*target, via_types.clone()));
                has_incoming.insert(*target);
            }
        }
    }

    if edges.is_empty() {
        return Vec::new();
    }

    // Start from sources (no incoming edges).
    let sources: Vec<SymbolId> = module_symbols
        .iter()
        .filter(|s| !has_incoming.contains(s))
        .copied()
        .collect();

    let mut chain = Vec::new();
    let mut visited = HashSet::new();
    let mut queue: Vec<SymbolId> = if sources.is_empty() {
        // Cycle — just pick first
        vec![module_symbols[0]]
    } else {
        sources
    };

    while let Some(current) = queue.first().copied() {
        queue.remove(0);
        if !visited.insert(current) {
            continue;
        }
        let name = engine.resolve_label(current);
        let via_type = if let Some(targets) = edges.get(&current) {
            // Pick the first via_type for the chain display.
            targets
                .first()
                .and_then(|(_, types)| types.first().cloned())
        } else {
            None
        };
        chain.push((name, via_type));

        if let Some(targets) = edges.get(&current) {
            for (target, _) in targets {
                if !visited.contains(target) {
                    queue.push(*target);
                }
            }
        }
    }

    chain
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn semantic_predicates_init() {
        let engine = test_engine();
        let preds = SemanticPredicates::init(&engine).unwrap();

        // All predicates should be distinct.
        let ids = [
            preds.has_role,
            preds.role_confidence,
            preds.importance,
            preds.produces_type,
            preds.consumes_type,
            preds.flows_to,
            preds.flow_via_type,
        ];
        let unique: HashSet<_> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "all semantic predicates must be unique"
        );
    }

    #[test]
    fn semantic_predicates_idempotent() {
        let engine = test_engine();
        let first = SemanticPredicates::init(&engine).unwrap();
        let second = SemanticPredicates::init(&engine).unwrap();
        assert_eq!(first.has_role, second.has_role);
        assert_eq!(first.flows_to, second.flows_to);
    }

    #[test]
    fn role_archetypes_are_distinct() {
        let engine = test_engine();
        let archetypes = build_role_archetypes(&engine);

        // Should build at least some archetypes.
        assert!(!archetypes.is_empty(), "should build archetype vectors");

        // Pairwise similarity should be < 0.6.
        let ops = engine.ops();
        for i in 0..archetypes.len() {
            for j in (i + 1)..archetypes.len() {
                let sim = ops
                    .similarity(&archetypes[i].vec, &archetypes[j].vec)
                    .unwrap();
                assert!(
                    sim < 0.6,
                    "archetypes '{}' and '{}' too similar: {sim:.3}",
                    archetypes[i].role,
                    archetypes[j].role,
                );
            }
        }
    }

    #[test]
    fn enrich_empty_kg() {
        let engine = test_engine();
        let result = enrich(&engine).unwrap();
        assert_eq!(result.roles_enriched, 0);
        assert_eq!(result.importance_enriched, 0);
        assert_eq!(result.flows_detected, 0);
    }

    #[test]
    fn enrich_with_code_triples() {
        let engine = test_engine();

        // Set up code predicates.
        let code_preds = CodePredicates::init(&engine).unwrap();

        // Create a module with submodules.
        let vsa = engine.resolve_or_create_entity("Vsa").unwrap();
        let encode = engine.resolve_or_create_entity("encode").unwrap();
        let item_memory = engine.resolve_or_create_entity("item_memory").unwrap();

        // Module structure.
        engine
            .add_triple(&Triple::new(vsa, code_preds.contains_mod, encode))
            .unwrap();
        engine
            .add_triple(&Triple::new(vsa, code_preds.contains_mod, item_memory))
            .unwrap();

        // Functions.
        let encode_fn = engine.resolve_or_create_entity("encode_symbol").unwrap();
        engine
            .add_triple(&Triple::new(encode, code_preds.defines_fn, encode_fn))
            .unwrap();

        // Types.
        let hypervec = engine.resolve_or_create_entity("HyperVec").unwrap();
        engine
            .add_triple(&Triple::new(encode, code_preds.defines_struct, hypervec))
            .unwrap();

        // Run enrichment.
        let result = enrich(&engine).unwrap();

        // Should have enriched at least the Vsa module.
        assert!(
            result.roles_enriched > 0,
            "should classify at least one role"
        );
        assert!(result.importance_enriched > 0, "should compute importance");

        // Verify role triple exists.
        let sem_preds = SemanticPredicates::init(&engine).unwrap();
        let role = lookup_role(&engine, vsa, &sem_preds);
        assert!(role.is_some(), "Vsa should have a role");

        // Verify importance triple exists.
        let imp = lookup_importance(&engine, vsa, &sem_preds);
        assert!(imp.is_some(), "Vsa should have importance");
    }

    #[test]
    fn provenance_created_for_enrichment() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(tempfile::TempDir::new().unwrap().keep()),
            ..Default::default()
        })
        .unwrap();

        let code_preds = CodePredicates::init(&engine).unwrap();
        let vsa = engine.resolve_or_create_entity("Vsa").unwrap();
        let encode = engine.resolve_or_create_entity("encode").unwrap();
        engine
            .add_triple(&Triple::new(vsa, code_preds.contains_mod, encode))
            .unwrap();

        let sem_preds = SemanticPredicates::init(&engine).unwrap();
        let _ = classify_and_persist_roles(&engine, &sem_preds).unwrap();

        // Check provenance was created.
        let prov_records = engine.provenance_of(vsa);
        assert!(
            prov_records.is_ok(),
            "should have provenance records for enrichment",
        );
    }

    #[test]
    fn lookup_helpers_return_none_without_enrichment() {
        let engine = test_engine();
        let sem_preds = SemanticPredicates::init(&engine).unwrap();
        let sym = engine.resolve_or_create_entity("test").unwrap();

        assert!(lookup_role(&engine, sym, &sem_preds).is_none());
        assert!(lookup_importance(&engine, sym, &sem_preds).is_none());
        assert!(lookup_flows(&engine, sym, &sem_preds).is_empty());
    }
}
