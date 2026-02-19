//! Library learning cycle: wake-sleep abstraction discovery.
//!
//! Periodically analyzes recently generated/ingested code, discovers recurring
//! structural patterns via anti-unification on `SimplifiedAst` trees, and stores
//! them as new `CodeTemplate` entries in the KG. This implements a
//! DreamCoder/LILO-inspired library learning loop.
//!
//! ## Workflow
//!
//! 1. **Collect**: Query KG for recent code entities via `CodeGenerated`/`SchemaDiscovered` provenance
//! 2. **Reconstruct**: Build `SimplifiedAst` skeletons from KG `code:*` triples
//! 3. **Discover**: Group by variant, anti-unify within groups, score abstractions
//! 4. **Store**: Convert top abstractions to `CodeTemplate`, store in KG with provenance

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tools::code_predicates::CodePredicates;
use crate::agent::tools::pattern_mine::{SimplifiedAst, extract_simplified_contexts};
use crate::compartment::ContextDomain;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::reason::anti_unify::{
    AntiUnifyConfig, DiscoveredAbstraction, discover_abstractions, score_abstractions,
};
use crate::symbol::{SymbolId, SymbolKind};
use crate::vsa::code_encode::encode_code_vector;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Summary of a library learning cycle.
#[derive(Debug, Clone)]
pub struct LibraryLearningResult {
    /// Number of candidate patterns found before filtering.
    pub candidates_found: usize,
    /// Number of abstractions that passed scoring and were stored.
    pub abstractions_stored: usize,
    /// Names/fingerprints of the stored templates.
    pub template_names: Vec<String>,
    /// Human-readable summary.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// LibraryLearner
// ---------------------------------------------------------------------------

/// Orchestrates the wake-sleep library learning cycle.
#[derive(Debug, Clone)]
pub struct LibraryLearner {
    /// Anti-unification and scoring configuration.
    pub config: AntiUnifyConfig,
}

impl LibraryLearner {
    /// Create a new library learner with the given configuration.
    pub fn new(config: AntiUnifyConfig) -> Self {
        Self { config }
    }

    /// Create a library learner with default configuration.
    pub fn with_defaults() -> Self {
        Self {
            config: AntiUnifyConfig::default(),
        }
    }

    /// Collect `SimplifiedAst` skeletons from recent code entities in the KG.
    ///
    /// Looks at symbols derived via `CodeGenerated` and `SchemaDiscovered` provenance,
    /// then reconstructs `SimplifiedAst` from their `code:*` triples.
    fn collect_code_asts(&self, engine: &Engine) -> AgentResult<Vec<SimplifiedAst>> {
        let code_preds = CodePredicates::init(engine)?;
        let mut asts = Vec::new();

        // Collect symbols from CodeGenerated provenance
        let generated = engine
            .provenance_by_kind(&DerivationKind::CodeGenerated {
                scope: String::new(),
                source_count: 0,
            })
            .unwrap_or_default();

        // Collect symbols from SchemaDiscovered provenance (from code_ingest / pattern_mine)
        let discovered = engine
            .provenance_by_kind(&DerivationKind::SchemaDiscovered {
                pattern_type: String::new(),
            })
            .unwrap_or_default();

        let all_symbols: Vec<SymbolId> = generated
            .iter()
            .chain(discovered.iter())
            .map(|r| r.derived_id)
            .collect();

        // Reconstruct SimplifiedAst from KG triples
        for sym_id in all_symbols {
            if let Some(ast) = reconstruct_ast(engine, sym_id, &code_preds) {
                asts.push(ast);
            }
        }

        Ok(asts)
    }

    /// Discover abstractions from the collected code ASTs.
    pub fn discover(&self, engine: &Engine) -> AgentResult<Vec<DiscoveredAbstraction>> {
        let asts = self.collect_code_asts(engine)?;
        if asts.len() < 2 {
            return Ok(Vec::new());
        }

        let candidates = discover_abstractions(&asts);
        let scored = score_abstractions(&candidates, &self.config);
        Ok(scored)
    }

    /// Store discovered abstractions as KG entities with template triples and provenance.
    ///
    /// Returns the SymbolIds of the created template entities.
    pub fn store_as_templates(
        &self,
        engine: &Engine,
        abstractions: &[DiscoveredAbstraction],
    ) -> AgentResult<Vec<SymbolId>> {
        if abstractions.is_empty() {
            return Ok(Vec::new());
        }

        // Ensure template predicates exist
        let template_preds = TemplatePredicates::init(engine)?;
        let _mt = ensure_patterns_microtheory(engine)?;
        let compartment = "mt:patterns";

        let mut created = Vec::new();

        for abs in abstractions {
            let label = format!("learned-template:{}", abs.fingerprint);

            // Create (or reuse) the symbol
            let sym = engine
                .create_symbol(SymbolKind::Entity, &label)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "library_learn".into(),
                    message: format!("failed to create template symbol: {e}"),
                })?;

            // Store template metadata triples
            let fp_val = engine
                .create_symbol(SymbolKind::Entity, format!("fp:{}", abs.fingerprint))
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "library_learn".into(),
                    message: format!("failed to create fingerprint value: {e}"),
                })?;
            let cat_val = engine
                .create_symbol(SymbolKind::Entity, format!("cat:{}", abs.category))
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "library_learn".into(),
                    message: format!("failed to create category value: {e}"),
                })?;
            let occ_val = engine
                .create_symbol(
                    SymbolKind::Entity,
                    format!("val:{}", abs.occurrences),
                )
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "library_learn".into(),
                    message: format!("failed to create occurrence value: {e}"),
                })?;

            let triples = [
                Triple::new(sym.id, template_preds.structure, fp_val.id),
                Triple::new(sym.id, template_preds.category, cat_val.id),
                Triple::new(sym.id, template_preds.occurrences, occ_val.id),
            ];

            for mut triple in triples {
                triple.compartment_id = Some(compartment.to_string());
                engine.add_triple(&triple).map_err(|e| AgentError::ToolExecution {
                    tool_name: "library_learn".into(),
                    message: format!("failed to add template triple: {e}"),
                })?;
            }

            // Encode as VSA vector for similarity retrieval
            let approx = crate::reason::anti_unify::approximate_simplified_pub(&abs.pattern);
            let contexts = extract_simplified_contexts(&approx, &abs.fingerprint);
            if !contexts.is_empty() {
                let ops = engine.ops();
                if let Ok(vec) = encode_code_vector(ops, &contexts) {
                    engine.item_memory().insert(sym.id, vec);
                }
            }

            // Record provenance
            let mut prov = ProvenanceRecord::new(
                sym.id,
                DerivationKind::LibraryLearning {
                    pattern_name: abs.fingerprint.clone(),
                    occurrences: abs.occurrences,
                    compression: abs.compression,
                },
            );
            let _ = engine.store_provenance(&mut prov);

            created.push(sym.id);
        }

        Ok(created)
    }

    /// Run a full library learning cycle: discover → score → store.
    pub fn run_cycle(&self, engine: &Engine) -> AgentResult<LibraryLearningResult> {
        let abstractions = self.discover(engine)?;
        let candidates_found = abstractions.len();

        let stored_ids = self.store_as_templates(engine, &abstractions)?;
        let abstractions_stored = stored_ids.len();

        let template_names: Vec<String> = abstractions
            .iter()
            .take(abstractions_stored)
            .map(|a| a.fingerprint.clone())
            .collect();

        let summary = if abstractions_stored == 0 {
            "Library learning: no new abstractions discovered.".to_string()
        } else {
            format!(
                "Library learning: discovered {} candidates, stored {} templates: {}",
                candidates_found,
                abstractions_stored,
                template_names.join(", "),
            )
        };

        Ok(LibraryLearningResult {
            candidates_found,
            abstractions_stored,
            template_names,
            summary,
        })
    }
}

// ---------------------------------------------------------------------------
// AST reconstruction from KG triples
// ---------------------------------------------------------------------------

/// Reconstruct a `SimplifiedAst` skeleton from the `code:*` triples of a symbol.
///
/// Uses the outgoing triples to infer the structure: `code:has-param` count,
/// `code:has-field` count, `code:has-variant` count, `code:has-method` count,
/// `code:implements-trait`, `code:returns-type`, and `code:derives-trait`.
fn reconstruct_ast(
    engine: &Engine,
    sym_id: SymbolId,
    preds: &CodePredicates,
) -> Option<SimplifiedAst> {
    let triples = engine.triples_from(sym_id);
    if triples.is_empty() {
        return None;
    }

    let param_count = triples.iter().filter(|t| t.predicate == preds.has_param).count();
    let field_count = triples.iter().filter(|t| t.predicate == preds.has_field).count();
    let variant_count = triples.iter().filter(|t| t.predicate == preds.has_variant).count();
    let method_count = triples.iter().filter(|t| t.predicate == preds.has_method).count();
    let has_return = triples.iter().any(|t| t.predicate == preds.returns_type);
    let derive_count = triples.iter().filter(|t| t.predicate == preds.derives_trait).count();
    let defines_fn = triples.iter().any(|t| t.predicate == preds.defines_fn);
    let defines_struct = triples.iter().any(|t| t.predicate == preds.defines_struct);
    let defines_enum = triples.iter().any(|t| t.predicate == preds.defines_enum);

    // Infer the trait name from implements-trait triples
    let trait_name = triples
        .iter()
        .find(|t| t.predicate == preds.implements_trait)
        .and_then(|t| engine.get_symbol_meta(t.object).ok())
        .map(|m| m.label);

    // Determine the most likely kind based on triple patterns
    if param_count > 0 || (has_return && !defines_fn && field_count == 0) {
        // Looks like a function
        Some(SimplifiedAst::Function {
            param_count,
            has_return,
            body: vec![],
        })
    } else if defines_fn || method_count > 0 {
        // Looks like a module or impl block
        if trait_name.is_some() || method_count > 0 {
            Some(SimplifiedAst::Impl {
                method_count,
                trait_name,
            })
        } else {
            // Module-like container
            Some(SimplifiedAst::Block {
                children: vec![],
            })
        }
    } else if defines_struct || field_count > 0 {
        Some(SimplifiedAst::Struct {
            field_count,
            derive_count,
        })
    } else if defines_enum || variant_count > 0 {
        Some(SimplifiedAst::Enum {
            variant_count,
            derive_count,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Template predicates
// ---------------------------------------------------------------------------

/// Well-known predicates for learned template storage.
#[derive(Debug, Clone)]
struct TemplatePredicates {
    /// "template:structure" — the pattern fingerprint
    structure: SymbolId,
    /// "template:category" — inferred category
    category: SymbolId,
    /// "template:occurrences" — how often the pattern appeared
    occurrences: SymbolId,
}

impl TemplatePredicates {
    fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            structure: engine.resolve_or_create_relation("template:structure")?,
            category: engine.resolve_or_create_relation("template:category")?,
            occurrences: engine.resolve_or_create_relation("template:occurrences")?,
        })
    }
}

/// Ensure the `mt:patterns` microtheory exists.
fn ensure_patterns_microtheory(engine: &Engine) -> AgentResult<SymbolId> {
    let label = "mt:patterns";
    if let Ok(id) = engine.lookup_symbol(label) {
        return Ok(id);
    }

    // Ensure base code mt exists first
    let base_label = "mt:rust-code";
    let base_mt = if let Ok(id) = engine.lookup_symbol(base_label) {
        id
    } else {
        let mt = engine
            .create_context(base_label, ContextDomain::Code, &[])
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "library_learn".into(),
                message: format!("failed to create base code microtheory: {e}"),
            })?;
        mt.id
    };

    let mt = engine
        .create_context(label, ContextDomain::Code, &[base_mt])
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "library_learn".into(),
            message: format!("failed to create patterns microtheory: {e}"),
        })?;
    Ok(mt.id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::reason::anti_unify::{GeneralizedAst, AstSlot, AntiUnifyVar, generalized_fingerprint};
    use crate::vsa::Dimension;

    /// Create an engine with a temp data dir so provenance is available.
    fn test_engine_persistent() -> (Engine, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        (engine, dir)
    }

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    /// Seed the KG with code entities that have CodeGenerated provenance.
    fn seed_code_entities(engine: &Engine) -> Vec<SymbolId> {
        let preds = CodePredicates::init(engine).unwrap();
        let mut ids = Vec::new();

        // Create 5 function-like entities with varying param counts
        for i in 0..5 {
            let sym = engine
                .create_symbol(SymbolKind::Entity, format!("code:fn_{i}"))
                .unwrap();

            // Add has-param triples (1 param each)
            let param = engine
                .create_symbol(SymbolKind::Entity, format!("param:p{i}"))
                .unwrap();
            engine
                .add_triple(&Triple::new(sym.id, preds.has_param, param.id))
                .unwrap();

            // Add returns-type triple
            let ret_type = engine
                .create_symbol(SymbolKind::Entity, format!("type:ret{i}"))
                .unwrap();
            engine
                .add_triple(&Triple::new(sym.id, preds.returns_type, ret_type.id))
                .unwrap();

            // Record provenance
            let mut prov = ProvenanceRecord::new(
                sym.id,
                DerivationKind::CodeGenerated {
                    scope: "function".into(),
                    source_count: 1,
                },
            );
            let _ = engine.store_provenance(&mut prov);

            ids.push(sym.id);
        }

        ids
    }

    #[test]
    fn discover_from_similar_functions() {
        let (engine, _dir) = test_engine_persistent();
        let _ids = seed_code_entities(&engine);

        let learner = LibraryLearner::new(AntiUnifyConfig {
            min_occurrences: 2,
            min_compression: 1.0,
            max_abstractions: 5,
            max_holes: 4,
        });

        let abstractions = learner.discover(&engine).unwrap();
        // All 5 are fn(1,ret) → should produce at least 1 abstraction
        assert!(
            !abstractions.is_empty(),
            "should discover abstractions from 5 similar functions"
        );
    }

    #[test]
    fn store_as_template() {
        let (engine, _dir) = test_engine_persistent();
        let _ids = seed_code_entities(&engine);

        let learner = LibraryLearner::new(AntiUnifyConfig {
            min_occurrences: 2,
            min_compression: 1.0,
            max_abstractions: 5,
            max_holes: 4,
        });

        let abstractions = learner.discover(&engine).unwrap();
        assert!(!abstractions.is_empty());

        let stored = learner.store_as_templates(&engine, &abstractions).unwrap();
        assert!(!stored.is_empty());

        // Verify the symbol exists in the KG
        for sym_id in &stored {
            let meta = engine.get_symbol_meta(*sym_id).unwrap();
            assert!(
                meta.label.starts_with("learned-template:"),
                "symbol should have learned-template label: {}",
                meta.label
            );
        }
    }

    #[test]
    fn run_cycle_integration() {
        let (engine, _dir) = test_engine_persistent();
        let _ids = seed_code_entities(&engine);

        let learner = LibraryLearner::new(AntiUnifyConfig {
            min_occurrences: 2,
            min_compression: 1.0,
            max_abstractions: 5,
            max_holes: 4,
        });

        let result = learner.run_cycle(&engine).unwrap();
        assert!(
            result.candidates_found > 0,
            "cycle should discover candidates from 5 seeded functions: {result:?}"
        );
    }

    #[test]
    fn learned_template_instantiation() {
        use crate::grammar::templates::CodeTemplate;

        // Create a DiscoveredAbstraction that looks like a function pattern
        let abs = DiscoveredAbstraction {
            pattern: GeneralizedAst::Function {
                param_count: AstSlot::Var(AntiUnifyVar {
                    name: "?v0".into(),
                    instances: vec!["1".into(), "2".into()],
                }),
                has_return: AstSlot::Bool(true),
                body: vec![],
            },
            fingerprint: "fn(?v0,ret)".into(),
            holes: 1,
            occurrences: 5,
            nodes: 1,
            compression: 5.0,
            category: "function".into(),
            example_sources: vec!["fn(1,ret)".into(), "fn(2,ret)".into()],
        };

        let template = CodeTemplate::from_abstraction(&abs);
        assert!(template.is_some(), "should create template from function abstraction");

        let t = template.unwrap();
        assert!(t.name.starts_with("learned:"), "name should start with 'learned:': {}", t.name);
        assert_eq!(t.category, "learned-function");

        // Instantiate with default params
        let code = t.instantiate(&std::collections::HashMap::new()).unwrap();
        assert!(code.contains("fn "), "should generate function scaffold: {code}");
        assert!(code.contains("Learned pattern:"), "should include pattern comment: {code}");
    }

    #[test]
    fn library_learning_provenance() {
        let (engine, _dir) = test_engine_persistent();
        let _ids = seed_code_entities(&engine);

        let learner = LibraryLearner::new(AntiUnifyConfig {
            min_occurrences: 2,
            min_compression: 1.0,
            max_abstractions: 5,
            max_holes: 4,
        });

        let abstractions = learner.discover(&engine).unwrap();
        let stored = learner.store_as_templates(&engine, &abstractions).unwrap();

        if !stored.is_empty() {
            // Check provenance was recorded
            let prov_records = engine.provenance_of(stored[0]).unwrap();
            assert!(
                prov_records
                    .iter()
                    .any(|r| matches!(r.kind, DerivationKind::LibraryLearning { .. })),
                "should have LibraryLearning provenance"
            );
        }
    }

    #[test]
    fn run_cycle_empty_kg() {
        let engine = test_engine();

        let learner = LibraryLearner::with_defaults();
        let result = learner.run_cycle(&engine).unwrap();
        assert_eq!(result.abstractions_stored, 0);
        assert!(result.summary.contains("no new abstractions"));
    }
}
