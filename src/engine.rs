//! Engine facade: top-level API for the akh-medu system.
//!
//! The `Engine` owns all subsystems and provides the public interface
//! for ingesting knowledge, querying, and managing the system.

use std::path::PathBuf;
use std::sync::Arc;

use egg::{AstSize, Extractor, Rewrite, Runner};

use crate::error::{AkhResult, EngineError, ProvenanceError, ReasonError, SymbolError};
use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::{ConcreteGrammar, LinContext, ParseContext};
use crate::grammar::custom::CustomGrammar;
use crate::grammar::entity_resolution::{
    EntityResolver, EquivalenceStats, LearnedEquivalence,
};
use crate::grammar::error::GrammarResult;
use crate::grammar::lexer::Language;
use crate::grammar::parser::{parse_prose, ParseResult};
use crate::grammar::GrammarRegistry;
use crate::graph::analytics;
use crate::export::{ProvenanceExport, SymbolExport, TripleExport};
use crate::graph::index::KnowledgeGraph;
use crate::graph::sparql::SparqlStore;
use crate::graph::traverse::{TraversalConfig, TraversalResult};
use crate::graph::Triple;
use crate::infer::engine::InferEngine;
use crate::infer::{InferenceQuery, InferenceResult};
use crate::pipeline::{Pipeline, PipelineContext, PipelineData, PipelineOutput};
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceLedger, ProvenanceRecord};
use crate::reason::AkhLang;
use crate::registry::SymbolRegistry;
use crate::simd;
use crate::skills::manager::SkillManager;
use crate::skills::SkillInfo;
use crate::store::TieredStore;
use crate::symbol::{AtomicSymbolAllocator, SymbolId, SymbolKind, SymbolMeta};
use crate::vsa::item_memory::{ItemMemory, SearchResult};
use crate::vsa::ops::VsaOps;
use crate::vsa::{Dimension, Encoding, HyperVec};

/// Configuration for the akh-medu engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Hypervector dimension (default: 10,000).
    pub dimension: Dimension,
    /// Encoding scheme.
    pub encoding: Encoding,
    /// Data directory for persistence. `None` for memory-only mode.
    pub data_dir: Option<PathBuf>,
    /// Maximum memory budget in MB for skillpacks.
    pub max_memory_mb: usize,
    /// Maximum expected symbols (capacity hint for item memory).
    pub max_symbols: usize,
    /// Default language for parsing. `Auto` means detect from text.
    pub language: Language,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dimension: Dimension::DEFAULT,
            encoding: Encoding::Bipolar,
            data_dir: None,
            max_memory_mb: 1024,
            max_symbols: 1_000_000,
            language: Language::Auto,
        }
    }
}

/// The akh-medu neuro-symbolic AI engine.
///
/// Owns all subsystems: VSA operations, item memory, knowledge graph,
/// storage tiers, symbol allocator, provenance ledger, and skill manager.
pub struct Engine {
    config: EngineConfig,
    ops: Arc<VsaOps>,
    item_memory: Arc<ItemMemory>,
    knowledge_graph: Arc<KnowledgeGraph>,
    sparql: Option<SparqlStore>,
    store: Arc<TieredStore>,
    symbol_allocator: Arc<AtomicSymbolAllocator>,
    registry: SymbolRegistry,
    provenance_ledger: Option<ProvenanceLedger>,
    skill_manager: Option<SkillManager>,
    grammar_registry: GrammarRegistry,
    entity_resolver: EntityResolver,
    compartment_manager: Option<crate::compartment::CompartmentManager>,
}

impl Engine {
    /// Create a new engine with the given configuration.
    pub fn new(config: EngineConfig) -> AkhResult<Self> {
        if config.dimension.0 == 0 {
            return Err(EngineError::InvalidConfig {
                message: "dimension must be > 0".into(),
            }
            .into());
        }

        let kernel = simd::best_kernel();
        tracing::info!(
            isa = %kernel.isa_level(),
            dim = config.dimension.0,
            encoding = %config.encoding,
            "initializing akh-medu engine"
        );

        let ops = Arc::new(VsaOps::new(kernel, config.dimension, config.encoding));
        let item_memory = Arc::new(ItemMemory::new(
            config.dimension,
            config.encoding,
            config.max_symbols,
        ));
        let knowledge_graph = Arc::new(KnowledgeGraph::new());

        let (store, sparql, provenance_ledger, skill_manager) = if let Some(ref dir) =
            config.data_dir
        {
            std::fs::create_dir_all(dir).map_err(|_| EngineError::DataDir {
                path: dir.display().to_string(),
            })?;
            let store = TieredStore::with_persistence(dir, "symbols")
                .map_err(|e| EngineError::InvalidConfig {
                    message: format!("failed to create tiered store: {e}"),
                })?;
            let sparql_dir = dir.join("oxigraph");
            let sparql = SparqlStore::open(&sparql_dir).map_err(|e| {
                EngineError::InvalidConfig {
                    message: format!("failed to create SPARQL store: {e}"),
                }
            })?;

            // Initialize provenance ledger from the durable store's database.
            let ledger = if let Some(ref durable) = store.durable {
                let db = durable.database_arc();
                match ProvenanceLedger::open(db) {
                    Ok(l) => Some(l),
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to open provenance ledger, running without");
                        None
                    }
                }
            } else {
                None
            };

            // Initialize skill manager.
            let skills_dir = dir.join("skills");
            let skill_mgr = SkillManager::new(skills_dir, config.max_memory_mb);
            // Auto-discover, but don't fail if directory doesn't exist yet.
            if let Err(e) = skill_mgr.discover() {
                tracing::debug!(error = %e, "skill discovery skipped");
            }

            (store, Some(sparql), ledger, Some(skill_mgr))
        } else {
            (TieredStore::memory_only(), None, None, None)
        };

        let store = Arc::new(store);

        // Restore knowledge graph from persistent SPARQL store.
        if let Some(ref sparql) = sparql {
            let restored = sparql.all_triples().unwrap_or_else(|e| {
                tracing::warn!(error = %e, "KG restore from SPARQL skipped");
                vec![]
            });
            if !restored.is_empty() {
                let count = knowledge_graph.bulk_load(&restored).unwrap_or(0);
                tracing::info!(triples = count, "knowledge graph restored from SPARQL");
            }
        }

        // Restore registry and allocator from persistent storage if available.
        let registry = SymbolRegistry::restore(&store).unwrap_or_else(|e| {
            tracing::debug!(error = %e, "registry restore skipped, starting fresh");
            SymbolRegistry::new()
        });

        let symbol_allocator = {
            let restored_next = store
                .get_meta(b"sym_allocator_next")
                .ok()
                .flatten()
                .and_then(|bytes| bincode::deserialize::<u64>(&bytes).ok());
            match restored_next {
                Some(next) => Arc::new(AtomicSymbolAllocator::starting_from(next)),
                None => Arc::new(AtomicSymbolAllocator::new()),
            }
        };

        let grammar_registry = GrammarRegistry::new();

        // Restore learned equivalences from persistent storage.
        let entity_resolver = EntityResolver::load_from_store(&store);

        // Initialize compartment manager if data_dir has a compartments/ subdir.
        let compartment_manager = config.data_dir.as_ref().map(|dir| {
            let compartments_dir = dir.join("compartments");
            let mgr = crate::compartment::CompartmentManager::new(compartments_dir);
            if let Err(e) = mgr.discover() {
                tracing::debug!(error = %e, "compartment discovery skipped");
            }
            mgr
        });

        Ok(Self {
            config,
            ops,
            item_memory,
            knowledge_graph,
            sparql,
            store,
            symbol_allocator,
            registry,
            provenance_ledger,
            skill_manager,
            grammar_registry,
            entity_resolver,
            compartment_manager,
        })
    }

    /// Allocate a new symbol with the given kind and label.
    pub fn create_symbol(&self, kind: SymbolKind, label: impl Into<String>) -> AkhResult<SymbolMeta> {
        let id = self.symbol_allocator.next_id()?;
        let meta = SymbolMeta::new(id, kind, label);

        // Register in the bidirectional registry.
        self.registry.register(meta.clone())?;

        // Store metadata in tiered store.
        let encoded = bincode::serialize(&meta).map_err(|e| {
            crate::error::StoreError::Serialization {
                message: format!("failed to serialize symbol meta: {e}"),
            }
        })?;
        self.store.put(id, encoded);

        // Create hypervector in item memory.
        self.item_memory.get_or_create(&self.ops, id);

        Ok(meta)
    }

    /// Add a triple to the knowledge graph.
    pub fn add_triple(&self, triple: &Triple) -> AkhResult<()> {
        self.knowledge_graph.insert_triple(triple)?;

        // Ensure all symbols in the triple have hypervectors
        self.item_memory.get_or_create(&self.ops, triple.subject);
        self.item_memory.get_or_create(&self.ops, triple.predicate);
        self.item_memory.get_or_create(&self.ops, triple.object);

        // Sync to SPARQL store if persistent
        if let Some(ref sparql) = self.sparql {
            sparql.insert_triple(triple)?;
        }

        Ok(())
    }

    /// Search for similar symbols by hypervector.
    pub fn search_similar(&self, query: &HyperVec, top_k: usize) -> AkhResult<Vec<SearchResult>> {
        Ok(self.item_memory.search(query, top_k)?)
    }

    /// Search for symbols similar to a given symbol.
    pub fn search_similar_to(
        &self,
        symbol: SymbolId,
        top_k: usize,
    ) -> AkhResult<Vec<SearchResult>> {
        let vec = self
            .item_memory
            .get(symbol)
            .ok_or(crate::error::VsaError::HypervectorNotFound {
                symbol_id: symbol.get(),
            })?;
        self.search_similar(&vec, top_k)
    }

    // -----------------------------------------------------------------------
    // Rule management
    // -----------------------------------------------------------------------

    /// Collect all rewrite rules: built-in + skill-provided.
    pub fn all_rules(&self) -> Vec<Rewrite<AkhLang, ()>> {
        let mut rules = crate::reason::builtin_rules();
        if let Some(ref mgr) = self.skill_manager {
            rules.extend(mgr.active_rules());
        }
        rules
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Run spreading-activation inference from the given query.
    ///
    /// Uses all active rules (built-in + skills) and optionally persists
    /// provenance records to the ledger.
    pub fn infer(&self, query: &InferenceQuery) -> AkhResult<InferenceResult> {
        let rules = self.all_rules();
        let infer_engine = InferEngine::new(
            Arc::clone(&self.ops),
            Arc::clone(&self.item_memory),
            Arc::clone(&self.knowledge_graph),
        );
        let mut result = infer_engine.infer_with_rules(query, &rules)?;

        // Persist provenance records if ledger is available.
        if let Some(ref ledger) = self.provenance_ledger {
            if let Err(e) = ledger.store_batch(&mut result.provenance) {
                tracing::warn!(error = %e, "failed to persist provenance records");
            }
        }

        Ok(result)
    }

    /// Analogy inference: "A is to B as C is to ?".
    pub fn infer_analogy(
        &self,
        a: SymbolId,
        b: SymbolId,
        c: SymbolId,
        top_k: usize,
    ) -> AkhResult<Vec<(SymbolId, f32)>> {
        let infer_engine = InferEngine::new(
            Arc::clone(&self.ops),
            Arc::clone(&self.item_memory),
            Arc::clone(&self.knowledge_graph),
        );
        Ok(infer_engine.infer_analogy(a, b, c, top_k)?)
    }

    /// Recover the role-filler for a (subject, predicate) pair.
    pub fn recover_filler(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        top_k: usize,
    ) -> AkhResult<Vec<(SymbolId, f32)>> {
        let infer_engine = InferEngine::new(
            Arc::clone(&self.ops),
            Arc::clone(&self.item_memory),
            Arc::clone(&self.knowledge_graph),
        );
        Ok(infer_engine.recover_filler(subject, predicate, top_k)?)
    }

    // -----------------------------------------------------------------------
    // Provenance
    // -----------------------------------------------------------------------

    /// Store a provenance record in the ledger.
    pub fn store_provenance(&self, record: &mut ProvenanceRecord) -> AkhResult<ProvenanceId> {
        let ledger = self
            .provenance_ledger
            .as_ref()
            .ok_or(ProvenanceError::NoPersistence)?;
        Ok(ledger.store(record)?)
    }

    /// Get a provenance record by ID.
    pub fn get_provenance(&self, id: ProvenanceId) -> AkhResult<ProvenanceRecord> {
        let ledger = self
            .provenance_ledger
            .as_ref()
            .ok_or(ProvenanceError::NoPersistence)?;
        Ok(ledger.get(id)?)
    }

    /// Get all provenance records for a derived symbol.
    pub fn provenance_of(&self, symbol: SymbolId) -> AkhResult<Vec<ProvenanceRecord>> {
        let ledger = self
            .provenance_ledger
            .as_ref()
            .ok_or(ProvenanceError::NoPersistence)?;
        Ok(ledger.by_derived(symbol)?)
    }

    /// Get all provenance records that depend on a given source symbol.
    pub fn dependents_of(&self, symbol: SymbolId) -> AkhResult<Vec<ProvenanceRecord>> {
        let ledger = self
            .provenance_ledger
            .as_ref()
            .ok_or(ProvenanceError::NoPersistence)?;
        Ok(ledger.by_source(symbol)?)
    }

    /// Get all provenance records of a given derivation kind.
    pub fn provenance_by_kind(&self, kind: &DerivationKind) -> AkhResult<Vec<ProvenanceRecord>> {
        let ledger = self
            .provenance_ledger
            .as_ref()
            .ok_or(ProvenanceError::NoPersistence)?;
        Ok(ledger.by_kind(kind)?)
    }

    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------

    /// Run a pipeline with the given initial data.
    pub fn run_pipeline(
        &self,
        pipeline: &Pipeline,
        initial: PipelineData,
    ) -> AkhResult<PipelineOutput> {
        let ctx = PipelineContext {
            ops: Arc::clone(&self.ops),
            item_memory: Arc::clone(&self.item_memory),
            knowledge_graph: Arc::clone(&self.knowledge_graph),
            rules: self.all_rules(),
        };
        Ok(pipeline.run(&ctx, initial)?)
    }

    /// Run the built-in query pipeline with the given seeds.
    pub fn query_pipeline(&self, seeds: Vec<SymbolId>) -> AkhResult<PipelineOutput> {
        let pipeline = Pipeline::query_pipeline();
        self.run_pipeline(&pipeline, PipelineData::Seeds(seeds))
    }

    // -----------------------------------------------------------------------
    // Skills
    // -----------------------------------------------------------------------

    /// Load a skill: discover → warm → activate.
    /// Automatically detects label-based triples and resolves them.
    pub fn load_skill(&self, name: &str) -> AkhResult<crate::skills::SkillActivation> {
        let mgr = self
            .skill_manager
            .as_ref()
            .ok_or(crate::error::SkillError::NotFound {
                name: name.into(),
            })?;

        // Check if the skill has label-based triples.
        let skill_dir = mgr.skills_dir().join(name);
        let triples_path = skill_dir.join("triples.json");
        let has_label_triples = if triples_path.exists() {
            std::fs::read_to_string(&triples_path)
                .ok()
                .and_then(|c| serde_json::from_str::<Vec<serde_json::Value>>(&c).ok())
                .is_some_and(|v| v.first().is_some_and(|e| e.get("subject").is_some()))
        } else {
            false
        };

        if has_label_triples {
            return self.load_skill_with_labels(name);
        }

        // Standard numeric path.
        let _ = mgr.discover();
        mgr.warm(name)?;
        Ok(mgr.activate(name, &self.knowledge_graph)?)
    }

    /// Unload (deactivate) a skill.
    pub fn unload_skill(&self, name: &str) -> AkhResult<()> {
        let mgr = self
            .skill_manager
            .as_ref()
            .ok_or(crate::error::SkillError::NotFound {
                name: name.into(),
            })?;
        Ok(mgr.deactivate(name)?)
    }

    /// List all known skills.
    pub fn list_skills(&self) -> Vec<SkillInfo> {
        self.skill_manager
            .as_ref()
            .map(|mgr| mgr.list())
            .unwrap_or_default()
    }

    /// Get info about a specific skill.
    pub fn skill_info(&self, name: &str) -> AkhResult<SkillInfo> {
        let mgr = self
            .skill_manager
            .as_ref()
            .ok_or(crate::error::SkillError::NotFound {
                name: name.into(),
            })?;
        Ok(mgr.get_info(name)?)
    }

    // -----------------------------------------------------------------------
    // Introspection: symbol lookups
    // -----------------------------------------------------------------------

    /// Look up a symbol by label (case-insensitive).
    pub fn lookup_symbol(&self, label: &str) -> AkhResult<SymbolId> {
        self.registry
            .lookup(label)
            .ok_or_else(|| SymbolError::LabelNotFound { label: label.into() }.into())
    }

    /// Get metadata for a symbol by ID.
    pub fn get_symbol_meta(&self, id: SymbolId) -> AkhResult<SymbolMeta> {
        self.registry
            .get(id)
            .ok_or_else(|| SymbolError::NotFound { symbol_id: id.get() }.into())
    }

    /// List all registered symbols with metadata.
    pub fn all_symbols(&self) -> Vec<SymbolMeta> {
        self.registry.all()
    }

    /// Resolve a name-or-id string: try parsing as u64 first, then label lookup.
    pub fn resolve_symbol(&self, name_or_id: &str) -> AkhResult<SymbolId> {
        // Try numeric ID first.
        if let Ok(raw) = name_or_id.trim().parse::<u64>() {
            if let Some(id) = SymbolId::new(raw) {
                return Ok(id);
            }
        }
        // Fall back to label lookup.
        self.lookup_symbol(name_or_id)
    }

    /// Resolve a label for display, falling back to `sym:{id}`.
    pub fn resolve_label(&self, id: SymbolId) -> String {
        self.registry.resolve_label(id)
    }

    // -----------------------------------------------------------------------
    // Introspection: triples
    // -----------------------------------------------------------------------

    /// Check if a specific triple exists.
    pub fn has_triple(&self, s: SymbolId, p: SymbolId, o: SymbolId) -> bool {
        self.knowledge_graph
            .objects_of(s, p)
            .contains(&o)
    }

    /// Get all triples where symbol is subject.
    pub fn triples_from(&self, symbol: SymbolId) -> Vec<Triple> {
        self.knowledge_graph.triples_from(symbol)
    }

    /// Get all triples where symbol is object.
    pub fn triples_to(&self, symbol: SymbolId) -> Vec<Triple> {
        self.knowledge_graph.triples_to(symbol)
    }

    /// Get all triples in the knowledge graph.
    pub fn all_triples(&self) -> Vec<Triple> {
        self.knowledge_graph.all_triples()
    }

    /// Execute a SPARQL SELECT query.
    pub fn sparql_query(&self, sparql: &str) -> AkhResult<Vec<Vec<(String, String)>>> {
        let store = self.sparql.as_ref().ok_or(EngineError::InvalidConfig {
            message: "SPARQL queries require persistence (--data-dir)".into(),
        })?;
        Ok(store.query_select(sparql)?)
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    /// Export the symbol table with resolved labels.
    pub fn export_symbol_table(&self) -> Vec<SymbolExport> {
        self.registry
            .all()
            .into_iter()
            .map(|m| SymbolExport {
                id: m.id.get(),
                label: m.label.clone(),
                kind: m.kind.to_string(),
                created_at: m.created_at,
            })
            .collect()
    }

    /// Export all triples with resolved labels.
    pub fn export_triples(&self) -> Vec<TripleExport> {
        self.knowledge_graph
            .all_triples()
            .into_iter()
            .map(|t| TripleExport {
                subject_id: t.subject.get(),
                subject_label: self.registry.resolve_label(t.subject),
                predicate_id: t.predicate.get(),
                predicate_label: self.registry.resolve_label(t.predicate),
                object_id: t.object.get(),
                object_label: self.registry.resolve_label(t.object),
                confidence: t.confidence,
            })
            .collect()
    }

    /// Export the provenance chain for a symbol.
    pub fn export_provenance_chain(&self, symbol: SymbolId) -> AkhResult<Vec<ProvenanceExport>> {
        let records = self.provenance_of(symbol)?;
        Ok(records
            .into_iter()
            .map(|r| {
                let kind_desc = format!("{:?}", r.kind);
                ProvenanceExport {
                    id: r.id.map(|p| p.get()).unwrap_or(0),
                    derived_id: r.derived_id.get(),
                    derived_label: self.registry.resolve_label(r.derived_id),
                    kind: kind_desc,
                    confidence: r.confidence,
                    depth: r.depth,
                    sources: r.sources.iter().map(|s| s.get()).collect(),
                }
            })
            .collect())
    }

    // -----------------------------------------------------------------------
    // Label-based ingest
    // -----------------------------------------------------------------------

    /// Resolve a symbol by label, or create it as Entity if it doesn't exist.
    pub fn resolve_or_create_entity(&self, label: &str) -> AkhResult<SymbolId> {
        match self.registry.lookup(label) {
            Some(id) => Ok(id),
            None => Ok(self.create_symbol(SymbolKind::Entity, label)?.id),
        }
    }

    /// Resolve a symbol by label, or create it as Relation if it doesn't exist.
    pub fn resolve_or_create_relation(&self, label: &str) -> AkhResult<SymbolId> {
        match self.registry.lookup(label) {
            Some(id) => Ok(id),
            None => Ok(self.create_symbol(SymbolKind::Relation, label)?.id),
        }
    }

    /// Batch ingest triples from label-based format.
    /// Auto-creates symbols that don't exist. Predicates become Relations, rest become Entities.
    /// Returns (created_symbols_count, ingested_triples_count).
    pub fn ingest_label_triples(
        &self,
        triples: &[(String, String, String, f32)],
    ) -> AkhResult<(usize, usize)> {
        let symbols_before = self.registry.all().len();
        let mut ingested = 0usize;

        for (subject, predicate, object, confidence) in triples {
            let s = self.resolve_or_create_entity(subject)?;
            let p = self.resolve_or_create_relation(predicate)?;
            let o = self.resolve_or_create_entity(object)?;

            let triple = Triple::new(s, p, o).with_confidence(*confidence);
            self.add_triple(&triple)?;
            ingested += 1;
        }

        let symbols_after = self.registry.all().len();
        let created = symbols_after - symbols_before;
        Ok((created, ingested))
    }

    // -----------------------------------------------------------------------
    // Graph traversal
    // -----------------------------------------------------------------------

    /// Traverse the knowledge graph from seed symbols using BFS.
    pub fn traverse(
        &self,
        seeds: &[SymbolId],
        config: TraversalConfig,
    ) -> AkhResult<TraversalResult> {
        Ok(crate::graph::traverse::traverse_bfs(
            &self.knowledge_graph,
            seeds,
            &config,
        )?)
    }

    /// Convenience: extract subgraph from seeds with default config.
    pub fn extract_subgraph(
        &self,
        seeds: &[SymbolId],
        max_depth: usize,
    ) -> AkhResult<TraversalResult> {
        Ok(crate::graph::traverse::extract_subgraph(
            &self.knowledge_graph,
            seeds,
            max_depth,
        )?)
    }

    // -----------------------------------------------------------------------
    // Reasoning
    // -----------------------------------------------------------------------

    /// Simplify a symbolic expression using e-graph rewriting.
    /// Parses the expression as AkhLang, runs equality saturation with all active rules,
    /// extracts the lowest-cost equivalent expression.
    pub fn simplify_expression(&self, expr: &str) -> AkhResult<String> {
        let parsed: egg::RecExpr<AkhLang> = expr.parse().map_err(|e| {
            ReasonError::ParseError {
                message: format!("{e}"),
            }
        })?;

        let rules = self.all_rules();
        let runner = Runner::default().with_expr(&parsed).run(&rules);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_cost, best) = extractor.find_best(runner.roots[0]);
        Ok(best.to_string())
    }

    // -----------------------------------------------------------------------
    // Skill loading with labels
    // -----------------------------------------------------------------------

    /// Load a skill with label-based triple resolution.
    /// Pre-resolves labels to symbol IDs, then delegates to standard SkillManager.
    pub fn load_skill_with_labels(&self, name: &str) -> AkhResult<crate::skills::SkillActivation> {
        let mgr = self
            .skill_manager
            .as_ref()
            .ok_or(crate::error::SkillError::NotFound {
                name: name.into(),
            })?;

        // Ensure discovered and warmed.
        let _ = mgr.discover();
        mgr.warm(name)?;

        // Read the skill's triples file before activation.
        let skill_dir = mgr.skills_dir().join(name);
        let triples_path = skill_dir.join("triples.json");
        let mut label_triples_count = 0usize;

        if triples_path.exists() {
            let content = std::fs::read_to_string(&triples_path).map_err(|e| {
                crate::error::SkillError::Io {
                    skill_id: name.into(),
                    source: e,
                }
            })?;

            let raw: Vec<serde_json::Value> =
                serde_json::from_str(&content).map_err(|e| {
                    crate::error::SkillError::InvalidManifest {
                        path: triples_path.display().to_string(),
                        message: format!("triples parse error: {e}"),
                    }
                })?;

            // Detect label-based format by checking first element.
            let is_label_format = raw
                .first()
                .is_some_and(|v| v.get("subject").is_some());

            if is_label_format {
                for val in &raw {
                    let subject = val["subject"].as_str().unwrap_or("");
                    let predicate = val["predicate"].as_str().unwrap_or("");
                    let object = val["object"].as_str().unwrap_or("");
                    let confidence = val["confidence"].as_f64().unwrap_or(1.0) as f32;

                    if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                        let s = self.resolve_or_create_entity(subject)?;
                        let p = self.resolve_or_create_relation(predicate)?;
                        let o = self.resolve_or_create_entity(object)?;

                        let triple = Triple::new(s, p, o).with_confidence(confidence);
                        let _ = self.add_triple(&triple); // ignore duplicates
                        label_triples_count += 1;
                    }
                }
            }
        }

        // Now activate via the standard path (which handles numeric triples and rules).
        let mut activation = mgr.activate(name, &self.knowledge_graph)?;

        // If we loaded label triples, add them to the activation count.
        if label_triples_count > 0 {
            activation.triples_loaded += label_triples_count;
        }

        Ok(activation)
    }

    // -----------------------------------------------------------------------
    // Graph analytics
    // -----------------------------------------------------------------------

    /// Compute degree centrality for all nodes.
    pub fn degree_centrality(&self) -> Vec<analytics::DegreeCentrality> {
        analytics::degree_centrality(&self.knowledge_graph)
    }

    /// Compute PageRank scores.
    pub fn pagerank(
        &self,
        damping: f64,
        iterations: usize,
    ) -> AkhResult<Vec<analytics::PageRankScore>> {
        Ok(analytics::pagerank(&self.knowledge_graph, damping, iterations)?)
    }

    /// Find strongly connected components.
    pub fn strongly_connected_components(
        &self,
    ) -> AkhResult<Vec<analytics::ConnectedComponent>> {
        Ok(analytics::strongly_connected_components(&self.knowledge_graph)?)
    }

    /// Find shortest path (by hop count) between two symbols.
    pub fn shortest_path(
        &self,
        from: SymbolId,
        to: SymbolId,
    ) -> AkhResult<Option<Vec<SymbolId>>> {
        Ok(analytics::shortest_path(&self.knowledge_graph, from, to)?)
    }

    // -----------------------------------------------------------------------
    // Autonomous reasoning
    // -----------------------------------------------------------------------

    /// Run forward-chaining inference rules on the knowledge graph.
    pub fn run_rules(
        &self,
        config: crate::autonomous::rule_engine::RuleEngineConfig,
    ) -> AkhResult<crate::autonomous::rule_engine::RuleEngineResult> {
        let engine = crate::autonomous::rule_engine::RuleEngine::new(config)
            .with_rules(crate::autonomous::rules::RuleSet::builtin());
        Ok(engine.run(self)?)
    }

    /// Run both builtin ontological + code-specific inference rules.
    pub fn run_code_rules(
        &self,
        config: crate::autonomous::rule_engine::RuleEngineConfig,
    ) -> AkhResult<crate::autonomous::rule_engine::RuleEngineResult> {
        let engine = crate::autonomous::rule_engine::RuleEngine::new(config)
            .with_rules(crate::autonomous::rules::RuleSet::builtin())
            .with_rules(crate::autonomous::rules::RuleSet::code_rules());
        Ok(engine.run(self)?)
    }

    /// Analyze knowledge gaps around the given goal symbols.
    pub fn analyze_gaps(
        &self,
        goals: &[SymbolId],
        config: crate::autonomous::gap::GapAnalysisConfig,
    ) -> AkhResult<crate::autonomous::gap::GapAnalysisResult> {
        Ok(crate::autonomous::gap::analyze_gaps(self, goals, &config)?)
    }

    /// Discover schema patterns from the knowledge graph.
    pub fn discover_schema(
        &self,
        config: crate::autonomous::schema::SchemaDiscoveryConfig,
    ) -> AkhResult<crate::autonomous::schema::SchemaDiscoveryResult> {
        Ok(crate::autonomous::schema::discover_schema(self, &config)?)
    }

    // -----------------------------------------------------------------------
    // Equivalence learning
    // -----------------------------------------------------------------------

    /// Run all equivalence learning strategies on current engine state.
    ///
    /// Discovers new cross-lingual mappings from KG structure and VSA
    /// similarity, then persists results to the durable store.
    /// Returns the number of new equivalences discovered.
    pub fn learn_equivalences(&mut self) -> AkhResult<usize> {
        let total = self.entity_resolver.learn_from_kg(
            &self.knowledge_graph,
            &self.registry,
        ) + self.entity_resolver.learn_from_vsa(
            &self.ops,
            &self.item_memory,
            &self.registry,
            0.65,
        );

        self.entity_resolver.persist_to_store(&self.store)?;
        Ok(total)
    }

    /// Get equivalence statistics.
    pub fn equivalence_stats(&self) -> EquivalenceStats {
        self.entity_resolver.stats()
    }

    /// Export all learned equivalences.
    pub fn export_equivalences(&self) -> Vec<LearnedEquivalence> {
        self.entity_resolver.export_learned()
    }

    /// Import equivalences and persist to durable store.
    pub fn import_equivalences(&mut self, equivs: &[LearnedEquivalence]) -> AkhResult<()> {
        self.entity_resolver.import_equivalences(equivs);
        self.entity_resolver.persist_to_store(&self.store)?;
        Ok(())
    }

    /// Get a reference to the entity resolver.
    pub fn entity_resolver(&self) -> &EntityResolver {
        &self.entity_resolver
    }

    /// Get a mutable reference to the entity resolver.
    pub fn entity_resolver_mut(&mut self) -> &mut EntityResolver {
        &mut self.entity_resolver
    }

    // -----------------------------------------------------------------------
    // Grammar API
    // -----------------------------------------------------------------------

    /// Get the grammar registry.
    pub fn grammar_registry(&self) -> &GrammarRegistry {
        &self.grammar_registry
    }

    /// Get a mutable reference to the grammar registry.
    pub fn grammar_registry_mut(&mut self) -> &mut GrammarRegistry {
        &mut self.grammar_registry
    }

    /// Parse prose input into a [`ParseResult`] using the grammar parser.
    ///
    /// Uses the engine's configured language. Automatically provides the
    /// engine's registry, VSA ops, and item memory for token resolution.
    pub fn parse(&self, input: &str) -> ParseResult {
        self.parse_with_language(input, self.config.language)
    }

    /// Parse prose input with an explicit language override.
    pub fn parse_with_language(&self, input: &str, language: Language) -> ParseResult {
        let ctx = ParseContext::with_engine_and_language(
            self.registry(),
            self.ops(),
            self.item_memory(),
            language,
        );
        parse_prose(input, &ctx)
    }

    /// Parse a mixed-language corpus by detecting language per sentence.
    ///
    /// Each sentence is detected independently and parsed with the
    /// appropriate language lexicon.
    pub fn parse_mixed_corpus(&self, input: &str) -> Vec<(String, Language, ParseResult)> {
        use crate::grammar::detect::detect_per_sentence;

        detect_per_sentence(input)
            .into_iter()
            .map(|(sentence, detection)| {
                let result = self.parse_with_language(&sentence, detection.language);
                (sentence, detection.language, result)
            })
            .collect()
    }

    /// Linearize an abstract syntax tree through a named grammar archetype.
    ///
    /// If `grammar_name` is `None`, uses the default grammar.
    pub fn linearize(&self, tree: &AbsTree, grammar_name: Option<&str>) -> GrammarResult<String> {
        let name = grammar_name.unwrap_or(self.grammar_registry.default_name());
        let grammar = self.grammar_registry.get(name)?;
        let ctx = LinContext::with_registry(self.registry());
        grammar.linearize(tree, &ctx)
    }

    /// Load a custom grammar from a TOML definition string.
    ///
    /// Returns the registered name of the grammar.
    pub fn load_custom_grammar(&mut self, toml_content: &str) -> GrammarResult<String> {
        let grammar = CustomGrammar::from_toml(toml_content)?;
        let name = grammar.name().to_string();
        self.grammar_registry.register(Box::new(grammar));
        Ok(name)
    }

    /// Parse prose, extract facts, ground them, and commit triples to the KG.
    ///
    /// Returns a summary of what was ingested. Sentences that don't parse into
    /// structured facts are silently skipped.
    pub fn ingest_prose(&self, input: &str) -> AkhResult<ProseIngestResult> {
        let result = self.parse(input);

        let facts = match result {
            ParseResult::Facts(facts) => facts,
            ParseResult::Freeform { partial, .. } => {
                if partial.is_empty() {
                    // Fall through to sentence splitting for multi-sentence input
                    return self.ingest_prose_sentences(input);
                }
                partial
            }
            _ => {
                return Ok(ProseIngestResult {
                    triples_ingested: 0,
                    symbols_created: 0,
                    trees: vec![],
                })
            }
        };

        // Ground each fact against the registry
        let grounded: Vec<AbsTree> = facts.iter().map(|f| f.ground(&self.registry)).collect();

        // Extract and commit triples from grounded trees
        let mut triples_ingested = 0usize;
        let symbols_before = self.registry.len();

        for tree in &grounded {
            triples_ingested += self.commit_abs_tree(tree)?;
        }

        let symbols_created = self.registry.len() - symbols_before;
        Ok(ProseIngestResult {
            triples_ingested,
            symbols_created,
            trees: grounded,
        })
    }

    /// Walk an [`AbsTree`] and commit each Triple node to the knowledge graph.
    fn commit_abs_tree(&self, tree: &AbsTree) -> AkhResult<usize> {
        match tree {
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                let s_label = subject.label().unwrap_or("?");
                let p_label = predicate.label().unwrap_or("?");
                let o_label = object.label().unwrap_or("?");

                let s = self.resolve_or_create_entity(s_label)?;
                let p = self.resolve_or_create_relation(p_label)?;
                let o = self.resolve_or_create_entity(o_label)?;

                self.add_triple(&Triple::new(s, p, o))?;
                Ok(1)
            }
            AbsTree::WithConfidence { inner, confidence } => {
                if let AbsTree::Triple {
                    subject,
                    predicate,
                    object,
                } = inner.as_ref()
                {
                    let s = self.resolve_or_create_entity(subject.label().unwrap_or("?"))?;
                    let p = self.resolve_or_create_relation(predicate.label().unwrap_or("?"))?;
                    let o = self.resolve_or_create_entity(object.label().unwrap_or("?"))?;
                    self.add_triple(&Triple::new(s, p, o).with_confidence(*confidence))?;
                    return Ok(1);
                }
                self.commit_abs_tree(inner)
            }
            AbsTree::Conjunction { items, .. } => {
                let mut count = 0;
                for item in items {
                    count += self.commit_abs_tree(item)?;
                }
                Ok(count)
            }
            _ => Ok(0),
        }
    }

    /// Split multi-sentence input on `.` boundaries and parse each individually.
    fn ingest_prose_sentences(&self, input: &str) -> AkhResult<ProseIngestResult> {
        let mut total_triples = 0usize;
        let symbols_before = self.registry.len();
        let mut all_trees = Vec::new();

        for sentence in input.split('.').map(str::trim).filter(|s| !s.is_empty()) {
            let result = self.parse(sentence);
            if let ParseResult::Facts(facts) = result {
                for fact in &facts {
                    let grounded = fact.ground(&self.registry);
                    total_triples += self.commit_abs_tree(&grounded)?;
                    all_trees.push(grounded);
                }
            }
        }

        let symbols_created = self.registry.len() - symbols_before;
        Ok(ProseIngestResult {
            triples_ingested: total_triples,
            symbols_created,
            trees: all_trees,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get the VSA operations handle.
    pub fn ops(&self) -> &VsaOps {
        &self.ops
    }

    /// Get the item memory handle.
    pub fn item_memory(&self) -> &ItemMemory {
        &self.item_memory
    }

    /// Get the knowledge graph handle.
    pub fn knowledge_graph(&self) -> &KnowledgeGraph {
        &self.knowledge_graph
    }

    /// Get the SPARQL store handle.
    pub fn sparql(&self) -> Option<&SparqlStore> {
        self.sparql.as_ref()
    }

    /// Get the tiered store handle.
    pub fn store(&self) -> &TieredStore {
        &self.store
    }

    /// Get the symbol registry handle.
    pub fn registry(&self) -> &SymbolRegistry {
        &self.registry
    }

    /// Get the engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get the compartment manager (if data_dir is configured).
    pub fn compartments(&self) -> Option<&crate::compartment::CompartmentManager> {
        self.compartment_manager.as_ref()
    }

    /// Get system info (node count, triple count, symbol count, etc.)
    pub fn info(&self) -> EngineInfo {
        let provenance_count = self
            .provenance_ledger
            .as_ref()
            .and_then(|l| l.len().ok())
            .unwrap_or(0);
        let skill_count = self.list_skills().len();

        EngineInfo {
            dimension: self.config.dimension.0,
            encoding: self.config.encoding.to_string(),
            isa_level: format!("{}", self.ops.isa_level()),
            symbol_count: self.item_memory.len(),
            node_count: self.knowledge_graph.node_count(),
            triple_count: self.knowledge_graph.triple_count(),
            store_hot_entries: self.store.hot_len(),
            persistent: self.config.data_dir.is_some(),
            provenance_count,
            skill_count,
        }
    }

    /// Persist current state (registry, allocator, equivalences, knowledge graph → SPARQL).
    pub fn persist(&self) -> AkhResult<()> {
        // Persist symbol registry.
        self.registry.persist(&self.store)?;

        // Persist allocator next-ID so new symbols resume correctly after restart.
        let next = self.symbol_allocator.peek_next();
        let encoded = bincode::serialize(&next).map_err(|e| {
            crate::error::StoreError::Serialization {
                message: format!("failed to serialize allocator state: {e}"),
            }
        })?;
        self.store.put_meta(b"sym_allocator_next", &encoded)?;

        // Persist learned equivalences.
        self.entity_resolver.persist_to_store(&self.store)?;

        // Sync knowledge graph to SPARQL store.
        if let Some(ref sparql) = self.sparql {
            sparql.sync_from(&self.knowledge_graph)?;
        }
        Ok(())
    }
}

/// Summary information about the engine state.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub dimension: usize,
    pub encoding: String,
    pub isa_level: String,
    pub symbol_count: usize,
    pub node_count: usize,
    pub triple_count: usize,
    pub store_hot_entries: usize,
    pub persistent: bool,
    pub provenance_count: usize,
    pub skill_count: usize,
}

impl std::fmt::Display for EngineInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "akh-medu engine info")?;
        writeln!(f, "  dimension:    {}", self.dimension)?;
        writeln!(f, "  encoding:     {}", self.encoding)?;
        writeln!(f, "  symbols:      {}", self.symbol_count)?;
        writeln!(f, "  nodes:        {}", self.node_count)?;
        writeln!(f, "  triples:      {}", self.triple_count)?;
        writeln!(f, "  hot entries:  {}", self.store_hot_entries)?;
        writeln!(f, "  provenance:   {}", self.provenance_count)?;
        writeln!(f, "  skills:       {}", self.skill_count)?;
        writeln!(f, "  persistent:   {}", self.persistent)?;
        Ok(())
    }
}

/// Result of ingesting prose through the grammar parser.
#[derive(Debug)]
pub struct ProseIngestResult {
    /// Number of triples committed to the knowledge graph.
    pub triples_ingested: usize,
    /// Number of new symbols created during ingest.
    pub symbols_created: usize,
    /// The grounded abstract syntax trees that were extracted.
    pub trees: Vec<AbsTree>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("config", &self.config)
            .field("item_memory", &self.item_memory)
            .field("knowledge_graph", &self.knowledge_graph)
            .field("registry", &self.registry)
            .field("learned_equivalences", &self.entity_resolver.learned_count())
            .field("compartment_manager", &self.compartment_manager)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_memory_only_engine() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let info = engine.info();
        assert_eq!(info.dimension, 1000);
        assert!(!info.persistent);
    }

    #[test]
    fn create_symbol_and_search() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        let _moon = engine.create_symbol(SymbolKind::Entity, "Moon").unwrap();
        let _star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();

        // Search for sun's vector — should find itself
        let results = engine.search_similar_to(sun.id, 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].symbol_id, sun.id);
    }

    #[test]
    fn add_triple_and_query() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();

        engine
            .add_triple(&Triple::new(sun.id, is_a.id, star.id))
            .unwrap();

        let objects = engine.knowledge_graph().objects_of(sun.id, is_a.id);
        assert_eq!(objects, vec![star.id]);
    }

    #[test]
    fn engine_with_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();

        let info = engine.info();
        assert!(info.persistent);
    }

    #[test]
    fn zero_dimension_rejected() {
        let result = Engine::new(EngineConfig {
            dimension: Dimension(0),
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn provenance_requires_persistence() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let sym = SymbolId::new(1).unwrap();
        let result = engine.provenance_of(sym);
        assert!(result.is_err());
    }

    #[test]
    fn parse_returns_facts_for_declarative_input() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let result = engine.parse("Dogs are mammals");
        assert!(matches!(result, ParseResult::Facts(_)));
    }

    #[test]
    fn linearize_triple_through_engine() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        use crate::grammar::abs::AbsTree;
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let prose = engine.linearize(&tree, Some("formal")).unwrap();
        assert!(!prose.is_empty());
        // Formal archetype should mention the entities
        let lower = prose.to_lowercase();
        assert!(lower.contains("dog"));
        assert!(lower.contains("mammal"));
    }

    #[test]
    fn ingest_prose_creates_triples() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let result = engine.ingest_prose("Dogs are mammals").unwrap();
        assert!(
            result.triples_ingested >= 1,
            "expected at least 1 triple, got {}",
            result.triples_ingested,
        );
        assert!(result.symbols_created >= 2, "expected at least 2 new symbols");
        assert!(!result.trees.is_empty());

        // Verify the triple exists in the KG
        let dog = engine.lookup_symbol("Dogs").unwrap();
        let triples = engine.triples_from(dog);
        assert!(!triples.is_empty(), "Dog should have outgoing triples");
    }

    #[test]
    fn ingest_prose_compound_sentence() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        // Compound "and" sentence should produce 2 triples from a single parse
        let result = engine
            .ingest_prose("Dogs are mammals and cats are mammals")
            .unwrap();
        assert!(
            result.triples_ingested >= 2,
            "expected at least 2 triples from compound sentence, got {}",
            result.triples_ingested,
        );
    }

    #[test]
    fn ingest_prose_sentence_splitting() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        // Two separate sentences that individually parse but don't as a whole
        let result = engine
            .ingest_prose("Oxygen is an element. Water contains hydrogen.")
            .unwrap();
        assert!(
            result.triples_ingested >= 1,
            "expected at least 1 triple from sentence splitting, got {}",
            result.triples_ingested,
        );
    }

    #[test]
    fn grammar_registry_accessible_from_engine() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let reg = engine.grammar_registry();
        let names = reg.list();
        assert!(names.contains(&"formal"));
        assert!(names.contains(&"terse"));
        assert!(names.contains(&"narrative"));
    }
}
