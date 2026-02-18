//! Engine facade: top-level API for the akh-medu system.
//!
//! The `Engine` owns all subsystems and provides the public interface
//! for ingesting knowledge, querying, and managing the system.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use egg::{AstSize, Extractor, Rewrite, Runner};

use crate::error::{AkhResult, EngineError, ProvenanceError, ReasonError, SymbolError};
use crate::workspace::WorkspaceError;
use crate::export::{ProvenanceExport, SymbolExport, TripleExport};
use crate::grammar::GrammarRegistry;
use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::{ConcreteGrammar, LinContext, ParseContext};
use crate::grammar::custom::CustomGrammar;
use crate::grammar::entity_resolution::{EntityResolver, EquivalenceStats, LearnedEquivalence};
use crate::grammar::error::GrammarResult;
use crate::grammar::lexer::Language;
use crate::grammar::parser::{ParseResult, parse_prose};
use crate::graph::Triple;
use crate::graph::analytics;
use crate::graph::index::KnowledgeGraph;
use crate::graph::sparql::SparqlStore;
use crate::graph::traverse::{TraversalConfig, TraversalResult};
use crate::infer::engine::InferEngine;
use crate::infer::{InferenceQuery, InferenceResult};
use crate::pipeline::{Pipeline, PipelineContext, PipelineData, PipelineOutput};
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceLedger, ProvenanceRecord};
use crate::reason::AkhLang;
use crate::registry::SymbolRegistry;
use crate::simd;
use crate::skills::SkillInfo;
use crate::skills::manager::SkillManager;
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
    entity_resolver: RwLock<EntityResolver>,
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
            let store = TieredStore::with_persistence(dir, "symbols").map_err(|e| {
                EngineError::InvalidConfig {
                    message: format!("failed to create tiered store: {e}"),
                }
            })?;
            let sparql_dir = dir.join("oxigraph");
            let sparql =
                SparqlStore::open(&sparql_dir).map_err(|e| EngineError::InvalidConfig {
                    message: format!("failed to create SPARQL store: {e}"),
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
        let entity_resolver = RwLock::new(EntityResolver::load_from_store(&store));

        // Initialize compartment manager if data_dir has a compartments/ subdir.
        let compartment_manager = config.data_dir.as_ref().map(|dir| {
            let compartments_dir = dir.join("compartments");
            let mgr = crate::compartment::CompartmentManager::new(compartments_dir);
            if let Err(e) = mgr.discover() {
                tracing::debug!(error = %e, "compartment discovery skipped");
            }
            mgr
        });

        let engine = Self {
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
        };

        // Auto-activate all discovered skills so installed == active.
        engine.activate_installed_skills();

        Ok(engine)
    }

    /// Activate all discovered (Cold) skills, making them Hot.
    ///
    /// Called during engine startup so that every skill present in the
    /// workspace's `skills/` directory is immediately usable.
    fn activate_installed_skills(&self) {
        let skill_ids: Vec<String> = self
            .skill_manager
            .as_ref()
            .map(|mgr| {
                mgr.list()
                    .into_iter()
                    .filter(|s| s.state == crate::skills::SkillState::Cold)
                    .map(|s| s.id)
                    .collect()
            })
            .unwrap_or_default();

        for id in skill_ids {
            match self.load_skill(&id) {
                Ok(activation) => {
                    tracing::info!(
                        skill = id.as_str(),
                        triples = activation.triples_loaded,
                        rules = activation.rules_loaded,
                        "auto-activated skill"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        skill = id.as_str(),
                        error = %e,
                        "failed to auto-activate skill"
                    );
                }
            }
        }
    }

    /// Allocate a new symbol with the given kind and label.
    pub fn create_symbol(
        &self,
        kind: SymbolKind,
        label: impl Into<String>,
    ) -> AkhResult<SymbolMeta> {
        let id = self.symbol_allocator.next_id()?;
        let meta = SymbolMeta::new(id, kind, label);

        // Register in the bidirectional registry.
        self.registry.register(meta.clone())?;

        // Store metadata in tiered store.
        let encoded =
            bincode::serialize(&meta).map_err(|e| crate::error::StoreError::Serialization {
                message: format!("failed to serialize symbol meta: {e}"),
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
        let vec =
            self.item_memory
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
            .ok_or(crate::error::SkillError::NotFound { name: name.into() })?;

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
            .ok_or(crate::error::SkillError::NotFound { name: name.into() })?;
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
            .ok_or(crate::error::SkillError::NotFound { name: name.into() })?;
        Ok(mgr.get_info(name)?)
    }

    /// Install a skill from a payload: write files to disk, then load.
    pub fn install_skill(
        &self,
        payload: &crate::skills::SkillInstallPayload,
    ) -> AkhResult<crate::skills::SkillActivation> {
        let mgr = self.skill_manager.as_ref().ok_or(EngineError::InvalidConfig {
            message: "skill installation requires persistence (--data-dir)".into(),
        })?;

        let skill_dir = mgr.skills_dir().join(&payload.manifest.id);
        std::fs::create_dir_all(&skill_dir).map_err(|e| EngineError::DataDir {
            path: format!("{}: {e}", skill_dir.display()),
        })?;

        // Write skill.json (manifest).
        let manifest_json =
            serde_json::to_string_pretty(&payload.manifest).map_err(|e| {
                EngineError::InvalidConfig {
                    message: format!("failed to serialize manifest: {e}"),
                }
            })?;
        std::fs::write(skill_dir.join("skill.json"), manifest_json).map_err(|e| {
            EngineError::DataDir {
                path: format!("{}/skill.json: {e}", skill_dir.display()),
            }
        })?;

        // Write triples.json.
        let triples_json =
            serde_json::to_string_pretty(&payload.triples).map_err(|e| {
                EngineError::InvalidConfig {
                    message: format!("failed to serialize triples: {e}"),
                }
            })?;
        std::fs::write(skill_dir.join("triples.json"), triples_json).map_err(|e| {
            EngineError::DataDir {
                path: format!("{}/triples.json: {e}", skill_dir.display()),
            }
        })?;

        // Write rules.txt if non-empty.
        if !payload.rules.is_empty() {
            std::fs::write(skill_dir.join("rules.txt"), &payload.rules).map_err(|e| {
                EngineError::DataDir {
                    path: format!("{}/rules.txt: {e}", skill_dir.display()),
                }
            })?;
        }

        // Load via existing path (handles label detection + loading).
        self.load_skill(&payload.manifest.id)
    }

    // -----------------------------------------------------------------------
    // Introspection: symbol lookups
    // -----------------------------------------------------------------------

    /// Look up a symbol by label (case-insensitive).
    pub fn lookup_symbol(&self, label: &str) -> AkhResult<SymbolId> {
        self.registry.lookup(label).ok_or_else(|| {
            SymbolError::LabelNotFound {
                label: label.into(),
            }
            .into()
        })
    }

    /// Get metadata for a symbol by ID.
    pub fn get_symbol_meta(&self, id: SymbolId) -> AkhResult<SymbolMeta> {
        self.registry.get(id).ok_or_else(|| {
            SymbolError::NotFound {
                symbol_id: id.get(),
            }
            .into()
        })
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
        self.knowledge_graph.objects_of(s, p).contains(&o)
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
        let parsed: egg::RecExpr<AkhLang> = expr.parse().map_err(|e| ReasonError::ParseError {
            message: format!("{e}"),
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
            .ok_or(crate::error::SkillError::NotFound { name: name.into() })?;

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

            let raw: Vec<serde_json::Value> = serde_json::from_str(&content).map_err(|e| {
                crate::error::SkillError::InvalidManifest {
                    path: triples_path.display().to_string(),
                    message: format!("triples parse error: {e}"),
                }
            })?;

            // Detect label-based format by checking first element.
            let is_label_format = raw.first().is_some_and(|v| v.get("subject").is_some());

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
        Ok(analytics::pagerank(
            &self.knowledge_graph,
            damping,
            iterations,
        )?)
    }

    /// Find strongly connected components.
    pub fn strongly_connected_components(&self) -> AkhResult<Vec<analytics::ConnectedComponent>> {
        Ok(analytics::strongly_connected_components(
            &self.knowledge_graph,
        )?)
    }

    /// Find shortest path (by hop count) between two symbols.
    pub fn shortest_path(&self, from: SymbolId, to: SymbolId) -> AkhResult<Option<Vec<SymbolId>>> {
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
    pub fn learn_equivalences(&self) -> AkhResult<usize> {
        let total = {
            let mut resolver = self.entity_resolver.write().unwrap();
            resolver.learn_from_kg(&self.knowledge_graph, &self.registry)
                + resolver.learn_from_vsa(&self.ops, &self.item_memory, &self.registry, 0.65)
                + resolver.learn_from_library(
                    &self.ops,
                    &self.item_memory,
                    &self.registry,
                    &self.knowledge_graph,
                    0.65,
                )
        };

        self.entity_resolver
            .read()
            .unwrap()
            .persist_to_store(&self.store)?;
        Ok(total)
    }

    /// Get equivalence statistics.
    pub fn equivalence_stats(&self) -> EquivalenceStats {
        self.entity_resolver.read().unwrap().stats()
    }

    /// Export all learned equivalences.
    pub fn export_equivalences(&self) -> Vec<LearnedEquivalence> {
        self.entity_resolver.read().unwrap().export_learned()
    }

    /// Import equivalences and persist to durable store.
    pub fn import_equivalences(&self, equivs: &[LearnedEquivalence]) -> AkhResult<()> {
        self.entity_resolver
            .write()
            .unwrap()
            .import_equivalences(equivs);
        self.entity_resolver
            .read()
            .unwrap()
            .persist_to_store(&self.store)?;
        Ok(())
    }

    /// Get a read lock on the entity resolver.
    pub fn entity_resolver(&self) -> std::sync::RwLockReadGuard<'_, EntityResolver> {
        self.entity_resolver.read().unwrap()
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
                });
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
    // Microtheories (Phase 9a)
    // -----------------------------------------------------------------------

    /// Create a new microtheory (reasoning context) as a first-class Entity.
    ///
    /// Registers the microtheory in the KG with `ctx:domain` metadata and optional
    /// parent contexts via `ctx:specializes`. Returns the microtheory's SymbolId.
    ///
    /// # Errors
    /// - Returns `CompartmentError::ContextCycle` if adding a parent would create a cycle.
    pub fn create_context(
        &self,
        label: &str,
        domain: crate::compartment::ContextDomain,
        parents: &[SymbolId],
    ) -> AkhResult<crate::compartment::Microtheory> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        let ctx = self.resolve_or_create_entity(label)?;
        let domain_entity = self.resolve_or_create_entity(&domain.to_string())?;

        // Assert ctx:domain
        self.add_triple(&Triple::new(ctx, preds.domain, domain_entity))?;

        // Assert ctx:specializes for each parent (with cycle check)
        for &parent in parents {
            // Check that parent doesn't eventually specialize ctx (would create a cycle)
            let parent_ancestors =
                crate::compartment::microtheory::resolve_ancestors(self, parent, preds.specializes);
            if parent_ancestors.contains(&ctx) || parent == ctx {
                let ctx_label = self.resolve_label(ctx);
                return Err(crate::compartment::CompartmentError::ContextCycle {
                    context: ctx_label,
                }
                .into());
            }
            self.add_triple(&Triple::new(ctx, preds.specializes, parent))?;
        }

        // Compute initial ancestor cache
        let ancestors =
            crate::compartment::microtheory::resolve_ancestors(self, ctx, preds.specializes);

        Ok(crate::compartment::Microtheory {
            id: ctx,
            domain,
            ancestors,
        })
    }

    /// Add a domain assumption to a context.
    ///
    /// Domain assumptions are triples that are implicitly true within the context.
    /// The assumption is stored as `context ctx:assumes assumption_entity`.
    pub fn add_context_assumption(
        &self,
        context: SymbolId,
        assumption: SymbolId,
    ) -> AkhResult<()> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        self.add_triple(&Triple::new(context, preds.assumes, assumption))
    }

    /// Query triples visible in a given context, including inherited ancestors.
    ///
    /// Searches the current context and all ancestor contexts via `ctx:specializes`
    /// transitive closure. Results from more specific contexts appear first.
    pub fn query_in_context(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        context: SymbolId,
    ) -> AkhResult<Vec<Triple>> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        Ok(crate::compartment::microtheory::triples_in_context(
            self,
            subject,
            predicate,
            context,
            preds.specializes,
        ))
    }

    /// Query all triples for a subject visible in a given context.
    pub fn query_all_in_context(
        &self,
        subject: SymbolId,
        context: SymbolId,
    ) -> AkhResult<Vec<Triple>> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        Ok(crate::compartment::microtheory::all_triples_in_context(
            self,
            subject,
            context,
            preds.specializes,
        ))
    }

    /// Add a lifting rule between two contexts.
    ///
    /// Lifting rules govern when entailments propagate between contexts.
    pub fn add_lifting_rule(&self, rule: &crate::compartment::LiftingRule) -> AkhResult<()> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        let condition_entity =
            self.resolve_or_create_entity(&rule.condition.to_string())?;

        self.add_triple(&Triple::new(rule.from, preds.lifts_to, rule.to))?;
        self.add_triple(&Triple::new(
            rule.from,
            preds.lifting_condition,
            condition_entity,
        ))
    }

    /// Get the transitive ancestor chain for a context.
    pub fn context_ancestors(&self, context: SymbolId) -> AkhResult<Vec<SymbolId>> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        Ok(crate::compartment::microtheory::resolve_ancestors(
            self,
            context,
            preds.specializes,
        ))
    }

    /// Check if two contexts are declared disjoint.
    pub fn contexts_are_disjoint(
        &self,
        ctx_a: SymbolId,
        ctx_b: SymbolId,
    ) -> AkhResult<bool> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        Ok(crate::compartment::microtheory::contexts_are_disjoint(
            self, ctx_a, ctx_b, preds.disjoint,
        ))
    }

    /// Apply all lifting rules from a source context and insert propagated triples.
    ///
    /// Returns the number of triples propagated.
    pub fn apply_lifting_rules(&self, from_context: SymbolId) -> AkhResult<usize> {
        let preds = crate::compartment::ContextPredicates::resolve(self)?;
        let rules =
            crate::compartment::microtheory::lifting_rules_from(self, from_context, &preds);

        let mut propagated = 0;
        for rule in &rules {
            let triples = crate::compartment::microtheory::apply_lifting_rule(
                self,
                rule,
                preds.specializes,
            );
            for triple in &triples {
                self.add_triple(triple)?;

                // Record provenance for lifted triples
                if let Some(ref ledger) = self.provenance_ledger {
                    let mut record = ProvenanceRecord::new(
                        triple.subject,
                        DerivationKind::ContextLifting {
                            from_context: rule.from,
                            to_context: rule.to,
                            condition: rule.condition.to_string(),
                        },
                    )
                    .with_sources(vec![triple.subject, triple.predicate, triple.object]);
                    let _ = ledger.store(&mut record);
                }

                propagated += 1;
            }
        }

        Ok(propagated)
    }

    // -----------------------------------------------------------------------
    // Truth Maintenance System (Phase 9c)
    // -----------------------------------------------------------------------

    /// Remove a triple from the knowledge graph.
    ///
    /// Returns true if the triple was found and removed.
    pub fn remove_triple(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        object: SymbolId,
    ) -> bool {
        self.knowledge_graph.remove_triple(subject, predicate, object)
    }

    // -----------------------------------------------------------------------
    // Predicate Hierarchy (Phase 9b)
    // -----------------------------------------------------------------------

    /// Declare that `specific` is a specialization of `general` (genlPreds).
    ///
    /// After adding, `general(X, Y)` queries will also find `specific(X, Y)` triples.
    pub fn add_predicate_generalization(
        &self,
        specific: SymbolId,
        general: SymbolId,
    ) -> AkhResult<()> {
        let preds = crate::graph::predicate_hierarchy::HierarchyPredicates::resolve(self)?;
        self.add_triple(&Triple::new(specific, preds.generalizes, general))
    }

    /// Declare that `predicate` has an inverse `inverse_pred` (genlInverse).
    ///
    /// `predicate(X, Y)` ↔ `inverse_pred(Y, X)`.
    pub fn add_predicate_inverse(
        &self,
        predicate: SymbolId,
        inverse_pred: SymbolId,
    ) -> AkhResult<()> {
        let preds = crate::graph::predicate_hierarchy::HierarchyPredicates::resolve(self)?;
        self.add_triple(&Triple::new(predicate, preds.inverse, inverse_pred))
    }

    /// Build the predicate hierarchy from current KG state.
    ///
    /// Returns the cached hierarchy for use in hierarchy-aware queries.
    pub fn build_predicate_hierarchy(
        &self,
    ) -> AkhResult<crate::graph::predicate_hierarchy::PredicateHierarchy> {
        crate::graph::predicate_hierarchy::PredicateHierarchy::build(self)
    }

    /// Query objects using predicate hierarchy (specialization + inverse inference).
    ///
    /// Returns `(actual_predicate, object)` pairs. The `actual_predicate` may differ
    /// from the queried predicate if the result was found via a specialization or inverse.
    pub fn query_with_hierarchy(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        hierarchy: &crate::graph::predicate_hierarchy::PredicateHierarchy,
    ) -> Vec<(SymbolId, SymbolId)> {
        crate::graph::predicate_hierarchy::objects_with_hierarchy_and_inverse(
            self, subject, predicate, hierarchy,
        )
    }

    // -----------------------------------------------------------------------
    // Defeasible Reasoning (Phase 9d)
    // -----------------------------------------------------------------------

    /// Resolve defeasible predicates from the registry.
    pub fn defeasible_predicates(
        &self,
    ) -> AkhResult<crate::graph::defeasible::DefeasiblePredicates> {
        crate::graph::defeasible::DefeasiblePredicates::resolve(self)
    }

    /// Mark a (subject, predicate) pair as monotonically true.
    ///
    /// Monotonic assertions are never overridden by defaults, even when
    /// a more specific type provides a conflicting value.
    pub fn mark_monotonic(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
    ) -> AkhResult<()> {
        let preds = crate::graph::defeasible::DefeasiblePredicates::resolve(self)?;
        crate::graph::defeasible::mark_monotonic(self, subject, predicate, &preds)
    }

    /// Register an exception: `general` defeasible:except `specific`.
    ///
    /// This declares that `specific`'s assertions override `general`'s
    /// for conflicting predicates.
    pub fn register_exception(
        &self,
        general: SymbolId,
        specific: SymbolId,
    ) -> AkhResult<()> {
        let preds = crate::graph::defeasible::DefeasiblePredicates::resolve(self)?;
        crate::graph::defeasible::register_exception(self, general, specific, &preds)
    }

    /// Query with defeasible conflict resolution.
    ///
    /// Searches the type hierarchy for conflicting answers to `(subject, predicate, ?)`
    /// and returns the winning answer using specificity, monotonicity, exceptions,
    /// recency, and confidence as override criteria.
    pub fn query_defeasible(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
    ) -> AkhResult<Option<crate::graph::defeasible::DefeasibleResult>> {
        let preds = crate::graph::defeasible::DefeasiblePredicates::resolve(self)?;
        Ok(crate::graph::defeasible::query_defeasible(
            self, subject, predicate, &preds,
        ))
    }

    /// Resolve a conflict among explicitly provided competing triples.
    pub fn resolve_conflict(
        &self,
        candidates: &[Triple],
    ) -> AkhResult<Option<crate::graph::defeasible::DefeasibleResult>> {
        let preds = crate::graph::defeasible::DefeasiblePredicates::resolve(self)?;
        Ok(crate::graph::defeasible::resolve_conflict(
            self, candidates, &preds,
        ))
    }

    // -----------------------------------------------------------------------
    // Competitive Reasoner Dispatch (Phase 9f)
    // -----------------------------------------------------------------------

    /// Create a reasoner registry pre-populated with all built-in reasoners.
    pub fn reasoner_registry(
        &self,
        budget: std::time::Duration,
    ) -> crate::dispatch::ReasonerRegistry {
        crate::dispatch::ReasonerRegistry::with_builtins(budget)
    }

    /// Dispatch a problem through the competitive reasoner system.
    ///
    /// Collects bids from all registered reasoners, runs the cheapest
    /// applicable one, and falls back to the next bidder on failure.
    pub fn dispatch(
        &self,
        problem: &crate::dispatch::Problem,
    ) -> AkhResult<(crate::dispatch::ReasonerOutput, crate::dispatch::DispatchTrace)> {
        let registry = crate::dispatch::ReasonerRegistry::with_builtins(
            std::time::Duration::from_secs(5),
        );
        Ok(registry.dispatch(problem, self)?)
    }

    /// Dispatch with a custom time budget per reasoner.
    pub fn dispatch_with_budget(
        &self,
        problem: &crate::dispatch::Problem,
        budget: std::time::Duration,
    ) -> AkhResult<(crate::dispatch::ReasonerOutput, crate::dispatch::DispatchTrace)> {
        let registry = crate::dispatch::ReasonerRegistry::with_builtins(budget);
        Ok(registry.dispatch_with_budget(problem, self, budget)?)
    }

    // -----------------------------------------------------------------------
    // Role assignment
    // -----------------------------------------------------------------------

    /// Assign a role to this workspace's agent. Write-once: errors if already assigned.
    pub fn assign_role(&self, role: &str) -> AkhResult<()> {
        if let Some(existing) = self
            .store
            .get_meta(b"ws:role")?
            .map(|b| String::from_utf8_lossy(&b).to_string())
        {
            return Err(WorkspaceError::RoleAlreadyAssigned {
                current_role: existing,
            }
            .into());
        }
        self.ingest_label_triples(&[("self".into(), "has-role".into(), role.into(), 1.0)])?;
        self.store.put_meta(b"ws:role", role.as_bytes())?;
        Ok(())
    }

    /// Get the assigned role, if any.
    pub fn assigned_role(&self) -> Option<String> {
        self.store
            .get_meta(b"ws:role")
            .ok()
            .flatten()
            .map(|b| String::from_utf8_lossy(&b).to_string())
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
        let encoded =
            bincode::serialize(&next).map_err(|e| crate::error::StoreError::Serialization {
                message: format!("failed to serialize allocator state: {e}"),
            })?;
        self.store.put_meta(b"sym_allocator_next", &encoded)?;

        // Persist learned equivalences.
        self.entity_resolver
            .read()
            .unwrap()
            .persist_to_store(&self.store)?;

        // Sync knowledge graph to SPARQL store.
        if let Some(ref sparql) = self.sparql {
            sparql.sync_from(&self.knowledge_graph)?;
        }
        Ok(())
    }
}

/// Summary information about the engine state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
            .field(
                "learned_equivalences",
                &self.entity_resolver.read().unwrap().learned_count(),
            )
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
        assert!(
            result.symbols_created >= 2,
            "expected at least 2 new symbols"
        );
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
