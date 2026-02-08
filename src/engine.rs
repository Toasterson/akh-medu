//! Engine facade: top-level API for the akh-medu system.
//!
//! The `Engine` owns all subsystems and provides the public interface
//! for ingesting knowledge, querying, and managing the system.

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{AkhResult, EngineError};
use crate::graph::index::KnowledgeGraph;
use crate::graph::sparql::SparqlStore;
use crate::graph::Triple;
use crate::infer::engine::InferEngine;
use crate::infer::{InferenceQuery, InferenceResult};
use crate::simd;
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dimension: Dimension::DEFAULT,
            encoding: Encoding::Bipolar,
            data_dir: None,
            max_memory_mb: 1024,
            max_symbols: 1_000_000,
        }
    }
}

/// The akh-medu neuro-symbolic AI engine.
///
/// Owns all subsystems: VSA operations, item memory, knowledge graph,
/// storage tiers, and symbol allocator.
pub struct Engine {
    config: EngineConfig,
    ops: Arc<VsaOps>,
    item_memory: Arc<ItemMemory>,
    knowledge_graph: Arc<KnowledgeGraph>,
    sparql: Option<SparqlStore>,
    store: Arc<TieredStore>,
    symbol_allocator: Arc<AtomicSymbolAllocator>,
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

        let (store, sparql) = if let Some(ref dir) = config.data_dir {
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
            (store, Some(sparql))
        } else {
            (TieredStore::memory_only(), None)
        };

        Ok(Self {
            config,
            ops,
            item_memory,
            knowledge_graph,
            sparql,
            store: Arc::new(store),
            symbol_allocator: Arc::new(AtomicSymbolAllocator::new()),
        })
    }

    /// Allocate a new symbol with the given kind and label.
    pub fn create_symbol(&self, kind: SymbolKind, label: impl Into<String>) -> AkhResult<SymbolMeta> {
        let id = self.symbol_allocator.next_id()?;
        let meta = SymbolMeta::new(id, kind, label);

        // Store metadata
        let encoded = bincode::serialize(&meta).map_err(|e| {
            crate::error::StoreError::Serialization {
                message: format!("failed to serialize symbol meta: {e}"),
            }
        })?;
        self.store.put(id, encoded);

        // Create hypervector in item memory
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

    /// Get the engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get system info (node count, triple count, symbol count, etc.)
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            dimension: self.config.dimension.0,
            encoding: self.config.encoding.to_string(),
            isa_level: self.ops.dim().0.to_string(), // placeholder
            symbol_count: self.item_memory.len(),
            node_count: self.knowledge_graph.node_count(),
            triple_count: self.knowledge_graph.triple_count(),
            store_hot_entries: self.store.hot_len(),
            persistent: self.config.data_dir.is_some(),
        }
    }

    /// Persist current state (sync knowledge graph to SPARQL store).
    pub fn persist(&self) -> AkhResult<()> {
        if let Some(ref sparql) = self.sparql {
            sparql.sync_from(&self.knowledge_graph)?;
        }
        Ok(())
    }

    /// Run spreading-activation inference from the given query.
    pub fn infer(&self, query: &InferenceQuery) -> AkhResult<InferenceResult> {
        let infer_engine = InferEngine::new(
            Arc::clone(&self.ops),
            Arc::clone(&self.item_memory),
            Arc::clone(&self.knowledge_graph),
        );
        Ok(infer_engine.infer(query)?)
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
        writeln!(f, "  persistent:   {}", self.persistent)?;
        Ok(())
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("config", &self.config)
            .field("item_memory", &self.item_memory)
            .field("knowledge_graph", &self.knowledge_graph)
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
}
