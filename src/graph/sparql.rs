//! Persistent SPARQL RDF graph backed by oxigraph.
//!
//! Provides durable storage of triples and SPARQL query capabilities.
//! The in-memory `KnowledgeGraph` can be synced to this store for persistence.

use oxigraph::model::{GraphNameRef, NamedNode, Quad};
use oxigraph::sparql::{QueryResults, SparqlEvaluator};
use oxigraph::store::Store;

use crate::error::GraphError;
use crate::symbol::SymbolId;

use super::Triple;
use super::index::{GraphResult, KnowledgeGraph};

/// IRI namespace for akh-medu symbols.
const AKH_NS: &str = "https://akh-medu.dev/sym/";

/// IRI namespace for compartment named graphs.
const COMPARTMENT_NS: &str = "https://akh-medu.dev/compartment/";

/// Persistent SPARQL-capable RDF store.
pub struct SparqlStore {
    store: Store,
}

impl SparqlStore {
    /// Create a new in-memory SPARQL store (no persistence).
    pub fn in_memory() -> GraphResult<Self> {
        let store = Store::new().map_err(|e| GraphError::Sparql {
            message: format!("failed to create oxigraph store: {e}"),
        })?;
        Ok(Self { store })
    }

    /// Open or create a persistent SPARQL store at the given path.
    ///
    /// When the `sparql-persist` feature is disabled (no libclang / RocksDB available),
    /// this silently falls back to an in-memory store so the rest of the engine
    /// can still compile and run. Production deployments should enable the feature.
    pub fn open(path: &std::path::Path) -> GraphResult<Self> {
        #[cfg(feature = "sparql-persist")]
        {
            std::fs::create_dir_all(path).map_err(|e| GraphError::Sparql {
                message: format!("failed to create oxigraph directory: {e}"),
            })?;
            let store = Store::open(path).map_err(|e| GraphError::Sparql {
                message: format!("failed to open oxigraph store at {}: {e}", path.display()),
            })?;
            return Ok(Self { store });
        }
        // sparql-persist not enabled — use an in-memory store instead.
        #[cfg(not(feature = "sparql-persist"))]
        {
            let _ = path;
            Self::in_memory()
        }
    }

    /// Convert a SymbolId to an IRI NamedNode.
    fn symbol_to_iri(symbol: SymbolId) -> NamedNode {
        NamedNode::new(format!("{AKH_NS}{}", symbol.get())).expect("valid IRI")
    }

    /// Convert a compartment ID to its named-graph IRI.
    pub(crate) fn compartment_to_iri(compartment_id: &str) -> GraphResult<NamedNode> {
        NamedNode::new(format!("{COMPARTMENT_NS}{compartment_id}")).map_err(|e| {
            GraphError::Sparql {
                message: format!("invalid compartment IRI for compartment {compartment_id:?}: {e}"),
            }
        })
    }

    /// Try to parse a SymbolId from an IRI string.
    pub(crate) fn iri_to_symbol(iri: &str) -> Option<SymbolId> {
        let id_str = iri.strip_prefix(AKH_NS)?;
        let raw: u64 = id_str.parse().ok()?;
        SymbolId::new(raw)
    }

    /// Insert a triple into the SPARQL store.
    ///
    /// If the triple has a `compartment_id`, it is inserted into the corresponding
    /// named graph. Otherwise it goes into the default graph.
    pub fn insert_triple(&self, triple: &Triple) -> GraphResult<()> {
        self.insert_triple_in_graph(triple, triple.compartment_id.as_deref())
    }

    /// Insert a triple into a specific compartment graph (or DefaultGraph if None).
    pub fn insert_triple_in_graph(
        &self,
        triple: &Triple,
        compartment_id: Option<&str>,
    ) -> GraphResult<()> {
        let subject = Self::symbol_to_iri(triple.subject);
        let predicate = Self::symbol_to_iri(triple.predicate);
        let object = Self::symbol_to_iri(triple.object);

        match compartment_id {
            Some(id) => {
                let graph_iri = Self::compartment_to_iri(id)?;
                let quad = Quad::new(subject, predicate, object, graph_iri.as_ref());
                self.store.insert(&quad).map_err(|e| GraphError::Sparql {
                    message: format!("insert into graph {id} failed: {e}"),
                })?;
            }
            None => {
                let quad = Quad::new(subject, predicate, object, GraphNameRef::DefaultGraph);
                self.store.insert(&quad).map_err(|e| GraphError::Sparql {
                    message: format!("insert failed: {e}"),
                })?;
            }
        }

        Ok(())
    }

    /// Execute a SPARQL SELECT query scoped to a specific compartment graph.
    ///
    /// Wraps the query with a `FROM <graph_iri>` clause.
    pub fn query_in_graph(
        &self,
        sparql: &str,
        compartment_id: &str,
    ) -> GraphResult<Vec<Vec<(String, String)>>> {
        let graph_iri = Self::compartment_to_iri(compartment_id)?;
        // Inject FROM clause after SELECT
        let scoped = if let Some(rest) = sparql.strip_prefix("SELECT") {
            format!("SELECT FROM <{}>{rest}", graph_iri.as_str())
        } else {
            sparql.to_string()
        };
        self.query_select(&scoped)
    }

    /// Remove a single triple from the default graph.
    pub fn remove_triple(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        object: SymbolId,
    ) -> GraphResult<()> {
        let s = Self::symbol_to_iri(subject);
        let p = Self::symbol_to_iri(predicate);
        let o = Self::symbol_to_iri(object);
        let quad = Quad::new(s, p, o, GraphNameRef::DefaultGraph);
        self.store.remove(&quad).map_err(|e| GraphError::Sparql {
            message: format!("remove triple failed: {e}"),
        })?;
        Ok(())
    }

    /// Remove all triples in a named compartment graph.
    pub fn remove_graph(&self, compartment_id: &str) -> GraphResult<()> {
        let graph_iri = Self::compartment_to_iri(compartment_id)?;
        let drop_query = format!("DROP GRAPH <{}>", graph_iri.as_str());
        // Oxigraph may not support DROP GRAPH via query(), so we use update().
        self.store
            .update(&drop_query)
            .map_err(|e| GraphError::Sparql {
                message: format!("failed to drop graph {compartment_id}: {e}"),
            })?;
        Ok(())
    }

    /// Retrieve all triples from the SPARQL store as in-memory Triple objects.
    /// Used to restore the KnowledgeGraph on engine restart.
    ///
    /// Queries both the default graph and all named compartment graphs using a
    /// SPARQL UNION so that skill/compartment triples (stored in named graphs
    /// via their compartment_id) are also restored. Without the UNION, only
    /// default-graph triples are returned and the in-memory KG is empty after
    /// restart, causing VSA grounding to find zero relations.
    ///
    /// Note: confidence values are not currently stored in SPARQL, so restored
    /// triples will have default confidence of 1.0.
    pub fn all_triples(&self) -> GraphResult<Vec<Triple>> {
        // The UNION includes default-graph triples ({ ?s ?p ?o }) and all
        // named-graph triples (GRAPH ?g { ?s ?p ?o }).  Deduplication is not
        // required because a triple is only ever inserted into one graph
        // (either DefaultGraph or a specific compartment named graph).
        let results = SparqlEvaluator::new()
            .parse_query("SELECT ?s ?p ?o WHERE { { ?s ?p ?o } UNION { GRAPH ?g { ?s ?p ?o } } }")
            .map_err(|e| GraphError::Sparql {
                message: format!("SPARQL parse failed: {e}"),
            })?
            .on_store(&self.store)
            .execute()
            .map_err(|e| GraphError::Sparql {
                message: format!("SPARQL all_triples query failed: {e}"),
            })?;

        let mut triples = Vec::new();
        match results {
            QueryResults::Solutions(solutions) => {
                for solution in solutions {
                    let solution = solution.map_err(|e| GraphError::Sparql {
                        message: format!("solution error: {e}"),
                    })?;
                    let s_term = solution.get("s");
                    let p_term = solution.get("p");
                    let o_term = solution.get("o");

                    if let (Some(s), Some(p), Some(o)) = (s_term, p_term, o_term) {
                        let s_iri = s
                            .to_string()
                            .trim_matches('<')
                            .trim_matches('>')
                            .to_string();
                        let p_iri = p
                            .to_string()
                            .trim_matches('<')
                            .trim_matches('>')
                            .to_string();
                        let o_iri = o
                            .to_string()
                            .trim_matches('<')
                            .trim_matches('>')
                            .to_string();

                        if let (Some(subject), Some(predicate), Some(object)) = (
                            Self::iri_to_symbol(&s_iri),
                            Self::iri_to_symbol(&p_iri),
                            Self::iri_to_symbol(&o_iri),
                        ) {
                            triples.push(Triple::new(subject, predicate, object));
                        }
                    }
                }
            }
            _ => {
                return Err(GraphError::Sparql {
                    message: "unexpected result type from all_triples query".into(),
                });
            }
        }

        Ok(triples)
    }

    /// Atomically replace the SPARQL store with the in-memory knowledge graph.
    ///
    /// A plain insert-only sync leaves retracted or host-stale triples behind,
    /// which is especially dangerous after a Postgres cold wake.  One
    /// Oxigraph transaction makes the rich in-memory snapshot authoritative.
    pub fn sync_from(&self, kg: &KnowledgeGraph) -> GraphResult<usize> {
        let triples = kg.all_triples();
        let count = triples.len();
        // Build every quad before opening the replacement transaction.  In
        // particular, an invalid compartment must not clear the live store
        // before its named-graph IRI is rejected.
        let quads = triples
            .iter()
            .map(|triple| {
                let subject = Self::symbol_to_iri(triple.subject);
                let predicate = Self::symbol_to_iri(triple.predicate);
                let object = Self::symbol_to_iri(triple.object);
                match triple.compartment_id.as_deref() {
                    Some(id) => Ok(Quad::new(
                        subject,
                        predicate,
                        object,
                        Self::compartment_to_iri(id)?,
                    )),
                    None => Ok(Quad::new(
                        subject,
                        predicate,
                        object,
                        GraphNameRef::DefaultGraph,
                    )),
                }
            })
            .collect::<GraphResult<Vec<_>>>()?;
        let mut transaction = self
            .store
            .start_transaction()
            .map_err(|e| GraphError::Sparql {
                message: format!("start SPARQL replacement transaction failed: {e}"),
            })?;
        transaction.clear().map_err(|e| GraphError::Sparql {
            message: format!("clear SPARQL replacement transaction failed: {e}"),
        })?;
        for quad in &quads {
            transaction.insert(quad);
        }
        transaction.commit().map_err(|e| GraphError::Sparql {
            message: format!("commit SPARQL replacement transaction failed: {e}"),
        })?;
        Ok(count)
    }

    /// Execute a SPARQL SELECT query and return results as Vec of binding maps.
    pub fn query_select(&self, sparql: &str) -> GraphResult<Vec<Vec<(String, String)>>> {
        let results = SparqlEvaluator::new()
            .parse_query(sparql)
            .map_err(|e| GraphError::Sparql {
                message: format!("SPARQL parse failed: {e}"),
            })?
            .on_store(&self.store)
            .execute()
            .map_err(|e| GraphError::Sparql {
                message: format!("SPARQL query failed: {e}"),
            })?;

        match results {
            QueryResults::Solutions(solutions) => {
                let mut rows = Vec::new();
                for solution in solutions {
                    let solution = solution.map_err(|e| GraphError::Sparql {
                        message: format!("solution error: {e}"),
                    })?;
                    let mut row = Vec::new();
                    for (var, term) in solution.iter() {
                        row.push((var.to_string(), term.to_string()));
                    }
                    rows.push(row);
                }
                Ok(rows)
            }
            QueryResults::Boolean(b) => Ok(vec![vec![("result".to_string(), b.to_string())]]),
            QueryResults::Graph(_) => Err(GraphError::Sparql {
                message: "CONSTRUCT/DESCRIBE queries not supported via query_select".into(),
            }),
        }
    }

    /// Execute a SPARQL ASK query.
    pub fn query_ask(&self, sparql: &str) -> GraphResult<bool> {
        let results = SparqlEvaluator::new()
            .parse_query(sparql)
            .map_err(|e| GraphError::Sparql {
                message: format!("SPARQL parse failed: {e}"),
            })?
            .on_store(&self.store)
            .execute()
            .map_err(|e| GraphError::Sparql {
                message: format!("SPARQL query failed: {e}"),
            })?;
        match results {
            QueryResults::Boolean(b) => Ok(b),
            _ => Err(GraphError::Sparql {
                message: "expected boolean result from ASK query".into(),
            }),
        }
    }

    /// Get the number of triples in the store (all graphs).
    pub fn len(&self) -> GraphResult<usize> {
        let results = self.query_select(
            "SELECT (COUNT(*) AS ?count) WHERE { { ?s ?p ?o } UNION { GRAPH ?g { ?s ?p ?o } } }",
        )?;
        if let Some(row) = results.first()
            && let Some((_, val)) = row.first()
        {
            // oxigraph returns count as a typed literal like "\"3\"^^<http://...#integer>"
            let count_str = val
                .trim_matches('"')
                .split('^')
                .next()
                .unwrap_or("0")
                .trim_matches('"');
            return Ok(count_str.parse().unwrap_or(0));
        }
        Ok(0)
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> GraphResult<bool> {
        self.len().map(|n| n == 0)
    }

    /// Get internal store reference (for advanced oxigraph operations).
    pub fn store(&self) -> &Store {
        &self.store
    }
}

impl std::fmt::Debug for SparqlStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparqlStore").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    fn assert_invalid_compartment(error: GraphError) {
        match error {
            GraphError::Sparql { message } => {
                assert!(message.contains("invalid compartment IRI"), "{message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn iri_roundtrip() {
        let id = sym(42);
        let iri = SparqlStore::symbol_to_iri(id);
        let recovered = SparqlStore::iri_to_symbol(iri.as_str()).unwrap();
        assert_eq!(id, recovered);
    }

    #[test]
    fn insert_and_query() {
        let store = SparqlStore::in_memory().unwrap();
        let triple = Triple::new(sym(1), sym(2), sym(3));
        store.insert_triple(&triple).unwrap();

        let results = store
            .query_select("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn compartment_operations_reject_malformed_iris() {
        let store = SparqlStore::in_memory().unwrap();
        let triple = Triple::new(sym(1), sym(2), sym(3)).with_compartment("bad compartment".into());

        assert_invalid_compartment(store.insert_triple(&triple).unwrap_err());
        assert_invalid_compartment(
            store
                .query_in_graph("SELECT ?s ?p ?o WHERE { ?s ?p ?o }", "bad compartment")
                .unwrap_err(),
        );
        assert_invalid_compartment(store.remove_graph("bad compartment").unwrap_err());
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn sync_from_knowledge_graph() {
        let kg = KnowledgeGraph::new();
        kg.insert_triple(&Triple::new(sym(1), sym(10), sym(2)))
            .unwrap();
        kg.insert_triple(&Triple::new(sym(2), sym(10), sym(3)))
            .unwrap();

        let store = SparqlStore::in_memory().unwrap();
        store
            .insert_triple(&Triple::new(sym(99), sym(10), sym(100)))
            .unwrap();
        let synced = store.sync_from(&kg).unwrap();
        assert_eq!(synced, 2);
        assert_eq!(store.len().unwrap(), 2);
        assert!(
            !store
                .query_ask(&format!("ASK {{ <{AKH_NS}99> ?p ?o }}"))
                .unwrap()
        );
    }

    #[test]
    fn sync_from_rejects_malformed_compartment_without_replacing_store() {
        let store = SparqlStore::in_memory().unwrap();
        store
            .insert_triple(&Triple::new(sym(99), sym(10), sym(100)))
            .unwrap();

        let kg = KnowledgeGraph::new();
        kg.insert_triple(&Triple::new(sym(1), sym(10), sym(2)))
            .unwrap();
        kg.insert_triple(
            &Triple::new(sym(2), sym(10), sym(3)).with_compartment("bad compartment".into()),
        )
        .unwrap();

        assert_invalid_compartment(store.sync_from(&kg).unwrap_err());
        assert_eq!(store.len().unwrap(), 1);
        assert!(
            store
                .query_ask(&format!("ASK {{ <{AKH_NS}99> ?p ?o }}"))
                .unwrap()
        );
        assert!(
            !store
                .query_ask(&format!("ASK {{ <{AKH_NS}1> ?p ?o }}"))
                .unwrap()
        );
    }

    #[test]
    fn ask_query() {
        let store = SparqlStore::in_memory().unwrap();
        let triple = Triple::new(sym(1), sym(2), sym(3));
        store.insert_triple(&triple).unwrap();

        let iri = format!("{AKH_NS}1");
        let exists = store
            .query_ask(&format!("ASK {{ <{iri}> ?p ?o }}"))
            .unwrap();
        assert!(exists);

        let not_exists = store
            .query_ask(&format!("ASK {{ <{AKH_NS}999> ?p ?o }}"))
            .unwrap();
        assert!(!not_exists);
    }
}
