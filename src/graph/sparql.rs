//! Persistent SPARQL RDF graph backed by oxigraph.
//!
//! Provides durable storage of triples and SPARQL query capabilities.
//! The in-memory `KnowledgeGraph` can be synced to this store for persistence.

use oxigraph::model::{GraphNameRef, NamedNode, Quad};
use oxigraph::sparql::QueryResults;
use oxigraph::store::Store;

use crate::error::GraphError;
use crate::symbol::SymbolId;

use super::index::{GraphResult, KnowledgeGraph};
use super::Triple;

/// IRI namespace for akh-medu symbols.
const AKH_NS: &str = "https://akh-medu.dev/sym/";

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
    pub fn open(path: &std::path::Path) -> GraphResult<Self> {
        std::fs::create_dir_all(path).map_err(|e| GraphError::Sparql {
            message: format!("failed to create oxigraph directory: {e}"),
        })?;
        let store = Store::open(path).map_err(|e| GraphError::Sparql {
            message: format!("failed to open oxigraph store at {}: {e}", path.display()),
        })?;
        Ok(Self { store })
    }

    /// Convert a SymbolId to an IRI NamedNode.
    fn symbol_to_iri(symbol: SymbolId) -> NamedNode {
        NamedNode::new(format!("{AKH_NS}{}", symbol.get())).expect("valid IRI")
    }

    /// Try to parse a SymbolId from an IRI string.
    pub(crate) fn iri_to_symbol(iri: &str) -> Option<SymbolId> {
        let id_str = iri.strip_prefix(AKH_NS)?;
        let raw: u64 = id_str.parse().ok()?;
        SymbolId::new(raw)
    }

    /// Insert a triple into the SPARQL store.
    pub fn insert_triple(&self, triple: &Triple) -> GraphResult<()> {
        let subject = Self::symbol_to_iri(triple.subject);
        let predicate = Self::symbol_to_iri(triple.predicate);
        let object = Self::symbol_to_iri(triple.object);

        let quad = Quad::new(
            subject,
            predicate,
            object,
            GraphNameRef::DefaultGraph,
        );

        self.store.insert(&quad).map_err(|e| GraphError::Sparql {
            message: format!("insert failed: {e}"),
        })?;

        Ok(())
    }

    /// Retrieve all triples from the SPARQL store as in-memory Triple objects.
    /// Used to restore the KnowledgeGraph on engine restart.
    ///
    /// Note: confidence values are not currently stored in SPARQL, so restored
    /// triples will have default confidence of 1.0.
    // TODO: Store confidence via reification or named graphs to preserve across restarts.
    pub fn all_triples(&self) -> GraphResult<Vec<Triple>> {
        let results = self.store.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }").map_err(|e| {
            GraphError::Sparql {
                message: format!("SPARQL all_triples query failed: {e}"),
            }
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
                        let s_iri = s.to_string().trim_matches('<').trim_matches('>').to_string();
                        let p_iri = p.to_string().trim_matches('<').trim_matches('>').to_string();
                        let o_iri = o.to_string().trim_matches('<').trim_matches('>').to_string();

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

    /// Sync all triples from an in-memory KnowledgeGraph to the SPARQL store.
    pub fn sync_from(&self, kg: &KnowledgeGraph) -> GraphResult<usize> {
        let triples = kg.all_triples();
        let count = triples.len();
        for triple in &triples {
            self.insert_triple(triple)?;
        }
        Ok(count)
    }

    /// Execute a SPARQL SELECT query and return results as Vec of binding maps.
    pub fn query_select(
        &self,
        sparql: &str,
    ) -> GraphResult<Vec<Vec<(String, String)>>> {
        let results = self.store.query(sparql).map_err(|e| GraphError::Sparql {
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
        let results = self.store.query(sparql).map_err(|e| GraphError::Sparql {
            message: format!("SPARQL query failed: {e}"),
        })?;
        match results {
            QueryResults::Boolean(b) => Ok(b),
            _ => Err(GraphError::Sparql {
                message: "expected boolean result from ASK query".into(),
            }),
        }
    }

    /// Get the number of triples in the store.
    pub fn len(&self) -> GraphResult<usize> {
        let results = self.query_select("SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }")?;
        if let Some(row) = results.first() {
            if let Some((_, val)) = row.first() {
                // oxigraph returns count as a typed literal like "\"3\"^^<http://...#integer>"
                let count_str = val
                    .trim_matches('"')
                    .split('^')
                    .next()
                    .unwrap_or("0")
                    .trim_matches('"');
                return Ok(count_str.parse().unwrap_or(0));
            }
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
    fn sync_from_knowledge_graph() {
        let kg = KnowledgeGraph::new();
        kg.insert_triple(&Triple::new(sym(1), sym(10), sym(2)))
            .unwrap();
        kg.insert_triple(&Triple::new(sym(2), sym(10), sym(3)))
            .unwrap();

        let store = SparqlStore::in_memory().unwrap();
        let synced = store.sync_from(&kg).unwrap();
        assert_eq!(synced, 2);
        assert_eq!(store.len().unwrap(), 2);
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
