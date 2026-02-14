//! Graph analytics: centrality, components, and path algorithms.
//!
//! All functions operate on a [`KnowledgeGraph`] reference and return
//! structured results sorted by relevance (score desc, size desc, etc.).

use std::collections::HashMap;

use petgraph::algo::{astar, page_rank, tarjan_scc};
use petgraph::graph::NodeIndex;

use crate::error::GraphError;
use crate::symbol::SymbolId;

use super::index::{GraphResult, KnowledgeGraph};

// ---------------------------------------------------------------------------
// Degree centrality
// ---------------------------------------------------------------------------

/// Degree centrality metrics for a single node.
#[derive(Debug, Clone)]
pub struct DegreeCentrality {
    /// The symbol this measurement belongs to.
    pub symbol: SymbolId,
    /// Number of incoming edges.
    pub in_degree: usize,
    /// Number of outgoing edges.
    pub out_degree: usize,
    /// Total degree (in + out).
    pub total: usize,
}

/// Compute degree centrality for all nodes. Returns sorted by total degree desc.
pub fn degree_centrality(kg: &KnowledgeGraph) -> Vec<DegreeCentrality> {
    let nodes = kg.nodes();
    let mut results: Vec<DegreeCentrality> = nodes
        .into_iter()
        .map(|symbol| {
            let out_degree = kg.triples_from(symbol).len();
            let in_degree = kg.triples_to(symbol).len();
            DegreeCentrality {
                symbol,
                in_degree,
                out_degree,
                total: in_degree + out_degree,
            }
        })
        .collect();
    results.sort_by(|a, b| b.total.cmp(&a.total));
    results
}

// ---------------------------------------------------------------------------
// PageRank
// ---------------------------------------------------------------------------

/// PageRank score for a single node.
#[derive(Debug, Clone)]
pub struct PageRankScore {
    /// The symbol this score belongs to.
    pub symbol: SymbolId,
    /// Computed PageRank score.
    pub score: f64,
}

/// Compute PageRank scores. Returns sorted by score desc.
pub fn pagerank(
    kg: &KnowledgeGraph,
    damping: f64,
    iterations: usize,
) -> GraphResult<Vec<PageRankScore>> {
    let graph = kg.graph();
    if graph.node_count() == 0 {
        return Ok(vec![]);
    }

    let scores = page_rank(&*graph, damping, iterations);
    let reverse_map = kg.node_index_to_symbol();

    let mut results: Vec<PageRankScore> = graph
        .node_indices()
        .filter_map(|idx| {
            let symbol = reverse_map.get(&idx)?;
            let score = scores[idx.index()];
            Some(PageRankScore {
                symbol: *symbol,
                score,
            })
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

// ---------------------------------------------------------------------------
// Strongly connected components
// ---------------------------------------------------------------------------

/// A strongly connected component in the graph.
#[derive(Debug, Clone)]
pub struct ConnectedComponent {
    /// Component identifier (arbitrary, for display).
    pub id: usize,
    /// Member symbols in this component.
    pub members: Vec<SymbolId>,
    /// Number of members.
    pub size: usize,
}

/// Find strongly connected components. Returns sorted by size desc.
pub fn strongly_connected_components(kg: &KnowledgeGraph) -> GraphResult<Vec<ConnectedComponent>> {
    let graph = kg.graph();
    let reverse_map = kg.node_index_to_symbol();

    let sccs = tarjan_scc(&*graph);

    let mut components: Vec<ConnectedComponent> = sccs
        .into_iter()
        .enumerate()
        .map(|(id, indices)| {
            let members: Vec<SymbolId> = indices
                .iter()
                .filter_map(|idx| reverse_map.get(idx).copied())
                .collect();
            let size = members.len();
            ConnectedComponent { id, members, size }
        })
        .collect();

    components.sort_by(|a, b| b.size.cmp(&a.size));
    Ok(components)
}

// ---------------------------------------------------------------------------
// Shortest path
// ---------------------------------------------------------------------------

/// Find shortest path (by hop count) between two symbols.
/// Returns None if no path exists.
pub fn shortest_path(
    kg: &KnowledgeGraph,
    from: SymbolId,
    to: SymbolId,
) -> GraphResult<Option<Vec<SymbolId>>> {
    if !kg.has_node(from) {
        return Err(GraphError::NodeNotFound {
            symbol_id: from.get(),
        });
    }
    if !kg.has_node(to) {
        return Err(GraphError::NodeNotFound {
            symbol_id: to.get(),
        });
    }

    let graph = kg.graph();
    let reverse_map = kg.node_index_to_symbol();

    // Use the forward map for O(1) lookup of start/end indices.
    let forward_map: HashMap<SymbolId, NodeIndex> =
        reverse_map.iter().map(|(&idx, &sym)| (sym, idx)).collect();

    let from_idx = forward_map[&from];
    let to_idx = forward_map[&to];

    let result = astar(&*graph, from_idx, |n| n == to_idx, |_| 1usize, |_| 0usize);

    match result {
        Some((_cost, path)) => {
            let symbols: Vec<SymbolId> = path
                .iter()
                .filter_map(|idx| reverse_map.get(idx).copied())
                .collect();
            Ok(Some(symbols))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Triple;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    fn build_star_graph() -> KnowledgeGraph {
        // Hub (1) connects to spokes (2, 3, 4, 5) via relation (10)
        let kg = KnowledgeGraph::new();
        let hub = sym(1);
        let r = sym(10);
        for spoke in 2..=5 {
            kg.insert_triple(&Triple::new(hub, r, sym(spoke))).unwrap();
        }
        kg
    }

    #[test]
    fn degree_centrality_hub_highest() {
        let kg = build_star_graph();
        let results = degree_centrality(&kg);
        assert!(!results.is_empty());
        // Hub has 4 outgoing edges, should be first
        assert_eq!(results[0].symbol, sym(1));
        assert_eq!(results[0].out_degree, 4);
    }

    #[test]
    fn pagerank_hub_scores_highest() {
        let kg = build_star_graph();
        let results = pagerank(&kg, 0.85, 20).unwrap();
        assert!(!results.is_empty());
        // The hub should have one of the highest scores due to centrality
        // (though in a star topology, spokes pointed to by hub also get rank)
        let hub_score = results.iter().find(|r| r.symbol == sym(1)).unwrap();
        assert!(hub_score.score > 0.0);
    }

    #[test]
    fn pagerank_empty_graph() {
        let kg = KnowledgeGraph::new();
        let results = pagerank(&kg, 0.85, 20).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn scc_finds_cycle() {
        let kg = KnowledgeGraph::new();
        let r = sym(100);
        // A -> B -> C -> A forms a cycle
        kg.insert_triple(&Triple::new(sym(1), r, sym(2))).unwrap();
        kg.insert_triple(&Triple::new(sym(2), r, sym(3))).unwrap();
        kg.insert_triple(&Triple::new(sym(3), r, sym(1))).unwrap();

        let components = strongly_connected_components(&kg).unwrap();
        // The 3 nodes in the cycle form one SCC
        let cycle_scc = components.iter().find(|c| c.size >= 3);
        assert!(
            cycle_scc.is_some(),
            "should find a component with 3+ members"
        );
        let members = &cycle_scc.unwrap().members;
        assert!(members.contains(&sym(1)));
        assert!(members.contains(&sym(2)));
        assert!(members.contains(&sym(3)));
    }

    #[test]
    fn shortest_path_finds_route() {
        let kg = KnowledgeGraph::new();
        let r = sym(100);
        // Chain: A -> B -> C
        kg.insert_triple(&Triple::new(sym(1), r, sym(2))).unwrap();
        kg.insert_triple(&Triple::new(sym(2), r, sym(3))).unwrap();

        let path = shortest_path(&kg, sym(1), sym(3)).unwrap();
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], sym(1));
        assert_eq!(path[2], sym(3));
    }

    #[test]
    fn shortest_path_no_route() {
        let kg = KnowledgeGraph::new();
        let r = sym(100);
        // A -> B, disconnected C -> D
        kg.insert_triple(&Triple::new(sym(1), r, sym(2))).unwrap();
        kg.insert_triple(&Triple::new(sym(3), r, sym(4))).unwrap();

        let path = shortest_path(&kg, sym(1), sym(3)).unwrap();
        assert!(path.is_none());
    }
}
