//! Multi-hop graph traversal and subgraph extraction.
//!
//! Provides BFS-based traversal from seed nodes with configurable depth limits,
//! predicate filters, and confidence thresholds.

use std::collections::{HashSet, VecDeque};

use crate::error::GraphError;
use crate::symbol::SymbolId;

use super::Triple;
use super::index::KnowledgeGraph;

/// Configuration for a graph traversal.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum hop depth from seed nodes.
    pub max_depth: usize,
    /// Only follow edges with these predicates (empty = follow all).
    pub predicate_filter: HashSet<SymbolId>,
    /// Minimum confidence to traverse an edge.
    pub min_confidence: f32,
    /// Maximum number of triples to collect.
    pub max_results: usize,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            predicate_filter: HashSet::new(),
            min_confidence: 0.0,
            max_results: 10_000,
        }
    }
}

/// Result of a multi-hop traversal.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// All triples discovered during traversal.
    pub triples: Vec<Triple>,
    /// All nodes visited.
    pub visited: HashSet<SymbolId>,
    /// Maximum depth actually reached.
    pub depth_reached: usize,
}

/// Perform a BFS traversal from seed nodes.
///
/// Explores outgoing edges from each seed, collecting triples up to `max_depth` hops.
pub fn traverse_bfs(
    graph: &KnowledgeGraph,
    seeds: &[SymbolId],
    config: &TraversalConfig,
) -> Result<TraversalResult, GraphError> {
    let mut visited: HashSet<SymbolId> = HashSet::new();
    let mut collected_triples: Vec<Triple> = Vec::new();
    let mut depth_reached: usize = 0;

    // BFS queue: (node, current_depth)
    let mut queue: VecDeque<(SymbolId, usize)> = VecDeque::new();

    for &seed in seeds {
        if visited.insert(seed) {
            queue.push_back((seed, 0));
        }
    }

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= config.max_depth {
            continue;
        }
        if collected_triples.len() >= config.max_results {
            break;
        }

        let outgoing = graph.triples_from(node);
        for triple in outgoing {
            // Apply predicate filter
            if !config.predicate_filter.is_empty()
                && !config.predicate_filter.contains(&triple.predicate)
            {
                continue;
            }

            // Apply confidence threshold
            if triple.confidence < config.min_confidence {
                continue;
            }

            collected_triples.push(triple.clone());
            depth_reached = depth_reached.max(depth + 1);

            if visited.insert(triple.object) {
                queue.push_back((triple.object, depth + 1));
            }
        }
    }

    Ok(TraversalResult {
        triples: collected_triples,
        visited,
        depth_reached,
    })
}

/// Extract the subgraph reachable from seeds within the configured depth.
///
/// Returns all unique nodes and triples in the reachable subgraph.
pub fn extract_subgraph(
    graph: &KnowledgeGraph,
    seeds: &[SymbolId],
    max_depth: usize,
) -> Result<TraversalResult, GraphError> {
    traverse_bfs(
        graph,
        seeds,
        &TraversalConfig {
            max_depth,
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    fn build_chain_graph() -> KnowledgeGraph {
        // A --r--> B --r--> C --r--> D
        let kg = KnowledgeGraph::new();
        let r = sym(100);
        kg.insert_triple(&Triple::new(sym(1), r, sym(2))).unwrap();
        kg.insert_triple(&Triple::new(sym(2), r, sym(3))).unwrap();
        kg.insert_triple(&Triple::new(sym(3), r, sym(4))).unwrap();
        kg
    }

    #[test]
    fn bfs_traversal_depth_1() {
        let kg = build_chain_graph();
        let result = traverse_bfs(
            &kg,
            &[sym(1)],
            &TraversalConfig {
                max_depth: 1,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.triples.len(), 1);
        assert!(result.visited.contains(&sym(1)));
        assert!(result.visited.contains(&sym(2)));
        assert_eq!(result.depth_reached, 1);
    }

    #[test]
    fn bfs_traversal_full_chain() {
        let kg = build_chain_graph();
        let result = traverse_bfs(
            &kg,
            &[sym(1)],
            &TraversalConfig {
                max_depth: 10,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.triples.len(), 3);
        assert_eq!(result.visited.len(), 4); // 1, 2, 3, 4 (predicate nodes are not followed)
        assert_eq!(result.depth_reached, 3);
    }

    #[test]
    fn predicate_filter() {
        let kg = KnowledgeGraph::new();
        let r1 = sym(100);
        let r2 = sym(101);
        kg.insert_triple(&Triple::new(sym(1), r1, sym(2))).unwrap();
        kg.insert_triple(&Triple::new(sym(1), r2, sym(3))).unwrap();

        let mut filter = HashSet::new();
        filter.insert(r1);
        let result = traverse_bfs(
            &kg,
            &[sym(1)],
            &TraversalConfig {
                max_depth: 2,
                predicate_filter: filter,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].object, sym(2));
    }

    #[test]
    fn confidence_threshold() {
        let kg = KnowledgeGraph::new();
        let r = sym(100);
        kg.insert_triple(&Triple::new(sym(1), r, sym(2)).with_confidence(0.9))
            .unwrap();
        kg.insert_triple(&Triple::new(sym(1), r, sym(3)).with_confidence(0.3))
            .unwrap();

        let result = traverse_bfs(
            &kg,
            &[sym(1)],
            &TraversalConfig {
                max_depth: 1,
                min_confidence: 0.5,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].object, sym(2));
    }

    #[test]
    fn extract_subgraph_test() {
        let kg = build_chain_graph();
        let result = extract_subgraph(&kg, &[sym(1)], 2).unwrap();
        assert_eq!(result.triples.len(), 2);
        assert_eq!(result.depth_reached, 2);
    }
}
