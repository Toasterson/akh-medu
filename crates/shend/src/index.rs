//! The resolution engine: parse → execute `rust.tsg` → stitch partial paths →
//! emit `code:resolves-to` records (explicitly unresolved when no binding is
//! found — never guessed, never name-matched silently).
//!
//! File-local and incremental (load-bearing WI-262 property): each file's
//! minimal partial-path set is computed from that file's stack graph alone —
//! no build, no dependencies present — and added to a shared
//! [`stack_graphs::stitching::Database`]. A reference whose path crosses into
//! an unindexed file yields no complete path **yet**; when the dependency is
//! indexed later, its partial paths compose in the database and the same
//! stitch completes. Nothing here performs whole-program analysis.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use stack_graphs::NoCancellation as GraphNoCancellation;
use stack_graphs::arena::Handle;
use stack_graphs::graph::{File, Node, StackGraph};
use stack_graphs::partial::PartialPaths;
use stack_graphs::stitching::{
    Database, DatabaseCandidates, ForwardPartialPathStitcher, StitcherConfig,
};
use tree_sitter_stack_graphs::NoCancellation as TsgNoCancellation;
use tree_sitter_stack_graphs::StackGraphLanguage;
use tree_sitter_stack_graphs::Variables;

use crate::provenance::{Provenance, Tier, Watermark};
use crate::store::{DefId, ResolutionStatus, ResolvesTo, Span};
use crate::{RUST_TSG, ShendError};

/// Per-file state retained after indexing.
#[derive(Debug)]
pub struct FileIndex {
    /// The stack-graph file handle for this source file.
    pub file: Handle<File>,
    /// sha256 of the source when it was indexed — the daemon re-keys on change.
    pub content_sha: String,
}

/// The name-binding resolver. One `Resolver` owns one [`StackGraph`], one
/// [`PartialPaths`] arena, and one shared [`Database`]; files are added with
/// [`Resolver::index_file`] (any order, dependencies absent is fine) and
/// queried with [`Resolver::resolutions_for_file`].
pub struct Resolver {
    sgl: StackGraphLanguage,
    graph: StackGraph,
    partials: PartialPaths,
    db: Database,
    files: HashMap<String, Handle<File>>,
}

impl Resolver {
    /// Construct a resolver for Rust sources. Parses and validates the
    /// embedded [`RUST_TSG`] ruleset against tree-sitter-rust — a ruleset that
    /// fails to load is a hard error here, not a silent gap downstream.
    pub fn new() -> Result<Self, ShendError> {
        let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        let sgl = StackGraphLanguage::from_str(language, RUST_TSG)
            .map_err(|e| ShendError::Ruleset(e.to_string()))?;
        Ok(Resolver {
            sgl,
            graph: StackGraph::new(),
            partials: PartialPaths::new(),
            db: Database::new(),
            files: HashMap::new(),
        })
    }

    /// Index a single Rust source file.
    ///
    /// `path` is the repository-relative path (`src/alpha.rs`) — the ruleset
    /// derives the module chain from it via the `FILE_PATH` global. Builds the
    /// file-local stack graph and adds the file's minimal partial-path set to
    /// the shared database. Re-indexing the same `path` replaces the previous
    /// build (the graph is re-created per call to `add_or_update_file` only
    /// when content changed — the caller decides via [`FileIndex::content_sha`]).
    pub fn index_file(&mut self, path: &str, source: &str) -> Result<FileIndex, ShendError> {
        let file = self.graph.get_or_create_file(path);
        let mut globals = Variables::new();
        // ROOT_NODE / JUMP_TO_SCOPE_NODE are injected by the builder; FILE_PATH
        // is injected from the file's name when absent — set it explicitly so
        // the module chain keys off the repo-relative path we were handed.
        globals
            .add(tree_sitter_stack_graphs::FILE_PATH_VAR.into(), path.into())
            .map_err(|e| ShendError::Ruleset(format!("FILE_PATH global: {e}")))?;
        self.sgl
            .build_stack_graph_into(&mut self.graph, file, source, &globals, &TsgNoCancellation)
            .map_err(|e| ShendError::Parse {
                file: path.to_string(),
                message: e.to_string(),
            })?;

        // Per-file minimal partial-path set, computed from this file's graph
        // alone. Incomplete (unresolvable-so-far) references simply contribute
        // no complete path yet; they are recorded explicitly on resolution.
        ForwardPartialPathStitcher::find_minimal_partial_path_set_in_file(
            &self.graph,
            &mut self.partials,
            file,
            StitcherConfig::default(),
            &GraphNoCancellation,
            |g, ps, p| {
                self.db.add_partial_path(g, ps, p.clone());
            },
        )
        .map_err(|e| ShendError::Parse {
            file: path.to_string(),
            message: format!("partial-path stitching: {e}"),
        })?;
        self.db.ensure_forwards(&mut self.partials);

        self.files.insert(path.to_string(), file);
        Ok(FileIndex {
            file,
            content_sha: crate::store::content_sha(source),
        })
    }

    /// Whether `path` has been indexed.
    pub fn is_indexed(&self, path: &str) -> bool {
        self.files.contains_key(path)
    }

    /// Resolve every reference in `path` against the shared database.
    ///
    /// Emits one [`ResolvesTo`] per reference node:
    /// - `status: resolved` with the [`DefId`] of the definition the complete
    ///   path terminates at (`tier: stackgraph`), or
    /// - `status: unresolved` with `target: None` — recorded explicitly, as
    ///   WI-262 requires. Nothing falls back to name matching.
    ///
    /// Takes `&mut self` because stitching appends candidate paths to the
    /// shared [`Database`]/[`PartialPaths`] arenas as a monotonic cache.
    pub fn resolutions_for_file(&mut self, path: &str, watermark: &Watermark) -> Vec<ResolvesTo> {
        let Some(&file) = self.files.get(path) else {
            return Vec::new();
        };
        let references: Vec<Handle<Node>> = self
            .graph
            .nodes_for_file(file)
            .filter(|n| self.graph[*n].is_reference())
            .collect();

        // reference node → completing definition nodes (the stitched complete
        // paths; each path starts at a reference endpoint and ends at a
        // definition endpoint).
        let mut bindings: BTreeMap<Handle<Node>, Vec<Handle<Node>>> = BTreeMap::new();
        {
            let mut candidates =
                DatabaseCandidates::new(&self.graph, &mut self.partials, &mut self.db);
            let _ = ForwardPartialPathStitcher::find_all_complete_partial_paths(
                &mut candidates,
                references.iter().copied(),
                StitcherConfig::default(),
                &GraphNoCancellation,
                |_g, _ps, p| {
                    bindings.entry(p.start_node).or_default().push(p.end_node);
                },
            );
        }

        let node_span = |n: Handle<Node>| -> Span {
            let si = self.graph.source_info(n);
            match si {
                Some(si) => Span {
                    start_row: si.span.start.line,
                    start_col: si.span.start.column.utf8_offset,
                    end_row: si.span.end.line,
                    end_col: si.span.end.column.utf8_offset,
                },
                None => Span {
                    start_row: 0,
                    start_col: 0,
                    end_row: 0,
                    end_col: 0,
                },
            }
        };
        let node_symbol = |n: Handle<Node>| -> String {
            self.graph[n]
                .symbol()
                .map(|s| self.graph[s].to_string())
                .unwrap_or_default()
        };

        references
            .iter()
            .map(|&r| {
                let prov = Provenance::new(Tier::Stackgraph, watermark.clone());
                let name = node_symbol(r);
                let reference_span = node_span(r);
                match bindings.get(&r).and_then(|defs| defs.first()) {
                    Some(&defn) if self.graph[defn].is_definition() => {
                        let def_file = self.graph[defn]
                            .file()
                            .map(|f| self.graph[f].name().to_string())
                            .unwrap_or_default();
                        ResolvesTo {
                            reference_file: path.to_string(),
                            reference_name: name,
                            reference_span,
                            target: Some(DefId {
                                module: module_of_file(&def_file),
                                file: def_file,
                                name: node_symbol(defn),
                                span: node_span(defn),
                            }),
                            status: ResolutionStatus::Resolved,
                            provenance: prov,
                        }
                    }
                    _ => ResolvesTo {
                        reference_file: path.to_string(),
                        reference_name: name,
                        reference_span,
                        target: None,
                        status: ResolutionStatus::Unresolved,
                        provenance: prov,
                    },
                }
            })
            .collect()
    }
}

/// `src/alpha.rs` → `crate::alpha`; `src/lib.rs`/`main.rs`/`mod.rs` → the
/// enclosing chain. Mirrors the ruleset's FILE_PATH module derivation so the
/// emitted [`DefId::module`] agrees with what the graph bound.
fn module_of_file(path: &str) -> String {
    let mut parts: Vec<&str> = path.split('/').collect();
    let stem = parts
        .last()
        .map(|f| f.strip_suffix(".rs").unwrap_or(f))
        .unwrap_or("");
    match stem {
        "lib" | "main" | "mod" => {
            parts.pop();
        }
        s => {
            let n = parts.len();
            parts[n - 1] = s;
        }
    }
    std::iter::once("crate")
        .chain(parts.into_iter().filter(|p| !p.is_empty()))
        .collect::<Vec<_>>()
        .join("::")
}

/// Naive name-matching baseline for the WI-262 disambiguation counterexample:
/// resolve a reference by name alone across every indexed definition. Returns
/// the set of candidate targets — with two same-named definitions this is
/// ambiguous (proves why name matching is the wrong answer), and with zero it
/// is empty (and must not be turned into a silent guess).
pub fn naive_name_match(
    defs: impl IntoIterator<Item = (String /*file*/, String /*name*/)>,
    reference_name: &str,
) -> BTreeSet<String> {
    defs.into_iter()
        .filter(|(_, n)| n == reference_name)
        .map(|(f, _)| f)
        .collect()
}
