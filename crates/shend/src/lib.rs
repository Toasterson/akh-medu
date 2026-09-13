//! shend — the resolution stage of the ADR 057 ingester pipeline
//! (parse → **resolve** → project), WI-262.
//!
//! A `Reference` becomes a *binding* instead of a name:
//!
//! - [`index::Resolver`] parses Rust source with tree-sitter, executes the
//!   per-language tree-sitter-graph ruleset ([`RUST_TSG`]) to build a
//!   file-local stack graph, stitches per-file minimal partial-path sets into
//!   a shared [`stack_graphs::stitching::Database`], and resolves each
//!   reference node to a definition node.
//! - Resolved references are recorded as `code:resolves-to` edges
//!   ([`store::ResolvesTo`], status `resolved`); a reference with no complete
//!   stitched path is recorded **explicitly unresolved** — never guessed,
//!   never silently omitted.
//! - [`store::ResolutionStore::record_resolution`] derives `code:depends-on`
//!   edges at file and crate granularity from resolved cross-file references.
//! - Every edge carries [`provenance::Provenance`]: the resolution
//!   [`provenance::Tier`] (`syntactic` | `stackgraph` | `scip`) and the
//!   `(repo, commit_sha)` [`provenance::Watermark`], so a tier can be
//!   bulk-retracted and re-asserted in isolation
//!   ([`store::ResolutionStore::retract_tier`] /
//!   [`store::ResolutionStore::retract_watermark`]).
//!
//! Resolution is **file-local and incremental**: a file indexes without being
//! built and without its dependencies present; partial resolutions compose in
//! the shared database once a dependency is indexed. Code structure is kept
//! out of belief space (`akhomed.facts`) per ADR 049/057 — this crate only
//! produces resolution edges with provenance.

pub mod index;
pub mod provenance;
pub mod store;

pub use index::{FileIndex, Resolver};
pub use provenance::{Provenance, Tier, Watermark};
pub use store::{
    DefId, DependsOn, ResolutionStatus, ResolutionStore, ResolvesTo, ShendError, Span, content_sha,
};

/// The Rust tree-sitter-graph ruleset (WI-262 covers Rust only; the mechanism
/// is polyglot and per-language rules land one at a time). Loaded and executed
/// by [`index::Resolver::new`] — it is validated at construction time.
pub const RUST_TSG: &str = include_str!("rules/rust.tsg");
