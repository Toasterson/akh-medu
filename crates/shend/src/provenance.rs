//! Tiered provenance for resolution edges.
//!
//! Every `code:resolves-to` / `code:depends-on` edge carries the resolution
//! tier that produced it and a `(repo, commit_sha)` watermark. A tier can be
//! bulk-retracted and re-asserted when its producer improves, without
//! disturbing the others (ADR draft §Decision.1, WI-262).

use serde::{Deserialize, Serialize};

/// The resolution tier that produced an edge.
///
/// Ordered from weakest to strongest. A higher tier may supersede a lower
/// one for the same reference, but each tier's edges remain independently
/// retractable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// File-local, syntactic binding discovered without any cross-file or
    /// whole-program analysis — e.g. a definition and reference in the same
    /// scope of one file. Always available; survives retraction of the
    /// stronger tiers.
    Syntactic,
    /// Binding discovered by file-local stack-graph stitching composed
    /// incrementally across files (the WI-262 default). Per-file minimal
    /// partial-path sets are stitched together without a build and without
    /// the whole program present.
    Stackgraph,
    /// Binding imported from a SCIP index (WI 2). Not produced by this crate
    /// today; reserved so the tier model accommodates it without a schema
    /// change.
    Scip,
}

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::Syntactic, Tier::Stackgraph, Tier::Scip];

    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::Syntactic => "syntactic",
            Tier::Stackgraph => "stackgraph",
            Tier::Scip => "scip",
        }
    }
}

/// The `(repo, commit_sha)` watermark identifying the producer revision that
/// asserted an edge. Retraction and re-assertion are scoped to a watermark so
/// an improved producer can replace its own prior facts in isolation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Watermark {
    /// Repository identifier (e.g. a canonical remote URL or project slug).
    pub repo: String,
    /// Commit SHA of the indexed revision.
    pub commit_sha: String,
}

impl Watermark {
    pub fn new(repo: impl Into<String>, commit_sha: impl Into<String>) -> Self {
        Watermark {
            repo: repo.into(),
            commit_sha: commit_sha.into(),
        }
    }
}

/// Provenance attached to every resolution edge: the tier that produced it and
/// the watermark of the producer revision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Provenance {
    pub tier: Tier,
    pub watermark: Watermark,
}

impl Provenance {
    pub fn new(tier: Tier, watermark: Watermark) -> Self {
        Provenance { tier, watermark }
    }
}
