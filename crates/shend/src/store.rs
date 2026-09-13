//! `ResolutionStore` — the durable, tier-aware store for `code:resolves-to`
//! and `code:depends-on` edges.
//!
//! This is the type the ADR 057 daemon hosts. It is a *library* store: the
//! daemon will wrap it in its `codex.*`/AGE projection. It deliberately keeps
//! code structure out of belief space (ADR 049/057 standing rule) — these are
//! resolution edges with provenance, not asserted belief triples.
//!
//! Snapshot persistence uses `bincode` (the `serde`/`serde_json` pair support
//! human-readable inspection and the JSON-lines projection the daemon emits).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::provenance::{Provenance, Tier, Watermark};

/// A source span (0-based, byte-oriented; matches tree-sitter's `Point`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start_row: usize,
    pub start_col: usize,
    pub end_row: usize,
    pub end_col: usize,
}

/// A definition of a symbol, located in a file at a span.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DefId {
    /// Repository-relative file path (e.g. `src/alpha.rs`).
    pub file: String,
    /// The symbol name.
    pub name: String,
    /// Module path the definition lives in (e.g. `crate::alpha`).
    pub module: String,
    pub span: Span,
}

/// Status of a `code:resolves-to` edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResolutionStatus {
    /// The reference resolved to exactly one definition.
    Resolved,
    /// The reference could not be resolved by any available tier. Recorded
    /// explicitly — never guessed, never silently omitted.
    Unresolved,
}

/// A `code:resolves-to` edge: a reference site and its binding (or explicit
/// absence of one), with provenance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResolvesTo {
    /// The file that contains the reference.
    pub reference_file: String,
    /// The referenced name as written.
    pub reference_name: String,
    /// Span of the reference site.
    pub reference_span: Span,
    /// The resolved definition, when resolved.
    pub target: Option<DefId>,
    pub status: ResolutionStatus,
    pub provenance: Provenance,
}

/// A `code:depends-on` edge derived from resolved references. What blast radius
/// reads (WI 2). Granularity is file→file and crate→crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DependsOn {
    /// The dependent file (the one containing the reference).
    pub from_file: String,
    /// The file that provides the definition.
    pub to_file: String,
    /// The crate the dependent file belongs to (its `src/` prefix root).
    pub from_crate: String,
    /// The crate providing the definition.
    pub to_crate: String,
    pub provenance: Provenance,
}

/// The store. Holds edges partitioned by tier so retraction is cheap and exact.
#[derive(Debug, Default)]
pub struct ResolutionStore {
    resolves: Vec<ResolvesTo>,
    depends: Vec<DependsOn>,
}

/// Content-identity hash for a file, used to key incremental re-indexing.
pub fn content_sha(source: &str) -> String {
    let mut h = Sha256::new();
    h.update(source.as_bytes());
    format!("{:x}", h.finalize())
}

fn crate_of(path: &str) -> String {
    let root = path.split('/').next().unwrap_or("");
    if root.is_empty() || root == path {
        "<crate>".to_string()
    } else {
        root.to_string()
    }
}

impl ResolutionStore {
    pub fn new() -> Self {
        ResolutionStore {
            resolves: Vec::new(),
            depends: Vec::new(),
        }
    }

    /// Record a resolution edge. If it carries a new cross-file dependency and
    /// is resolved, a `code:depends-on` edge is derived automatically.
    pub fn record_resolution(&mut self, edge: ResolvesTo) {
        if edge.status == ResolutionStatus::Resolved {
            if let Some(target) = &edge.target {
                if target.file != edge.reference_file {
                    self.depends.push(DependsOn {
                        from_file: edge.reference_file.clone(),
                        to_file: target.file.clone(),
                        from_crate: crate_of(&edge.reference_file),
                        to_crate: crate_of(&target.file),
                        provenance: edge.provenance.clone(),
                    });
                }
            }
        }
        self.resolves.push(edge);
    }

    /// Bulk-retract every edge produced by `tier`. Other tiers are untouched.
    pub fn retract_tier(&mut self, tier: Tier) {
        self.resolves.retain(|e| e.provenance.tier != tier);
        self.depends.retain(|e| e.provenance.tier != tier);
    }

    /// Bulk-retract every edge at a watermark for a single tier.
    pub fn retract_watermark(&mut self, tier: Tier, watermark: &Watermark) {
        self.resolves
            .retain(|e| !(e.provenance.tier == tier && &e.provenance.watermark == watermark));
        self.depends
            .retain(|e| !(e.provenance.tier == tier && &e.provenance.watermark == watermark));
    }

    pub fn resolves(&self) -> &[ResolvesTo] {
        &self.resolves
    }

    pub fn depends_on(&self) -> &[DependsOn] {
        &self.depends
    }

    /// All resolved edges at a given tier.
    pub fn resolved_at(&self, tier: Tier) -> impl Iterator<Item = &ResolvesTo> {
        self.resolves
            .iter()
            .filter(move |e| e.provenance.tier == tier && e.status == ResolutionStatus::Resolved)
    }

    /// All explicitly-unresolved reference records at a given tier.
    pub fn unresolved_at(&self, tier: Tier) -> impl Iterator<Item = &ResolvesTo> {
        self.resolves
            .iter()
            .filter(move |e| e.provenance.tier == tier && e.status == ResolutionStatus::Unresolved)
    }

    /// Distinct cross-crate dependency pairs `(from_crate, to_crate)`.
    pub fn crate_dependencies(&self) -> BTreeSet<(String, String)> {
        self.depends
            .iter()
            .filter(|d| d.from_crate != d.to_crate)
            .map(|d| (d.from_crate.clone(), d.to_crate.clone()))
            .collect()
    }

    /// Distinct cross-file dependency pairs `(from_file, to_file)`.
    pub fn file_dependencies(&self) -> BTreeSet<(String, String)> {
        self.depends
            .iter()
            .filter(|d| d.from_file != d.to_file)
            .map(|d| (d.from_file.clone(), d.to_file.clone()))
            .collect()
    }

    /// Serialize a snapshot (bincode).
    pub fn snapshot(&self) -> Result<Vec<u8>, ShendError> {
        #[derive(Serialize)]
        struct Snap<'a> {
            resolves: &'a [ResolvesTo],
            depends: &'a [DependsOn],
        }
        bincode::serialize(&Snap {
            resolves: &self.resolves,
            depends: &self.depends,
        })
        .map_err(|e| ShendError::Serialize(e.to_string()))
    }

    /// Emit the JSON-lines projection the daemon hands to its AGE writer.
    pub fn to_json_lines(&self) -> Result<String, ShendError> {
        let mut out = String::new();
        for e in &self.resolves {
            out.push_str(
                &serde_json::to_string(e).map_err(|e| ShendError::Serialize(e.to_string()))?,
            );
            out.push('\n');
        }
        for d in &self.depends {
            out.push_str(
                &serde_json::to_string(d).map_err(|e| ShendError::Serialize(e.to_string()))?,
            );
            out.push('\n');
        }
        Ok(out)
    }
}

/// Errors surfaced by the resolution stage.
#[derive(Debug, thiserror::Error)]
pub enum ShendError {
    #[error("parse error in {file}: {message}")]
    Parse { file: String, message: String },
    #[error("tsg ruleset error: {0}")]
    Ruleset(String),
    #[error("serialization error: {0}")]
    Serialize(String),
}
