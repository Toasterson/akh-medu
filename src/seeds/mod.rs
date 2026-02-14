//! Seed packs: knowledge bootstrapping for workspaces.
//!
//! A seed pack is a TOML-defined bundle of triples that can be applied to an
//! engine to bootstrap fundamental knowledge. Three packs are bundled into the
//! binary: `identity`, `ontology`, and `common-sense`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use miette::Diagnostic;
use serde::Deserialize;
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;

// ── Errors ──────────────────────────────────────────────────────────────

#[derive(Debug, Error, Diagnostic)]
pub enum SeedError {
    #[error("seed pack not found: \"{id}\"")]
    #[diagnostic(
        code(akh::seed::not_found),
        help(
            "List available packs with `akh-medu seed list`. Check the seeds directory at ~/.local/share/akh-medu/seeds/"
        )
    )]
    NotFound { id: String },

    #[error("failed to parse seed pack \"{id}\": {message}")]
    #[diagnostic(
        code(akh::seed::parse),
        help("Check the seed.toml syntax. See `docs/seed-packs.md` for the format reference.")
    )]
    Parse { id: String, message: String },

    #[error("failed to read seed file: {path}")]
    #[diagnostic(code(akh::seed::io), help("Ensure the file exists and is readable."))]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to apply seed \"{id}\": {message}")]
    #[diagnostic(
        code(akh::seed::apply),
        help(
            "Check that the engine is writable and triple subjects/predicates/objects are valid."
        )
    )]
    Apply { id: String, message: String },
}

pub type SeedResult<T> = std::result::Result<T, SeedError>;

// ── Seed pack data model ────────────────────────────────────────────────

/// A seed pack: TOML-defined knowledge bundle.
#[derive(Debug, Clone)]
pub struct SeedPack {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub triples: Vec<SeedTriple>,
    /// Source: `Bundled` or `External(path)`.
    pub source: SeedSource,
    /// Optional partition to route triples into (SPARQL named graph).
    pub partition: Option<String>,
}

/// Where a seed pack came from.
#[derive(Debug, Clone)]
pub enum SeedSource {
    /// Bundled into the binary via `include_str!`.
    Bundled,
    /// Loaded from an external directory.
    External(PathBuf),
}

/// A triple in a seed pack.
#[derive(Debug, Clone, Deserialize)]
pub struct SeedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    0.8
}

/// Report after applying a seed pack.
#[derive(Debug, Clone)]
pub struct SeedReport {
    pub id: String,
    pub triples_applied: usize,
    pub triples_skipped: usize,
    pub already_applied: bool,
}

// ── TOML deserialization helpers ─────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct SeedToml {
    seed: SeedMeta,
    #[serde(default)]
    triples: Vec<SeedTriple>,
}

#[derive(Debug, Deserialize)]
struct SeedMeta {
    id: String,
    name: String,
    version: String,
    description: String,
    #[serde(default)]
    partition: Option<String>,
}

// ── Bundled seed packs ──────────────────────────────────────────────────

const IDENTITY_TOML: &str = include_str!("../../data/seeds/identity/seed.toml");
const ONTOLOGY_TOML: &str = include_str!("../../data/seeds/ontology/seed.toml");
const COMMON_SENSE_TOML: &str = include_str!("../../data/seeds/common-sense/seed.toml");

fn parse_seed_toml(toml_str: &str, source: SeedSource) -> SeedResult<SeedPack> {
    let parsed: SeedToml = toml::from_str(toml_str).map_err(|e| SeedError::Parse {
        id: "(unknown)".into(),
        message: e.to_string(),
    })?;
    Ok(SeedPack {
        id: parsed.seed.id,
        name: parsed.seed.name,
        version: parsed.seed.version,
        description: parsed.seed.description,
        triples: parsed.triples,
        source,
        partition: parsed.seed.partition,
    })
}

fn bundled_packs() -> Vec<SeedPack> {
    [
        (IDENTITY_TOML, "identity"),
        (ONTOLOGY_TOML, "ontology"),
        (COMMON_SENSE_TOML, "common-sense"),
    ]
    .iter()
    .filter_map(
        |(toml, id)| match parse_seed_toml(toml, SeedSource::Bundled) {
            Ok(pack) => Some(pack),
            Err(e) => {
                tracing::warn!(seed = id, "Failed to parse bundled seed: {e}");
                None
            }
        },
    )
    .collect()
}

// ── Seed Registry ───────────────────────────────────────────────────────

/// Registry of available seed packs (bundled + discovered from disk).
pub struct SeedRegistry {
    packs: HashMap<String, SeedPack>,
}

impl SeedRegistry {
    /// Create a registry with only bundled packs.
    pub fn bundled() -> Self {
        let packs = bundled_packs()
            .into_iter()
            .map(|p| (p.id.clone(), p))
            .collect();
        Self { packs }
    }

    /// Discover seed packs from a directory (in addition to bundled packs).
    ///
    /// Each subdirectory containing a `seed.toml` is loaded as a pack.
    pub fn discover(seeds_dir: &Path) -> Self {
        let mut registry = Self::bundled();

        if let Ok(entries) = std::fs::read_dir(seeds_dir) {
            for entry in entries.flatten() {
                let seed_file = entry.path().join("seed.toml");
                if seed_file.is_file() {
                    match std::fs::read_to_string(&seed_file) {
                        Ok(content) => {
                            match parse_seed_toml(&content, SeedSource::External(entry.path())) {
                                Ok(pack) => {
                                    registry.packs.insert(pack.id.clone(), pack);
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        path = %seed_file.display(),
                                        "Failed to parse seed pack: {e}"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                path = %seed_file.display(),
                                "Failed to read seed file: {e}"
                            );
                        }
                    }
                }
            }
        }

        registry
    }

    /// List all available seed packs.
    pub fn list(&self) -> Vec<&SeedPack> {
        let mut packs: Vec<&SeedPack> = self.packs.values().collect();
        packs.sort_by(|a, b| a.id.cmp(&b.id));
        packs
    }

    /// Get a seed pack by ID.
    pub fn get(&self, id: &str) -> SeedResult<&SeedPack> {
        self.packs
            .get(id)
            .ok_or_else(|| SeedError::NotFound { id: id.to_string() })
    }

    /// Apply a single seed pack to an engine. Idempotent via `akh:seed-applied` tracking.
    pub fn apply(&self, pack_id: &str, engine: &Engine) -> SeedResult<SeedReport> {
        let pack = self.get(pack_id)?;
        apply_seed_pack(pack, engine)
    }

    /// Apply multiple seed packs. Returns a report per pack.
    pub fn apply_all(&self, pack_ids: &[String], engine: &Engine) -> SeedResult<Vec<SeedReport>> {
        let mut reports = Vec::new();
        for id in pack_ids {
            reports.push(self.apply(id, engine)?);
        }
        Ok(reports)
    }
}

// ── Application logic ───────────────────────────────────────────────────

/// Well-known predicate for tracking applied seeds.
const SEED_APPLIED_PREDICATE: &str = "akh:seed-applied";

/// Check if a seed pack has already been applied to this engine (public API).
pub fn is_seed_applied_public(engine: &Engine, seed_id: &str) -> bool {
    is_seed_applied(engine, seed_id)
}

/// Check if a seed pack has already been applied to this engine.
fn is_seed_applied(engine: &Engine, seed_id: &str) -> bool {
    // Look for a triple: akh-medu --akh:seed-applied--> {seed_id}
    if let Ok(akh_id) = engine.resolve_symbol("akh-medu") {
        let triples = engine.triples_from(akh_id);
        triples.iter().any(|t| {
            let pred_label = engine.resolve_label(t.predicate);
            let obj_label = engine.resolve_label(t.object);
            pred_label == SEED_APPLIED_PREDICATE && obj_label == seed_id
        })
    } else {
        false
    }
}

/// Mark a seed as applied by adding the tracking triple.
fn mark_seed_applied(engine: &Engine, seed_id: &str) -> SeedResult<()> {
    let akh = engine
        .resolve_or_create_entity("akh-medu")
        .map_err(|e| SeedError::Apply {
            id: seed_id.to_string(),
            message: format!("failed to create akh-medu entity: {e}"),
        })?;

    let pred = engine
        .resolve_or_create_relation(SEED_APPLIED_PREDICATE)
        .map_err(|e| SeedError::Apply {
            id: seed_id.to_string(),
            message: format!("failed to create seed-applied predicate: {e}"),
        })?;

    let seed_sym = engine
        .resolve_or_create_entity(seed_id)
        .map_err(|e| SeedError::Apply {
            id: seed_id.to_string(),
            message: format!("failed to create seed entity: {e}"),
        })?;

    engine
        .add_triple(&Triple::new(akh, pred, seed_sym))
        .map_err(|e| SeedError::Apply {
            id: seed_id.to_string(),
            message: format!("failed to add tracking triple: {e}"),
        })?;

    Ok(())
}

/// Apply a seed pack's triples to the engine.
fn apply_seed_pack(pack: &SeedPack, engine: &Engine) -> SeedResult<SeedReport> {
    // Check idempotency.
    if is_seed_applied(engine, &pack.id) {
        return Ok(SeedReport {
            id: pack.id.clone(),
            triples_applied: 0,
            triples_skipped: pack.triples.len(),
            already_applied: true,
        });
    }

    let mut applied = 0;
    let mut skipped = 0;

    for st in &pack.triples {
        // If the subject is being declared as a relation, create it as Relation kind.
        let subj = if st.predicate == "is-a" && st.object == "relation" {
            engine.resolve_or_create_relation(&st.subject)
        } else {
            engine.resolve_or_create_entity(&st.subject)
        }
        .map_err(|e| SeedError::Apply {
            id: pack.id.clone(),
            message: format!("subject '{}': {e}", st.subject),
        })?;

        let pred = engine
            .resolve_or_create_relation(&st.predicate)
            .map_err(|e| SeedError::Apply {
                id: pack.id.clone(),
                message: format!("predicate '{}': {e}", st.predicate),
            })?;

        let obj = engine
            .resolve_or_create_entity(&st.object)
            .map_err(|e| SeedError::Apply {
                id: pack.id.clone(),
                message: format!("object '{}': {e}", st.object),
            })?;

        let triple = Triple::new(subj, pred, obj).with_confidence(st.confidence);
        // Route to partition's named graph if specified, otherwise default graph.
        if let Some(ref partition_name) = pack.partition {
            if let Some(sparql) = engine.sparql() {
                match sparql.insert_triple_in_graph(&triple, Some(partition_name)) {
                    Ok(_) => applied += 1,
                    Err(_) => skipped += 1,
                }
            } else {
                // No SPARQL store available, fall back to default graph.
                match engine.add_triple(&triple) {
                    Ok(_) => applied += 1,
                    Err(_) => skipped += 1,
                }
            }
        } else {
            match engine.add_triple(&triple) {
                Ok(_) => applied += 1,
                Err(_) => skipped += 1,
            }
        }
    }

    // Mark as applied.
    mark_seed_applied(engine, &pack.id)?;

    Ok(SeedReport {
        id: pack.id.clone(),
        triples_applied: applied,
        triples_skipped: skipped,
        already_applied: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    #[test]
    fn bundled_packs_parse() {
        let packs = bundled_packs();
        assert_eq!(packs.len(), 3);
        assert!(packs.iter().any(|p| p.id == "identity"));
        assert!(packs.iter().any(|p| p.id == "ontology"));
        assert!(packs.iter().any(|p| p.id == "common-sense"));
    }

    #[test]
    fn registry_lists_all_bundled() {
        let reg = SeedRegistry::bundled();
        let list = reg.list();
        assert!(list.len() >= 3);
    }

    #[test]
    fn apply_identity_seed() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let reg = SeedRegistry::bundled();
        let report = reg.apply("identity", &engine).unwrap();
        assert!(!report.already_applied);
        assert!(report.triples_applied > 0);

        // Verify akh-medu symbol exists.
        assert!(engine.resolve_symbol("akh-medu").is_ok());

        // Verify idempotency.
        let report2 = reg.apply("identity", &engine).unwrap();
        assert!(report2.already_applied);
        assert_eq!(report2.triples_applied, 0);
    }

    #[test]
    fn apply_all_bundled_seeds() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let reg = SeedRegistry::bundled();
        let ids: Vec<String> = reg.list().iter().map(|p| p.id.clone()).collect();
        let reports = reg.apply_all(&ids, &engine).unwrap();
        assert_eq!(reports.len(), 3);

        let total: usize = reports.iter().map(|r| r.triples_applied).sum();
        assert!(
            total > 30,
            "Expected 30+ triples from all seeds, got {total}"
        );
    }
}
