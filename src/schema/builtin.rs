//! Built-in default schema packs: memory-hygiene defaults every akh gets
//! without an operator installing a pack (Phase 53 follow-up, Anima WI
//! 01a0d569-1cf1 item 7; `docs/ai/phases/phase-53/README.md`).
//!
//! ## Admission
//!
//! ADR 081 decision 8 requires a human confirm before a pack enters the
//! shared registry. A built-in pack never enters the registry: its source
//! (`data/schema-packs/fleet-memory.json`) is compiled into the binary with
//! `include_str!`, so the code review that merges the file is its admission,
//! and a binary can only install the exact revision it was built from. The
//! registry, `akh schema admit` and wake-time convergence
//! (`registry_pg::converge_workspace`) are untouched.
//!
//! ## The default tier
//!
//! A built-in pack installs under its own compartment
//! (`mt:schema/builtin/<name>@<version>`) and its own predicates
//! (`onto:default-relation`, `onto:default-cardinality`,
//! `onto:default-alias-of`; see [`super::compile`]). The read side
//! ([`super::cardinality`]) consults an akh's installed packs first, so an
//! akh-specific pack (for example `roman-memory`) always wins on a relation
//! both declare, whichever was installed first. Only relation declarations
//! (cardinality and aliases) can be built in: types are global and immutable
//! (ADR 081 decision 5), so a default tier for them would have no precedence
//! to give.
//!
//! ## Lifecycle ([`ensure_builtin_pack`])
//!
//! Run on every engine open (akhomed `get_engine`, which also covers wake and
//! relocation). Idempotent:
//!
//! - opted out (`[memory] default_pack = false`): every installed version is
//!   retired (its compartment and activation edge removed);
//! - a newer version is installed than this binary carries: it is kept and
//!   nothing is written (a rolled-back binary never downgrades an akh);
//! - this version is installed: nothing is written;
//! - otherwise older versions are retired first, then this version is
//!   installed and activated (`ctx:specializes` from the akh's
//!   effective-schema context, as for any installed pack).

use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use serde::Serialize;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::compile::{
    ARITY_2, CompiledSchemaPack, PRED_ALIAS_OF, PRED_ARITY, PRED_CARDINALITY,
    PRED_DEFAULT_ALIAS_OF, PRED_DEFAULT_CARDINALITY, PRED_DEFAULT_RELATION, PRED_IS_A,
    PortableTriple, RELATION_MARKER, compile,
};
use super::install::{INSTALLED_PREDICATE, effective_schema_label};
use super::{SchemaError, SchemaResult};

/// Compartment prefix of every built-in pack install. Distinct from a
/// registry pack's `mt:schema/<name>@<version>`, so a registry pack of the
/// same name can never be mistaken for (or retired as) the built-in one.
pub const BUILTIN_COMPARTMENT_PREFIX: &str = "mt:schema/builtin/";

/// The activation predicate (`ctx:specializes`, ADR 081 decision 6).
const PRED_SPECIALIZES: &str = "ctx:specializes";

/// A schema pack compiled into the binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinPack {
    /// `fleet-memory`: single-valued state predicates and their spellings.
    FleetMemory,
}

impl BuiltinPack {
    /// Every built-in pack, in install order.
    pub const ALL: [BuiltinPack; 1] = [BuiltinPack::FleetMemory];

    /// The pack name (`pack.name` in the source).
    pub const fn name(self) -> &'static str {
        match self {
            BuiltinPack::FleetMemory => "fleet-memory",
        }
    }

    /// The reviewed portable source embedded in the binary.
    pub const fn source(self) -> &'static str {
        match self {
            BuiltinPack::FleetMemory => {
                include_str!("../../data/schema-packs/fleet-memory.json")
            }
        }
    }

    /// Compile the embedded source and check that it can be installed as a
    /// default tier (its name matches, it declares relations only).
    ///
    /// # Errors
    ///
    /// Any compile diagnostic of the source, or
    /// [`SchemaError::NotDefaultTier`] / [`SchemaError::InvalidPackVersion`].
    /// Both are caught by this module's tests, so a shipped binary does not
    /// hit them.
    pub fn compile(self) -> SchemaResult<BuiltinCompiled> {
        let pack = compile(self.source())?;
        if pack.name != self.name() {
            return Err(SchemaError::NotDefaultTier {
                pack: pack.name.clone(),
                detail: format!(
                    "the source is named \"{}\", expected \"{}\"",
                    pack.name,
                    self.name()
                ),
            });
        }
        if !pack.assumes.is_empty() || !pack.types.is_empty() {
            return Err(SchemaError::NotDefaultTier {
                pack: pack.name.clone(),
                detail: "a built-in pack declares relations only (no `types`, no `assumes`)".into(),
            });
        }
        let version = pack.version.parse::<PackVersion>()?;
        let triples = default_tier_triples(&pack)?;
        Ok(BuiltinCompiled {
            pack: self,
            version,
            source: pack,
            triples,
        })
    }
}

impl fmt::Display for BuiltinPack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A built-in pack compiled and lowered to its default-tier triples.
#[derive(Debug, Clone)]
pub struct BuiltinCompiled {
    /// Which built-in pack.
    pub pack: BuiltinPack,
    /// Its parsed version.
    pub version: PackVersion,
    /// The ordinary compiled pack (relations, source checksum).
    pub source: CompiledSchemaPack,
    /// The declarations under the default-tier predicates.
    pub triples: Vec<PortableTriple>,
}

impl BuiltinCompiled {
    /// `name@version`, the installed-marker object.
    pub fn id(&self) -> String {
        format!("{}@{}", self.pack.name(), self.version)
    }

    /// The compartment this version installs under.
    pub fn microtheory_label(&self) -> String {
        builtin_microtheory_label(self.pack, &self.version)
    }
}

/// `mt:schema/builtin/<name>@<version>`.
pub fn builtin_microtheory_label(pack: BuiltinPack, version: &PackVersion) -> String {
    format!("{BUILTIN_COMPARTMENT_PREFIX}{}@{version}", pack.name())
}

/// Lower a compiled pack's relation declarations onto the default-tier
/// predicates. Arity edges are dropped (every relation is binary).
///
/// # Errors
///
/// [`SchemaError::NotDefaultTier`] for any triple that is not a relation,
/// cardinality or alias declaration (types, properties, argument types).
pub fn default_tier_triples(pack: &CompiledSchemaPack) -> SchemaResult<Vec<PortableTriple>> {
    let mut out = Vec::with_capacity(pack.triples.len());
    for t in &pack.triples {
        let predicate = match (t.predicate.as_str(), t.object.as_str()) {
            (PRED_IS_A, RELATION_MARKER) => PRED_DEFAULT_RELATION,
            (PRED_ARITY, ARITY_2) => continue,
            (PRED_CARDINALITY, _) => PRED_DEFAULT_CARDINALITY,
            (PRED_ALIAS_OF, _) => PRED_DEFAULT_ALIAS_OF,
            _ => {
                return Err(SchemaError::NotDefaultTier {
                    pack: pack.name.clone(),
                    detail: format!(
                        "({}, {}, {}) is not a relation, cardinality or alias declaration",
                        t.subject, t.predicate, t.object
                    ),
                });
            }
        };
        out.push(PortableTriple {
            subject: t.subject.clone(),
            predicate: predicate.to_string(),
            object: t.object.clone(),
        });
    }
    Ok(out)
}

/// A pack version: dot-separated unsigned integers (`1.0.0`). Trailing zero
/// components do not change the order (`1.0` == `1.0.0`). Pre-release and
/// build suffixes are rejected: a built-in pack only ships releases.
#[derive(Debug, Clone)]
pub struct PackVersion {
    text: String,
    /// Components with trailing zeros removed (the comparison key).
    key: Vec<u64>,
}

impl PackVersion {
    /// The version as written.
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl FromStr for PackVersion {
    type Err = SchemaError;

    fn from_str(s: &str) -> SchemaResult<Self> {
        let invalid = || SchemaError::InvalidPackVersion {
            version: s.to_string(),
        };
        let mut key = s
            .split('.')
            .map(|part| {
                if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid());
                }
                part.parse::<u64>().map_err(|_| invalid())
            })
            .collect::<SchemaResult<Vec<u64>>>()?;
        while key.last() == Some(&0) {
            key.pop();
        }
        Ok(Self {
            text: s.to_string(),
            key,
        })
    }
}

impl fmt::Display for PackVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl PartialEq for PackVersion {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for PackVersion {}

impl PartialOrd for PackVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PackVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl Serialize for PackVersion {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.text)
    }
}

/// Whether an akh gets the built-in default packs (`[memory] default_pack`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DefaultPackPolicy {
    /// Install and keep the built-in packs current (the default).
    #[default]
    Enabled,
    /// Opted out: retire any installed built-in pack.
    Disabled,
}

impl From<bool> for DefaultPackPolicy {
    fn from(enabled: bool) -> Self {
        if enabled {
            DefaultPackPolicy::Enabled
        } else {
            DefaultPackPolicy::Disabled
        }
    }
}

/// One installed version of a built-in pack, found in the KG.
#[derive(Debug, Clone)]
struct InstalledBuiltin {
    version: PackVersion,
    microtheory: SymbolId,
    label: String,
}

/// What [`ensure_builtin_pack`] did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum BuiltinPackOutcome {
    /// First install on this akh.
    Installed {
        /// `name@version`.
        pack: String,
        /// Declaration triples written.
        triples_applied: usize,
    },
    /// Older versions were retired and this version installed.
    Upgraded {
        /// The retired `name@version`s.
        from: Vec<String>,
        /// The installed `name@version`.
        to: String,
        /// Declaration triples written.
        triples_applied: usize,
        /// Triples removed with the retired versions.
        triples_removed: usize,
    },
    /// This version is already installed: nothing written.
    Current {
        /// `name@version`.
        pack: String,
    },
    /// A newer version than this binary carries is installed and kept.
    NewerKept {
        /// The installed `name@version`.
        installed: String,
        /// The version this binary carries.
        builtin: String,
    },
    /// Opted out: installed versions (if any) were retired.
    Disabled {
        /// The retired `name@version`s (empty when none was installed).
        retired: Vec<String>,
        /// Triples removed.
        triples_removed: usize,
    },
}

impl BuiltinPackOutcome {
    /// Whether the call wrote to the graph.
    pub fn mutated(&self) -> bool {
        match self {
            BuiltinPackOutcome::Installed { .. } | BuiltinPackOutcome::Upgraded { .. } => true,
            BuiltinPackOutcome::Disabled { retired, .. } => !retired.is_empty(),
            BuiltinPackOutcome::Current { .. } | BuiltinPackOutcome::NewerKept { .. } => false,
        }
    }
}

fn engine_err(pack: BuiltinPack, message: impl fmt::Display) -> SchemaError {
    SchemaError::Install {
        pack: pack.name().to_string(),
        message: message.to_string(),
    }
}

/// Every installed version of `pack`, oldest first. An installed marker
/// whose version does not parse is skipped with a warning (it cannot be
/// ordered, so it is neither kept as "newer" nor retired).
fn installed_versions(engine: &Engine, pack: BuiltinPack) -> Vec<InstalledBuiltin> {
    let Ok(installed) = engine.lookup_symbol(INSTALLED_PREDICATE) else {
        return Vec::new();
    };
    let prefix = format!("{BUILTIN_COMPARTMENT_PREFIX}{}@", pack.name());
    let mut versions: Vec<InstalledBuiltin> = engine
        .triples_with_predicate(installed)
        .into_iter()
        .filter_map(|marker| {
            let label = engine.resolve_label(marker.subject);
            let text = label.strip_prefix(&prefix)?;
            match text.parse::<PackVersion>() {
                Ok(version) => Some(InstalledBuiltin {
                    version,
                    microtheory: marker.subject,
                    label: label.clone(),
                }),
                Err(error) => {
                    tracing::warn!(
                        compartment = %label,
                        %error,
                        "built-in schema pack marker with an unparseable version, left alone"
                    );
                    None
                }
            }
        })
        .collect();
    versions.sort_by(|a, b| a.version.cmp(&b.version));
    versions.dedup_by(|a, b| a.microtheory == b.microtheory);
    versions
}

/// Remove one installed version: its activation edge, then its compartment.
fn retire(
    engine: &Engine,
    pack: BuiltinPack,
    importer: Option<SymbolId>,
    installed: &InstalledBuiltin,
) -> SchemaResult<usize> {
    let mut removed = 0;
    if let (Some(importer), Ok(specializes)) = (importer, engine.lookup_symbol(PRED_SPECIALIZES))
        && engine
            .triples_from(importer)
            .iter()
            .any(|t| t.predicate == specializes && t.object == installed.microtheory)
    {
        engine
            .remove_triple(importer, specializes, installed.microtheory)
            .map_err(|e| engine_err(pack, format!("deactivating {}: {e}", installed.label)))?;
        removed += 1;
    }
    removed += engine
        .remove_triples_in_compartment_checked(&installed.label)
        .map_err(|e| engine_err(pack, format!("retiring {}: {e}", installed.label)))?;
    Ok(removed)
}

/// Write `compiled`'s default-tier triples, the installed marker and the
/// activation edge.
fn install(engine: &Engine, compiled: &BuiltinCompiled, importer: SymbolId) -> SchemaResult<usize> {
    let pack = compiled.pack;
    let mt_label = compiled.microtheory_label();
    let mut applied = 0;
    for t in &compiled.triples {
        // Every default-tier subject is a predicate spelling (a relation or
        // an alias), minted as a Relation like `install_pack` does.
        let subject = engine
            .resolve_or_create_relation(&t.subject)
            .map_err(|e| engine_err(pack, format!("subject '{}': {e}", t.subject)))?;
        let predicate = engine
            .resolve_or_create_relation(&t.predicate)
            .map_err(|e| engine_err(pack, format!("predicate '{}': {e}", t.predicate)))?;
        let object = engine
            .resolve_or_create_entity(&t.object)
            .map_err(|e| engine_err(pack, format!("object '{}': {e}", t.object)))?;
        engine
            .add_triple(&Triple::new(subject, predicate, object).with_compartment(mt_label.clone()))
            .map_err(|e| {
                engine_err(
                    pack,
                    format!("triple ({}, {}, {}): {e}", t.subject, t.predicate, t.object),
                )
            })?;
        applied += 1;
    }
    let mt = engine
        .resolve_or_create_entity(&mt_label)
        .map_err(|e| engine_err(pack, format!("microtheory entity: {e}")))?;
    let marker = engine
        .resolve_or_create_relation(INSTALLED_PREDICATE)
        .map_err(|e| engine_err(pack, format!("installed predicate: {e}")))?;
    let id = engine
        .resolve_or_create_entity(&compiled.id())
        .map_err(|e| engine_err(pack, format!("version entity: {e}")))?;
    engine
        .add_triple(&Triple::new(mt, marker, id).with_compartment(mt_label.clone()))
        .map_err(|e| engine_err(pack, format!("installed marker: {e}")))?;
    let specializes = engine
        .resolve_or_create_relation(PRED_SPECIALIZES)
        .map_err(|e| engine_err(pack, format!("{PRED_SPECIALIZES}: {e}")))?;
    engine
        .add_triple(&Triple::new(importer, specializes, mt))
        .map_err(|e| engine_err(pack, format!("activation edge: {e}")))?;
    Ok(applied)
}

/// Bring `pack` on `engine` (the akh `workspace`) in line with `policy`.
/// See the module docs for the rules. Safe to call on every engine open.
///
/// # Errors
///
/// A compile diagnostic of the embedded source (caught by tests), or
/// [`SchemaError::Install`] when an engine write fails (for example the
/// engine is quiesced for a handoff). A failed call leaves the akh with at
/// most a partial install of this version, which the next call finishes:
/// the installed marker is written last.
pub fn ensure_builtin_pack(
    engine: &Engine,
    pack: BuiltinPack,
    workspace: &str,
    policy: DefaultPackPolicy,
) -> SchemaResult<BuiltinPackOutcome> {
    let installed = installed_versions(engine, pack);
    let importer_label = effective_schema_label(workspace);

    if policy == DefaultPackPolicy::Disabled {
        let importer = engine.lookup_symbol(&importer_label).ok();
        let mut triples_removed = 0;
        for version in &installed {
            triples_removed += retire(engine, pack, importer, version)?;
        }
        return Ok(BuiltinPackOutcome::Disabled {
            retired: installed
                .iter()
                .map(|v| format!("{}@{}", pack.name(), v.version))
                .collect(),
            triples_removed,
        });
    }

    let compiled = pack.compile()?;
    if let Some(newest) = installed.last() {
        match newest.version.cmp(&compiled.version) {
            Ordering::Greater => {
                return Ok(BuiltinPackOutcome::NewerKept {
                    installed: format!("{}@{}", pack.name(), newest.version),
                    builtin: compiled.id(),
                });
            }
            Ordering::Equal => {
                return Ok(BuiltinPackOutcome::Current {
                    pack: compiled.id(),
                });
            }
            Ordering::Less => {}
        }
    }

    let importer = engine
        .resolve_or_create_entity(&importer_label)
        .map_err(|e| engine_err(pack, format!("importer entity: {e}")))?;
    // Retire first: two versions' identical declarations are one KG edge
    // (kept in the compartment that wrote it first), so installing before
    // retiring would let the retirement delete the new version's edges.
    let mut triples_removed = 0;
    for version in &installed {
        triples_removed += retire(engine, pack, Some(importer), version)?;
    }
    let triples_applied = install(engine, &compiled, importer)?;
    Ok(if installed.is_empty() {
        BuiltinPackOutcome::Installed {
            pack: compiled.id(),
            triples_applied,
        }
    } else {
        BuiltinPackOutcome::Upgraded {
            from: installed
                .iter()
                .map(|v| format!("{}@{}", pack.name(), v.version))
                .collect(),
            to: compiled.id(),
            triples_applied,
            triples_removed,
        }
    })
}

/// [`ensure_builtin_pack`] for every built-in pack, in [`BuiltinPack::ALL`]
/// order.
///
/// # Errors
///
/// The first pack that fails; packs before it stay ensured.
pub fn ensure_builtin_packs(
    engine: &Engine,
    workspace: &str,
    policy: DefaultPackPolicy,
) -> SchemaResult<Vec<(BuiltinPack, BuiltinPackOutcome)>> {
    BuiltinPack::ALL
        .into_iter()
        .map(|pack| ensure_builtin_pack(engine, pack, workspace, policy).map(|o| (pack, o)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ContradictionPolicy, EngineConfig};
    use crate::schema::cardinality::is_declared_relation;
    use crate::schema::{
        HygieneOptions, MergeRequest, MergeSelection, PredicateCardinality, declared_alias_target,
        declared_cardinality, execute_merge, install_pack, run_hygiene,
    };

    const WS: &str = "test-akh";
    const TEMPORAL: PredicateCardinality = PredicateCardinality::SingleValued { temporal: true };
    const SINGLE: PredicateCardinality = PredicateCardinality::SingleValued { temporal: false };

    fn engine() -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(dir.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (engine, dir)
    }

    fn ensure(engine: &Engine, policy: DefaultPackPolicy) -> BuiltinPackOutcome {
        ensure_builtin_pack(engine, BuiltinPack::FleetMemory, WS, policy).unwrap()
    }

    fn cardinality(engine: &Engine, label: &str) -> PredicateCardinality {
        engine
            .lookup_symbol(label)
            .map(|p| declared_cardinality(engine, p))
            .unwrap_or_default()
    }

    fn alias_target(engine: &Engine, label: &str) -> Option<String> {
        let alias = engine.lookup_symbol(label).ok()?;
        declared_alias_target(engine, alias).map(|t| engine.resolve_label(t))
    }

    fn put(engine: &Engine, s: &str, p: &str, o: &str, ts: u64) {
        engine
            .add_triple(&Triple {
                timestamp: ts,
                ..Triple::new(
                    engine.resolve_or_create_entity(s).unwrap(),
                    engine.resolve_or_create_relation(p).unwrap(),
                    engine.resolve_or_create_entity(o).unwrap(),
                )
            })
            .unwrap();
    }

    fn live(engine: &Engine, s: &str, p: &str) -> Vec<String> {
        let (Ok(s), Ok(p)) = (engine.lookup_symbol(s), engine.lookup_symbol(p)) else {
            return Vec::new();
        };
        let mut objects: Vec<String> = engine
            .triples_from(s)
            .into_iter()
            .filter(|t| t.predicate == p)
            .map(|t| engine.resolve_label(t.object))
            .collect();
        objects.sort();
        objects
    }

    /// Install a copy of the fleet pack under another version (optionally
    /// with one extra single-valued relation), as an older or newer binary
    /// would have.
    fn install_as_version(engine: &Engine, version: &str, extra_relation: Option<&str>) {
        let mut compiled = BuiltinPack::FleetMemory.compile().unwrap();
        compiled.version = version.parse().unwrap();
        if let Some(extra) = extra_relation {
            compiled.triples.push(PortableTriple {
                subject: extra.into(),
                predicate: PRED_DEFAULT_RELATION.into(),
                object: RELATION_MARKER.into(),
            });
            compiled.triples.push(PortableTriple {
                subject: extra.into(),
                predicate: PRED_DEFAULT_CARDINALITY.into(),
                object: super::super::compile::CARDINALITY_SINGLE_VALUED.into(),
            });
        }
        let importer = engine
            .resolve_or_create_entity(&effective_schema_label(WS))
            .unwrap();
        install(engine, &compiled, importer).unwrap();
    }

    #[test]
    fn embedded_fleet_pack_is_a_valid_default_tier() {
        let compiled = BuiltinPack::FleetMemory.compile().unwrap();
        assert_eq!(compiled.id(), "fleet-memory@1.0.0");
        assert_eq!(
            compiled.microtheory_label(),
            "mt:schema/builtin/fleet-memory@1.0.0"
        );
        assert!(compiled.triples.iter().all(|t| matches!(
            t.predicate.as_str(),
            PRED_DEFAULT_RELATION | PRED_DEFAULT_CARDINALITY | PRED_DEFAULT_ALIAS_OF
        )));
        // Every declared relation is single-valued: the fleet default only
        // carries state predicates.
        for relation in &compiled.source.relations {
            assert!(
                compiled
                    .triples
                    .iter()
                    .any(|t| &t.subject == relation && t.predicate == PRED_DEFAULT_CARDINALITY),
                "{relation} must declare a cardinality"
            );
            assert!(!relation.starts_with("agent:"));
        }
    }

    #[test]
    fn a_pack_with_types_cannot_be_a_default_tier() {
        let typed =
            compile(r#"{"pack": {"name": "t", "version": "1"}, "types": [{"name": "Thing"}]}"#)
                .unwrap();
        assert!(matches!(
            default_tier_triples(&typed),
            Err(SchemaError::NotDefaultTier { .. })
        ));
    }

    #[test]
    fn pack_versions_order_numerically() {
        let v = |s: &str| s.parse::<PackVersion>().unwrap();
        assert!(v("1.10.0") > v("1.9.3"));
        assert_eq!(v("1.0"), v("1.0.0"));
        assert!(v("2") > v("1.99"));
        for bad in ["", "1..0", "1.0.0-rc1", "v1", "1.x"] {
            assert!(matches!(
                bad.parse::<PackVersion>(),
                Err(SchemaError::InvalidPackVersion { .. })
            ));
        }
    }

    #[test]
    fn installs_on_a_fresh_akh_and_is_idempotent() {
        let (engine, _dir) = engine();
        assert_eq!(cardinality(&engine, "status"), PredicateCardinality::Multi);

        let first = ensure(&engine, DefaultPackPolicy::Enabled);
        let BuiltinPackOutcome::Installed {
            pack,
            triples_applied,
        } = &first
        else {
            panic!("expected a first install, got {first:?}");
        };
        assert_eq!(pack, "fleet-memory@1.0.0");
        assert!(*triples_applied > 20);
        assert_eq!(cardinality(&engine, "status"), TEMPORAL);
        assert_eq!(cardinality(&engine, "tool:danger_level"), SINGLE);
        assert_eq!(
            cardinality(&engine, "agent:has_tag"),
            PredicateCardinality::Multi
        );
        assert_eq!(
            alias_target(&engine, "next-check").as_deref(),
            Some("next_check")
        );
        assert_eq!(
            alias_target(&engine, "has next check").as_deref(),
            Some("next_check")
        );
        let next_check = engine.lookup_symbol("next_check").unwrap();
        assert!(is_declared_relation(&engine, next_check));

        // Recorded like any installed pack: marker + activation edge.
        let mt = engine
            .lookup_symbol("mt:schema/builtin/fleet-memory@1.0.0")
            .unwrap();
        let importer = engine.lookup_symbol(&effective_schema_label(WS)).unwrap();
        let specializes = engine.lookup_symbol(PRED_SPECIALIZES).unwrap();
        assert!(
            engine
                .triples_from(importer)
                .iter()
                .any(|t| t.predicate == specializes && t.object == mt)
        );

        let before = engine.all_triples().len();
        let second = ensure(&engine, DefaultPackPolicy::Enabled);
        assert_eq!(
            second,
            BuiltinPackOutcome::Current {
                pack: "fleet-memory@1.0.0".into()
            }
        );
        assert!(!second.mutated());
        assert_eq!(engine.all_triples().len(), before);
    }

    #[test]
    fn opt_out_installs_nothing_and_retires_an_installed_copy() {
        let (fresh, _d1) = engine();
        assert_eq!(
            ensure(&fresh, DefaultPackPolicy::Disabled),
            BuiltinPackOutcome::Disabled {
                retired: vec![],
                triples_removed: 0
            }
        );
        assert!(fresh.all_triples().is_empty());

        let (engine, _d2) = engine();
        ensure(&engine, DefaultPackPolicy::Enabled);
        let outcome = ensure(&engine, DefaultPackPolicy::Disabled);
        let BuiltinPackOutcome::Disabled {
            retired,
            triples_removed,
        } = &outcome
        else {
            panic!("expected Disabled, got {outcome:?}");
        };
        assert_eq!(retired, &vec!["fleet-memory@1.0.0".to_string()]);
        assert!(*triples_removed > 20);
        assert_eq!(cardinality(&engine, "status"), PredicateCardinality::Multi);
        assert_eq!(alias_target(&engine, "next-check"), None);
        assert!(
            engine.all_triples().is_empty(),
            "nothing of the pack is left"
        );
        // Opting back in reinstalls.
        assert!(matches!(
            ensure(&engine, DefaultPackPolicy::Enabled),
            BuiltinPackOutcome::Installed { .. }
        ));
    }

    #[test]
    fn a_newer_installed_version_is_kept() {
        let (engine, _dir) = engine();
        install_as_version(&engine, "9.0.0", Some("future_state"));
        let before = engine.all_triples().len();
        assert_eq!(
            ensure(&engine, DefaultPackPolicy::Enabled),
            BuiltinPackOutcome::NewerKept {
                installed: "fleet-memory@9.0.0".into(),
                builtin: "fleet-memory@1.0.0".into(),
            }
        );
        assert_eq!(engine.all_triples().len(), before);
        assert_eq!(cardinality(&engine, "future_state"), SINGLE);
        assert!(
            engine
                .lookup_symbol("mt:schema/builtin/fleet-memory@1.0.0")
                .is_err()
        );
    }

    #[test]
    fn an_older_installed_version_is_upgraded() {
        let (engine, _dir) = engine();
        install_as_version(&engine, "0.9.0", Some("obsolete_state"));
        assert_eq!(cardinality(&engine, "obsolete_state"), SINGLE);

        let outcome = ensure(&engine, DefaultPackPolicy::Enabled);
        let BuiltinPackOutcome::Upgraded { from, to, .. } = &outcome else {
            panic!("expected Upgraded, got {outcome:?}");
        };
        assert_eq!(from, &vec!["fleet-memory@0.9.0".to_string()]);
        assert_eq!(to, "fleet-memory@1.0.0");
        // The relation 1.0.0 dropped is no longer declared; the ones both
        // versions declare survived the retirement of the old compartment.
        assert_eq!(
            cardinality(&engine, "obsolete_state"),
            PredicateCardinality::Multi
        );
        assert_eq!(cardinality(&engine, "status"), TEMPORAL);
        assert_eq!(
            alias_target(&engine, "next-check").as_deref(),
            Some("next_check")
        );
        assert!(matches!(
            ensure(&engine, DefaultPackPolicy::Enabled),
            BuiltinPackOutcome::Current { .. }
        ));
    }

    /// An akh's own pack wins on every relation both declare, whichever is
    /// installed first.
    #[test]
    fn an_akh_specific_pack_wins_on_overlapping_relations() {
        const AKH_PACK: &str = r#"{
            "pack": {"name": "akh-own", "version": "1.0.0", "scope": "own"},
            "relations": [
                {"name": "status", "cardinality": "single_valued"},
                {"name": "state", "cardinality": "multi"},
                {"name": "next_check", "cardinality": "single_valued", "temporal": true,
                 "aliases": ["nc"]},
                {"name": "next-action"},
                {"name": "phase", "aliases": ["current_phase"]}
            ]
        }"#;
        for fleet_first in [true, false] {
            let (engine, _dir) = engine();
            let own = compile(AKH_PACK).unwrap();
            if fleet_first {
                ensure(&engine, DefaultPackPolicy::Enabled);
                install_pack(&engine, &own).unwrap();
            } else {
                install_pack(&engine, &own).unwrap();
                ensure(&engine, DefaultPackPolicy::Enabled);
            }
            let order = if fleet_first {
                "fleet first"
            } else {
                "akh first"
            };
            // Cardinality: the akh pack's declaration, even an explicit multi.
            assert_eq!(cardinality(&engine, "status"), SINGLE, "{order}");
            assert_eq!(
                cardinality(&engine, "state"),
                PredicateCardinality::Multi,
                "{order}"
            );
            // Relations the akh pack does not mention keep the default.
            assert_eq!(cardinality(&engine, "has_state"), TEMPORAL, "{order}");
            // Aliases: both tiers' spellings of a shared relation apply.
            assert_eq!(alias_target(&engine, "nc").as_deref(), Some("next_check"));
            assert_eq!(
                alias_target(&engine, "next-check").as_deref(),
                Some("next_check")
            );
            // A spelling the akh pack declares as its own relation is not
            // rewritten by the default alias.
            assert_eq!(alias_target(&engine, "next-action"), None, "{order}");
            let next_action_spelling = engine.lookup_symbol("next-action").unwrap();
            assert!(is_declared_relation(&engine, next_action_spelling));
            // A default relation the akh pack re-declares as an alias follows
            // the akh pack.
            assert_eq!(
                alias_target(&engine, "current_phase").as_deref(),
                Some("phase"),
                "{order}"
            );
            let current_phase = engine.lookup_symbol("current_phase").unwrap();
            assert!(!is_declared_relation(&engine, current_phase), "{order}");
        }
    }

    /// The Roman pack next to the fleet pack: every roman-memory relation
    /// keeps roman-memory's declaration, and retiring the fleet pack never
    /// removes a roman-memory edge (the two tiers never share an edge).
    #[test]
    fn roman_memory_is_undisturbed_by_the_fleet_pack() {
        let roman_source = include_str!("../../deploy/roman-mini/schemas/roman-memory.json");
        let roman = compile(roman_source).unwrap();
        let parsed = super::super::parse_source(roman_source).unwrap();
        for fleet_first in [true, false] {
            let (engine, _dir) = engine();
            if fleet_first {
                ensure(&engine, DefaultPackPolicy::Enabled);
                install_pack(&engine, &roman).unwrap();
            } else {
                install_pack(&engine, &roman).unwrap();
                ensure(&engine, DefaultPackPolicy::Enabled);
            }
            let roman_edges: Vec<Triple> = engine
                .all_triples()
                .into_iter()
                .filter(|t| t.compartment_id.as_deref() == Some(&roman.microtheory_label()))
                .collect();
            let check = |engine: &Engine| {
                for r in &parsed.relations {
                    let expected = PredicateCardinality::SingleValued {
                        temporal: r.temporal,
                    };
                    assert_eq!(cardinality(engine, &r.name), expected, "{}", r.name);
                    for alias in &r.aliases {
                        assert_eq!(
                            alias_target(engine, alias).as_deref(),
                            Some(r.name.as_str()),
                            "{alias}"
                        );
                    }
                }
            };
            check(&engine);
            ensure(&engine, DefaultPackPolicy::Disabled);
            check(&engine);
            for edge in &roman_edges {
                assert!(
                    engine
                        .triples_from(edge.subject)
                        .iter()
                        .any(|t| t.predicate == edge.predicate && t.object == edge.object),
                    "roman-memory edge removed by the fleet retirement"
                );
            }
        }
    }

    /// A backlog written before the pack existed converges: exact spelling
    /// collisions merge onto the declared relation, each subject keeps its
    /// newest value, declared aliases merge, and a second pass is a no-op.
    #[test]
    fn hygiene_converges_on_fleet_pack_predicates() {
        let (engine, _dir) = engine();
        put(&engine, "wi-1", "status", "open", 10);
        put(&engine, "wi-1", "status", "review", 20);
        put(&engine, "wi-1", "status", "merged", 30);
        put(&engine, "wi-2", "next_check", "monday", 10);
        put(&engine, "wi-2", "next-check", "tuesday", 20);
        put(&engine, "wi-2", "has_next_check", "wednesday", 30);
        put(&engine, "tool:task_list", "tool:danger_level", "Safe", 10);
        put(
            &engine,
            "tool:task_list",
            "tool:danger_level",
            "Cautious",
            20,
        );
        put(&engine, "wi-1", "agent:has_tag", "a", 1);
        put(&engine, "wi-1", "agent:has_tag", "b", 2);

        ensure(&engine, DefaultPackPolicy::Enabled);
        let options = HygieneOptions {
            policy: ContradictionPolicy::Warn,
            max_mutations: crate::schema::DEFAULT_HYGIENE_BATCH,
            dry_run: false,
        };
        let first = run_hygiene(&engine, options).unwrap();
        assert!(!first.truncated);
        assert!(first.superseded >= 3, "{first:?}");
        execute_merge(
            &engine,
            &MergeRequest {
                selection: MergeSelection::Declared,
                dry_run: false,
                limit: crate::schema::DEFAULT_MERGE_LIMIT,
            },
        )
        .unwrap();
        run_hygiene(&engine, options).unwrap();
        let idempotent = run_hygiene(&engine, options).unwrap();
        assert_eq!(idempotent.mutations, 0);

        assert_eq!(live(&engine, "wi-1", "status"), vec!["merged"]);
        assert_eq!(live(&engine, "wi-2", "next_check"), vec!["wednesday"]);
        assert!(live(&engine, "wi-2", "next-check").is_empty());
        assert!(live(&engine, "wi-2", "has_next_check").is_empty());
        assert_eq!(
            live(&engine, "tool:task_list", "tool:danger_level"),
            vec!["Cautious"]
        );
        assert_eq!(live(&engine, "wi-1", "agent:has_tag"), vec!["a", "b"]);

        // Insert-time supersession once the pack is in place.
        put(&engine, "wi-1", "status", "deployed", 40);
        assert_eq!(live(&engine, "wi-1", "status"), vec!["deployed"]);
        put(&engine, "wi-3", "current state", "idle", 1);
        assert_eq!(live(&engine, "wi-3", "current_state"), vec!["idle"]);
    }
}
