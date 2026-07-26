//! Confirm-time compiler: portable source → native-triple byte blob.
//!
//! ADR 081 decision 9: the registry holds **native-triple byte blobs**
//! "compiled once, at the human-confirm step, from a portable GitHub-curated
//! source"; per-akh install is then a dumb byte-fetch + bulk-insert. The
//! compiled triples are **label-based** (not `SymbolId`-based) because
//! `SymbolId`s are per-akh — labels are the relocatable, cross-akh names,
//! exactly like seed packs.

use serde::{Deserialize, Serialize};

use super::source::{PackScope, SchemaPackSource, parse_source};
use super::{SchemaError, SchemaResult};

// ---------------------------------------------------------------------------
// Lowering vocabulary (well-known labels)
// ---------------------------------------------------------------------------

/// Marker entity: `(T, is-a, onto:Type)` declares a global entity type.
pub const TYPE_MARKER: &str = "onto:Type";
/// Marker entity for relations (seed convention: `(r, is-a, relation)` mints
/// the subject as `SymbolKind::Relation` on install).
pub const RELATION_MARKER: &str = "relation";
/// `is-a` — subtyping / instance-of (the only extension mechanism).
pub const PRED_IS_A: &str = "is-a";
/// `(T, onto:property, p)` — additive property attachment.
pub const PRED_PROPERTY: &str = "onto:property";
/// `(T, onto:required, p)` — property required by the type's shape.
pub const PRED_REQUIRED: &str = "onto:required";
/// `(p, onto:value-type, onto:value/…)` — Obsidian value type of a property.
pub const PRED_VALUE_TYPE: &str = "onto:value-type";
/// `(r, onto:arity, onto:arity/2)` — declared arity of a relation.
pub const PRED_ARITY: &str = "onto:arity";
/// Entity naming binary arity.
pub const ARITY_2: &str = "onto:arity/2";
/// `(r, onto:arg1type, T)` — required subject type (matches `ArityPredicates`).
pub const PRED_ARG1_TYPE: &str = "onto:arg1type";
/// `(r, onto:arg2type, T)` — required object type (matches `ArityPredicates`).
pub const PRED_ARG2_TYPE: &str = "onto:arg2type";

/// Current blob format version. Bump on any incompatible change to
/// [`CompiledSchemaPack`]'s layout.
pub const BLOB_FORMAT_VERSION: u16 = 1;

// ---------------------------------------------------------------------------
// Compiled artifact
// ---------------------------------------------------------------------------

/// A label-based native triple, insertable into any akh without translation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortableTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl PortableTriple {
    fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            subject: s.into(),
            predicate: p.into(),
            object: o.into(),
        }
    }
}

/// The compiled schema pack: what the registry blob column holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledSchemaPack {
    /// Blob format version ([`BLOB_FORMAT_VERSION`] at compile time).
    pub format_version: u16,
    /// Pack name (registry-row `name`).
    pub name: String,
    /// Pack version (registry-row `version`).
    pub version: String,
    /// Registry-row `scope`.
    pub scope: PackScope,
    /// Registry-row `source_url` (where the portable source is curated).
    pub source_url: Option<String>,
    /// blake3 hex of the exact portable source bytes this blob was compiled
    /// from — ties the artifact back to the reviewed text.
    pub source_checksum: String,
    /// External type names this pack extends (verified against the registry
    /// at confirm time — the compiler cannot see other packs offline).
    pub assumes: Vec<String>,
    /// Type names this pack declares (for the immutable-global-name guard
    /// without scanning triples).
    pub types: Vec<String>,
    /// Relation names this pack declares.
    pub relations: Vec<String>,
    /// The native triples, insertable as-is under the pack's microtheory.
    pub triples: Vec<PortableTriple>,
}

impl CompiledSchemaPack {
    /// The microtheory compartment label this pack installs under.
    /// Deterministic from (name, version) so triple identities of installed
    /// schema triples agree across akhs.
    pub fn microtheory_label(&self) -> String {
        format!("mt:schema/{}@{}", self.name, self.version)
    }

    /// Serialize to the registry blob bytes (bincode).
    pub fn to_blob(&self) -> SchemaResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| SchemaError::Encode {
            message: e.to_string(),
        })
    }

    /// Deserialize from registry blob bytes, checking the format version.
    pub fn from_blob(bytes: &[u8]) -> SchemaResult<Self> {
        let pack: Self = bincode::deserialize(bytes).map_err(|e| SchemaError::Decode {
            message: e.to_string(),
        })?;
        if pack.format_version > BLOB_FORMAT_VERSION {
            return Err(SchemaError::FormatVersion {
                found: pack.format_version,
                supported: BLOB_FORMAT_VERSION,
            });
        }
        Ok(pack)
    }
}

/// blake3 hex checksum of a byte string (used for both the source checksum
/// embedded in the artifact and the registry-row blob checksum).
pub fn checksum_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

/// Compile a portable schema-pack source text into a [`CompiledSchemaPack`].
///
/// Parses + validates ([`parse_source`]) then lowers declarations into the
/// native-triple vocabulary documented in [`crate::schema`].
pub fn compile(source_text: &str) -> SchemaResult<CompiledSchemaPack> {
    let source = parse_source(source_text)?;
    Ok(lower(&source, source_text))
}

fn lower(source: &SchemaPackSource, source_text: &str) -> CompiledSchemaPack {
    let mut triples = Vec::new();

    for t in &source.types {
        // (T, is-a, onto:Type) — global type declaration.
        triples.push(PortableTriple::new(&t.name, PRED_IS_A, TYPE_MARKER));
        // (Sub, is-a, Super) — subtype edge.
        if let Some(ref sup) = t.is_a {
            triples.push(PortableTriple::new(&t.name, PRED_IS_A, sup));
        }
        for p in &t.properties {
            triples.push(PortableTriple::new(&t.name, PRED_PROPERTY, &p.name));
            if p.required {
                triples.push(PortableTriple::new(&t.name, PRED_REQUIRED, &p.name));
            }
            triples.push(PortableTriple::new(
                &p.name,
                PRED_VALUE_TYPE,
                p.value_type.entity_label(),
            ));
        }
    }

    for r in &source.relations {
        // Seed convention: minted as SymbolKind::Relation at install.
        triples.push(PortableTriple::new(&r.name, PRED_IS_A, RELATION_MARKER));
        triples.push(PortableTriple::new(&r.name, PRED_ARITY, ARITY_2));
        if let Some(ref t1) = r.arg1_type {
            triples.push(PortableTriple::new(&r.name, PRED_ARG1_TYPE, t1));
        }
        if let Some(ref t2) = r.arg2_type {
            triples.push(PortableTriple::new(&r.name, PRED_ARG2_TYPE, t2));
        }
    }

    CompiledSchemaPack {
        format_version: BLOB_FORMAT_VERSION,
        name: source.pack.name.clone(),
        version: source.pack.version.clone(),
        scope: source.pack.scope,
        source_url: source.pack.source_url.clone(),
        source_checksum: checksum_hex(source_text.as_bytes()),
        assumes: source.assumes.clone(),
        types: source.types.iter().map(|t| t.name.clone()).collect(),
        relations: source.relations.iter().map(|r| r.name.clone()).collect(),
        triples,
    }
}

// ---------------------------------------------------------------------------
// Registry seam (ADR 081 decision 9 — Postgres half is Slice 2 server work)
// ---------------------------------------------------------------------------

/// A row of the schema-pack registry table:
/// `{id, version, name, scope, source_url, checksum, confirmed_by, blob}`.
/// `id` is assigned by the store on admission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryRow {
    pub name: String,
    pub version: String,
    pub scope: PackScope,
    pub source_url: Option<String>,
    /// blake3 hex of `blob` — per-akh fetch verifies this before install.
    pub checksum: String,
    /// The human who confirmed admission (mandatory: ADR 081 decision 8 —
    /// admission to the shared registry always requires a human confirm).
    pub confirmed_by: String,
    /// The compiled native-triple blob ([`CompiledSchemaPack::to_blob`]).
    pub blob: Vec<u8>,
}

impl RegistryRow {
    /// Build a registry row from a compiled pack at the human-confirm step.
    pub fn from_pack(pack: &CompiledSchemaPack, confirmed_by: impl Into<String>) -> SchemaResult<Self> {
        let blob = pack.to_blob()?;
        Ok(Self {
            name: pack.name.clone(),
            version: pack.version.clone(),
            scope: pack.scope,
            source_url: pack.source_url.clone(),
            checksum: checksum_hex(&blob),
            confirmed_by: confirmed_by.into(),
            blob,
        })
    }

    /// Verify + decode the blob (checksum, then format version).
    pub fn decode(&self) -> SchemaResult<CompiledSchemaPack> {
        let actual = checksum_hex(&self.blob);
        if actual != self.checksum {
            return Err(SchemaError::Decode {
                message: format!(
                    "blob checksum mismatch: row says {}, blob hashes to {actual}",
                    self.checksum
                ),
            });
        }
        CompiledSchemaPack::from_blob(&self.blob)
    }
}

/// The schema-pack registry seam (ADR 081 decision 9).
///
/// The production implementation ([`super::registry_pg::PgSchemaRegistry`],
/// behind the `postgres` feature) is a dedicated `akhomed-store` Postgres
/// table with the operator-confirm inflow and workspace-scoped activation.
/// This trait is the boundary the compiler/install side codes against; tests
/// use an in-memory impl (see the `tests` module below).
pub trait SchemaRegistry {
    /// Admit a confirmed pack to the registry. Must reject a (name, version)
    /// that is already admitted, and must reject any row whose declared type
    /// names collide with types declared by an already-admitted pack
    /// (global immutability, ADR 081 decision 5). Implementations should
    /// call [`check_admission`] first to get both checks for free.
    fn admit(&self, row: RegistryRow) -> SchemaResult<()>;

    /// Fetch an admitted pack by (name, version).
    fn fetch(&self, name: &str, version: &str) -> SchemaResult<Option<RegistryRow>>;

    /// List admitted packs as (name, version) pairs.
    fn list(&self) -> SchemaResult<Vec<(String, String)>>;
}

/// Validate a row against `registry`'s already-admitted rows before
/// insertion — the two guards [`SchemaRegistry::admit`] documents, factored
/// out so every backend enforces them identically:
///
/// 1. reject a duplicate `(name, version)`
/// 2. reject any type name the row declares that a *different* already-
///    admitted pack also declares (ADR 081 decision 5: global, immutable,
///    additive type names — enforced at the registry gate, not only at
///    per-akh install time).
///
/// Backends call this at the top of their `admit` implementation, then do
/// the backend-specific write.
pub fn check_admission(registry: &dyn SchemaRegistry, row: &RegistryRow) -> SchemaResult<()> {
    if registry.fetch(&row.name, &row.version)?.is_some() {
        return Err(SchemaError::AlreadyAdmitted {
            name: row.name.clone(),
            version: row.version.clone(),
        });
    }

    let pack = row.decode()?;
    for (other_name, other_version) in registry.list()? {
        if other_name == row.name && other_version == row.version {
            continue; // the duplicate case, already handled above
        }
        let Some(other_row) = registry.fetch(&other_name, &other_version)? else {
            continue; // raced deletion between list() and fetch(); ignore
        };
        let other_pack = other_row.decode()?;
        for type_name in &pack.types {
            if other_pack.types.contains(type_name) {
                return Err(SchemaError::TypeRedefinition {
                    type_name: type_name.clone(),
                    existing_compartment: other_pack.microtheory_label(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"{
        "pack": {"name": "core-test", "version": "0.1.0",
                 "source_url": "https://example.org/core-test.json"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "name", "value_type": "text", "required": true},
                {"name": "url", "value_type": "link"}]},
            {"name": "Organization"},
            {"name": "Employee", "is_a": "Person", "properties": [
                {"name": "employer", "value_type": "link", "required": true}]}
        ],
        "relations": [
            {"name": "worksFor", "arg1_type": "Person", "arg2_type": "Organization"}
        ]
    }"#;

    #[test]
    fn compiles_expected_triples() {
        let pack = compile(SRC).unwrap();
        assert_eq!(pack.format_version, BLOB_FORMAT_VERSION);
        assert_eq!(pack.name, "core-test");
        assert_eq!(pack.microtheory_label(), "mt:schema/core-test@0.1.0");
        assert_eq!(pack.types, vec!["Person", "Organization", "Employee"]);
        assert_eq!(pack.relations, vec!["worksFor"]);

        let has = |s: &str, p: &str, o: &str| {
            pack.triples
                .iter()
                .any(|t| t.subject == s && t.predicate == p && t.object == o)
        };
        // Type declarations.
        assert!(has("Person", PRED_IS_A, TYPE_MARKER));
        assert!(has("Organization", PRED_IS_A, TYPE_MARKER));
        assert!(has("Employee", PRED_IS_A, TYPE_MARKER));
        // Subtype edge.
        assert!(has("Employee", PRED_IS_A, "Person"));
        // Property shape.
        assert!(has("Person", PRED_PROPERTY, "name"));
        assert!(has("Person", PRED_REQUIRED, "name"));
        assert!(has("Person", PRED_PROPERTY, "url"));
        assert!(!has("Person", PRED_REQUIRED, "url"));
        assert!(has("name", PRED_VALUE_TYPE, "onto:value/text"));
        assert!(has("url", PRED_VALUE_TYPE, "onto:value/link"));
        assert!(has("Employee", PRED_PROPERTY, "employer"));
        // Relation shape.
        assert!(has("worksFor", PRED_IS_A, RELATION_MARKER));
        assert!(has("worksFor", PRED_ARITY, ARITY_2));
        assert!(has("worksFor", PRED_ARG1_TYPE, "Person"));
        assert!(has("worksFor", PRED_ARG2_TYPE, "Organization"));
    }

    #[test]
    fn compilation_is_deterministic() {
        let a = compile(SRC).unwrap();
        let b = compile(SRC).unwrap();
        assert_eq!(a.to_blob().unwrap(), b.to_blob().unwrap());
        assert_eq!(a.source_checksum, b.source_checksum);
    }

    #[test]
    fn blob_round_trips() {
        let pack = compile(SRC).unwrap();
        let blob = pack.to_blob().unwrap();
        let back = CompiledSchemaPack::from_blob(&blob).unwrap();
        assert_eq!(back.name, pack.name);
        assert_eq!(back.triples, pack.triples);
    }

    #[test]
    fn from_blob_rejects_newer_format() {
        let mut pack = compile(SRC).unwrap();
        pack.format_version = BLOB_FORMAT_VERSION + 1;
        let blob = pack.to_blob().unwrap();
        assert!(matches!(
            CompiledSchemaPack::from_blob(&blob),
            Err(SchemaError::FormatVersion { .. })
        ));
    }

    #[test]
    fn from_blob_rejects_garbage() {
        assert!(matches!(
            CompiledSchemaPack::from_blob(b"not a blob"),
            Err(SchemaError::Decode { .. })
        ));
    }

    #[test]
    fn bundled_seed_pack_fixture_compiles() {
        let text = include_str!("../../data/schema-packs/schema-org-core.json");
        let pack = compile(text).unwrap();
        assert_eq!(pack.name, "schema-org-core");
        assert_eq!(pack.scope, PackScope::Shared);
        // The seed core: schema.org Thing subset + Akh-native types (ADR 081 dec. 4).
        for ty in [
            "Thing",
            "Person",
            "Organization",
            "Place",
            "Event",
            "CreativeWork",
            "Akh",
            "Goal",
            "Skill",
            "Decision",
        ] {
            assert!(pack.types.iter().any(|t| t == ty), "missing type {ty}");
        }
        assert!(pack.relations.iter().any(|r| r == "worksFor"));
        // Blob round-trips.
        let blob = pack.to_blob().unwrap();
        let back = CompiledSchemaPack::from_blob(&blob).unwrap();
        assert_eq!(back.triples, pack.triples);
    }

    #[test]
    fn registry_row_checksum_guards_blob() {
        let pack = compile(SRC).unwrap();
        let mut row = RegistryRow::from_pack(&pack, "toasty").unwrap();
        assert_eq!(row.confirmed_by, "toasty");
        assert!(row.decode().is_ok());
        // Tamper with the blob → checksum mismatch.
        row.blob[0] ^= 0xFF;
        assert!(matches!(row.decode(), Err(SchemaError::Decode { .. })));
    }

    // -----------------------------------------------------------------
    // `SchemaRegistry` trait contract — proved against an in-memory fake.
    // Every real backend (the Postgres `PgSchemaRegistry`, `postgres`
    // feature) is expected to satisfy the exact same contract; these tests
    // exercise it without any database.
    // -----------------------------------------------------------------

    use std::sync::Mutex;

    /// A trivial in-memory [`SchemaRegistry`] — a `Vec<RegistryRow>` behind a
    /// mutex, admitting via [`check_admission`] exactly like the Postgres
    /// backend does. Used only by tests.
    #[derive(Default)]
    struct InMemoryRegistry {
        rows: Mutex<Vec<RegistryRow>>,
    }

    impl SchemaRegistry for InMemoryRegistry {
        fn admit(&self, row: RegistryRow) -> SchemaResult<()> {
            check_admission(self, &row)?;
            self.rows.lock().unwrap().push(row);
            Ok(())
        }

        fn fetch(&self, name: &str, version: &str) -> SchemaResult<Option<RegistryRow>> {
            Ok(self
                .rows
                .lock()
                .unwrap()
                .iter()
                .find(|r| r.name == name && r.version == version)
                .cloned())
        }

        fn list(&self) -> SchemaResult<Vec<(String, String)>> {
            Ok(self
                .rows
                .lock()
                .unwrap()
                .iter()
                .map(|r| (r.name.clone(), r.version.clone()))
                .collect())
        }
    }

    #[test]
    fn registry_admits_and_lists_a_pack() {
        let registry = InMemoryRegistry::default();
        let pack = compile(SRC).unwrap();
        let row = RegistryRow::from_pack(&pack, "toasty").unwrap();

        registry.admit(row).unwrap();

        assert_eq!(
            registry.list().unwrap(),
            vec![("core-test".to_string(), "0.1.0".to_string())]
        );
        let fetched = registry.fetch("core-test", "0.1.0").unwrap().unwrap();
        assert_eq!(fetched.confirmed_by, "toasty");
        assert!(registry.fetch("core-test", "9.9.9").unwrap().is_none());
    }

    #[test]
    fn registry_rejects_duplicate_name_and_version() {
        let registry = InMemoryRegistry::default();
        let pack = compile(SRC).unwrap();
        registry
            .admit(RegistryRow::from_pack(&pack, "toasty").unwrap())
            .unwrap();

        let err = registry
            .admit(RegistryRow::from_pack(&pack, "someone-else").unwrap())
            .unwrap_err();
        assert!(matches!(
            err,
            SchemaError::AlreadyAdmitted { ref name, ref version }
                if name == "core-test" && version == "0.1.0"
        ));
    }

    #[test]
    fn registry_rejects_type_collision_across_different_packs() {
        let registry = InMemoryRegistry::default();
        let first = compile(SRC).unwrap(); // declares Person, Organization, Employee
        registry
            .admit(RegistryRow::from_pack(&first, "toasty").unwrap())
            .unwrap();

        // A different pack (different name/version) re-declaring "Person" must
        // be rejected at the registry gate, not only at per-akh install time.
        let rival = compile(
            r#"{"pack": {"name": "rival", "version": "1.0.0"},
                "types": [{"name": "Person"}]}"#,
        )
        .unwrap();
        let err = registry
            .admit(RegistryRow::from_pack(&rival, "toasty").unwrap())
            .unwrap_err();
        assert!(matches!(
            err,
            SchemaError::TypeRedefinition { ref type_name, .. } if type_name == "Person"
        ));
    }

    #[test]
    fn registry_allows_additive_extension_via_is_a() {
        let registry = InMemoryRegistry::default();
        let core = compile(SRC).unwrap();
        registry
            .admit(RegistryRow::from_pack(&core, "toasty").unwrap())
            .unwrap();

        // A pack that only *extends* an existing type via `is_a` (no
        // redeclared type names) is admitted cleanly.
        let extension = compile(
            r#"{"pack": {"name": "hr", "version": "1.0.0"},
                "assumes": ["Person"],
                "types": [{"name": "Manager", "is_a": "Person"}]}"#,
        )
        .unwrap();
        registry
            .admit(RegistryRow::from_pack(&extension, "toasty").unwrap())
            .unwrap();

        let mut names: Vec<_> = registry.list().unwrap().into_iter().map(|(n, _)| n).collect();
        names.sort();
        assert_eq!(names, vec!["core-test".to_string(), "hr".to_string()]);
    }

    #[test]
    fn registry_allows_reinstalling_the_same_pack_version_is_still_a_conflict() {
        // NOTE: admission (registry-level) is stricter than install
        // (engine-level): reinstalling the exact same pack@version at the
        // engine is idempotent (`install::reinstall_same_pack_is_idempotent`),
        // but *admitting* the same (name, version) twice is always rejected —
        // the registry row is meant to be written exactly once.
        let registry = InMemoryRegistry::default();
        let pack = compile(SRC).unwrap();
        let row = RegistryRow::from_pack(&pack, "toasty").unwrap();
        registry.admit(row.clone()).unwrap();
        assert!(matches!(
            registry.admit(row).unwrap_err(),
            SchemaError::AlreadyAdmitted { .. }
        ));
    }
}
