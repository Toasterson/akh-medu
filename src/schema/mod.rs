//! Schema packs: the installable type system for Akh memory (ADR 081).
//!
//! A **schema pack** is a typed microtheory shipped as an artifact: a
//! human-authored portable source (curated schema.org subset + Akh-native
//! types, `is-a` subtyping, `onto:` property shapes) that is **compiled
//! once, at the human-confirm step**, into a native-triple byte blob. Per-akh
//! install is then a dumb byte-fetch + bulk-insert-as-microtheory — no
//! bespoke translation code (ADR 081 decisions 4–6, 8, 9).
//!
//! This module is the **offline compiler + install half** of Slice 2 in the
//! work breakdown (`docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`):
//!
//! - [`source`] — the portable source model + parser/validator
//!   (JSON front-end; Turtle/YAML front-ends can be added behind
//!   [`source::parse_source`]).
//! - [`compile`] — lowering to [`compile::CompiledSchemaPack`] (label-based
//!   native triples), the bincode blob format, and the [`compile::SchemaRegistry`]
//!   seam the Postgres registry implements later.
//! - [`install`] — bulk-insert a compiled pack into an [`crate::engine::Engine`]
//!   as a microtheory compartment, with the global-immutable-type-name guard
//!   (ADR 081 decision 5), plus `ctx:specializes` activation.
//!
//! ## Native-triple vocabulary (the lowering target)
//!
//! | Triple | Meaning |
//! |---|---|
//! | `(T, is-a, onto:Type)` | `T` is a globally-unique entity type |
//! | `(Sub, is-a, Super)` | subtype extension (the only extension mechanism) |
//! | `(T, onto:property, p)` | type `T` carries property `p` (additive) |
//! | `(T, onto:required, p)` | property `p` is required by `T`'s shape |
//! | `(p, onto:value-type, onto:value/text)` | Obsidian value type of `p` |
//! | `(r, is-a, relation)` | `r` is a relation (seed convention — minted as `SymbolKind::Relation`) |
//! | `(r, onto:arity, onto:arity/2)` | declared arity (binary only today) |
//! | `(r, onto:arg1type, T)` / `(r, onto:arg2type, T)` | argument type shapes (reuses `ArityPredicates` labels) |
//!
//! References: ADR 081 (`docs/ai/decisions/081-akh-memory-schema-typing.md`),
//! plan (`docs/ai/plans/2026-06-30-akh-memory-schema.md`), Anima project
//! `019d9844-ce74-7993-a9f0-c953b8f04a4e`.

pub mod compile;
pub mod conflict;
pub mod coverage;
pub mod digest;
pub mod install;
pub mod project;
#[cfg(feature = "postgres")]
pub mod registry_pg;
pub mod shape;
pub mod source;

use miette::Diagnostic;
use thiserror::Error;

pub use compile::{
    CompiledSchemaPack, PortableTriple, RegistryRow, SchemaRegistry, check_admission, compile,
};
pub use conflict::{
    AssertionConflict, ResolvedAssertion, SourcedAssertion, assert_with_source,
    conflicting_assertions, resolve_conflict,
};
pub use coverage::{
    CompartmentBreakdown, CoverageReport, SchemaView, TripleClass, classify_triple,
    coverage_report, coverage_report_top_n,
};
pub use digest::{PropertyDigest, RelationDigest, SchemaDigest, digest};
pub use install::{InstallReport, activate_pack, install_pack};
pub use project::{
    MANIFEST_COMPARTMENT, PRED_PROJECTION_ASSERTED, ProjectionDiff, ProjectionReport, TypeOutcome,
    TypedNoteView, TypedPropertyProjection, note_of_entity, project_note, reproject_note,
    typed_view,
};
#[cfg(feature = "postgres")]
pub use registry_pg::PgSchemaRegistry;
pub use shape::{
    EffectiveProperty, EffectivePropertySet, ShapeViolation, check_value_shape,
    effective_properties, effective_properties_of_type, is_installed_type, type_chain,
};
pub use source::{PackScope, SchemaPackSource, ValueType, parse_source};

/// Errors from schema-pack parsing, compilation, and installation.
#[derive(Debug, Error, Diagnostic)]
pub enum SchemaError {
    #[error("failed to parse schema pack source: {message}")]
    #[diagnostic(
        code(akh::schema::parse),
        help(
            "The source must be a JSON document with a `pack` object ({{name, version}}) and \
             optional `types`, `relations`, and `assumes` arrays. See \
             `data/schema-packs/schema-org-core.json` for a reference pack."
        )
    )]
    Parse { message: String },

    #[error("pack metadata invalid: {message}")]
    #[diagnostic(
        code(akh::schema::invalid_meta),
        help(
            "`pack.name` must be non-empty without whitespace (it becomes part of the \
             microtheory label), and `pack.version` must be non-empty."
        )
    )]
    InvalidMeta { message: String },

    #[error("duplicate type declaration: \"{name}\"")]
    #[diagnostic(
        code(akh::schema::duplicate_type),
        help(
            "Type names are globally unique and immutable (ADR 081 decision 5). Declare each \
             type once; extend behaviour with an `is_a` subtype instead of redefining."
        )
    )]
    DuplicateType { name: String },

    #[error("duplicate relation declaration: \"{name}\"")]
    #[diagnostic(
        code(akh::schema::duplicate_relation),
        help("Relation names are globally unique. Declare each relation once per pack.")
    )]
    DuplicateRelation { name: String },

    #[error("duplicate property \"{property}\" on type \"{type_name}\"")]
    #[diagnostic(
        code(akh::schema::duplicate_property),
        help("List each property once per type. Subtypes add properties additively via `is_a`.")
    )]
    DuplicateProperty { type_name: String, property: String },

    #[error("type \"{type_name}\" extends unknown supertype \"{super_name}\"")]
    #[diagnostic(
        code(akh::schema::unknown_supertype),
        help(
            "Either declare \"{super_name}\" in this pack's `types`, or list it in `assumes` \
             if it is provided by another (already-admitted) pack the registry can check at \
             confirm time."
        )
    )]
    UnknownSupertype {
        type_name: String,
        super_name: String,
    },

    #[error("relation \"{relation}\" constrains arg{arg} to unknown type \"{type_name}\"")]
    #[diagnostic(
        code(akh::schema::unknown_arg_type),
        help(
            "Argument type constraints must reference a type declared in this pack or listed \
             in `assumes`."
        )
    )]
    UnknownArgType {
        relation: String,
        arg: u8,
        type_name: String,
    },

    #[error("relation \"{relation}\" declares unsupported arity {arity}")]
    #[diagnostic(
        code(akh::schema::unsupported_arity),
        help("Triples are binary: only arity 2 is supported (omit `arity` to default to 2).")
    )]
    UnsupportedArity { relation: String, arity: u64 },

    #[error("reserved name: \"{name}\"")]
    #[diagnostic(
        code(akh::schema::reserved_name),
        help(
            "Names under the engine namespaces (`onto:`, `ctx:`, `agent:`, `doc:`, `akh:`, \
             `rel:`, `mt:`) and the bare word `relation` are reserved for the substrate and \
             cannot be declared by a pack."
        )
    )]
    ReservedName { name: String },

    #[error("failed to encode compiled schema pack: {message}")]
    #[diagnostic(
        code(akh::schema::encode),
        help("This is an internal serialization failure — please report it.")
    )]
    Encode { message: String },

    #[error("failed to decode schema pack blob: {message}")]
    #[diagnostic(
        code(akh::schema::decode),
        help(
            "The blob may be truncated, corrupted, or produced by an incompatible compiler. \
             Verify the registry checksum matches, then recompile from the portable source."
        )
    )]
    Decode { message: String },

    #[error(
        "unsupported schema pack blob format version {found} (this build reads up to {supported})"
    )]
    #[diagnostic(
        code(akh::schema::format_version),
        help(
            "The blob was compiled by a newer akh-medu. Upgrade this binary, or recompile the \
             pack's portable source with this build's `akh schema compile`."
        )
    )]
    FormatVersion { found: u16, supported: u16 },

    #[error(
        "type \"{type_name}\" is already defined by \"{existing_compartment}\" — global type \
         names are immutable"
    )]
    #[diagnostic(
        code(akh::schema::type_redefinition),
        help(
            "ADR 081 decision 5: a type name may never be redefined. Extend it with an `is-a` \
             subtype in a new pack, or bump this pack's version only if the definition is \
             byte-identical (reinstall of the same pack@version is idempotent and allowed)."
        )
    )]
    TypeRedefinition {
        type_name: String,
        existing_compartment: String,
    },

    #[error("failed to install schema pack \"{pack}\": {message}")]
    #[diagnostic(
        code(akh::schema::install),
        help(
            "Check that the engine is writable; the pack blob decoded cleanly, so this is an engine-side failure."
        )
    )]
    Install { pack: String, message: String },

    #[error("failed to project note \"{note_id}\" into the typed KG: {message}")]
    #[diagnostic(
        code(akh::schema::projection),
        help(
            "Projection never blocks on typing problems by design (ADR 081 decision 7 — \
             unknown types/keys land in frontier instead). This error means a genuine \
             engine-side failure (symbol allocation, constraint violation, or the note was \
             never projected when reading a typed_view back) — check the message for which."
        )
    )]
    Projection { note_id: String, message: String },

    #[error("io error on {path}: {source}")]
    #[diagnostic(
        code(akh::schema::io),
        help("Ensure the file exists and is readable/writable.")
    )]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("schema pack \"{name}@{version}\" is already admitted to the registry")]
    #[diagnostic(
        code(akh::schema::already_admitted),
        help(
            "Registry rows are immutable once admitted (ADR 081 decision 9). Bump the pack's \
             version to publish a new revision — reinstalling the existing pack@version stays \
             idempotent at the per-akh install step, so nothing is lost by leaving it as-is."
        )
    )]
    AlreadyAdmitted { name: String, version: String },

    #[error("schema-pack registry operation failed: {message}")]
    #[diagnostic(
        code(akh::schema::registry),
        help(
            "This is a registry-backend failure (e.g. the Postgres connection). Check the \
             daemon's `[postgres]` configuration and connectivity, then retry."
        )
    )]
    Registry { message: String },

    #[error("subtype cycle detected while resolving \"{type_name}\": {}", chain.join(" -> "))]
    #[diagnostic(
        code(akh::schema::subtype_cycle),
        help(
            "The `is-a` chain loops back on itself instead of terminating at a root type. This \
             can only happen across independently-compiled packs (each pack's own subtype edges \
             are validated acyclic within the pack at parse time; the registry does not yet \
             check cross-pack cycles at admission time — see ADR 081 decision 5). Fix one \
             pack's `is_a` declaration to break the cycle and reinstall; until then, notes of \
             this type project with an empty effective property set (every field lands in \
             frontier) rather than hanging or failing the write."
        )
    )]
    SubtypeCycle {
        type_name: String,
        chain: Vec<String>,
    },

    #[error("schema introspection failed: {message}")]
    #[diagnostic(
        code(akh::schema::introspection),
        help(
            "This is a read-only query against the installed schema graph (effective-property \
             resolution or the schema digest). Check that at least one schema pack is installed \
             in this engine, then retry."
        )
    )]
    Introspection { message: String },
}

/// Result alias for schema-pack operations.
pub type SchemaResult<T> = std::result::Result<T, SchemaError>;
