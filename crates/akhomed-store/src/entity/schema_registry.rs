//! Entity: `akhomed_schema_registry` — the admitted schema-pack registry
//! (ADR 081 decision 9). One row per admitted `(name, version)` pack: the
//! compiled native-triple blob plus its provenance (curated source, blake3
//! checksum, confirming human). Rows are insert-only — a `(name, version)`
//! pair is never overwritten once admitted (see
//! [`crate::schema_registry::admit_pack`]'s duplicate guard); a new revision
//! of a pack is a new row with a bumped `version`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_schema_registry")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,
    /// Pack name (globally unique together with `version`).
    pub name: String,
    pub version: String,
    /// `PackScope` as text (`"shared"` | `"own"`) — the main crate's
    /// `schema::PackScope` enum, kept out of this leaf crate's dependencies.
    pub scope: String,
    /// Where the portable source is curated (e.g. a GitHub URL).
    pub source_url: Option<String>,
    /// blake3 hex of `blob` — per-akh fetch verifies this before install.
    pub checksum: String,
    /// The human who confirmed admission (ADR 081 decision 8: admission to
    /// the shared registry always requires a human confirm).
    pub confirmed_by: String,
    /// The compiled native-triple blob (bincode; `CompiledSchemaPack::to_blob`).
    pub blob: Vec<u8>,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
