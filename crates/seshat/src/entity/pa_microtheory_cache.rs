//! Entity: `pa_microtheory_cache` — cached LLM-synthesized microtheories.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "pa_microtheory_cache")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// SHA-256 of the normalized query text (cache key).
    pub query_hash: String,

    /// Original query text.
    #[sea_orm(column_type = "Text")]
    pub query_text: String,

    /// Compartment ID that was created for this microtheory.
    pub compartment_id: String,

    /// Synthesized triples as JSON array of `{subject, predicate, object, confidence}`.
    #[sea_orm(column_type = "JsonBinary")]
    pub triples_json: Json,

    /// UUIDs of the source chunks used for synthesis (JSONB array).
    #[sea_orm(column_type = "JsonBinary")]
    pub chunk_ids_used: Json,

    /// Which LLM model was used for synthesis.
    pub model_used: String,

    /// Overall confidence of the synthesized microtheory.
    pub confidence: f32,

    /// When this cache entry was created.
    pub created_at: DateTimeUtc,

    /// Number of times this cache entry has been accessed.
    pub access_count: i32,

    /// Last time this cache entry was accessed.
    pub last_accessed_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}

/// A single triple in the synthesized microtheory.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct LabelTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}
