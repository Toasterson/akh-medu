//! Entity: `heka_drawer_embeddings` — 384-dim pgvector embeddings for chunks.
//!
//! The `embedding` column is `vector(384)` in PostgreSQL (pgvector extension).
//! SeaORM does not natively support the pgvector `vector` type, so this entity
//! maps the column as a `String` for metadata operations. Actual vector queries
//! (cosine similarity search) should use parameterized SQL (to be added in a
//! later phase), following the Seshat pattern in `crates/seshat/src/query.rs`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "heka_drawer_embeddings")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Foreign key to `heka_drawer_chunks.id`.
    pub chunk_id: Uuid,

    /// Name of the embedding model used (e.g., "all-MiniLM-L6-v2").
    pub model_name: String,

    /// The embedding vector.
    ///
    /// Stored as pgvector `vector(384)` in PostgreSQL.
    /// Mapped as a String here for SeaORM compatibility; the actual vector
    /// is inserted and queried via parameterized SQL (Phase 42b+).
    /// pgvector's text representation is `[0.1,0.2,...,0.384]`.
    #[sea_orm(column_type = "custom(\"vector(384)\")")]
    pub embedding: String,

    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::heka_drawer_chunks::Entity",
        from = "Column::ChunkId",
        to = "super::heka_drawer_chunks::Column::Id"
    )]
    Chunk,
}

impl Related<super::heka_drawer_chunks::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Chunk.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}

/// Format a `Vec<f32>` as a pgvector text representation: `[0.1,0.2,...]`.
pub fn vec_to_pgvector(v: &[f32]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(v.len() * 8 + 2);
    s.push('[');
    for (i, val) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{val}");
    }
    s.push(']');
    s
}

/// Parse a pgvector text representation `[0.1,0.2,...]` into a `Vec<f32>`.
pub fn pgvector_to_vec(s: &str) -> Vec<f32> {
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(',')
        .filter_map(|v| v.trim().parse::<f32>().ok())
        .collect()
}