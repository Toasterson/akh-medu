//! Entity: `th_chunks` — embedding units.
//!
//! Turn-boundary chunks (~1500 chars) over `prompt`/`reply` events only —
//! **not** `thought` (near-empty: Codex `reasoning` is 100% encrypted, 91.8%
//! of Claude `thinking` blocks are empty) and not tool output or renders, per
//! the [embedding-pipeline ticket](../../../docs/wayfinder/transcript-harvest/tickets/embedding-pipeline.md).
//!
//! `embedding` is `vector(384)` in Postgres (pgvector), matching `akh-embed`'s
//! `EMBEDDING_DIM`. SeaORM has no native pgvector column type, so — exactly as
//! `crates/seshat/src/entity/pa_embedding.rs` does — this entity maps the
//! column as `String` for metadata operations; similarity search goes through
//! parameterized raw SQL using the pgvector text form (`[0.1,0.2,...]`).
//! `embedded_at` is `NULL` until the derivation worker embeds the chunk; a
//! failed embedding writes a `degraded` marker via `th_events` rather than
//! leaving the chunk silently pending (same poison-record concern as
//! normalization).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "th_chunks")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,

    pub derivation_version: i32,
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    /// Inclusive `seq` range of the events folded into this chunk — the
    /// pointer a semantic hit widens into `get_lines`.
    pub seq_from: i64,
    pub seq_to: i64,

    pub text: String,
    /// Postgres `vector(384)`, mapped as pgvector's text representation
    /// (`[0.1,0.2,...]`) since SeaORM cannot express the native type.
    #[sea_orm(column_type = "custom(\"vector(384)\")", nullable)]
    pub embedding: Option<String>,
    #[sea_orm(nullable)]
    pub embedded_at: Option<DateTimeUtc>,

    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}

/// Format a `Vec<f32>` as a pgvector text literal: `[0.1,0.2,...]`.
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

/// Parse a pgvector text literal (`[0.1,0.2,...]`) into a `Vec<f32>`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pgvector_roundtrip() {
        let v = vec![0.1_f32, -0.25, 3.5];
        let text = vec_to_pgvector(&v);
        let back = pgvector_to_vec(&text);
        assert_eq!(back.len(), v.len());
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_vector_roundtrip() {
        assert_eq!(vec_to_pgvector(&[]), "[]");
        assert!(pgvector_to_vec("[]").is_empty());
    }
}
