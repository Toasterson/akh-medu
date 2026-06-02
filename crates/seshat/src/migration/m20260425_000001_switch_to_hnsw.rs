//! Switch the pgvector similarity-search index from IVFFlat to HNSW.
//!
//! Why: the original migration created
//!
//!   CREATE INDEX idx_pa_embeddings_vector
//!     ON pa_embeddings USING ivfflat (embedding vector_cosine_ops)
//!     WITH (lists = 100);
//!
//! ...up-front on an empty table. IVFFlat clusters rows into N lists
//! at build time; with zero rows the clusters are empty, and with
//! a handful of rows each cluster has a sliver of data. The default
//! `ivfflat.probes = 1` then visits one cluster per query and
//! returns nearly nothing — we caught this as `seshat_search`
//! returning `[]` on a 3-chunk archive despite `embedded_chunks = 3`.
//!
//! HNSW (pgvector ≥ 0.5) doesn't have the "needs training data"
//! failure mode and scales well for the sub-100k-row regime we're
//! in. Slower inserts, larger index on disk — both acceptable for
//! a read-heavy RAG corpus where we ingest in manageable batches.
//!
//! The DROP is `IF EXISTS` so it's idempotent even if the old index
//! was already yanked manually (which it was in the live cluster
//! on 2026-04-23 via psql). HNSW create is also IF NOT EXISTS.

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260425_000001_switch_to_hnsw"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // Drop the old IVFFlat index.
        db.execute_unprepared("DROP INDEX IF EXISTS idx_pa_embeddings_vector")
            .await?;

        // Build HNSW in its place. The `m` and `ef_construction` knobs
        // use pgvector defaults (16 and 64 respectively) — good enough
        // for the expected workload; tuneable later without another
        // migration via `CREATE INDEX ... WITH (m = …)`.
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_pa_embeddings_vector \
             ON pa_embeddings USING hnsw (embedding vector_cosine_ops)",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        db.execute_unprepared("DROP INDEX IF EXISTS idx_pa_embeddings_vector")
            .await?;

        // Restore the (known-broken) IVFFlat so `down` is actually a
        // mirror of `up`. The choice to revert all the way is mostly
        // symbolic: the previous migration is what established this
        // shape, and we want `down`/`up` cycles to return to a state
        // that matches what `MigratorTrait::status` expects.
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_pa_embeddings_vector \
             ON pa_embeddings USING ivfflat (embedding vector_cosine_ops) \
             WITH (lists = 100)",
        )
        .await?;

        Ok(())
    }
}
