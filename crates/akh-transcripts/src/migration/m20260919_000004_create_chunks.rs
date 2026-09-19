//! Migration: `th_chunks` — embedding units.
//!
//! Every column except `embedding` is expressible with the SeaORM schema
//! builder; `embedding vector(384)` and its HNSW index are not (SeaORM has no
//! native pgvector column type), so those two steps use raw SQL — the same
//! split `crates/seshat/src/migration/m20260409_000001_create_seshat.rs` and
//! `m20260425_000001_switch_to_hnsw.rs` use for `pa_embeddings`. Unlike
//! Seshat's history, this migration goes straight to HNSW: Seshat's own
//! post-mortem in `m20260425_000001_switch_to_hnsw.rs` documents why IVFFlat's
//! "needs training data" failure mode makes it the wrong default to start
//! from, so there is no reason to repeat that path here.
//!
//! This migration assumes `vector` was already enabled by
//! `m20260919_000001_create_raw_records` (which fails the whole migration run
//! if pgvector is unavailable), so it does not re-check availability.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum Chunks {
    #[sea_orm(iden = "th_chunks")]
    Table,
    Id,
    DerivationVersion,
    Source,
    SessionId,
    StreamId,
    SeqFrom,
    SeqTo,
    Text,
    EmbeddedAt,
    CreatedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Chunks::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Chunks::Id)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(
                        ColumnDef::new(Chunks::DerivationVersion)
                            .integer()
                            .not_null(),
                    )
                    .col(ColumnDef::new(Chunks::Source).text().not_null())
                    .col(ColumnDef::new(Chunks::SessionId).text().not_null())
                    .col(ColumnDef::new(Chunks::StreamId).text().not_null())
                    .col(ColumnDef::new(Chunks::SeqFrom).big_integer().not_null())
                    .col(ColumnDef::new(Chunks::SeqTo).big_integer().not_null())
                    .col(ColumnDef::new(Chunks::Text).text().not_null())
                    .col(ColumnDef::new(Chunks::EmbeddedAt).timestamp_with_time_zone())
                    .col(
                        ColumnDef::new(Chunks::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null()
                            .default(Expr::cust("now()")),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_th_chunks_identity")
                    .table(Chunks::Table)
                    .col(Chunks::DerivationVersion)
                    .col(Chunks::Source)
                    .col(Chunks::SessionId)
                    .col(Chunks::StreamId)
                    .col(Chunks::SeqFrom)
                    .unique()
                    .to_owned(),
            )
            .await?;

        let db = manager.get_connection();

        // pgvector column: not expressible via the SeaORM schema builder.
        db.execute_unprepared(
            "ALTER TABLE th_chunks ADD COLUMN IF NOT EXISTS embedding vector(384)",
        )
        .await?;

        // HNSW, cosine ops — matches akh-embed's EMBEDDING_DIM (384) and the
        // read surface's cosine query. `IF NOT EXISTS` for the same
        // concurrent-migrator reasons as the rest of the guarded DDL here.
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_th_chunks_embedding \
             ON th_chunks USING hnsw (embedding vector_cosine_ops)",
        )
        .await?;

        // Partial index: find unembedded chunks efficiently — the derivation
        // worker's anti-join for the embedding stage, mirroring Seshat's
        // `idx_pa_chunks_unembedded`.
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_th_chunks_unembedded \
             ON th_chunks (derivation_version) WHERE embedded_at IS NULL",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Chunks::Table).to_owned())
            .await
    }
}
