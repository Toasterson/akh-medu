//! Initial migration: create pgvector extension and all Seshat tables.

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260409_000001_create_seshat"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Enable pgvector extension.
        let db = manager.get_connection();
        db.execute_unprepared("CREATE EXTENSION IF NOT EXISTS vector")
            .await?;

        // --- pa_documents ---
        manager
            .create_table(
                Table::create()
                    .table(PaDocuments::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(PaDocuments::Id).uuid().not_null().primary_key().default(Expr::cust("gen_random_uuid()")))
                    .col(ColumnDef::new(PaDocuments::Title).text().not_null())
                    .col(ColumnDef::new(PaDocuments::Slug).text().not_null().unique_key())
                    .col(ColumnDef::new(PaDocuments::SourcePath).text())
                    .col(ColumnDef::new(PaDocuments::Format).text().not_null())
                    .col(ColumnDef::new(PaDocuments::Tags).json_binary().not_null().default(Expr::cust("'[]'::jsonb")))
                    .col(ColumnDef::new(PaDocuments::Metadata).json_binary().not_null().default(Expr::cust("'{}'::jsonb")))
                    .col(ColumnDef::new(PaDocuments::ChunkCount).integer().not_null().default(0))
                    .col(ColumnDef::new(PaDocuments::IngestedAt).timestamp_with_time_zone().not_null().default(Expr::cust("now()")))
                    .col(ColumnDef::new(PaDocuments::IngestedBy).text())
                    .to_owned(),
            )
            .await?;

        // --- pa_chunks ---
        manager
            .create_table(
                Table::create()
                    .table(PaChunks::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(PaChunks::Id).uuid().not_null().primary_key().default(Expr::cust("gen_random_uuid()")))
                    .col(ColumnDef::new(PaChunks::DocumentId).uuid().not_null())
                    .col(ColumnDef::new(PaChunks::ChunkIndex).integer().not_null())
                    .col(ColumnDef::new(PaChunks::Text).text().not_null())
                    .col(ColumnDef::new(PaChunks::WordCount).integer().not_null())
                    .col(ColumnDef::new(PaChunks::Chapter).integer())
                    .col(ColumnDef::new(PaChunks::Section).integer())
                    .col(ColumnDef::new(PaChunks::Heading).text())
                    .col(ColumnDef::new(PaChunks::EmbeddedAt).timestamp_with_time_zone())
                    .foreign_key(
                        ForeignKey::create()
                            .from(PaChunks::Table, PaChunks::DocumentId)
                            .to(PaDocuments::Table, PaDocuments::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Index: lookup chunks by document
        manager
            .create_index(
                Index::create()
                    .name("idx_pa_chunks_doc")
                    .table(PaChunks::Table)
                    .col(PaChunks::DocumentId)
                    .to_owned(),
            )
            .await?;

        // Partial index: find unembedded chunks efficiently
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_pa_chunks_unembedded \
             ON pa_chunks(document_id) WHERE embedded_at IS NULL",
        )
        .await?;

        // --- pa_embeddings ---
        // Use raw SQL because SeaORM's ColumnDef doesn't support pgvector types.
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS pa_embeddings (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                chunk_id    UUID NOT NULL REFERENCES pa_chunks(id) ON DELETE CASCADE,
                model_name  TEXT NOT NULL,
                embedding   vector(384) NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;

        // Index: lookup embeddings by chunk
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_pa_embeddings_chunk \
             ON pa_embeddings(chunk_id)",
        )
        .await?;

        // IVFFlat index for cosine similarity search.
        // Note: this index requires the table to contain some rows before
        // it can be built with `lists > 0`. We create it here so it's ready
        // once data arrives. For empty tables, PostgreSQL falls back to seqscan.
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS idx_pa_embeddings_vector \
             ON pa_embeddings USING ivfflat (embedding vector_cosine_ops) \
             WITH (lists = 100)",
        )
        .await?;

        // --- pa_microtheory_cache ---
        manager
            .create_table(
                Table::create()
                    .table(PaMicrotheoryCache::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(PaMicrotheoryCache::Id).uuid().not_null().primary_key().default(Expr::cust("gen_random_uuid()")))
                    .col(ColumnDef::new(PaMicrotheoryCache::QueryHash).text().not_null())
                    .col(ColumnDef::new(PaMicrotheoryCache::QueryText).text().not_null())
                    .col(ColumnDef::new(PaMicrotheoryCache::CompartmentId).text().not_null())
                    .col(ColumnDef::new(PaMicrotheoryCache::TriplesJson).json_binary().not_null())
                    .col(ColumnDef::new(PaMicrotheoryCache::ChunkIdsUsed).json_binary().not_null())
                    .col(ColumnDef::new(PaMicrotheoryCache::ModelUsed).text().not_null())
                    .col(ColumnDef::new(PaMicrotheoryCache::Confidence).float().not_null())
                    .col(ColumnDef::new(PaMicrotheoryCache::CreatedAt).timestamp_with_time_zone().not_null().default(Expr::cust("now()")))
                    .col(ColumnDef::new(PaMicrotheoryCache::AccessCount).integer().not_null().default(0))
                    .col(ColumnDef::new(PaMicrotheoryCache::LastAccessedAt).timestamp_with_time_zone().not_null().default(Expr::cust("now()")))
                    .to_owned(),
            )
            .await?;

        // Unique index for cache lookup by query hash.
        manager
            .create_index(
                Index::create()
                    .name("idx_pa_cache_query")
                    .table(PaMicrotheoryCache::Table)
                    .col(PaMicrotheoryCache::QueryHash)
                    .unique()
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // Drop in reverse dependency order.
        manager
            .drop_table(Table::drop().table(PaMicrotheoryCache::Table).to_owned())
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS pa_embeddings CASCADE")
            .await?;
        manager
            .drop_table(Table::drop().table(PaChunks::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(PaDocuments::Table).to_owned())
            .await?;

        Ok(())
    }
}

// --- Iden enums for SeaORM table/column references ---

#[derive(DeriveIden)]
enum PaDocuments {
    Table,
    Id,
    Title,
    Slug,
    SourcePath,
    Format,
    Tags,
    Metadata,
    ChunkCount,
    IngestedAt,
    IngestedBy,
}

#[derive(DeriveIden)]
enum PaChunks {
    Table,
    Id,
    DocumentId,
    ChunkIndex,
    Text,
    WordCount,
    Chapter,
    Section,
    Heading,
    EmbeddedAt,
}

#[derive(DeriveIden)]
enum PaMicrotheoryCache {
    Table,
    Id,
    QueryHash,
    QueryText,
    CompartmentId,
    TriplesJson,
    ChunkIdsUsed,
    ModelUsed,
    Confidence,
    CreatedAt,
    AccessCount,
    LastAccessedAt,
}
