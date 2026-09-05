//! Migration: `akhomed_schema_registry` — the admitted schema-pack registry
//! (ADR 081 decision 9, the Postgres half of Slice 2 —
//! `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`).
//!
//! One row per admitted `(name, version)` pack: the compiled native-triple
//! byte blob plus its provenance (curated source, blake3 checksum, confirming
//! human). `IF NOT EXISTS` on both the table and the unique index matches the
//! repo's guarded-DDL convention (see `m20260710_000001_harvest_watermark.rs`)
//! so a raced second `up` under `run_migrations_locked` no-ops instead of
//! dying on a duplicate-object error.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum SchemaRegistry {
    #[sea_orm(iden = "akhomed_schema_registry")]
    Table,
    Id,
    Name,
    Version,
    Scope,
    SourceUrl,
    Checksum,
    ConfirmedBy,
    Blob,
    CreatedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(SchemaRegistry::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(SchemaRegistry::Id)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(SchemaRegistry::Name).text().not_null())
                    .col(ColumnDef::new(SchemaRegistry::Version).text().not_null())
                    .col(ColumnDef::new(SchemaRegistry::Scope).text().not_null())
                    .col(ColumnDef::new(SchemaRegistry::SourceUrl).text())
                    .col(ColumnDef::new(SchemaRegistry::Checksum).text().not_null())
                    .col(
                        ColumnDef::new(SchemaRegistry::ConfirmedBy)
                            .text()
                            .not_null(),
                    )
                    .col(ColumnDef::new(SchemaRegistry::Blob).binary().not_null())
                    .col(
                        ColumnDef::new(SchemaRegistry::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;

        // A (name, version) pair is admitted at most once — the registry's
        // duplicate-admission guard (`akhomed_store::schema_registry::admit_pack`)
        // checks this before inserting, but the unique index is the durable
        // backstop against a race between two concurrent admits.
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .unique()
                    .name("idx_akhomed_schema_registry_name_version")
                    .table(SchemaRegistry::Table)
                    .col(SchemaRegistry::Name)
                    .col(SchemaRegistry::Version)
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(
                Table::drop()
                    .table(SchemaRegistry::Table)
                    .if_exists()
                    .to_owned(),
            )
            .await
    }
}
