//! Migration: portable snapshots of each Akh's local redb provenance ledger.
//!
//! Local provenance IDs begin at one independently for every Akh, so this
//! table deliberately uses `(akh_id, provenance_id)` as its primary key.  It
//! complements the legacy globally-keyed provenance table and is the source
//! used by snapshot/wake to rebuild exact fact links without ID reuse.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum ProvenanceSnapshot {
    #[sea_orm(iden = "akhomed_provenance_snapshot")]
    Table,
    AkhId,
    ProvenanceId,
    WorkspaceId,
    DerivedSymbol,
    Sources,
    Derivation,
    Confidence,
    Depth,
    CreatedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(ProvenanceSnapshot::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(ProvenanceSnapshot::AkhId).uuid().not_null())
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::ProvenanceId)
                            .big_integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::WorkspaceId)
                            .uuid()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::DerivedSymbol)
                            .big_integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::Sources)
                            .json()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::Derivation)
                            .json()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::Confidence)
                            .float()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::Depth)
                            .big_integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceSnapshot::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .primary_key(
                        Index::create()
                            .col(ProvenanceSnapshot::AkhId)
                            .col(ProvenanceSnapshot::ProvenanceId),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_provenance_snapshot_workspace")
                    .table(ProvenanceSnapshot::Table)
                    .col(ProvenanceSnapshot::WorkspaceId)
                    .col(ProvenanceSnapshot::AkhId)
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(
                Table::drop()
                    .table(ProvenanceSnapshot::Table)
                    .if_exists()
                    .to_owned(),
            )
            .await
    }
}
