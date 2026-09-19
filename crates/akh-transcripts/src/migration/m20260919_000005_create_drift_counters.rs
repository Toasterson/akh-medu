//! Migration: `th_drift_counters` — unknown/degraded record counts.
//!
//! Keyed `(source, record_type, schema_marker)`, per the normal-form ticket's
//! dispatch decision.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum DriftCounters {
    #[sea_orm(iden = "th_drift_counters")]
    Table,
    Source,
    RecordType,
    SchemaMarker,
    UnknownCount,
    DegradedCount,
    LastSeen,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(DriftCounters::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(DriftCounters::Source).text().not_null())
                    .col(ColumnDef::new(DriftCounters::RecordType).text().not_null())
                    .col(
                        ColumnDef::new(DriftCounters::SchemaMarker)
                            .text()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(DriftCounters::UnknownCount)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(DriftCounters::DegradedCount)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(DriftCounters::LastSeen)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .primary_key(
                        Index::create()
                            .col(DriftCounters::Source)
                            .col(DriftCounters::RecordType)
                            .col(DriftCounters::SchemaMarker),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(DriftCounters::Table).to_owned())
            .await
    }
}
