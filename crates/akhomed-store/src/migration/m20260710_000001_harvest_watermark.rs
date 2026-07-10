//! Migration: add `harvest_watermark` to `akhomed_akh_state` (ADR 091 / the
//! transcript→episodic capture write-side slice).
//!
//! Tracks how far the sleep-cycle "harvest" phase has walked the `kind='turn'`
//! rows in `akhomed_mutation_log` (the daily-log projection) and minted their
//! `episode:turn:*` index. Distinct from the existing `watermark` column, which
//! is the *wake-replay* boundary over all mutation kinds — `harvest_watermark`
//! only ever advances via the dedicated setter
//! ([`crate::snapshots::set_harvest_watermark`]), never via the general
//! `akh_state` snapshot upsert (see the comment on
//! [`crate::snapshots::upsert_akh_state`] for the clobber-guard rationale).

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum AkhState {
    #[sea_orm(iden = "akhomed_akh_state")]
    Table,
    HarvestWatermark,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(AkhState::Table)
                    .add_column(
                        ColumnDef::new(AkhState::HarvestWatermark)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(AkhState::Table)
                    .drop_column(AkhState::HarvestWatermark)
                    .to_owned(),
            )
            .await
    }
}
