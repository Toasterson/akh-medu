//! Migration: add `librarian_watermark` to `akhomed_akh_state` (ADR 081
//! slice 4, `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md` work-item 4
//! — the daily Librarian projection).
//!
//! Tracks how far the daily-Librarian pass has walked the same `kind='turn'`
//! WAL tail the sleep-cycle harvest phase reads (ADR 091), plus any note
//! material, extracting typed triples against the in-force schema. Deliberately
//! a **separate** watermark from `harvest_watermark`: the two passes progress
//! independently (harvest is cheap/always-on bookkeeping that mints episode
//! indices; the Librarian pass is an LLM-cost extraction run that is off by
//! default and runs on its own cadence — ADR 081 decision 10), so one must
//! never gate or advance the other. Only ever advances via the dedicated
//! setter ([`crate::snapshots::set_librarian_watermark`]), never via the
//! general `akh_state` snapshot upsert — same clobber-guard rationale as
//! `harvest_watermark` (see the comment on
//! [`crate::snapshots::upsert_akh_state`]).
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum AkhState {
    #[sea_orm(iden = "akhomed_akh_state")]
    Table,
    LibrarianWatermark,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // `IF NOT EXISTS` matches the repo's guarded-DDL convention (see
        // `m20260710_000001_harvest_watermark`): the migration suite runs
        // under concurrent callers (advisory-locked in prod, but
        // plain-parallel in tests), so a raced second `up` must no-op instead
        // of dying with 42701 duplicate-column.
        manager
            .alter_table(
                Table::alter()
                    .table(AkhState::Table)
                    .add_column_if_not_exists(
                        ColumnDef::new(AkhState::LibrarianWatermark)
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
                    .drop_column(AkhState::LibrarianWatermark)
                    .to_owned(),
            )
            .await
    }
}
