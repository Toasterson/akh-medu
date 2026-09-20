//! Migration: `th_events` gains `subseq` and `disposition`, `kind` becomes
//! nullable.
//!
//! Two gaps in the increment-1 schema, both surfaced by the
//! [derivation-worker ticket](../../../docs/wayfinder/transcript-harvest/tickets/derivation-worker.md):
//!
//! 1. **One raw record can yield several events** — a Claude `user` record
//!    with three `tool_result` blocks in its content array — but the
//!    increment-1 primary key `(derivation_version, source, session_id,
//!    stream_id, seq)` has no room for more than one row per raw record.
//!    `subseq` (the block index inside a record; `0` for records that yield
//!    at most one event) joins the key:
//!    `(derivation_version, source, session_id, stream_id, seq, subseq)`.
//! 2. **Records that yield no event at all must still satisfy the
//!    anti-join**, or a `Noise`/`Undetermined` record — or one whose
//!    normalizer panics outright — re-pends on every tick forever, which is
//!    the exact poison-record failure mode §2 of the derivation-worker
//!    ticket designs against. `disposition` (`event | noise | undetermined |
//!    failed`) records which [`crate::normalize::Outcome`] arm produced the
//!    row; `kind` is only ever populated for `disposition = 'event'`, so it
//!    must become nullable.
//!
//! Safe as a plain `ALTER TABLE`: `th_events` is empty in every environment
//! this migration has shipped to (the derivation worker that would populate
//! it is this same increment), so there is no backfill step and no rows to
//! reconcile against the new, wider primary key.
//!
//! The `disposition` check constraint and the primary-key swap are
//! Postgres-specific raw SQL — sea-query has no portable "drop and recreate
//! a composite primary key" operation — guarded the same way
//! `m20260620_000002_mutation_log_hypertable.rs` guards its own PK swap: a
//! no-op on non-Postgres backends. The two new columns themselves are added
//! through the portable schema builder.

use sea_orm::{ConnectionTrait, DbBackend};
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum Events {
    #[sea_orm(iden = "th_events")]
    Table,
    Subseq,
    Disposition,
}

const CHECK_CONSTRAINT: &str = "th_events_disposition_check";
const OLD_PKEY: &str = "th_events_pkey";

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Events::Table)
                    .add_column(
                        ColumnDef::new(Events::Subseq)
                            .integer()
                            .not_null()
                            .default(0),
                    )
                    .add_column(
                        ColumnDef::new(Events::Disposition)
                            .text()
                            .not_null()
                            .default("event"),
                    )
                    .to_owned(),
            )
            .await?;

        if manager.get_database_backend() != DbBackend::Postgres {
            // Portable backends keep the increment-1 primary key and a
            // `NOT NULL` `kind` column; only Postgres carries this store in
            // practice (harvest-pg), so this path exists solely so a future
            // portable-backend test of this migrator does not fail outright.
            return Ok(());
        }
        let db = manager.get_connection();

        db.execute_unprepared(&format!(
            "ALTER TABLE th_events ADD CONSTRAINT {CHECK_CONSTRAINT} \
             CHECK (disposition IN ('event', 'noise', 'undetermined', 'failed'))"
        ))
        .await?;

        // `kind` is populated only for disposition = 'event'; every other
        // disposition writes NULL rather than an invented pseudo-kind, per
        // the normal-form ticket's closed-ten-kinds stance.
        db.execute_unprepared("ALTER TABLE th_events ALTER COLUMN kind DROP NOT NULL")
            .await?;

        db.execute_unprepared(&format!("ALTER TABLE th_events DROP CONSTRAINT {OLD_PKEY}"))
            .await?;
        db.execute_unprepared(
            "ALTER TABLE th_events ADD PRIMARY KEY \
             (derivation_version, source, session_id, stream_id, seq, subseq)",
        )
        .await?;

        tracing::info!(
            "th_events: added subseq + disposition, widened the primary key, and made kind nullable"
        );
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        if manager.get_database_backend() == DbBackend::Postgres {
            let db = manager.get_connection();
            db.execute_unprepared("ALTER TABLE th_events DROP CONSTRAINT IF EXISTS th_events_pkey")
                .await?;
            db.execute_unprepared(
                "ALTER TABLE th_events ADD PRIMARY KEY (derivation_version, source, session_id, stream_id, seq)",
            )
            .await?;
            // Best-effort: fails if any row written under this migration has
            // a NULL kind (any non-'event' disposition). Rolling back a
            // populated store is not a supported path; this is dev tooling.
            db.execute_unprepared("ALTER TABLE th_events ALTER COLUMN kind SET NOT NULL")
                .await?;
            db.execute_unprepared(&format!(
                "ALTER TABLE th_events DROP CONSTRAINT IF EXISTS {CHECK_CONSTRAINT}"
            ))
            .await?;
        }

        manager
            .alter_table(
                Table::alter()
                    .table(Events::Table)
                    .drop_column(Events::Disposition)
                    .drop_column(Events::Subseq)
                    .to_owned(),
            )
            .await
    }
}
