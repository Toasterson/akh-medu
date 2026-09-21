//! Migration: `th_raw_records` — the verbatim system of record, and pgvector.
//!
//! Two independent extension concerns live in this first migration because
//! both gate what the rest of the schema can promise
//! ([store-schema ticket](../../../docs/wayfinder/transcript-harvest/tickets/store-schema.md),
//! decision 6 — "degradation is asymmetric and deliberate"):
//!
//! - **`vector` (pgvector) is a hard requirement.** `th_chunks` (a later
//!   migration) carries a `vector(384)` column and the MCP read surface
//!   advertises semantic search, so a missing extension must fail loudly here
//!   rather than let a later migration fail with an opaque "type vector does
//!   not exist". [`crate::pool::run_migrations_locked`] additionally checks
//!   this *before* invoking the migrator, returning a rich
//!   [`crate::error::TranscriptError::PgvectorMissing`] diagnostic — this
//!   migration-level check is the safety net for callers that invoke
//!   [`sea_orm_migration::MigratorTrait::up`] directly.
//! - **`timescaledb` is optional.** If present, `th_raw_records` becomes a
//!   hypertable partitioned on `observed_at` with a compression policy
//!   (segmented by `(source, session_id)`, ordered by `seq`, after 30 days).
//!   If absent, warn and stay a plain table with the **identical primary
//!   key** — everything still works, only compression is unavailable. Pattern
//!   copied from
//!   `akhomed-store`'s `m20260620_000002_mutation_log_hypertable.rs`.
//!
//! The primary key is `(observed_at, source, session_id, stream_id, seq)`:
//! TimescaleDB requires the partition column (`observed_at`) in every unique
//! index, so it cannot stand outside the natural
//! `(source, session_id, stream_id, seq)` key from the
//! [session-identity ticket](../../../docs/wayfinder/transcript-harvest/tickets/session-identity.md).
//! Because this table is created fresh with that composite key, there is no
//! retrofit step (unlike the `akhomed_mutation_log` precedent, which had to
//! drop and recreate its primary key).

use sea_orm::{ConnectionTrait, DbBackend, Statement};
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum RawRecords {
    #[sea_orm(iden = "th_raw_records")]
    Table,
    ObservedAt,
    Source,
    SessionId,
    StreamId,
    Seq,
    StreamKind,
    Machine,
    Raw,
    Payload,
    RecordType,
    SchemaMarker,
    OriginPath,
    ObservedAtSynthetic,
    HarvestedAt,
}

/// Name of the wrapper image bundling both TimescaleDB and pgvector, quoted in
/// every diagnostic that mentions where to get the missing extension.
pub(crate) const WRAPPER_IMAGE: &str = "code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1";

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let backend = manager.get_database_backend();
        let db = manager.get_connection();

        if backend == DbBackend::Postgres {
            ensure_pgvector_or_fail(db).await?;
        }

        manager
            .create_table(
                Table::create()
                    .table(RawRecords::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(RawRecords::ObservedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(ColumnDef::new(RawRecords::Source).text().not_null())
                    .col(ColumnDef::new(RawRecords::SessionId).text().not_null())
                    .col(ColumnDef::new(RawRecords::StreamId).text().not_null())
                    .col(ColumnDef::new(RawRecords::Seq).big_integer().not_null())
                    .col(ColumnDef::new(RawRecords::StreamKind).text().not_null())
                    .col(ColumnDef::new(RawRecords::Machine).text().not_null())
                    .col(ColumnDef::new(RawRecords::Raw).text().not_null())
                    .col(ColumnDef::new(RawRecords::Payload).json_binary())
                    .col(ColumnDef::new(RawRecords::RecordType).text())
                    .col(ColumnDef::new(RawRecords::SchemaMarker).text())
                    .col(ColumnDef::new(RawRecords::OriginPath).text().not_null())
                    .col(
                        ColumnDef::new(RawRecords::ObservedAtSynthetic)
                            .boolean()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(RawRecords::HarvestedAt)
                            .timestamp_with_time_zone()
                            .not_null()
                            .default(Expr::cust("now()")),
                    )
                    .primary_key(
                        Index::create()
                            .col(RawRecords::ObservedAt)
                            .col(RawRecords::Source)
                            .col(RawRecords::SessionId)
                            .col(RawRecords::StreamId)
                            .col(RawRecords::Seq),
                    )
                    .to_owned(),
            )
            .await?;

        if backend != DbBackend::Postgres {
            return Ok(()); // portable backends: stays a plain table
        }

        let timescaledb_available = extension_available(db, "timescaledb").await?;
        if !timescaledb_available {
            tracing::warn!(
                "timescaledb extension not available; th_raw_records stays a plain table \
                 (identical primary key, no compression). Use the {WRAPPER_IMAGE} image to \
                 get it."
            );
            return Ok(());
        }

        db.execute_unprepared("CREATE EXTENSION IF NOT EXISTS timescaledb")
            .await?;
        db.execute_unprepared(
            "SELECT create_hypertable('th_raw_records', 'observed_at', if_not_exists => true)",
        )
        .await?;
        db.execute_unprepared(
            "ALTER TABLE th_raw_records SET ( \
                timescaledb.compress, \
                timescaledb.compress_segmentby = 'source, session_id', \
                timescaledb.compress_orderby = 'seq' \
             )",
        )
        .await?;
        db.execute_unprepared(
            "SELECT add_compression_policy('th_raw_records', INTERVAL '30 days')",
        )
        .await?;
        tracing::info!(
            "th_raw_records converted to a TimescaleDB hypertable with a 30-day compression policy"
        );
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(RawRecords::Table).to_owned())
            .await
    }
}

/// `SELECT 1 FROM pg_available_extensions WHERE name = $1` — true if the
/// extension can be created (it may already be created, or merely installable).
pub(crate) async fn extension_available<C: ConnectionTrait>(
    db: &C,
    name: &str,
) -> Result<bool, DbErr> {
    Ok(db
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT 1 FROM pg_available_extensions WHERE name = $1",
            [name.into()],
        ))
        .await?
        .is_some())
}

/// `CREATE EXTENSION IF NOT EXISTS vector`, or a loud [`DbErr`] naming the
/// wrapper image if it is not even installable on this server.
///
/// This is the migration-level safety net; the primary, richly-diagnosed
/// failure path is [`crate::pool::run_migrations_locked`]'s pre-flight check,
/// which returns [`crate::error::TranscriptError::PgvectorMissing`] before the
/// migrator ever runs.
async fn ensure_pgvector_or_fail<C: ConnectionTrait>(db: &C) -> Result<(), DbErr> {
    if !extension_available(db, "vector").await? {
        return Err(DbErr::Custom(format!(
            "the `vector` (pgvector) extension is not available on this Postgres server; \
             the transcript store requires it for semantic search over harvested prose. \
             Use an image that bundles it, e.g. {WRAPPER_IMAGE}."
        )));
    }
    db.execute_unprepared("CREATE EXTENSION IF NOT EXISTS vector")
        .await?;
    Ok(())
}
