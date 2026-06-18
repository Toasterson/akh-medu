//! Migration: create the `akhomed` schema and its tables.
//!
//! Uses raw SQL via `execute_unprepared` rather than the SeaORM schema builder.
//! The schema is rich (cross-schema-friendly UUID FKs, `BIGINT[]` arrays, JSONB,
//! partial-null soft deletes, a multi-column unique constraint) and the plan
//! ships canonical DDL; raw SQL maps onto it 1:1 with no translation risk.
//! sea-orm-migration wraps each migration in a transaction on Postgres, so a
//! mid-migration failure rolls the whole thing back.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        db.execute_unprepared("CREATE SCHEMA IF NOT EXISTS akhomed")
            .await?;

        // --- akhomed.akh — Akh identities (PK is the identity, not the host) ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS akhomed.akh (
                id            UUID PRIMARY KEY,
                name          TEXT NOT NULL,
                home_host     TEXT,
                awakened_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_seen     TIMESTAMPTZ
            )",
        )
        .await?;

        // --- akhomed.facts — every triple ever asserted by any Akh ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS akhomed.facts (
                fact_id       BIGSERIAL PRIMARY KEY,
                akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
                workspace_id  UUID NOT NULL,
                subject       BIGINT NOT NULL,
                predicate     BIGINT NOT NULL,
                object        BIGINT NOT NULL,
                confidence    REAL NOT NULL DEFAULT 1.0,
                asserted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                retracted_at  TIMESTAMPTZ,
                provenance_id BIGINT,
                compartment   TEXT,
                CONSTRAINT facts_unique UNIQUE (workspace_id, subject, predicate, object)
            )",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS facts_by_workspace ON akhomed.facts(workspace_id);
             CREATE INDEX IF NOT EXISTS facts_by_subject   ON akhomed.facts(workspace_id, subject);
             CREATE INDEX IF NOT EXISTS facts_by_predicate ON akhomed.facts(workspace_id, predicate);
             CREATE INDEX IF NOT EXISTS facts_by_object    ON akhomed.facts(workspace_id, object)",
        )
        .await?;

        // --- akhomed.mutation_log — the streaming WAL ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS akhomed.mutation_log (
                seq           BIGSERIAL PRIMARY KEY,
                akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
                workspace_id  UUID NOT NULL,
                kind          TEXT NOT NULL,
                payload       JSONB NOT NULL,
                applied_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS mutation_log_by_akh ON akhomed.mutation_log(akh_id, seq)",
        )
        .await?;

        // --- akhomed.snapshots — periodic full snapshots ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS akhomed.snapshots (
                snapshot_id   BIGSERIAL PRIMARY KEY,
                akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                watermark     BIGINT NOT NULL,
                working_mem   BYTEA NOT NULL,
                episodes      BYTEA NOT NULL,
                state_blob    BYTEA NOT NULL,
                psyche        JSONB
            )",
        )
        .await?;
        db.execute_unprepared(
            "CREATE INDEX IF NOT EXISTS snapshots_by_akh ON akhomed.snapshots(akh_id, created_at DESC)",
        )
        .await?;

        // --- akhomed.psyche — live current-row psyche per Akh ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS akhomed.psyche (
                akh_id            UUID PRIMARY KEY REFERENCES akhomed.akh(id),
                persona           JSONB NOT NULL,
                shadow            JSONB NOT NULL,
                archetypes        JSONB NOT NULL,
                self_integration  JSONB NOT NULL,
                awakened          BOOLEAN NOT NULL DEFAULT false,
                updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;

        // --- akhomed.provenance — provenance ledger ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS akhomed.provenance (
                provenance_id BIGSERIAL PRIMARY KEY,
                akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
                workspace_id  UUID NOT NULL,
                subject       BIGINT NOT NULL,
                derivation    JSONB NOT NULL,
                inputs        BIGINT[] NOT NULL,
                confidence    REAL NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;

        // --- akhomed.episodes — durable episodic memory ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS akhomed.episodes (
                episode_id    BIGSERIAL PRIMARY KEY,
                akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
                workspace_id  UUID NOT NULL,
                started_at    TIMESTAMPTZ NOT NULL,
                ended_at      TIMESTAMPTZ,
                payload       JSONB NOT NULL
            )",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // CASCADE drops every table, index, and constraint in the schema.
        manager
            .get_connection()
            .execute_unprepared("DROP SCHEMA IF EXISTS akhomed CASCADE")
            .await?;
        Ok(())
    }
}
