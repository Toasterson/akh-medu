//! Migration: create the `workspaces` schema and its tables.
//!
//! Depends on `akhomed.akh` existing (membership + active_microtheories
//! reference it), which the prior migration guarantees.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        db.execute_unprepared("CREATE SCHEMA IF NOT EXISTS workspaces")
            .await?;

        // --- workspaces.workspace — workspace metadata ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS workspaces.workspace (
                id            UUID PRIMARY KEY,
                name          TEXT NOT NULL UNIQUE,
                dimension     INTEGER NOT NULL DEFAULT 10000,
                encoding      TEXT NOT NULL DEFAULT 'bipolar',
                language      TEXT NOT NULL DEFAULT 'auto',
                seed_packs    TEXT[] NOT NULL DEFAULT '{}',
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )",
        )
        .await?;

        // --- workspaces.membership — which Akhs are in which workspaces ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS workspaces.membership (
                workspace_id  UUID NOT NULL REFERENCES workspaces.workspace(id),
                akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
                joined_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                role          TEXT NOT NULL DEFAULT 'member',
                PRIMARY KEY (workspace_id, akh_id)
            )",
        )
        .await?;

        // --- workspaces.active_microtheories — activated microtheories ---
        db.execute_unprepared(
            "CREATE TABLE IF NOT EXISTS workspaces.active_microtheories (
                workspace_id   UUID NOT NULL REFERENCES workspaces.workspace(id),
                microtheory_id TEXT NOT NULL,
                loaded_by      UUID NOT NULL REFERENCES akhomed.akh(id),
                loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (workspace_id, microtheory_id)
            )",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared("DROP SCHEMA IF EXISTS workspaces CASCADE")
            .await?;
        Ok(())
    }
}
