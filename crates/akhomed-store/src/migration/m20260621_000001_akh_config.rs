//! Migration: `akhomed_akh_config` — the akh-level + remaining workspace config
//! that has no home in `workspaces_workspace` (Phase 38f).
//!
//! `workspaces_workspace` holds the engine geometry (dimension/encoding/language)
//! and `workspaces_workspace_seed_pack` holds the seed-pack list. This table holds
//! the rest of `AkhConfig`: memory/symbol/triple budgets, shared partitions, the
//! pinned model + reasoning effort, the authoritative system prompt, is-main,
//! chat backend, path mode, related workspaces, the Ollama endpoint, autonomy and
//! delegation. Self-contained values are stored as JSON text (backend-portable;
//! no native arrays/json so a future SQLite/libSQL backend is a drop-in).
//!
//! Keyed by `workspace_id` (the deterministic v5 UUID `workspace_id_for(name)`),
//! consistent with `workspaces_workspace.id` and `workspaces_workspace_seed_pack`.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum AkhConfig {
    #[sea_orm(iden = "akhomed_akh_config")]
    Table,
    WorkspaceId,
    MaxMemoryMb,
    MaxSymbols,
    MaxTriples,
    SharedPartitions,
    Model,
    ReasoningEffort,
    SystemPrompt,
    IsMain,
    Backend,
    PathMode,
    RelatedWorkspaces,
    OllamaEndpoint,
    Autonomy,
    Delegation,
    UpdatedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(AkhConfig::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(AkhConfig::WorkspaceId)
                            .uuid()
                            .not_null()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(AkhConfig::MaxMemoryMb).big_integer().not_null())
                    .col(ColumnDef::new(AkhConfig::MaxSymbols).big_integer().not_null())
                    .col(ColumnDef::new(AkhConfig::MaxTriples).big_integer().not_null())
                    .col(ColumnDef::new(AkhConfig::SharedPartitions).text().not_null())
                    .col(ColumnDef::new(AkhConfig::Model).text().null())
                    .col(ColumnDef::new(AkhConfig::ReasoningEffort).text().null())
                    .col(ColumnDef::new(AkhConfig::SystemPrompt).text().null())
                    .col(ColumnDef::new(AkhConfig::IsMain).boolean().not_null())
                    .col(ColumnDef::new(AkhConfig::Backend).text().not_null())
                    .col(ColumnDef::new(AkhConfig::PathMode).text().not_null())
                    .col(ColumnDef::new(AkhConfig::RelatedWorkspaces).text().not_null())
                    .col(ColumnDef::new(AkhConfig::OllamaEndpoint).text().not_null())
                    .col(ColumnDef::new(AkhConfig::Autonomy).text().not_null())
                    .col(ColumnDef::new(AkhConfig::Delegation).text().null())
                    .col(
                        ColumnDef::new(AkhConfig::UpdatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(AkhConfig::Table).if_exists().to_owned())
            .await?;
        Ok(())
    }
}
