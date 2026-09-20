//! Migration: `th_sessions` — derived per-session metadata.
//!
//! One row per `(source, session_id)`, rolled up by the derivation worker from
//! that session's `th_events` rows for the read surface's `list_sessions`.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum Sessions {
    #[sea_orm(iden = "th_sessions")]
    Table,
    Source,
    SessionId,
    Title,
    Repo,
    Workspace,
    Branch,
    FirstSeen,
    LastSeen,
    StreamCount,
    RecordCount,
    DerivationVersion,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Sessions::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Sessions::Source).text().not_null())
                    .col(ColumnDef::new(Sessions::SessionId).text().not_null())
                    .col(ColumnDef::new(Sessions::Title).text())
                    .col(ColumnDef::new(Sessions::Repo).text())
                    .col(ColumnDef::new(Sessions::Workspace).text())
                    .col(ColumnDef::new(Sessions::Branch).text())
                    .col(
                        ColumnDef::new(Sessions::FirstSeen)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(Sessions::LastSeen)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(Sessions::StreamCount)
                            .integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Sessions::RecordCount)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Sessions::DerivationVersion)
                            .integer()
                            .not_null(),
                    )
                    .primary_key(
                        Index::create()
                            .col(Sessions::Source)
                            .col(Sessions::SessionId),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_th_sessions_repo_last_seen")
                    .table(Sessions::Table)
                    .col(Sessions::Repo)
                    .col(Sessions::LastSeen)
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Sessions::Table).to_owned())
            .await
    }
}
