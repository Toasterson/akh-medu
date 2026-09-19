//! Migration: `th_events` — the normalized cross-tool layer.
//!
//! Deliberately a **plain table** (see `entity::event` for why): its unique
//! key is `(derivation_version, source, session_id, stream_id, seq)`, which
//! the derivation worker's anti-join discovery depends on being index-backed
//! — unindexed it "degrades badly at the 66k-record backfill, which is the
//! very first run"
//! ([derivation-worker ticket](../../../docs/wayfinder/transcript-harvest/tickets/derivation-worker.md)).
//! That unique key is also this table's primary key, so it is index-backed by
//! construction.
//!
//! Two supporting indexes match the store-schema ticket's read-shape list:
//! `(kind, observed_at)` for "every `tool_call` last week", and
//! `(repo, observed_at)` for per-project session listing.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum Events {
    #[sea_orm(iden = "th_events")]
    Table,
    DerivationVersion,
    Source,
    SessionId,
    StreamId,
    Seq,
    ObservedAt,
    Kind,
    Subkind,
    Role,
    Text,
    ToolName,
    ToolCallId,
    AgentId,
    ParentAgentId,
    Render,
    Degraded,
    DegradedReason,
    SecretFlags,
    Repo,
    Workspace,
    Branch,
    DerivedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Events::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Events::DerivationVersion)
                            .integer()
                            .not_null(),
                    )
                    .col(ColumnDef::new(Events::Source).text().not_null())
                    .col(ColumnDef::new(Events::SessionId).text().not_null())
                    .col(ColumnDef::new(Events::StreamId).text().not_null())
                    .col(ColumnDef::new(Events::Seq).big_integer().not_null())
                    .col(
                        ColumnDef::new(Events::ObservedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(ColumnDef::new(Events::Kind).text().not_null())
                    .col(ColumnDef::new(Events::Subkind).text())
                    .col(ColumnDef::new(Events::Role).text())
                    .col(ColumnDef::new(Events::Text).text())
                    .col(ColumnDef::new(Events::ToolName).text())
                    .col(ColumnDef::new(Events::ToolCallId).text())
                    .col(ColumnDef::new(Events::AgentId).text())
                    .col(ColumnDef::new(Events::ParentAgentId).text())
                    .col(ColumnDef::new(Events::Render).text())
                    .col(
                        ColumnDef::new(Events::Degraded)
                            .boolean()
                            .not_null()
                            .default(false),
                    )
                    .col(ColumnDef::new(Events::DegradedReason).text())
                    .col(ColumnDef::new(Events::SecretFlags).json_binary())
                    .col(ColumnDef::new(Events::Repo).text())
                    .col(ColumnDef::new(Events::Workspace).text())
                    .col(ColumnDef::new(Events::Branch).text())
                    .col(
                        ColumnDef::new(Events::DerivedAt)
                            .timestamp_with_time_zone()
                            .not_null()
                            .default(Expr::cust("now()")),
                    )
                    .primary_key(
                        Index::create()
                            .col(Events::DerivationVersion)
                            .col(Events::Source)
                            .col(Events::SessionId)
                            .col(Events::StreamId)
                            .col(Events::Seq),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_th_events_kind_observed_at")
                    .table(Events::Table)
                    .col(Events::Kind)
                    .col(Events::ObservedAt)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_th_events_repo_observed_at")
                    .table(Events::Table)
                    .col(Events::Repo)
                    .col(Events::ObservedAt)
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Events::Table).to_owned())
            .await
    }
}
