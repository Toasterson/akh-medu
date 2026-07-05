//! Migration: workspace metadata + membership (flat `workspaces_*` tables, builder-only).

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum Workspace {
    #[sea_orm(iden = "workspaces_workspace")]
    Table,
    Id,
    Name,
    Dimension,
    Encoding,
    Language,
    CreatedAt,
}

#[derive(DeriveIden)]
enum WorkspaceSeedPack {
    #[sea_orm(iden = "workspaces_workspace_seed_pack")]
    Table,
    WorkspaceId,
    Position,
    PackName,
}

#[derive(DeriveIden)]
enum Membership {
    #[sea_orm(iden = "workspaces_membership")]
    Table,
    WorkspaceId,
    AkhId,
    JoinedAt,
    Role,
}

#[derive(DeriveIden)]
enum ActiveMicrotheories {
    #[sea_orm(iden = "workspaces_active_microtheories")]
    Table,
    WorkspaceId,
    MicrotheoryId,
    LoadedBy,
    LoadedAt,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Workspace::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Workspace::Id).uuid().not_null().primary_key())
                    .col(ColumnDef::new(Workspace::Name).text().not_null().unique_key())
                    .col(ColumnDef::new(Workspace::Dimension).integer().not_null())
                    .col(ColumnDef::new(Workspace::Encoding).text().not_null())
                    .col(ColumnDef::new(Workspace::Language).text().not_null())
                    .col(ColumnDef::new(Workspace::CreatedAt).timestamp_with_time_zone().not_null())
                    .to_owned(),
            )
            .await?;

        // workspace_seed_pack — ordered seed packs per workspace,
        // PK (workspace_id, position). Normalises the former `seed_packs` JSON list.
        manager
            .create_table(
                Table::create()
                    .table(WorkspaceSeedPack::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(WorkspaceSeedPack::WorkspaceId).uuid().not_null())
                    .col(ColumnDef::new(WorkspaceSeedPack::Position).integer().not_null())
                    .col(ColumnDef::new(WorkspaceSeedPack::PackName).text().not_null())
                    .primary_key(
                        Index::create()
                            .col(WorkspaceSeedPack::WorkspaceId)
                            .col(WorkspaceSeedPack::Position),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_workspaces_seed_pack_by_workspace")
                    .table(WorkspaceSeedPack::Table)
                    .col(WorkspaceSeedPack::WorkspaceId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Membership::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Membership::WorkspaceId).uuid().not_null())
                    .col(ColumnDef::new(Membership::AkhId).uuid().not_null())
                    .col(ColumnDef::new(Membership::JoinedAt).timestamp_with_time_zone().not_null())
                    .col(ColumnDef::new(Membership::Role).text().not_null())
                    .primary_key(Index::create().col(Membership::WorkspaceId).col(Membership::AkhId))
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(ActiveMicrotheories::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(ActiveMicrotheories::WorkspaceId).uuid().not_null())
                    .col(ColumnDef::new(ActiveMicrotheories::MicrotheoryId).text().not_null())
                    .col(ColumnDef::new(ActiveMicrotheories::LoadedBy).uuid().not_null())
                    .col(ColumnDef::new(ActiveMicrotheories::LoadedAt).timestamp_with_time_zone().not_null())
                    .primary_key(
                        Index::create()
                            .col(ActiveMicrotheories::WorkspaceId)
                            .col(ActiveMicrotheories::MicrotheoryId),
                    )
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        for table in [
            ActiveMicrotheories::Table.into_table_ref(),
            Membership::Table.into_table_ref(),
            WorkspaceSeedPack::Table.into_table_ref(),
            Workspace::Table.into_table_ref(),
        ] {
            manager.drop_table(Table::drop().table(table).if_exists().to_owned()).await?;
        }
        Ok(())
    }
}
