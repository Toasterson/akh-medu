//! Migration: core akhomed tables (identity, registry, KG, WAL, runtime state).
//!
//! Built with the SeaORM schema builder (no raw SQL, no Postgres schemas, no
//! native arrays, no DB-side defaults) so the schema is portable to SQLite /
//! libSQL-Turso. Referential integrity is maintained app-side (no FKs) to keep
//! migration ordering simple and SQLite-friendly. Table names are flat and
//! prefixed (`akhomed_*`) since SQLite has no schema namespaces.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum Akh {
    #[sea_orm(iden = "akhomed_akh")]
    Table,
    Id,
    Name,
    HomeHost,
    AwakenedAt,
    LastSeen,
}

#[derive(DeriveIden)]
enum Symbols {
    #[sea_orm(iden = "akhomed_symbols")]
    Table,
    AkhId,
    Id,
    Kind,
    GlyphCodepoint,
    Label,
}

#[derive(DeriveIden)]
enum Facts {
    #[sea_orm(iden = "akhomed_facts")]
    Table,
    AkhId,
    WorkspaceId,
    Subject,
    Predicate,
    Object,
    Confidence,
    AssertedAt,
    RetractedAt,
    ProvenanceId,
    Compartment,
}

#[derive(DeriveIden)]
enum MutationLog {
    #[sea_orm(iden = "akhomed_mutation_log")]
    Table,
    Seq,
    AkhId,
    WorkspaceId,
    Kind,
    Payload,
    AppliedAt,
}

#[derive(DeriveIden)]
enum AkhState {
    #[sea_orm(iden = "akhomed_akh_state")]
    Table,
    AkhId,
    Watermark,
    AllocatorNext,
    CycleCount,
    SessionStartCycle,
    SleepPhase,
    LastSleepCycle,
    Psyche,
    Extra,
    UpdatedAt,
}

#[allow(clippy::enum_variant_names)]
#[derive(DeriveIden)]
enum Provenance {
    #[sea_orm(iden = "akhomed_provenance")]
    Table,
    ProvenanceId,
    AkhId,
    WorkspaceId,
    Subject,
    Derivation,
    Confidence,
    CreatedAt,
}

#[derive(DeriveIden)]
enum ProvenanceInput {
    #[sea_orm(iden = "akhomed_provenance_input")]
    Table,
    ProvenanceId,
    Position,
    InputSymbolId,
}

#[derive(DeriveIden)]
enum Episodes {
    #[sea_orm(iden = "akhomed_episodes")]
    Table,
    EpisodeId,
    AkhId,
    WorkspaceId,
    StartedAt,
    EndedAt,
    Payload,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Akh::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Akh::Id).uuid().not_null().primary_key())
                    .col(ColumnDef::new(Akh::Name).text().not_null())
                    .col(ColumnDef::new(Akh::HomeHost).text())
                    .col(
                        ColumnDef::new(Akh::AwakenedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(ColumnDef::new(Akh::LastSeen).timestamp_with_time_zone())
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Symbols::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Symbols::AkhId).uuid().not_null())
                    .col(ColumnDef::new(Symbols::Id).big_integer().not_null())
                    .col(ColumnDef::new(Symbols::Kind).text().not_null())
                    .col(ColumnDef::new(Symbols::GlyphCodepoint).integer())
                    .col(ColumnDef::new(Symbols::Label).text().not_null())
                    .primary_key(Index::create().col(Symbols::AkhId).col(Symbols::Id))
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_symbols_label")
                    .table(Symbols::Table)
                    .col(Symbols::AkhId)
                    .col(Symbols::Label)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Facts::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Facts::WorkspaceId).uuid().not_null())
                    .col(ColumnDef::new(Facts::Subject).big_integer().not_null())
                    .col(ColumnDef::new(Facts::Predicate).big_integer().not_null())
                    .col(ColumnDef::new(Facts::Object).big_integer().not_null())
                    .col(ColumnDef::new(Facts::AkhId).uuid().not_null())
                    .col(ColumnDef::new(Facts::Confidence).float().not_null())
                    .col(
                        ColumnDef::new(Facts::AssertedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(ColumnDef::new(Facts::RetractedAt).timestamp_with_time_zone())
                    .col(ColumnDef::new(Facts::ProvenanceId).big_integer())
                    .col(ColumnDef::new(Facts::Compartment).text())
                    // Content-addressed primary key (a triple is unique per workspace).
                    .primary_key(
                        Index::create()
                            .col(Facts::WorkspaceId)
                            .col(Facts::Subject)
                            .col(Facts::Predicate)
                            .col(Facts::Object),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_facts_by_workspace")
                    .table(Facts::Table)
                    .col(Facts::WorkspaceId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(MutationLog::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(MutationLog::Seq)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(MutationLog::AkhId).uuid().not_null())
                    .col(ColumnDef::new(MutationLog::WorkspaceId).uuid().not_null())
                    .col(ColumnDef::new(MutationLog::Kind).text().not_null())
                    .col(ColumnDef::new(MutationLog::Payload).json().not_null())
                    .col(
                        ColumnDef::new(MutationLog::AppliedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_mutation_log_by_akh")
                    .table(MutationLog::Table)
                    .col(MutationLog::AkhId)
                    .col(MutationLog::Seq)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(AkhState::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(AkhState::AkhId)
                            .uuid()
                            .not_null()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(AkhState::Watermark).big_integer().not_null())
                    .col(
                        ColumnDef::new(AkhState::AllocatorNext)
                            .big_integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(AkhState::CycleCount)
                            .big_integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(AkhState::SessionStartCycle)
                            .big_integer()
                            .not_null(),
                    )
                    .col(ColumnDef::new(AkhState::SleepPhase).text().not_null())
                    .col(
                        ColumnDef::new(AkhState::LastSleepCycle)
                            .big_integer()
                            .not_null(),
                    )
                    .col(ColumnDef::new(AkhState::Psyche).json())
                    .col(ColumnDef::new(AkhState::Extra).json())
                    .col(
                        ColumnDef::new(AkhState::UpdatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Provenance::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Provenance::ProvenanceId)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Provenance::AkhId).uuid().not_null())
                    .col(ColumnDef::new(Provenance::WorkspaceId).uuid().not_null())
                    .col(ColumnDef::new(Provenance::Subject).big_integer().not_null())
                    .col(ColumnDef::new(Provenance::Derivation).json().not_null())
                    .col(ColumnDef::new(Provenance::Confidence).float().not_null())
                    .col(
                        ColumnDef::new(Provenance::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;

        // provenance_input — ordered input symbols per provenance record,
        // PK (provenance_id, position). Normalises the former `inputs` JSON list.
        manager
            .create_table(
                Table::create()
                    .table(ProvenanceInput::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(ProvenanceInput::ProvenanceId)
                            .big_integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceInput::Position)
                            .integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(ProvenanceInput::InputSymbolId)
                            .big_integer()
                            .not_null(),
                    )
                    .primary_key(
                        Index::create()
                            .col(ProvenanceInput::ProvenanceId)
                            .col(ProvenanceInput::Position),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_provenance_input_by_provenance")
                    .table(ProvenanceInput::Table)
                    .col(ProvenanceInput::ProvenanceId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Episodes::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Episodes::EpisodeId)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Episodes::AkhId).uuid().not_null())
                    .col(ColumnDef::new(Episodes::WorkspaceId).uuid().not_null())
                    .col(
                        ColumnDef::new(Episodes::StartedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(ColumnDef::new(Episodes::EndedAt).timestamp_with_time_zone())
                    .col(ColumnDef::new(Episodes::Payload).json().not_null())
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        for table in [
            Episodes::Table.into_table_ref(),
            ProvenanceInput::Table.into_table_ref(),
            Provenance::Table.into_table_ref(),
            AkhState::Table.into_table_ref(),
            MutationLog::Table.into_table_ref(),
            Facts::Table.into_table_ref(),
            Symbols::Table.into_table_ref(),
            Akh::Table.into_table_ref(),
        ] {
            manager
                .drop_table(Table::drop().table(table).if_exists().to_owned())
                .await?;
        }
        Ok(())
    }
}
