//! Add the durable Akh residency fencing generation.
//!
//! Every successful session-lock acquisition increments this value while
//! holding the same transaction advisory lock used by snapshot publication.
//! A snapshot captured by an earlier owner is therefore rejected even when it
//! has the same WAL watermark as the current owner's image.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[derive(DeriveIden)]
enum Akh {
    #[sea_orm(iden = "akhomed_akh")]
    Table,
    ResidencyEpoch,
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Akh::Table)
                    .add_column_if_not_exists(
                        ColumnDef::new(Akh::ResidencyEpoch)
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
                    .table(Akh::Table)
                    .drop_column(Akh::ResidencyEpoch)
                    .to_owned(),
            )
            .await
    }
}
