//! Typed API for the schema-pack registry (ADR 081 decision 9) and
//! workspace-scoped pack activation.
//!
//! All SeaORM access lives here, inside the leaf crate that owns the schema —
//! the main crate's `schema::registry_pg` adapter (and the daemon's ACP
//! handlers) call these functions with plain data, so neither takes a direct
//! ORM dependency.
//!
//! **Admission** ([`admit_pack`]) writes one immutable row per `(name,
//! version)`; re-admitting the same pair is rejected with
//! [`StoreError::Conflict`] rather than silently overwriting — the registry
//! is the durable record of a human-confirmed artifact (ADR 081 decision 8).
//!
//! **Activation** reuses the existing `workspaces_active_microtheories` table
//! ([`crate::entity::active_microtheory`]) instead of a bespoke table: a
//! schema pack's compartment is, structurally, just a microtheory, and that
//! table already tracks "which microtheory is active in which workspace,
//! loaded by which Akh". [`record_pack_activation`] and
//! [`list_active_microtheories`] are thin wrappers over it. The wake-time
//! lazy-convergence hook ([`crate::schema` in the main crate, `snapshot::wake`])
//! reads [`list_active_microtheories`], filters for the `mt:schema/` prefix,
//! and reinstalls each pack — safe and idempotent because schema-pack types
//! are immutable and additive (ADR 081 decision 5).

use sea_orm::sea_query::OnConflict;
use sea_orm::{
    ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder,
};
use uuid::Uuid;

use crate::entity::{active_microtheory, schema_registry};
use crate::error::{StoreError, StoreResult};

/// A schema-pack row ready to admit. Store-facing plain data — mirrors the
/// main crate's `schema::RegistryRow` without this leaf crate depending on
/// it (`scope` travels as the already-`Display`-formatted text, e.g.
/// `"shared"` / `"own"`).
#[derive(Debug, Clone)]
pub struct SchemaPackRow {
    pub name: String,
    pub version: String,
    pub scope: String,
    pub source_url: Option<String>,
    pub checksum: String,
    pub confirmed_by: String,
    pub blob: Vec<u8>,
}

/// Admit a confirmed schema pack to the registry.
///
/// ADR 081 decision 8: the mandatory human-confirm step already happened by
/// the time this is called (the CLI/ACP boundary enforces `--yes`) — this
/// just persists the row. Rejects a duplicate `(name, version)` with
/// [`StoreError::Conflict`]; the unique index on the table is the durable
/// backstop against a race between two concurrent admits of the same pair.
/// Returns the new row's `id`.
pub async fn admit_pack(db: &DatabaseConnection, row: SchemaPackRow) -> StoreResult<i64> {
    if get_pack(db, &row.name, &row.version).await?.is_some() {
        return Err(StoreError::Conflict {
            message: format!(
                "schema pack \"{}@{}\" is already admitted to the registry",
                row.name, row.version
            ),
        });
    }
    let model = schema_registry::ActiveModel {
        id: sea_orm::ActiveValue::NotSet,
        name: Set(row.name),
        version: Set(row.version),
        scope: Set(row.scope),
        source_url: Set(row.source_url),
        checksum: Set(row.checksum),
        confirmed_by: Set(row.confirmed_by),
        blob: Set(row.blob),
        created_at: Set(chrono::Utc::now()),
    };
    let result = schema_registry::Entity::insert(model).exec(db).await;
    match result {
        Ok(inserted) => Ok(inserted.last_insert_id),
        // A raced concurrent admit of the same (name, version) hits the
        // unique index after our pre-check above passed; surface it as the
        // same duplicate-admission conflict rather than a raw DB error.
        Err(sea_orm::DbErr::Exec(ref e))
            if e.to_string().to_lowercase().contains("unique") =>
        {
            Err(StoreError::Conflict {
                message: "schema pack admission raced a concurrent admit of the same \
                          (name, version); the other admit won"
                    .to_string(),
            })
        }
        Err(e) => Err(e.into()),
    }
}

/// Fetch an admitted pack row by exact `(name, version)`.
pub async fn get_pack(
    db: &DatabaseConnection,
    name: &str,
    version: &str,
) -> StoreResult<Option<schema_registry::Model>> {
    Ok(schema_registry::Entity::find()
        .filter(schema_registry::Column::Name.eq(name))
        .filter(schema_registry::Column::Version.eq(version))
        .one(db)
        .await?)
}

/// List all admitted registry rows, most recently admitted first.
pub async fn list_packs(db: &DatabaseConnection) -> StoreResult<Vec<schema_registry::Model>> {
    Ok(schema_registry::Entity::find()
        .order_by_desc(schema_registry::Column::CreatedAt)
        .all(db)
        .await?)
}

/// The most recently admitted row for `name` (registry rows are never
/// mutated, so "latest" means "most recently inserted", not a semver
/// compare across `version` strings).
pub async fn latest_pack(
    db: &DatabaseConnection,
    name: &str,
) -> StoreResult<Option<schema_registry::Model>> {
    Ok(schema_registry::Entity::find()
        .filter(schema_registry::Column::Name.eq(name))
        .order_by_desc(schema_registry::Column::CreatedAt)
        .one(db)
        .await?)
}

/// Record that `microtheory_id` (a schema pack's `microtheory_label()`, e.g.
/// `"mt:schema/schema-org-core@1.0.0"`) is active in `workspace_id`.
///
/// Reuses the generic `workspaces_active_microtheories` table — a schema
/// pack's compartment is just a microtheory. Idempotent: re-activating the
/// same `(workspace_id, microtheory_id)` updates `loaded_by`/`loaded_at`
/// rather than erroring.
pub async fn record_pack_activation(
    db: &DatabaseConnection,
    workspace_id: Uuid,
    microtheory_id: &str,
    loaded_by: Uuid,
    loaded_at: chrono::DateTime<chrono::Utc>,
) -> StoreResult<()> {
    let model = active_microtheory::ActiveModel {
        workspace_id: Set(workspace_id),
        microtheory_id: Set(microtheory_id.to_owned()),
        loaded_by: Set(loaded_by),
        loaded_at: Set(loaded_at),
    };
    let result = active_microtheory::Entity::insert(model)
        .on_conflict(
            OnConflict::columns([
                active_microtheory::Column::WorkspaceId,
                active_microtheory::Column::MicrotheoryId,
            ])
            .update_columns([
                active_microtheory::Column::LoadedBy,
                active_microtheory::Column::LoadedAt,
            ])
            .to_owned(),
        )
        .exec(db)
        .await;
    match result {
        Ok(_) => Ok(()),
        // An update-on-conflict that changes nothing can surface as
        // RecordNotInserted; that is still the success case (mirrors
        // `memberships::join_workspace`'s idempotent-rejoin handling).
        Err(sea_orm::DbErr::RecordNotInserted) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// List the microtheory ids currently active in a workspace — the source of
/// truth the wake-time lazy-convergence hook reconciles against. Includes
/// every activated microtheory, not only schema packs; callers filter for
/// the `mt:schema/` prefix.
pub async fn list_active_microtheories(
    db: &DatabaseConnection,
    workspace_id: Uuid,
) -> StoreResult<Vec<String>> {
    Ok(active_microtheory::Entity::find()
        .filter(active_microtheory::Column::WorkspaceId.eq(workspace_id))
        .all(db)
        .await?
        .into_iter()
        .map(|m| m.microtheory_id)
        .collect())
}
