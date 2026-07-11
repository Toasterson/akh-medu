//! Postgres-backed [`SchemaRegistry`] (ADR 081 decision 9 — the registry half
//! of Slice 2, `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`).
//!
//! [`PgSchemaRegistry`] is a thin **synchronous** adapter over
//! `akhomed-store`'s async `schema_registry` API. [`SchemaRegistry`] is a
//! sync trait (the compiler/install side it serves is sync throughout), so
//! this bridges the async DB hop the same way `agent::harvest::harvest_daily_log`
//! does: `tokio::runtime::Handle::current().block_on(...)`, wrapped in
//! `tokio::task::block_in_place` when a runtime is already entered on this
//! thread (ADR 027 — never hold the agent mutex across an `.await`; the
//! store calls here run via `block_on` instead of this being `async fn`).
//! Every caller of this adapter runs inside a tokio runtime (daemon ACP
//! handlers, CLI commands built on the `tokio` runtime), matching every
//! other sync/async bridge in this codebase.
//!
//! [`converge_workspace`] is the separate **async** half: the wake-time
//! lazy-convergence hook (ADR 081 decision 9's "lazy convergence" — no
//! `pg_notify`, safe because schema-pack types are immutable+additive).
//! `snapshot::wake` is already `async fn`, so it calls this directly instead
//! of going through the sync trait.

use akhomed_store::DatabaseConnection;
use akhomed_store::uuid::Uuid;

use super::compile::{RegistryRow, SchemaRegistry, check_admission};
use super::install::install_pack;
use super::source::PackScope;
use super::{SchemaError, SchemaResult};

/// Bridges [`SchemaRegistry`]'s synchronous trait methods to `akhomed-store`'s
/// async API. Borrows the pool rather than owning it — callers construct one
/// of these per operation from the daemon's shared `Arc<DatabaseConnection>`.
pub struct PgSchemaRegistry<'a> {
    db: &'a DatabaseConnection,
}

impl<'a> PgSchemaRegistry<'a> {
    pub fn new(db: &'a DatabaseConnection) -> Self {
        Self { db }
    }
}

/// Run `fut` to completion from a synchronous context on the current tokio
/// runtime. Mirrors `agent::harvest::harvest_daily_log`'s and
/// `PgTurnBodyReader::turn_body`'s discipline (see `src/bin/akhomed.rs`):
/// `block_in_place` when a runtime is already entered on this thread (the
/// common case), a plain `block_on` otherwise.
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    let handle = tokio::runtime::Handle::current();
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(|| handle.block_on(fut))
    } else {
        handle.block_on(fut)
    }
}

fn store_err(e: akhomed_store::StoreError) -> SchemaError {
    // A `Conflict` from the store (duplicate admission racing past our
    // pre-check) maps to the same domain error `check_admission` raises for
    // the common case, so callers see one error shape either way.
    match &e {
        akhomed_store::StoreError::Conflict { .. } => {
            // The message already names the pack; a generic name/version
            // isn't recoverable here, so surface it as a registry failure
            // with the store's own descriptive text rather than guessing.
            SchemaError::Registry {
                message: e.to_string(),
            }
        }
        _ => SchemaError::Registry {
            message: e.to_string(),
        },
    }
}

/// `PackScope` → the text form the store column holds. Public so ACP
/// handlers (`src/bin/akhomed.rs`) building a [`RegistryRow`] from wire
/// params reuse the exact same mapping.
pub fn scope_to_string(scope: PackScope) -> String {
    scope.to_string()
}

/// Inverse of [`scope_to_string`]. Unrecognized text defaults to
/// [`PackScope::Shared`] (the registry's default scope) rather than erroring
/// — a scope column is advisory metadata, not a correctness-critical field.
pub fn scope_from_string(s: &str) -> PackScope {
    match s {
        "own" => PackScope::Own,
        _ => PackScope::Shared,
    }
}

fn row_to_store(row: &RegistryRow) -> akhomed_store::SchemaPackRow {
    akhomed_store::SchemaPackRow {
        name: row.name.clone(),
        version: row.version.clone(),
        scope: scope_to_string(row.scope),
        source_url: row.source_url.clone(),
        checksum: row.checksum.clone(),
        confirmed_by: row.confirmed_by.clone(),
        blob: row.blob.clone(),
    }
}

fn model_to_row(m: akhomed_store::entity::schema_registry::Model) -> RegistryRow {
    RegistryRow {
        name: m.name,
        version: m.version,
        scope: scope_from_string(&m.scope),
        source_url: m.source_url,
        checksum: m.checksum,
        confirmed_by: m.confirmed_by,
        blob: m.blob,
    }
}

impl<'a> SchemaRegistry for PgSchemaRegistry<'a> {
    fn admit(&self, row: RegistryRow) -> SchemaResult<()> {
        // `check_admission` re-derives (name, version) duplication + the
        // cross-pack type-collision guard against the live registry state —
        // the same contract the in-memory test fake enforces
        // (`compile::tests::InMemoryRegistry`). The store's own unique index
        // is the durable backstop against a race between two concurrent
        // admits (surfaced via `store_err` as `SchemaError::Registry`).
        check_admission(self, &row)?;
        block_on(akhomed_store::admit_pack(self.db, row_to_store(&row)))
            .map(|_id| ())
            .map_err(store_err)
    }

    fn fetch(&self, name: &str, version: &str) -> SchemaResult<Option<RegistryRow>> {
        let row = block_on(akhomed_store::get_pack(self.db, name, version)).map_err(store_err)?;
        Ok(row.map(model_to_row))
    }

    fn list(&self) -> SchemaResult<Vec<(String, String)>> {
        let rows = block_on(akhomed_store::list_packs(self.db)).map_err(store_err)?;
        Ok(rows.into_iter().map(|m| (m.name, m.version)).collect())
    }
}

/// The most recently admitted row for `name` (not part of the abstract
/// [`SchemaRegistry`] trait — `akh schema install <name>` without an
/// explicit `@version` uses this directly).
pub fn latest(db: &DatabaseConnection, name: &str) -> SchemaResult<Option<RegistryRow>> {
    block_on(akhomed_store::latest_pack(db, name))
        .map(|opt| opt.map(model_to_row))
        .map_err(store_err)
}

/// Record that `pack`'s microtheory is active in `workspace_id`, attributed
/// to `loaded_by` (the Akh performing the install).
pub fn record_activation(
    db: &DatabaseConnection,
    workspace_id: Uuid,
    microtheory_id: &str,
    loaded_by: Uuid,
) -> SchemaResult<()> {
    block_on(akhomed_store::record_pack_activation(
        db,
        workspace_id,
        microtheory_id,
        loaded_by,
        akhomed_store::chrono::Utc::now(),
    ))
    .map_err(store_err)
}

/// Prefix every schema-pack microtheory id carries
/// ([`super::compile::CompiledSchemaPack::microtheory_label`]) — used to
/// pick schema-pack activations out of the generic
/// `workspaces_active_microtheories` table.
const SCHEMA_MICROTHEORY_PREFIX: &str = "mt:schema/";

/// Split a schema-pack microtheory id (`"mt:schema/<name>@<version>"`) back
/// into `(name, version)`. `None` if `id` isn't a schema-pack microtheory id,
/// or is malformed (missing the `@version` suffix).
fn parse_schema_microtheory_id(id: &str) -> Option<(&str, &str)> {
    id.strip_prefix(SCHEMA_MICROTHEORY_PREFIX)?.rsplit_once('@')
}

/// Wake-time lazy convergence (ADR 081 decision 9): install every schema
/// pack recorded as active for `workspace_id` into `engine`.
///
/// Called from [`crate::snapshot::wake`], which is already `async fn` — this
/// runs on the same task, no sync bridge needed. Reinstalling an
/// already-installed pack@version is an idempotent no-op
/// ([`install_pack`]), and schema-pack types are immutable+additive, so
/// converging on every wake is always safe — this is exactly what "lazy
/// convergence" (deferred `pg_notify`) means for this subsystem: an akh
/// catches its workspace's active-pack list up to date on its own schedule
/// (wake), not via a live push.
///
/// Best-effort: a pack that fails to list/fetch/decode/install is logged and
/// skipped so one bad activation record cannot block wake for the rest of
/// the workspace's packs (mirrors `agent::harvest::harvest_daily_log`'s
/// skip-and-continue discipline). Returns the number of packs installed
/// (including no-op reinstalls of already-installed packs).
pub async fn converge_workspace(
    engine: &crate::engine::Engine,
    db: &DatabaseConnection,
    workspace_id: Uuid,
) -> usize {
    let microtheories = match akhomed_store::list_active_microtheories(db, workspace_id).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                %workspace_id,
                error = %e,
                "schema-pack convergence: could not list active microtheories"
            );
            return 0;
        }
    };

    let mut converged = 0usize;
    for mt_id in microtheories {
        let Some((name, version)) = parse_schema_microtheory_id(&mt_id) else {
            continue; // not a schema-pack activation (or malformed) — skip
        };
        match akhomed_store::get_pack(db, name, version).await {
            Ok(Some(model)) => {
                let row = model_to_row(model);
                match row.decode().and_then(|pack| install_pack(engine, &pack)) {
                    Ok(_) => converged += 1,
                    Err(e) => tracing::warn!(
                        pack = %mt_id,
                        error = %e,
                        "schema-pack convergence: install failed, skipping"
                    ),
                }
            }
            Ok(None) => tracing::warn!(
                pack = %mt_id,
                "schema-pack convergence: activated pack not found in registry, skipping"
            ),
            Err(e) => tracing::warn!(
                pack = %mt_id,
                error = %e,
                "schema-pack convergence: fetch failed, skipping"
            ),
        }
    }
    converged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_schema_microtheory_id() {
        assert_eq!(
            parse_schema_microtheory_id("mt:schema/schema-org-core@1.0.0"),
            Some(("schema-org-core", "1.0.0"))
        );
    }

    #[test]
    fn ignores_non_schema_microtheory_ids() {
        assert_eq!(parse_schema_microtheory_id("mt:other/thing@1.0.0"), None);
    }

    #[test]
    fn ignores_malformed_schema_microtheory_ids() {
        // Missing the `@version` suffix.
        assert_eq!(parse_schema_microtheory_id("mt:schema/no-version"), None);
    }
}
