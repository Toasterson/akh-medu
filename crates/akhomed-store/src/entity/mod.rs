//! SeaORM entities for the `akhomed.*` and `workspaces.*` Postgres schemas.
//!
//! Each entity declares its schema via `#[sea_orm(schema_name = "...")]` so
//! generated SQL is schema-qualified (`"akhomed"."facts"`, etc.). The DDL that
//! creates these tables lives in [`crate::migration`].

// akhomed.* — Akh-owned state
pub mod akh;
pub mod episode;
pub mod fact;
pub mod mutation_log;
pub mod provenance;
pub mod psyche;
pub mod snapshot;

// workspaces.* — workspace metadata + membership
pub mod active_microtheory;
pub mod membership;
pub mod workspace;
