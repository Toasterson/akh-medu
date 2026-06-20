//! SeaORM entities for the `akhomed.*` and `workspaces.*` schemas.
//!
//! Each entity declares its schema via `#[sea_orm(schema_name = "...")]`. The DDL
//! that creates these tables lives in [`crate::migration`]. The schema is kept
//! backend-portable (no native arrays, no DB-side defaults) so a future SQLite /
//! libSQL-Turso backend is a drop-in.

// akhomed.* — identity, KG, WAL, runtime state
pub mod akh;
pub mod akh_state;
pub mod episode;
pub mod fact;
pub mod mutation_log;
pub mod provenance;
pub mod provenance_input;
pub mod symbol;

// akhomed.* — cognitive state tables
pub mod agenda;
pub mod competence_calibration;
pub mod competence_category;
pub mod competence_tool;
pub mod curiosity_config;
pub mod drive;
pub mod effort_case;
pub mod failure_case;
pub mod goal;
pub mod goal_dependency;
pub mod goal_hierarchy;
pub mod learned_method;
pub mod learned_method_step;
pub mod sleep_metric;

// workspaces.* — workspace metadata + membership
pub mod active_microtheory;
pub mod membership;
pub mod workspace;
pub mod workspace_seed_pack;
