//! Migration: per-Akh cognitive-state tables (drives, agenda, curiosity,
//! goals, CBR cases, learned methods, competence, sleep metrics).
//!
//! Builder-only, flat names, no DB defaults — backend-portable.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

fn col(name: &str) -> ColumnDef {
    ColumnDef::new(Alias::new(name))
}

// Drop order matters only for FK-bearing schemas; we keep app-side integrity
// (no DB FKs), but listing the normalised edge/child tables first keeps the
// teardown intuitive (edges before their parents).
const TABLES: &[&str] = &[
    "akhomed_goal_hierarchy",
    "akhomed_goal_dependencies",
    "akhomed_learned_method_step",
    "akhomed_drives",
    "akhomed_agenda",
    "akhomed_curiosity_config",
    "akhomed_goals",
    "akhomed_failure_cases",
    "akhomed_effort_cases",
    "akhomed_learned_methods",
    "akhomed_competence_tools",
    "akhomed_competence_categories",
    "akhomed_competence_calibration",
    "akhomed_sleep_metrics",
];

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // drives — 4 rows/akh, PK (akh_id, kind)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_drives"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("kind").text().not_null())
                    .col(col("strength").float().not_null())
                    .col(col("last_computed").big_integer().not_null())
                    .col(col("last_triple_count").big_integer().not_null())
                    .col(col("last_symbol_count").big_integer().not_null())
                    .col(col("threshold").float().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("kind")))
                    .to_owned(),
            )
            .await?;

        // agenda — PK (akh_id, project_id)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_agenda"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("project_id").big_integer().not_null())
                    .col(col("priority").small_integer().not_null())
                    .col(col("is_active").boolean().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("project_id")))
                    .to_owned(),
            )
            .await?;

        // curiosity_config — 1 row/akh
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_curiosity_config"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null().primary_key())
                    .col(col("max_targets").integer().not_null())
                    .col(col("proximal_low").float().not_null())
                    .col(col("proximal_high").float().not_null())
                    .col(col("peak_fill_ratio").float().not_null())
                    .col(col("beyond_curiosity_factor").float().not_null())
                    .col(col("known_curiosity_factor").float().not_null())
                    .to_owned(),
            )
            .await?;

        // goals — PK (akh_id, id)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_goals"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("id").big_integer().not_null())
                    .col(col("description").text().not_null())
                    .col(col("status").text().not_null())
                    .col(col("priority").small_integer().not_null())
                    .col(col("success_criteria").text().not_null())
                    .col(col("parent").big_integer())
                    .col(col("created_at").big_integer().not_null())
                    .col(col("cycles_worked").integer().not_null())
                    .col(col("last_progress_cycle").big_integer().not_null())
                    .col(col("source").text())
                    .col(col("detail").json())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("id")))
                    .to_owned(),
            )
            .await?;

        // goal_hierarchy — parent→child edges, PK (akh_id, parent_id, child_id)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_goal_hierarchy"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("parent_id").big_integer().not_null())
                    .col(col("child_id").big_integer().not_null())
                    .col(col("position").integer().not_null())
                    .primary_key(
                        Index::create()
                            .col(Alias::new("akh_id"))
                            .col(Alias::new("parent_id"))
                            .col(Alias::new("child_id")),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_goal_hierarchy_parent")
                    .table(Alias::new("akhomed_goal_hierarchy"))
                    .col(Alias::new("akh_id"))
                    .col(Alias::new("parent_id"))
                    .to_owned(),
            )
            .await?;

        // goal_dependencies — "blocked-by" edges, PK (akh_id, goal_id, blocking_goal_id)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_goal_dependencies"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("goal_id").big_integer().not_null())
                    .col(col("blocking_goal_id").big_integer().not_null())
                    .primary_key(
                        Index::create()
                            .col(Alias::new("akh_id"))
                            .col(Alias::new("goal_id"))
                            .col(Alias::new("blocking_goal_id")),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_goal_dependencies_goal")
                    .table(Alias::new("akhomed_goal_dependencies"))
                    .col(Alias::new("akh_id"))
                    .col(Alias::new("goal_id"))
                    .to_owned(),
            )
            .await?;

        // failure_cases — PK (akh_id, seq)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_failure_cases"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("seq").integer().not_null())
                    .col(col("goal_description").text().not_null())
                    .col(col("tool_name").text().not_null())
                    .col(col("error_message").text().not_null())
                    .col(col("cycle").big_integer().not_null())
                    .col(col("resolved").boolean().not_null())
                    .col(col("resolution_tool").text())
                    .col(col("vector").binary().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("seq")))
                    .to_owned(),
            )
            .await?;

        // effort_cases — PK (akh_id, seq)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_effort_cases"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("seq").integer().not_null())
                    .col(col("description").text().not_null())
                    .col(col("goal_vector").binary().not_null())
                    .col(col("cycles_consumed").integer().not_null())
                    .col(col("succeeded").boolean().not_null())
                    .col(col("tool_usage").json().not_null())
                    .col(col("initial_coverage").integer().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("seq")))
                    .to_owned(),
            )
            .await?;

        // learned_methods — PK (akh_id, id)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_learned_methods"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("id").big_integer().not_null())
                    .col(col("name").text().not_null())
                    .col(col("precondition_pattern").text().not_null())
                    .col(col("precondition_vector").binary().not_null())
                    .col(col("confidence").float().not_null())
                    .col(col("usage_count").integer().not_null())
                    .col(col("success_count").integer().not_null())
                    .col(col("failure_count").integer().not_null())
                    .col(col("learned_at").big_integer().not_null())
                    .col(col("last_used").big_integer().not_null())
                    .col(col("source_goal_description").text().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("id")))
                    .to_owned(),
            )
            .await?;

        // learned_method_step — ordered method steps, PK (akh_id, method_id, step_index)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_learned_method_step"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("method_id").big_integer().not_null())
                    .col(col("step_index").integer().not_null())
                    .col(col("tool").text().not_null())
                    .col(col("arg_pattern").text().not_null())
                    .col(col("expected_outcome").text().not_null())
                    .primary_key(
                        Index::create()
                            .col(Alias::new("akh_id"))
                            .col(Alias::new("method_id"))
                            .col(Alias::new("step_index")),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name("idx_akhomed_learned_method_step_method")
                    .table(Alias::new("akhomed_learned_method_step"))
                    .col(Alias::new("akh_id"))
                    .col(Alias::new("method_id"))
                    .to_owned(),
            )
            .await?;

        // competence_tools — PK (akh_id, tool)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_competence_tools"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("tool").text().not_null())
                    .col(col("successes").integer().not_null())
                    .col(col("attempts").integer().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("tool")))
                    .to_owned(),
            )
            .await?;

        // competence_categories — PK (akh_id, category)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_competence_categories"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("category").text().not_null())
                    .col(col("completed").integer().not_null())
                    .col(col("attempted").integer().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("category")))
                    .to_owned(),
            )
            .await?;

        // competence_calibration — PK (akh_id, seq)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_competence_calibration"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("seq").integer().not_null())
                    .col(col("predicted").float().not_null())
                    .col(col("actual").float().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("seq")))
                    .to_owned(),
            )
            .await?;

        // sleep_metrics — PK (akh_id, cycle)
        manager
            .create_table(
                Table::create()
                    .table(Alias::new("akhomed_sleep_metrics"))
                    .if_not_exists()
                    .col(col("akh_id").uuid().not_null())
                    .col(col("cycle").big_integer().not_null())
                    .col(col("episodes_replayed").integer().not_null())
                    .col(col("duplicates_merged").integer().not_null())
                    .col(col("orphans_pruned").integer().not_null())
                    .col(col("dream_connections_found").integer().not_null())
                    .col(col("equivalences_discovered").integer().not_null())
                    .col(col("duration_cycles").big_integer().not_null())
                    .primary_key(Index::create().col(Alias::new("akh_id")).col(Alias::new("cycle")))
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        for t in TABLES.iter().rev() {
            manager
                .drop_table(Table::drop().table(Alias::new(*t)).if_exists().to_owned())
                .await?;
        }
        Ok(())
    }
}
