//! Entity: `th_drift_counters` — unknown/degraded record counts.
//!
//! Keyed `(source, record_type, schema_marker)`, per the
//! [normal-form ticket](../../../docs/wayfinder/transcript-harvest/tickets/event-normal-form.md)'s
//! dispatch decision: an unrecognised `(source, type, schema_marker)`
//! combination is stored and counted, never guessed at. A rising
//! `degraded_count` for a *known* combination is the alarm that the format
//! research's thresholds read — a normalizer that used to succeed on this
//! shape has started failing.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "th_drift_counters")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub source: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub record_type: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub schema_marker: String,

    /// Records of this `(source, record_type, schema_marker)` the normalizer
    /// has never seen classified — undetermined, per `CONTEXT.md`.
    pub unknown_count: i64,
    /// Records normalized but only partially recognised (`th_events.degraded`).
    pub degraded_count: i64,
    pub last_seen: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
