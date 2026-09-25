//! Wire types of akheventd's Anima source (Phase 53g PR 6, ADR 113
//! decision 10).
//!
//! The source replaces the Roman observer's Anima scan and the
//! `laya_controller` routing queue. It lists projects, their work items and
//! plans through Anima's MCP API, keeps one projection per entity, and emits
//! an [`AnimaChange`] event whenever an entity's projection changes. Routing
//! is not decided here: event routes carry data-driven `when` filters over
//! the change (see `akh-session-services::route_filter`).
use crate::{EventEnvelope, Fault};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// The kind of an observed Anima entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnimaEntityKind {
    WorkItem,
    Plan,
}

impl AnimaEntityKind {
    /// The entity id prefix: `work_item` or `plan`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WorkItem => "work_item",
            Self::Plan => "plan",
        }
    }
    /// `<kind>:<id>`, the entity's stable identity in events and state.
    pub fn entity(self, id: &str) -> String {
        format!("{}:{id}", self.as_str())
    }
}

/// Hydrated fields the source adds to a work item's projection. The field
/// names are the JSON keys route filters point at.
pub mod hydrated {
    /// Stable id of the current park episode (hex).
    pub const PARK_EPISODE: &str = "park_episode";
    /// Number of the work item's sessions that match the configured
    /// `live_session` filter.
    pub const LIVE_SESSIONS: &str = "live_sessions";
    /// Id of the newest thread message that matches the configured
    /// `thread_message` filter (every message when unset).
    pub const LATEST_THREAD_MESSAGE_ID: &str = "latest_thread_message_id";
}

/// Event data of `anima.work_item.changed` / `anima.plan.changed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnimaChange {
    pub kind: AnimaEntityKind,
    /// `<kind>:<id>`.
    pub entity: String,
    pub project_id: String,
    /// The previous projection; `null` for an entity first seen after the
    /// project's baseline.
    pub previous: Option<Value>,
    /// The current projection: the configured listing fields, overlaid with
    /// the hydrated work-item detail and the [`hydrated`] fields for work
    /// items that match the source's `hydrate_parked.when` filter.
    pub current: Value,
    /// Hex blake3 digest of the canonical current projection.
    pub projection_digest: String,
    /// How many times this entity's projection has changed since it was
    /// first recorded.
    pub revision: u64,
    /// The park episode this change belongs to, when the item is parked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub park_episode: Option<String>,
    /// The n-th emission within `park_episode` (1-based).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emission: Option<u32>,
    /// The per-episode emission cap in force.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_emissions: Option<u32>,
    pub observed_at_utc: String,
}

/// A would-be event recorded by a source in shadow mode instead of being
/// ingested (`GET /v1/anima/{source}/shadow`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowRecord {
    pub source: String,
    pub recorded_at_ms: u64,
    pub event: EventEnvelope,
    /// Routes whose source, type and `when` filter match the event.
    pub routes: Vec<ShadowRoute>,
}

/// One route a shadowed event would have been delivered on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowRoute {
    pub route: String,
    /// The target akh, for an akh-targeted route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub akh: Option<String>,
    /// The target automation, for an automation-targeted route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub automation: Option<String>,
}

/// A parked entity whose emissions in its current park episode hit the cap;
/// later changes in that episode were suppressed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmissionAttention {
    pub entity: String,
    pub park_episode: String,
    pub emitted: u32,
    pub suppressed: u32,
}

/// Enumeration coverage of one project in the latest pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectCoverage {
    pub name: String,
    pub at_ms: u64,
    pub observed: u64,
    /// The server's `total`, when it reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    pub pages: u16,
    pub plans: u64,
    /// The cursor ran out and every reported item was observed.
    pub complete: bool,
    /// The project's first complete pass happened (later new rows emit).
    pub baselined: bool,
}

/// Latest outcome of an Anima source (`GET /v1/anima/{source}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnimaSourceStatus {
    pub source: String,
    pub shadow: bool,
    /// The latest poll completed.
    pub ok: bool,
    pub at_ms: u64,
    /// When a poll last completed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_ok_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Fault>,
    pub projects: u64,
    #[serde(default)]
    pub coverage: BTreeMap<String, ProjectCoverage>,
    /// Totals since the daemon started this source.
    pub counters: AnimaCounters,
    /// Entities whose current park episode hit the emission cap.
    #[serde(default)]
    pub attention: Vec<EmissionAttention>,
}

/// Running totals of one source.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnimaCounters {
    pub emitted: u64,
    pub shadowed: u64,
    pub suppressed: u64,
    pub hydrated: u64,
    /// Hydrations postponed (per-tick bound or a failed detail read); the
    /// entity keeps its previous projection until a later pass.
    pub hydration_deferred: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn change_round_trips_and_omits_absent_episode_fields() {
        let change = AnimaChange {
            kind: AnimaEntityKind::WorkItem,
            entity: AnimaEntityKind::WorkItem.entity("w1"),
            project_id: "p1".into(),
            previous: None,
            current: json!({"id":"w1","current_phase":"working"}),
            projection_digest: "d".into(),
            revision: 0,
            park_episode: None,
            emission: None,
            max_emissions: None,
            observed_at_utc: "2026-09-26T00:00:00.000Z".into(),
        };
        let value = serde_json::to_value(&change).unwrap();
        assert_eq!(value["entity"], "work_item:w1");
        assert_eq!(value["kind"], "work_item");
        assert!(value.get("park_episode").is_none());
        assert_eq!(value["previous"], Value::Null);
        assert_eq!(
            serde_json::from_value::<AnimaChange>(value).unwrap(),
            change
        );
    }
}
