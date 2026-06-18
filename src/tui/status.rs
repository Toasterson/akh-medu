//! Typed client-side view of the `_akh/status` snapshot that backs the TUI
//! sidebar.
//!
//! The daemon answers `_akh/status` (see `crate::acp::dispatch_akh_agent`) with a
//! single JSON object combining goals, the active plan, tools, skills, projects,
//! and (eventually) token usage. The TUI deserializes that into these strongly-
//! typed views so the renderer consumes fields directly rather than poking at a
//! `serde_json::Value`. Unknown/extra fields are ignored, so the wire format can
//! grow without breaking the client.

use serde::Deserialize;

/// One sidebar refresh: the deserialized `_akh/status` snapshot.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct StatusSnapshot {
    /// True when the agent was busy in a turn at poll time (ADR 063 §2): the
    /// agent-derived fields below are empty, but `skills` is still populated
    /// (engine-derivable).
    #[serde(default)]
    pub stale: bool,
    /// OODA cycle count, when the agent could be read.
    #[serde(default)]
    pub cycle: Option<u64>,
    #[serde(default)]
    pub goals: Vec<GoalView>,
    #[serde(default)]
    pub plan: Option<PlanView>,
    #[serde(default)]
    pub tools: Vec<ToolView>,
    #[serde(default)]
    pub skills: Vec<SkillView>,
    #[serde(default)]
    pub projects: Vec<ProjectView>,
    /// Knowledge-graph size (engine-derivable; present even when `stale`).
    #[serde(default)]
    pub kg: Option<KgView>,
    /// Name of the active project / microtheory, when one is focused.
    #[serde(default)]
    pub active_project: Option<String>,
    /// Token usage — `None` until the model layer reports it (Phase D).
    #[serde(default)]
    pub usage: Option<UsageView>,
}

/// Knowledge-graph size for the cognitive "Mind" panel.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct KgView {
    #[serde(default)]
    pub symbols: u64,
    #[serde(default)]
    pub triples: u64,
}

/// A single goal row.
#[derive(Debug, Clone, Deserialize)]
pub struct GoalView {
    pub description: String,
    /// The serialized `GoalStatus`: a bare string for unit variants (`"Active"`)
    /// or a single-key object for data-carrying ones (`{"Failed": {...}}`). Use
    /// [`GoalView::status_label`] rather than reading this directly.
    #[serde(default)]
    pub status: serde_json::Value,
    #[serde(default)]
    pub priority: u8,
}

impl GoalView {
    /// Short status label, handling both the unit-variant string form and the
    /// tagged-object form of a serialized `GoalStatus`.
    #[must_use]
    pub fn status_label(&self) -> &str {
        match &self.status {
            serde_json::Value::String(s) => s.as_str(),
            serde_json::Value::Object(map) => {
                map.keys().next().map(String::as_str).unwrap_or("?")
            }
            _ => "?",
        }
    }
}

/// The active goal's plan.
#[derive(Debug, Clone, Deserialize)]
pub struct PlanView {
    pub goal: String,
    pub strategy: String,
    pub status: String,
    #[serde(default)]
    pub steps: Vec<PlanStepView>,
}

/// One step of the active plan.
#[derive(Debug, Clone, Deserialize)]
pub struct PlanStepView {
    pub tool: String,
    pub status: String,
}

/// A registered tool.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolView {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

/// A loaded skill / microtheory.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillView {
    pub name: String,
    #[serde(default)]
    pub state: String,
}

/// A project (named grouping of goals).
#[derive(Debug, Clone, Deserialize)]
pub struct ProjectView {
    pub name: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub goals: usize,
}

/// Token usage from the model layer (Phase D). `context_window` is `None` until
/// the window size is wired (Phase D follow-up via Ollama `/api/show`), in which
/// case the meter shows absolute counts without a percentage.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct UsageView {
    /// Cumulative tokens billed this session (prompt + completion).
    #[serde(default)]
    pub total_tokens: u64,
    /// Most recent turn's footprint — a proxy for current context fill.
    #[serde(default)]
    pub context_tokens: u64,
    /// Number of model turns recorded this session.
    #[serde(default)]
    pub turns: u64,
    /// Context-window size, when known, for the `% used` figure.
    #[serde(default)]
    pub context_window: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::StatusSnapshot;

    #[test]
    fn deserializes_full_status_payload() {
        let json = serde_json::json!({
            "workspace": "default",
            "stale": false,
            "cycle": 7,
            "goals": [
                { "description": "diagnose renewal", "status": "Active", "priority": 200 },
                { "description": "give up", "status": { "Failed": { "reason": "x" } }, "priority": 10 }
            ],
            "plan": {
                "goal": "diagnose renewal",
                "strategy": "probe then fix",
                "status": "active",
                "steps": [ { "tool": "grep", "status": "completed" } ]
            },
            "tools": [ { "name": "grep", "description": "search" } ],
            "skills": [ { "name": "k8s", "state": "Active" } ],
            "projects": [ { "name": "infra", "status": "active", "goals": 2 } ],
            "kg": { "symbols": 1200, "triples": 3400 },
            "active_project": "infra",
            "usage": { "total_tokens": 180, "context_tokens": 60, "turns": 2, "context_window": 8192 }
        });
        let snap: StatusSnapshot = serde_json::from_value(json).expect("deserialize");
        assert!(!snap.stale);
        assert_eq!(snap.cycle, Some(7));
        assert_eq!(snap.goals.len(), 2);
        assert_eq!(snap.goals[0].status_label(), "Active");
        assert_eq!(snap.goals[1].status_label(), "Failed");
        assert_eq!(snap.plan.as_ref().unwrap().steps.len(), 1);
        assert_eq!(snap.tools[0].name, "grep");
        assert_eq!(snap.skills[0].state, "Active");
        assert_eq!(snap.projects[0].goals, 2);
        let kg = snap.kg.expect("kg present");
        assert_eq!(kg.symbols, 1200);
        assert_eq!(kg.triples, 3400);
        assert_eq!(snap.active_project.as_deref(), Some("infra"));
        let usage = snap.usage.expect("usage present");
        assert_eq!(usage.total_tokens, 180);
        assert_eq!(usage.context_tokens, 60);
        assert_eq!(usage.context_window, Some(8192));
    }

    #[test]
    fn stale_payload_has_empty_agent_fields_but_parses() {
        let json = serde_json::json!({
            "workspace": "default",
            "stale": true,
            "cycle": null,
            "goals": [],
            "plan": null,
            "tools": [],
            "projects": [],
            "skills": [ { "name": "k8s", "state": "Active" } ],
            "usage": null
        });
        let snap: StatusSnapshot = serde_json::from_value(json).expect("deserialize");
        assert!(snap.stale);
        assert!(snap.goals.is_empty());
        assert_eq!(snap.skills.len(), 1);
    }
}
