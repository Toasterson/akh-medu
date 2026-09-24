//! `POST /v1/akhs/{akh}/events`: event admission into the akh's own reactor.
//!
//! Phase 53d (ADR 113 decision 4, Anima WI `01a0d469-f8f1`,
//! `docs/ai/phases/phase-53/README.md`). akheventd stays a thin ingress
//! adapter: it forwards each routed [`EventEnvelope`] here, and the akh
//! decides what runs. The event wakes every enabled reactor schedule whose
//! rule is `event { source: "<event source>/<event type>" }`; the schedule's
//! own action (an `llm_turn` with an exit condition, a trigger, …) does the
//! work on the next reactor tick, under the akh's daily budget.
//!
//! Status codes (the `/v1` Fault convention: JSON [`Fault`] body):
//!
//! | outcome | HTTP | `code` |
//! |---|---|---|
//! | admitted (or replayed id) | 200 | [`EventAdmission`] body, `duplicate` set on replay |
//! | principal lacks the `event` role or the akh grant | 403 | `forbidden` |
//! | malformed envelope | 400 | `invalid_request` |
//! | same id, different payload | 409 | `conflict` |
//! | no enabled schedule subscribes | 422 | [`NO_SUBSCRIBER`] (terminal for akheventd) |
//! | akh cannot load / a wake failed | 503 | `unavailable` (akheventd retries) |
use super::{Api, CoreApiError, CoreApp, CorePrincipal};
use crate::agent::reactor::{ReactorEvent, ReactorStore, ScheduleRule};
use crate::engine::Engine;
use akh_service_store::key;
use akh_session_types::{
    EventAdmission, EventEnvelope, Fault, NO_SUBSCRIBER, SCHEMA_VERSION, validate_key,
};
use axum::{Extension, Json, extract::State};
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

/// Largest event payload queued on a reactor schedule. Larger `data` is
/// replaced by a bounded preview, as board ingress does, so 256 pending
/// events cannot grow one schedule record without bound.
const MAX_QUEUED_EVENT_BYTES: usize = 128 * 1024;
const MAX_PREVIEW_CHARS: usize = 16 * 1024;

/// Service roles a core principal may hold beyond tool calls (Phase 53d).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreRole {
    /// May submit events to `POST /v1/akhs/{akh}/events` for its akhs.
    Event,
}

/// Errors of event admission.
#[derive(Debug, Error, Diagnostic)]
pub enum EventAdmissionError {
    #[error("core principal `{principal}` may not submit events")]
    #[diagnostic(
        code(akh::core_api::event_role_required),
        help(
            "Give the principal the event role in the core config: \
             `\"roles\": [\"event\"]` next to its `akhs`. The akh must also be listed in \
             `akhs`. akheventd authenticates with this principal's token file."
        )
    )]
    RoleRequired { principal: String },

    #[error("invalid event envelope: {message}")]
    #[diagnostic(
        code(akh::core_api::invalid_event),
        help(
            "Send an akh-session-types EventEnvelope (schema_version 1) with non-empty \
             `source`, `type` and `id` of at most 256 characters without control characters."
        )
    )]
    Invalid { message: String },

    #[error("no enabled reactor schedule of akh `{akh}` subscribes to `{subscription}`")]
    #[diagnostic(
        code(akh::core_api::no_subscriber),
        help(
            "Nothing was queued and retrying will not help. Create an event schedule with \
             `reactor_manage` (rule `{{\"kind\":\"event\",\"source\":\"{subscription}\"}}`) and enable \
             it, re-enable a paused one ({paused}), or remove the akheventd route that \
             targets this akh."
        )
    )]
    NoSubscriber {
        akh: String,
        /// `<event source>/<event type>`, the schedule rule's `source`.
        subscription: String,
        /// Matching schedules that exist but are disabled or paused.
        paused: String,
    },

    #[error("event id `{id}` from `{event_source}` was admitted before with a different payload")]
    #[diagnostic(
        code(akh::core_api::event_conflict),
        help(
            "Event ids are idempotency keys. Give a changed event a new id; a replay must \
             carry the identical envelope."
        )
    )]
    Conflict { event_source: String, id: String },

    #[error("could not queue the event on schedule `{schedule}` of akh `{akh}`: {message}")]
    #[diagnostic(
        code(akh::core_api::event_wake),
        help(
            "The schedule's queue is full or its store failed. Schedules already woken keep \
             the event (a retry is deduplicated). akheventd retries; check the akhomed log \
             for this workspace if it persists."
        )
    )]
    Wake {
        akh: String,
        schedule: String,
        message: String,
    },
}

impl EventAdmissionError {
    /// Wire code (`Fault.code`).
    pub fn wire_code(&self) -> &'static str {
        match self {
            Self::RoleRequired { .. } => "forbidden",
            Self::Invalid { .. } => "invalid_request",
            Self::NoSubscriber { .. } => NO_SUBSCRIBER,
            Self::Conflict { .. } => "conflict",
            Self::Wake { .. } => "unavailable",
        }
    }
}

impl From<EventAdmissionError> for Fault {
    fn from(error: EventAdmissionError) -> Self {
        let message = match error.help() {
            Some(help) => format!("{error} ({help})"),
            None => error.to_string(),
        };
        Fault::new(error.wire_code(), message)
    }
}

/// Durable replay record of one admitted event id.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdmittedEvent {
    fingerprint: String,
    admission: EventAdmission,
}

fn event_fingerprint(event: &EventEnvelope) -> Result<String, Fault> {
    let bytes = serde_json::to_vec(event).map_err(|e| Fault::invalid(e.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn validate(event: &EventEnvelope) -> Result<(), EventAdmissionError> {
    let invalid = |message: String| EventAdmissionError::Invalid { message };
    if event.schema_version != SCHEMA_VERSION {
        return Err(invalid(format!(
            "unsupported schema_version {}",
            event.schema_version
        )));
    }
    for (field, value) in [
        ("source", &event.source),
        ("type", &event.event_type),
        ("id", &event.id),
    ] {
        validate_key(value).map_err(|fault| invalid(format!("{field}: {}", fault.message)))?;
    }
    Ok(())
}

/// The payload queued on the schedule: the whole envelope, with `data`
/// replaced by a bounded preview when it is too large.
fn queued_payload(event: &EventEnvelope) -> Result<String, Fault> {
    let encoded = serde_json::to_string(event).map_err(|e| Fault::invalid(e.to_string()))?;
    if encoded.len() <= MAX_QUEUED_EVENT_BYTES {
        return Ok(encoded);
    }
    let mut bounded = event.clone();
    bounded.data = json!({
        "truncated": true,
        "original_bytes": encoded.len(),
        "preview": event.data.to_string().chars().take(MAX_PREVIEW_CHARS).collect::<String>(),
    });
    serde_json::to_string(&bounded).map_err(|e| Fault::invalid(e.to_string()))
}

/// Wake every enabled event schedule subscribed to `source`. Returns the
/// schedules that newly queued the event (a schedule that already holds or
/// processed the id is deduplicated by the reactor).
fn wake_subscribers(
    engine: &Engine,
    akh: &str,
    source: &str,
    event: ReactorEvent,
) -> Result<Vec<String>, EventAdmissionError> {
    let store = ReactorStore::new(engine);
    let wake_error = |schedule: &str, message: String| EventAdmissionError::Wake {
        akh: akh.into(),
        schedule: schedule.into(),
        message,
    };
    let subscribed: Vec<_> = store
        .list_checked()
        .map_err(|error| wake_error("*", error.to_string()))?
        .into_iter()
        .filter(
            |schedule| matches!(&schedule.rule, ScheduleRule::Event { source: s } if s == source),
        )
        .collect();
    let (enabled, paused): (Vec<_>, Vec<_>) = subscribed
        .into_iter()
        .partition(|schedule| schedule.enabled && !schedule.state.completed);
    if enabled.is_empty() {
        let paused = paused
            .iter()
            .map(|schedule| schedule.id.as_str())
            .collect::<Vec<_>>();
        return Err(EventAdmissionError::NoSubscriber {
            akh: akh.into(),
            subscription: source.into(),
            paused: if paused.is_empty() {
                "none exist".into()
            } else {
                paused.join(", ")
            },
        });
    }
    let mut woken = Vec::new();
    for schedule in enabled {
        if store
            .wake(&schedule.id, event.clone())
            .map_err(|error| wake_error(&schedule.id, error.to_string()))?
        {
            woken.push(schedule.id);
        }
    }
    Ok(woken)
}

/// `POST /v1/akhs/{akh}/events`.
pub(super) async fn submit(
    State(app): State<CoreApp>,
    Extension(p): Extension<CorePrincipal>,
    axum::extract::Path(akh): axum::extract::Path<String>,
    Json(event): Json<EventEnvelope>,
) -> Api<EventAdmission> {
    if !p.roles.contains(&CoreRole::Event) {
        return Err(Fault::from(EventAdmissionError::RoleRequired { principal: p.id }).into());
    }
    super::authorize(&p, &akh)?;
    validate(&event).map_err(Fault::from)?;
    let source = format!("{}/{}", event.source, event.event_type);
    let event_key = format!("{source}#{}", event.id);
    let fingerprint = event_fingerprint(&event)?;
    let record_key = key("core_event", &[&akh, &event.source, &event.id]);
    let replay = |old: AdmittedEvent| -> Api<EventAdmission> {
        if old.fingerprint != fingerprint {
            return Err(Fault::from(EventAdmissionError::Conflict {
                event_source: event.source.clone(),
                id: event.id.clone(),
            })
            .into());
        }
        Ok(Json(EventAdmission {
            duplicate: true,
            ..old.admission
        }))
    };
    if let Some(old) = app.store.get::<AdmittedEvent>(&record_key)? {
        return replay(old);
    }

    // Lazily loads the akh (its Agent owns the reactor the daemon drives).
    let engine = app.backend.engine(&akh).await.map_err(|message| {
        Fault::from(CoreApiError::BackendUnavailable {
            akh: akh.clone(),
            message,
        })
    })?;
    let queued = ReactorEvent {
        source: source.clone(),
        id: event.id.clone(),
        observed_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        payload: queued_payload(&event)?,
    };
    let woken = {
        let akh = akh.clone();
        let source = source.clone();
        tokio::task::spawn_blocking(move || wake_subscribers(&engine, &akh, &source, queued))
            .await
            .map_err(super::error)?
            .map_err(Fault::from)?
    };
    let admission = EventAdmission {
        event_key,
        duplicate: woken.is_empty(),
        schedules_woken: woken,
    };
    // First writer wins, so a concurrent replay cannot overwrite the record
    // of the submission that actually queued the event.
    let stored = app.store.transaction(|tx| {
        if let Some(old) = tx.get::<AdmittedEvent>(&record_key)? {
            return Ok(Some(old));
        }
        tx.put(
            &record_key,
            &AdmittedEvent {
                fingerprint: fingerprint.clone(),
                admission: admission.clone(),
            },
        )?;
        Ok(None)
    })?;
    match stored {
        Some(old) => replay(old),
        None => Ok(Json(admission)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CoreBackend, CoreConfig, CoreFuture, ToolScope, router};
    use super::*;
    use crate::agent::reactor::{ReactorAction, ReactorSchedule};
    use crate::agent::tool::ToolRegistry;
    use crate::akh_config::AkhConfig;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use serde_json::Value;
    use std::{collections::BTreeSet, path::Path, sync::Arc};
    use tower::ServiceExt;

    const TOKEN: &str = "core-event-credential-long-enough";

    struct Backend(Arc<Engine>);
    impl CoreBackend for Backend {
        fn engine<'a>(&'a self, _: &'a str) -> CoreFuture<'a, Arc<Engine>> {
            Box::pin(async { Ok(self.0.clone()) })
        }
        fn config<'a>(&'a self, _: &'a str) -> CoreFuture<'a, Arc<AkhConfig>> {
            Box::pin(async { Ok(Arc::new(AkhConfig::default())) })
        }
        fn tools<'a>(&'a self, _: &'a str) -> CoreFuture<'a, Arc<ToolRegistry>> {
            Box::pin(async { Ok(Arc::new(ToolRegistry::new())) })
        }
    }

    fn durable_engine() -> Arc<Engine> {
        Arc::new(
            Engine::new(crate::engine::EngineConfig {
                data_dir: Some(tempfile::tempdir().unwrap().keep()),
                ..Default::default()
            })
            .unwrap(),
        )
    }

    fn app(dir: &Path, engine: Arc<Engine>, roles: BTreeSet<CoreRole>) -> CoreApp {
        let token_file = dir.join("token");
        std::fs::write(&token_file, TOKEN).unwrap();
        CoreApp::new(
            CoreConfig {
                sqlite: false,
                listen: "127.0.0.1:0".parse().unwrap(),
                data_dir: dir.into(),
                principals: vec![CorePrincipal {
                    id: "events".into(),
                    token_file,
                    akhs: vec!["janus".into()],
                    tools: ToolScope::default(),
                    roles,
                }],
            },
            Arc::new(Backend(engine)),
        )
        .unwrap()
    }

    fn schedule(id: &str, source: &str) -> ReactorSchedule {
        ReactorSchedule::new(
            id,
            id,
            ScheduleRule::Event {
                source: source.into(),
            },
            ReactorAction::LlmTurn {
                prompt: "Handle the delegation.".into(),
                agent_id: None,
                allow_tools: false,
                require_mutation: false,
                exit_condition: String::new(),
                report_thread: None,
            },
            0,
        )
        .unwrap()
    }

    fn envelope(id: &str, text: &str) -> Value {
        json!({"schema_version": 1, "id": id, "source": "djehuty-inbox", "type": "delegation",
            "occurred_at": 1, "data": {"text": text}})
    }

    async fn submit_to(app: &CoreApp, akh: &str, body: Value) -> (StatusCode, Value) {
        let response = router(app.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/akhs/{akh}/events"))
                    .header("authorization", format!("Bearer {TOKEN}"))
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap())
            .unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn an_event_wakes_matching_schedules_and_a_replay_is_a_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let engine = durable_engine();
        let store = ReactorStore::new(&engine);
        store
            .put(&schedule("delegations", "djehuty-inbox/delegation"))
            .unwrap();
        store
            .put(&schedule("audit", "djehuty-inbox/delegation"))
            .unwrap();
        store.put(&schedule("alerts", "grafana/alert")).unwrap();
        let app = app(
            dir.path(),
            engine.clone(),
            BTreeSet::from([CoreRole::Event]),
        );

        let (status, body) = submit_to(&app, "janus", envelope("m-1", "please review")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let admission: EventAdmission = serde_json::from_value(body).unwrap();
        assert_eq!(admission.event_key, "djehuty-inbox/delegation#m-1");
        assert_eq!(admission.schedules_woken, ["audit", "delegations"]);
        assert!(!admission.duplicate);
        let queued = store.get("delegations").unwrap().state.pending_events;
        assert_eq!(queued.len(), 1);
        assert!(queued[0].payload.contains("please review"));
        assert!(store.get("alerts").unwrap().state.pending_events.is_empty());

        // The same id again: acknowledged as a duplicate, nothing re-queued.
        let (status, body) = submit_to(&app, "janus", envelope("m-1", "please review")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["duplicate"], true);
        assert_eq!(
            store.get("delegations").unwrap().state.pending_events.len(),
            1
        );
        // The same id with a different payload is a conflict.
        let (status, body) = submit_to(&app, "janus", envelope("m-1", "changed")).await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
    }

    #[tokio::test]
    async fn no_subscriber_is_a_diagnostic_and_nothing_is_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let engine = durable_engine();
        let mut paused = schedule("delegations", "djehuty-inbox/delegation");
        paused.enabled = false;
        ReactorStore::new(&engine).put(&paused).unwrap();
        let app = app(dir.path(), engine, BTreeSet::from([CoreRole::Event]));

        let (status, body) = submit_to(&app, "janus", envelope("m-2", "hello")).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["code"], NO_SUBSCRIBER);
        let message = body["message"].as_str().unwrap();
        assert!(message.contains("djehuty-inbox/delegation"), "{message}");
        assert!(message.contains("delegations"), "names the paused schedule");
        assert!(app.store.scan::<Value>("core_event/").unwrap().is_empty());
    }

    #[tokio::test]
    async fn submitting_events_needs_the_event_role_and_the_akh_grant() {
        let dir = tempfile::tempdir().unwrap();
        let engine = durable_engine();
        ReactorStore::new(&engine)
            .put(&schedule("delegations", "djehuty-inbox/delegation"))
            .unwrap();
        let no_role = app(dir.path(), engine.clone(), BTreeSet::new());
        let (status, body) = submit_to(&no_role, "janus", envelope("m-3", "x")).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        assert!(body["message"].as_str().unwrap().contains("roles"));

        let dir = tempfile::tempdir().unwrap();
        let with_role = app(dir.path(), engine, BTreeSet::from([CoreRole::Event]));
        let (status, _) = submit_to(&with_role, "minerva", envelope("m-3", "x")).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn roles_default_to_none_and_read_event() {
        let principal: CorePrincipal = serde_json::from_value(json!({
            "id": "sessions", "token_file": "/t", "akhs": ["a"]
        }))
        .unwrap();
        assert!(principal.roles.is_empty());
        let principal: CorePrincipal = serde_json::from_value(json!({
            "id": "events", "token_file": "/t", "akhs": ["a"], "roles": ["event"]
        }))
        .unwrap();
        assert!(principal.roles.contains(&CoreRole::Event));
    }

    #[test]
    fn oversized_event_data_is_bounded_before_queueing() {
        let event: EventEnvelope =
            serde_json::from_value(envelope("big", &"x".repeat(MAX_QUEUED_EVENT_BYTES * 2)))
                .unwrap();
        let payload = queued_payload(&event).unwrap();
        assert!(payload.len() < MAX_QUEUED_EVENT_BYTES);
        let bounded: EventEnvelope = serde_json::from_str(&payload).unwrap();
        assert_eq!(bounded.data["truncated"], true);
        assert_eq!(bounded.id, "big");
    }
}
