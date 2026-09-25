//! akheventd's deterministic Board watchdog (Phase 53g PR 6, ADR 113
//! decision 10). Replaces the Roman observer's `notify_watchdog`.
//!
//! Every `interval_secs` the watchdog derives a structural
//! [`WatchdogSignal`] from akheventd's own records:
//!
//! - each monitor source's availability (a reading older than `stale_secs` is
//!   unavailable), health states and feed availability;
//! - each Anima source's availability, mode and emission-cap attention list
//!   (entity + park episode, so a newly capped episode posts once);
//! - delivery failures: routes whose latest settled delivery was refused
//!   (anything but `superseded`), and routes with a delivery still retrying
//!   after [`RETRYING_ATTEMPTS`] attempts.
//!
//! When the signal differs from the last one posted it sends one
//! `coordination/status@v1` message to the configured Board thread: state
//! `blocked` when anything is degraded, else `complete`. At most one post per
//! interval; a notice that fails to post is kept with its idempotency key and
//! retried unchanged. Nothing is interpreted as text.
use crate::{
    config::WatchdogConfig,
    ingress::{Delivery, Events},
    store::key,
    types::anima::{AnimaSourceStatus, EmissionAttention},
    types::monitor::{FeedAvailability, HealthStatus, MonitorReading},
    types::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// A pending delivery with this many attempts counts as retrying.
pub const RETRYING_ATTEMPTS: u32 = 3;
/// The Board message protocol of a watchdog notice.
const MESSAGE_TYPE: &str = "coordination/status@v1";

fn signal_key() -> String {
    key("watchdog", &["signal"])
}
fn notice_key() -> String {
    key("watchdog", &["notice"])
}

/// A monitor source as the watchdog sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonitorSignal {
    /// A reading exists and is fresh.
    pub available: bool,
    #[serde(default)]
    pub health: BTreeMap<String, HealthStatus>,
    /// `None` when the feed is not configured (or no fresh reading exists).
    pub alertmanager: Option<bool>,
    pub prometheus: Option<bool>,
}

/// A capped park episode.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AttentionSignal {
    pub entity: String,
    pub park_episode: String,
}

/// An Anima source as the watchdog sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimaSignal {
    /// The latest poll completed and is fresh.
    pub available: bool,
    pub shadow: bool,
    pub attention: Vec<AttentionSignal>,
}

/// Everything the watchdog compares between intervals.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchdogSignal {
    pub monitors: BTreeMap<String, MonitorSignal>,
    pub anima: BTreeMap<String, AnimaSignal>,
    /// Route -> fault code of its latest settled, refused delivery.
    pub failing_routes: BTreeMap<String, String>,
    pub retrying_routes: BTreeSet<String>,
}

impl WatchdogSignal {
    /// Whether the notice reports `blocked`.
    pub fn degraded(&self) -> bool {
        self.monitors.values().any(|m| {
            !m.available
                || m.health.values().any(|h| *h != HealthStatus::Up)
                || m.alertmanager == Some(false)
                || m.prometheus == Some(false)
        }) || self
            .anima
            .values()
            .any(|a| !a.available || !a.attention.is_empty())
            || !self.failing_routes.is_empty()
            || !self.retrying_routes.is_empty()
    }
}

/// A notice built for a signal and not yet accepted by the Board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PendingNotice {
    signal: WatchdogSignal,
    message: Value,
}

/// What one watchdog tick did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TickOutcome {
    Unchanged,
    /// A notice for this signal was accepted by the Board.
    Posted {
        degraded: bool,
    },
}

/// Run the watchdog forever (spawned by [`Events::sources`]).
pub async fn run(service: Events, config: WatchdogConfig) {
    let interval = std::time::Duration::from_secs(config.interval_secs);
    loop {
        match tick(&service, &config).await {
            Ok(TickOutcome::Posted { degraded }) => {
                tracing::info!(thread=%config.thread, degraded, "watchdog notice posted")
            }
            Ok(TickOutcome::Unchanged) => {}
            Err(error) => tracing::warn!(
                code = %error.code,
                message = %error.message,
                "watchdog notice deferred; it is retried with the same idempotency key"
            ),
        }
        tokio::time::sleep(interval).await;
    }
}

/// Compare the current signal with the last posted one; post on a change.
pub async fn tick(service: &Events, config: &WatchdogConfig) -> Result<TickOutcome> {
    let signal = signal(service, config, now_ms())?;
    let pending: Option<PendingNotice> = service.store.get(&notice_key())?;
    if pending.is_none()
        && service.store.get::<WatchdogSignal>(&signal_key())?.as_ref() == Some(&signal)
    {
        return Ok(TickOutcome::Unchanged);
    }
    let notice = match pending {
        Some(notice) => notice,
        None => {
            let notice = PendingNotice {
                message: message(config, &signal),
                signal,
            };
            service
                .store
                .transaction(|tx| tx.put(&notice_key(), &notice))?;
            notice
        }
    };
    let path = format!(
        "/v1/threads/{}/messages",
        crate::http::segment(&config.thread)
    );
    let _: Value = service
        .client()
        .post(&config.endpoint, &path, &notice.message)
        .await?;
    service.store.transaction(|tx| {
        tx.put(&signal_key(), &notice.signal)?;
        tx.delete(&notice_key())
    })?;
    Ok(TickOutcome::Posted {
        degraded: notice.signal.degraded(),
    })
}

fn message(config: &WatchdogConfig, signal: &WatchdogSignal) -> Value {
    let evidence = serde_json::to_string(signal).unwrap_or_default();
    json!({
        "message_type": MESSAGE_TYPE,
        "payload": {
            "state": if signal.degraded() { "blocked" } else { "complete" },
            "detail": format!("{}{evidence}", config.detail_prefix),
        },
        "summary": config.summary,
        "retention": {"kind": "durable"},
        "idempotency_key": new_id(),
    })
}

/// The current signal, read from akheventd's own store.
pub fn signal(service: &Events, config: &WatchdogConfig, now: u64) -> Result<WatchdogSignal> {
    let fresh = |at_ms: u64| now.saturating_sub(at_ms) < config.stale_secs * 1000;
    let feed = |availability: FeedAvailability| match availability {
        FeedAvailability::NotConfigured => None,
        FeedAvailability::Available => Some(true),
        FeedAvailability::Unknown => Some(false),
    };
    let mut monitors = BTreeMap::new();
    for monitor in &service.config.monitors {
        let reading: Option<MonitorReading> = service
            .store
            .get(&crate::monitor::snapshot_key(&monitor.source))?;
        let entry = match reading.filter(|r| fresh(r.observed_at_ms)) {
            Some(reading) => MonitorSignal {
                available: true,
                health: reading.feeds.health,
                alertmanager: feed(reading.feeds.alertmanager),
                prometheus: feed(reading.feeds.prometheus),
            },
            None => MonitorSignal {
                available: false,
                health: BTreeMap::new(),
                alertmanager: None,
                prometheus: None,
            },
        };
        monitors.insert(monitor.source.clone(), entry);
    }
    let mut anima = BTreeMap::new();
    for source in &service.config.anima {
        let status: Option<AnimaSourceStatus> =
            crate::anima_source::status(service, &source.source)?;
        let mut attention: Vec<AttentionSignal> = status
            .as_ref()
            .map(|s| s.attention.iter().map(attention_signal).collect())
            .unwrap_or_default();
        attention.sort();
        attention.truncate(config.max_attention.into());
        anima.insert(
            source.source.clone(),
            AnimaSignal {
                available: status.as_ref().is_some_and(|s| s.ok && fresh(s.at_ms)),
                shadow: source.shadow,
                attention,
            },
        );
    }
    let (failing_routes, retrying_routes) = delivery_failures(service)?;
    Ok(WatchdogSignal {
        monitors,
        anima,
        failing_routes,
        retrying_routes,
    })
}

fn attention_signal(entry: &EmissionAttention) -> AttentionSignal {
    AttentionSignal {
        entity: entry.entity.clone(),
        park_episode: entry.park_episode.clone(),
    }
}

/// Routes whose latest settled delivery was refused, and routes with a
/// delivery still retrying.
fn delivery_failures(service: &Events) -> Result<(BTreeMap<String, String>, BTreeSet<String>)> {
    let deliveries: Vec<Delivery> = service.store.scan("delivery/")?;
    let mut latest: BTreeMap<&str, &Delivery> = BTreeMap::new();
    let mut retrying = BTreeSet::new();
    for delivery in &deliveries {
        if !delivery.terminal {
            if delivery.attempts >= RETRYING_ATTEMPTS {
                retrying.insert(delivery.route.clone());
            }
            continue;
        }
        let order = |d: &Delivery| (d.request.event.occurred_at, d.id.clone());
        match latest.get(delivery.route.as_str()) {
            Some(current) if order(current) >= order(delivery) => {}
            _ => {
                latest.insert(&delivery.route, delivery);
            }
        }
    }
    let failing = latest
        .into_iter()
        .filter_map(|(route, delivery)| {
            let error = delivery.error.as_ref()?;
            (error.code != "superseded" && !delivery.admitted())
                .then(|| (route.to_string(), error.code.clone()))
        })
        .collect();
    Ok((failing, retrying))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, store::Store};
    use axum::{
        Json, Router,
        extract::{Path, State},
        http::{HeaderMap, StatusCode},
        routing::post,
    };
    use std::sync::{Arc, Mutex};

    const BOARD_TOKEN: &str = "board-watchdog-token-0123456789-abcdef";
    const THREAD: &str = "01a0ab49-9cc4-7012-8785-1eada26fbd46";

    #[derive(Default)]
    struct Board {
        posts: Vec<(String, Value)>,
        fail: bool,
    }

    async fn messages(
        State(board): State<Arc<Mutex<Board>>>,
        Path(thread): Path<String>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> std::result::Result<Json<Value>, StatusCode> {
        assert_eq!(
            headers["authorization"].to_str().unwrap(),
            format!("Bearer {BOARD_TOKEN}")
        );
        let mut board = board.lock().unwrap();
        board.posts.push((thread, body));
        if board.fail {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        Ok(Json(json!({"id":"m1"})))
    }

    async fn fixture() -> (tempfile::TempDir, Events, WatchdogConfig, Arc<Mutex<Board>>) {
        let board = Arc::new(Mutex::new(Board::default()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new()
            .route("/v1/threads/{thread}/messages", post(messages))
            .with_state(board.clone());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let dir = tempfile::tempdir().unwrap();
        let operator = dir.path().join("operator.token");
        std::fs::write(&operator, "operator-token-0123456789-abcdef").unwrap();
        let board_token = dir.path().join("board.token");
        std::fs::write(&board_token, BOARD_TOKEN).unwrap();
        let config: Config = serde_json::from_value(json!({
            "service_id":"events","listen":"127.0.0.1:0","data_dir":dir.path(),
            "principals":[{"id":"operator","token_file":operator,"roles":["admin"]}],
            "endpoints":{"board-watchdog":{"url":url,"token_file":board_token},
                "health":{"url":"http://127.0.0.1:1","anonymous":true}},
            "routes":[{"id":"janus-inbox","source":"inbox","event_type":"delegation","automation":"janus"}],
            "monitors":[{"source":"roman-alerts","event_type":"monitoring.changed",
                "health":[{"name":"core","endpoint":"health"}]}],
            "watchdog":{"endpoint":"board-watchdog","thread":THREAD,
                "summary":"Roman workflow and monitoring availability changed",
                "detail_prefix":"evidence: "}
        }))
        .unwrap();
        config.validate().unwrap();
        let watchdog = config.watchdog.clone().unwrap();
        let events = Events::new(
            Arc::new(config),
            Store::open(&dir.path().join("events.redb")).unwrap(),
        )
        .unwrap();
        (dir, events, watchdog, board)
    }

    #[tokio::test]
    async fn posts_once_per_signal_change_and_retries_with_the_same_key() {
        let (_dir, events, config, board) = fixture().await;
        // No monitor reading yet: unavailable, degraded.
        let first = signal(&events, &config, now_ms()).unwrap();
        assert!(!first.monitors["roman-alerts"].available);
        assert!(first.degraded());
        assert_eq!(
            tick(&events, &config).await.unwrap(),
            TickOutcome::Posted { degraded: true }
        );
        assert_eq!(
            tick(&events, &config).await.unwrap(),
            TickOutcome::Unchanged
        );
        {
            let board = board.lock().unwrap();
            assert_eq!(board.posts.len(), 1);
            let (thread, body) = &board.posts[0];
            assert_eq!(thread, THREAD);
            assert_eq!(body["message_type"], MESSAGE_TYPE);
            assert_eq!(body["payload"]["state"], "blocked");
            assert!(
                body["payload"]["detail"]
                    .as_str()
                    .unwrap()
                    .starts_with("evidence: {")
            );
            assert_eq!(
                body["summary"],
                "Roman workflow and monitoring availability changed"
            );
        }

        // A refused delivery is a new failing route: one more notice. The
        // first attempt fails; the retry reuses the idempotency key.
        let event: EventEnvelope = serde_json::from_value(json!({"schema_version":1,
            "source":"inbox","type":"delegation","id":"d1","occurred_at":1,"data":{}}))
        .unwrap();
        let receipt = events
            .ingest(&events.source_principal("inbox"), event)
            .unwrap();
        events
            .store
            .transaction(|tx| {
                let k = key("delivery", &[&receipt.deliveries[0]]);
                let mut delivery: Delivery = tx.get(&k)?.unwrap();
                delivery.terminal = true;
                delivery.error = Some(Fault::new(NO_SUBSCRIBER, "no schedule"));
                tx.put(&k, &delivery)
            })
            .unwrap();
        board.lock().unwrap().fail = true;
        assert!(tick(&events, &config).await.is_err());
        board.lock().unwrap().fail = false;
        assert_eq!(
            tick(&events, &config).await.unwrap(),
            TickOutcome::Posted { degraded: true }
        );
        let board = board.lock().unwrap();
        assert_eq!(board.posts.len(), 3);
        assert_eq!(
            board.posts[1].1["idempotency_key"],
            board.posts[2].1["idempotency_key"]
        );
        assert_ne!(
            board.posts[0].1["idempotency_key"],
            board.posts[1].1["idempotency_key"]
        );
        let posted: WatchdogSignal = events.store.get(&signal_key()).unwrap().unwrap();
        assert_eq!(posted.failing_routes["janus-inbox"], NO_SUBSCRIBER);
    }

    #[tokio::test]
    async fn fresh_healthy_monitor_and_superseded_deliveries_are_not_degraded() {
        let (_dir, events, config, _board) = fixture().await;
        let snapshot = akh_session_types::monitor::MonitorSnapshot {
            observed_at_ms: now_ms(),
            health: [(
                "core".to_string(),
                akh_session_types::monitor::HealthObservation {
                    status: HealthStatus::Up,
                    latency: None,
                },
            )]
            .into(),
            alertmanager: akh_session_types::monitor::Feed::NotConfigured,
            prometheus: akh_session_types::monitor::Feed::NotConfigured,
        };
        crate::monitor::record(&events, &events.config.monitors[0], snapshot).unwrap();
        let event: EventEnvelope = serde_json::from_value(json!({"schema_version":1,
            "source":"inbox","type":"delegation","id":"d1","occurred_at":1,"data":{}}))
        .unwrap();
        let receipt = events
            .ingest(&events.source_principal("inbox"), event)
            .unwrap();
        events
            .store
            .transaction(|tx| {
                let k = key("delivery", &[&receipt.deliveries[0]]);
                let mut delivery: Delivery = tx.get(&k)?.unwrap();
                delivery.terminal = true;
                delivery.error = Some(Fault::new("superseded", "newer"));
                tx.put(&k, &delivery)
            })
            .unwrap();
        let current = signal(&events, &config, now_ms()).unwrap();
        assert!(current.monitors["roman-alerts"].available);
        assert_eq!(current.monitors["roman-alerts"].alertmanager, None);
        assert!(current.failing_routes.is_empty());
        assert!(!current.degraded());
        // The same reading, seen after `stale_secs`, is unavailable.
        let later = now_ms() + config.stale_secs * 1000 + 1;
        assert!(!signal(&events, &config, later).unwrap().monitors["roman-alerts"].available);
    }
}
