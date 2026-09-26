//! akheventd monitor source (Phase 53g PR 5, ADR 113 decision 10).
//!
//! Replaces the Roman observer's Python `monitoring()` loop. Each configured
//! [`MonitorSource`] polls its health endpoints and alert feeds every
//! `interval_secs` and records a [`MonitorSnapshot`] under
//! `monitor_snapshot/<source>`. The first observation is a silent baseline;
//! afterwards, whenever the snapshot's structural signature changes, one
//! `{previous, current}` event ([`MonitorChange`]) is ingested as
//! `<source>/<event_type>`. The event and the new snapshot commit in one
//! store transaction, so a crash can neither lose a change nor emit it twice.
//!
//! The event ID is the blake3 digest of the new signature, keyed by the
//! transition number: a state that recurs (firing, resolved, firing again)
//! gets a fresh ID instead of being refused as a reused ID, and a coalescing
//! route (`coalesce_pending = true`) supersedes undelivered older changes
//! exactly as it did for the observer's outbox.
//!
//! Nothing here reads text for meaning. A feed that cannot be read is
//! [`Feed::Unknown`], never an empty alert list.
use crate::{
    config::{AlertmanagerFeed, HealthCheck, MonitorSource, PrometheusFeed},
    http::Client,
    ingress::Events,
    store::key,
    types::monitor::{
        Alert, Feed, FeedFault, HealthObservation, HealthStatus, LabelSet, LatencyClass,
        MonitorChange, MonitorReading, MonitorSignature, MonitorSnapshot,
    },
    types::*,
};
use serde::{Deserialize, de::DeserializeOwned};
use std::time::{Duration, Instant};

/// Store key prefix of the latest reading per source.
pub const SNAPSHOT_KEY: &str = "monitor_snapshot";
/// Largest feed body the monitor reads.
const MAX_FEED_BYTES: usize = 4 * 1024 * 1024;
/// Keep emitted events safely under the ingress bound of 1 MiB.
const MAX_EVENT_DATA_BYTES: usize = 900 * 1024;
const SIGNATURE_DOMAIN: &[u8] = b"akh-monitor-signature/v1\n";
const EVENT_ID_DOMAIN: &[u8] = b"akh-monitor-event/v1\n";

/// `monitor_snapshot/<source>`.
pub fn snapshot_key(source: &str) -> String {
    key(SNAPSHOT_KEY, &[source])
}

/// Hex blake3 digest of a signature's canonical bytes.
pub fn signature_digest(signature: &MonitorSignature) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SIGNATURE_DOMAIN);
    hasher.update(&signature.canonical_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Event ID of the `transition`-th change, whose new signature digest is
/// `signature`.
pub fn event_id(signature: &str, transition: u64) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(EVENT_ID_DOMAIN);
    hasher.update(signature.as_bytes());
    hasher.update(&transition.to_be_bytes());
    hasher.finalize().to_hex().to_string()
}

/// What one poll did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollOutcome {
    /// First observation: recorded, nothing emitted.
    Baseline,
    /// Same signature: the reading's timestamps and display fields refreshed.
    Unchanged,
    /// The signature changed and one event was ingested.
    Changed {
        event_id: String,
        deliveries: Vec<String>,
    },
}

/// Poll `source` forever (spawned by [`Events::sources`]).
pub async fn run(service: Events, source: MonitorSource) {
    let interval = Duration::from_secs(source.interval_secs);
    loop {
        match poll(&service, &source).await {
            Ok(PollOutcome::Changed {
                event_id,
                deliveries,
            }) => tracing::info!(
                source = %source.source,
                event = %event_id,
                deliveries = deliveries.len(),
                "monitor signature changed; event ingested"
            ),
            Ok(_) => {}
            Err(error) => tracing::warn!(
                source = %source.source,
                code = %error.code,
                message = %error.message,
                "monitor poll deferred; the last snapshot is kept"
            ),
        }
        tokio::time::sleep(interval).await;
    }
}

/// Observe once and record the result.
pub async fn poll(service: &Events, source: &MonitorSource) -> Result<PollOutcome> {
    let snapshot = observe(service.client(), source).await;
    record(service, source, snapshot)
}

/// Record `snapshot` as the source's latest reading, emitting a change event
/// when its signature differs from the stored one.
pub fn record(
    service: &Events,
    source: &MonitorSource,
    snapshot: MonitorSnapshot,
) -> Result<PollOutcome> {
    let snapshot = snapshot.normalized();
    let signature = signature_digest(&snapshot.signature());
    let k = snapshot_key(&source.source);
    let previous: Option<MonitorReading> = service.store.get(&k)?;
    let reading = |transitions: u64, changed_at_ms: u64| MonitorReading {
        source: source.source.clone(),
        event_type: source.event_type.clone(),
        signature: signature.clone(),
        transitions,
        changed_at_ms,
        observed_at_ms: snapshot.observed_at_ms,
        feeds: snapshot.feeds(),
        snapshot: snapshot.clone(),
    };
    match previous {
        None => {
            let current = reading(0, snapshot.observed_at_ms);
            service.store.transaction(|tx| tx.put(&k, &current))?;
            Ok(PollOutcome::Baseline)
        }
        Some(previous) if previous.signature == signature => {
            let current = reading(previous.transitions, previous.changed_at_ms);
            service.store.transaction(|tx| tx.put(&k, &current))?;
            Ok(PollOutcome::Unchanged)
        }
        Some(previous) => {
            let current = reading(
                previous.transitions.saturating_add(1),
                snapshot.observed_at_ms,
            );
            let event = change_event(source, &previous, &current)?;
            let event_id = event.id.clone();
            let receipt =
                service.ingest_with(&service.source_principal(&source.source), event, |tx| {
                    tx.put(&k, &current)
                })?;
            Ok(PollOutcome::Changed {
                event_id,
                deliveries: receipt.deliveries,
            })
        }
    }
}

fn utc(ms: u64) -> String {
    i64::try_from(ms)
        .ok()
        .and_then(chrono::DateTime::from_timestamp_millis)
        .map(|at| at.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

fn elide_annotations(snapshot: &mut MonitorSnapshot) {
    if let Feed::Available { items } = &mut snapshot.alertmanager {
        for alert in items {
            alert.annotations.clear();
        }
    }
}

/// The change event for `previous` -> `current`.
fn change_event(
    source: &MonitorSource,
    previous: &MonitorReading,
    current: &MonitorReading,
) -> Result<EventEnvelope> {
    let mut change = MonitorChange {
        previous: previous.snapshot.clone(),
        current: current.snapshot.clone(),
        previous_signature: previous.signature.clone(),
        signature: current.signature.clone(),
        transition: current.transitions,
        observed_at_utc: utc(current.observed_at_ms),
        annotations_elided: false,
    };
    let encode = |change: &MonitorChange| {
        serde_json::to_value(change).map_err(|e| Fault::invalid(e.to_string()))
    };
    let mut data = encode(&change)?;
    if data.to_string().len() > MAX_EVENT_DATA_BYTES {
        // Annotations are display-only; the reader can fetch them from
        // `GET /v1/monitors/{source}`.
        elide_annotations(&mut change.previous);
        elide_annotations(&mut change.current);
        change.annotations_elided = true;
        data = encode(&change)?;
        if data.to_string().len() > MAX_EVENT_DATA_BYTES {
            return Err(Fault::invalid(format!(
                "monitor {} change exceeds the event size bound even without annotations; \
                 narrow the alert feed (for example an Alertmanager filter in `path`)",
                source.source
            )));
        }
    }
    Ok(EventEnvelope {
        schema_version: SCHEMA_VERSION,
        id: event_id(&current.signature, current.transitions),
        source: source.source.clone(),
        event_type: source.event_type.clone(),
        occurred_at: current.observed_at_ms,
        subject: None,
        correlation_id: None,
        data,
    })
}

/// Probe every health endpoint and read every feed of `source`, concurrently.
pub async fn observe(client: &Client, source: &MonitorSource) -> MonitorSnapshot {
    let timeout = Duration::from_secs(source.timeout_secs);
    let probes = futures_util::future::join_all(
        source
            .health
            .iter()
            .map(|check| probe(client, check, timeout)),
    );
    let alerts = async {
        match &source.alertmanager {
            None => Feed::NotConfigured,
            Some(feed) => alertmanager(client, feed, timeout).await,
        }
    };
    let series = async {
        match &source.prometheus {
            None => Feed::NotConfigured,
            Some(feed) => prometheus(client, feed, timeout).await,
        }
    };
    let (health, alertmanager, prometheus) = tokio::join!(probes, alerts, series);
    MonitorSnapshot {
        observed_at_ms: now_ms(),
        health: source
            .health
            .iter()
            .map(|check| check.name.clone())
            .zip(health)
            .collect(),
        alertmanager,
        prometheus,
    }
    .normalized()
}

/// Up on 2xx, down on a refused connection or non-2xx, unknown otherwise.
async fn probe(client: &Client, check: &HealthCheck, timeout: Duration) -> HealthObservation {
    let unknown = HealthObservation {
        status: HealthStatus::Unknown,
        latency: None,
    };
    let Ok(request) = client.request(&check.endpoint, reqwest::Method::GET, &check.path) else {
        return unknown;
    };
    let started = Instant::now();
    match request.timeout(timeout).send().await {
        Ok(response) => HealthObservation {
            status: if response.status().is_success() {
                HealthStatus::Up
            } else {
                HealthStatus::Down
            },
            latency: Some(LatencyClass::classify(
                u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            )),
        },
        Err(error) if error.is_timeout() => unknown,
        Err(error) if error.is_connect() => HealthObservation {
            status: HealthStatus::Down,
            latency: None,
        },
        Err(_) => unknown,
    }
}

/// Why a feed read failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Unavailable {
    reason: FeedFault,
    http_status: Option<u16>,
}

impl Unavailable {
    fn new(reason: FeedFault) -> Self {
        Self {
            reason,
            http_status: None,
        }
    }
    fn feed<T>(self) -> Feed<T> {
        Feed::Unknown {
            reason: self.reason,
            http_status: self.http_status,
        }
    }
}

async fn fetch_json<T: DeserializeOwned>(
    client: &Client,
    endpoint: &str,
    path: &str,
    timeout: Duration,
) -> std::result::Result<T, Unavailable> {
    let request = client
        .request(endpoint, reqwest::Method::GET, path)
        .map_err(|_| Unavailable::new(FeedFault::CredentialUnavailable))?;
    let mut response = request.timeout(timeout).send().await.map_err(|error| {
        Unavailable::new(if error.is_timeout() {
            FeedFault::Timeout
        } else {
            FeedFault::Unreachable
        })
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(Unavailable {
            reason: if matches!(status.as_u16(), 401 | 403) {
                FeedFault::Unauthorized
            } else {
                FeedFault::HttpStatus
            },
            http_status: Some(status.as_u16()),
        });
    }
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if body.len() + chunk.len() > MAX_FEED_BYTES {
                    return Err(Unavailable::new(FeedFault::TooLarge));
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(error) => {
                return Err(Unavailable::new(if error.is_timeout() {
                    FeedFault::Timeout
                } else {
                    FeedFault::Unreachable
                }));
            }
        }
    }
    serde_json::from_slice(&body).map_err(|_| Unavailable::new(FeedFault::InvalidResponse))
}

async fn alertmanager(
    client: &Client,
    feed: &AlertmanagerFeed,
    timeout: Duration,
) -> Feed<Vec<Alert>> {
    match fetch_json::<Vec<Alert>>(client, &feed.endpoint, &feed.path, timeout).await {
        Ok(items) => Feed::Available { items },
        Err(unavailable) => unavailable.feed(),
    }
}

#[derive(Deserialize)]
struct QueryReply {
    status: String,
    #[serde(default)]
    data: Option<QueryData>,
}
#[derive(Deserialize)]
struct QueryData {
    #[serde(default)]
    result: Vec<Series>,
}
#[derive(Deserialize)]
struct Series {
    #[serde(default)]
    metric: LabelSet,
}

async fn prometheus(
    client: &Client,
    feed: &PrometheusFeed,
    timeout: Duration,
) -> Feed<Vec<LabelSet>> {
    let query: String = url::form_urlencoded::byte_serialize(feed.query.as_bytes()).collect();
    let path = format!("{}?query={query}", feed.path);
    match fetch_json::<QueryReply>(client, &feed.endpoint, &path, timeout).await {
        Ok(QueryReply {
            status,
            data: Some(data),
        }) if status == "success" => Feed::Available {
            items: data.result.into_iter().map(|s| s.metric).collect(),
        },
        Ok(_) => Unavailable::new(FeedFault::QueryFailed).feed(),
        Err(unavailable) => unavailable.feed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, ingress::Delivery, store::Store};
    use axum::{
        Json, Router,
        extract::{Query, State},
        http::{HeaderMap, StatusCode},
        response::{IntoResponse, Response},
        routing::get,
    };
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    const FEED_TOKEN: &str = "grafana-service-account-token-0123456789";

    /// What the stub feeds answer.
    struct Stub {
        alerts: std::result::Result<Value, StatusCode>,
        series: Value,
        /// Authorization headers seen by the (anonymous) health endpoint.
        health_auth: Vec<String>,
        query: Vec<String>,
    }

    type Shared = Arc<Mutex<Stub>>;

    async fn healthz(State(stub): State<Shared>, headers: HeaderMap) -> StatusCode {
        if let Some(value) = headers.get("authorization") {
            stub.lock()
                .unwrap()
                .health_auth
                .push(value.to_str().unwrap_or_default().to_string());
        }
        StatusCode::OK
    }

    async fn alerts(State(stub): State<Shared>, headers: HeaderMap) -> Response {
        let bearer = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        if bearer != format!("Bearer {FEED_TOKEN}") {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        match &stub.lock().unwrap().alerts {
            Ok(value) => Json(value.clone()).into_response(),
            Err(status) => status.into_response(),
        }
    }

    async fn query(
        State(stub): State<Shared>,
        Query(params): Query<std::collections::BTreeMap<String, String>>,
    ) -> Json<Value> {
        let mut stub = stub.lock().unwrap();
        stub.query
            .push(params.get("query").cloned().unwrap_or_default());
        Json(stub.series.clone())
    }

    fn firing(labels: Value) -> Value {
        json!({"status":"success","data":{"resultType":"vector","result":[{"metric":labels,"value":[1,"1"]}]}})
    }

    fn alert(state: &str, silenced: &[&str], annotation: &str) -> Value {
        json!([{"fingerprint":"f1","labels":{"alertname":"DiskFull","severity":"page"},
            "annotations":{"value":annotation},"startsAt":"2026-09-26T00:00:00Z",
            "status":{"state":state,"silencedBy":silenced,"inhibitedBy":[]},
            "receivers":[{"name":"default"}],"updatedAt":"2026-09-26T00:00:01Z"}])
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        dir: std::path::PathBuf,
        stub: Shared,
        events: Events,
        source: MonitorSource,
        server: tokio::task::JoinHandle<()>,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    async fn fixture() -> Fixture {
        let stub = Arc::new(Mutex::new(Stub {
            alerts: Ok(alert("active", &[], "85")),
            series: firing(json!({"alertname":"DiskFull","alertstate":"firing"})),
            health_auth: vec![],
            query: vec![],
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new()
            .route("/healthz", get(healthz))
            .route("/api/alertmanager/grafana/api/v2/alerts", get(alerts))
            .route("/api/v1/query", get(query))
            .with_state(stub.clone());
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        // A port nobody listens on: connection refused, i.e. down.
        let refused = {
            let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}", probe.local_addr().unwrap())
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let operator = path.join("operator.token");
        std::fs::write(&operator, "operator-token-0123456789-abcdef").unwrap();
        let feed_token = path.join("grafana-sa.token");
        std::fs::write(&feed_token, FEED_TOKEN).unwrap();
        let config: Config = serde_json::from_value(json!({
            "service_id":"events","listen":"127.0.0.1:0","data_dir":path,
            "principals":[{"id":"operator","token_file":operator,"roles":["admin"]}],
            "endpoints":{
                "health-core":{"url":url,"anonymous":true},
                "health-gone":{"url":refused,"anonymous":true},
                "grafana":{"url":url,"token_file":feed_token},
                "metrics":{"url":url,"anonymous":true}
            },
            "routes":[{"id":"vesta-monitoring","source":"roman-alerts","event_type":"monitoring.changed",
                "automation":"vesta","coalesce_pending":true}],
            "monitors":[{"source":"roman-alerts","event_type":"monitoring.changed","interval_secs":60,
                "timeout_secs":2,
                "health":[{"name":"core","endpoint":"health-core"},{"name":"gone","endpoint":"health-gone"}],
                "alertmanager":{"endpoint":"grafana","path":"/api/alertmanager/grafana/api/v2/alerts"},
                "prometheus":{"endpoint":"metrics","query":"ALERTS{alertstate=\"firing\"}"}}]
        }))
        .unwrap();
        config.validate().unwrap();
        let source = config.monitors[0].clone();
        let events = Events::new(
            Arc::new(config),
            Store::open(&path.join("events.redb")).unwrap(),
        )
        .unwrap();
        Fixture {
            _dir: dir,
            dir: path,
            stub,
            events,
            source,
            server,
        }
    }

    impl Fixture {
        async fn poll(&self) -> PollOutcome {
            poll(&self.events, &self.source).await.unwrap()
        }
        fn reading(&self) -> MonitorReading {
            self.events
                .store
                .get(&snapshot_key("roman-alerts"))
                .unwrap()
                .unwrap()
        }
        fn delivery(&self, id: &str) -> Delivery {
            self.events
                .store
                .get(&key("delivery", &[id]))
                .unwrap()
                .unwrap()
        }
    }

    #[tokio::test]
    async fn baseline_then_structural_changes_emit_one_event_each() {
        let f = fixture().await;
        assert_eq!(f.poll().await, PollOutcome::Baseline);
        let baseline = f.reading();
        assert_eq!(baseline.transitions, 0);
        assert_eq!(baseline.feeds.health["core"], HealthStatus::Up);
        assert_eq!(baseline.feeds.health["gone"], HealthStatus::Down);
        assert_eq!(
            baseline.snapshot.alertmanager.items().unwrap()[0]
                .status
                .state,
            crate::types::monitor::AlertState::Active
        );
        assert_eq!(baseline.snapshot.prometheus.items().unwrap().len(), 1);
        // Anonymous endpoints never receive a credential; the query is encoded.
        assert!(f.stub.lock().unwrap().health_auth.is_empty());
        assert_eq!(
            f.stub.lock().unwrap().query[0],
            "ALERTS{alertstate=\"firing\"}"
        );

        // A new annotation value is not an incident transition.
        f.stub.lock().unwrap().alerts = Ok(alert("active", &[], "86"));
        assert_eq!(f.poll().await, PollOutcome::Unchanged);
        let refreshed = f.reading();
        assert_eq!(refreshed.signature, baseline.signature);
        assert_eq!(
            refreshed.snapshot.alertmanager.items().unwrap()[0].annotations["value"],
            "86"
        );

        // Silencing is.
        f.stub.lock().unwrap().alerts = Ok(alert("suppressed", &["silence-1"], "86"));
        let PollOutcome::Changed {
            event_id: silenced_id,
            deliveries,
        } = f.poll().await
        else {
            panic!("silencing must change the signature")
        };
        assert_eq!(deliveries.len(), 1);
        let delivery = f.delivery(&deliveries[0]);
        let data: MonitorChange =
            serde_json::from_value(delivery.request.event.data.clone()).unwrap();
        assert_eq!(data.transition, 1);
        assert_eq!(data.previous_signature, baseline.signature);
        assert_eq!(
            data.previous.alertmanager.items().unwrap()[0].status.state,
            crate::types::monitor::AlertState::Active
        );
        assert_eq!(
            data.current.alertmanager.items().unwrap()[0]
                .status
                .silenced_by,
            ["silence-1"]
        );
        assert_eq!(delivery.request.event.event_type, "monitoring.changed");
        assert_eq!(silenced_id, event_id(&f.reading().signature, 1));

        // Losing the alert feed is unknown, not zero alerts, and is a change.
        f.stub.lock().unwrap().alerts = Err(StatusCode::SERVICE_UNAVAILABLE);
        assert!(matches!(f.poll().await, PollOutcome::Changed { .. }));
        let lost = f.reading();
        assert_eq!(
            lost.snapshot.alertmanager,
            Feed::Unknown {
                reason: FeedFault::HttpStatus,
                http_status: Some(503)
            }
        );

        // The silenced state recurs: a fresh event ID, not a reused-ID refusal.
        f.stub.lock().unwrap().alerts = Ok(alert("suppressed", &["silence-1"], "86"));
        let PollOutcome::Changed {
            event_id: recurred, ..
        } = f.poll().await
        else {
            panic!("recovery of the feed must change the signature")
        };
        assert_ne!(recurred, silenced_id);
        assert_eq!(f.reading().transitions, 3);

        // The coalescing route keeps only the newest undelivered change.
        f.events.coalesce_pending().unwrap();
        let pending: Vec<Delivery> = f
            .events
            .store
            .scan::<Delivery>("delivery/")
            .unwrap()
            .into_iter()
            .filter(|d| !d.terminal)
            .collect();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].request.event.id, recurred);
    }

    #[tokio::test]
    async fn unreadable_credentials_and_failed_queries_are_unknown() {
        let f = fixture().await;
        std::fs::remove_file(f.dir.join("grafana-sa.token")).unwrap();
        f.stub.lock().unwrap().series = json!({"status":"error","errorType":"bad_data"});
        let snapshot = observe(f.events.client(), &f.source).await;
        assert_eq!(
            snapshot.alertmanager,
            Feed::Unknown {
                reason: FeedFault::CredentialUnavailable,
                http_status: None
            }
        );
        assert_eq!(
            snapshot.prometheus,
            Feed::Unknown {
                reason: FeedFault::QueryFailed,
                http_status: None
            }
        );
        std::fs::write(
            f.dir.join("grafana-sa.token"),
            "wrong-token-0123456789-abcdefgh",
        )
        .unwrap();
        let snapshot = observe(f.events.client(), &f.source).await;
        assert_eq!(
            snapshot.alertmanager,
            Feed::Unknown {
                reason: FeedFault::Unauthorized,
                http_status: Some(401)
            }
        );
    }

    #[tokio::test]
    async fn monitors_endpoint_serves_the_reading_to_operator_and_granted_event_roles() {
        use axum::body::Body;
        use tower::ServiceExt;
        let f = fixture().await;
        let tokens = [
            (
                "reader",
                "monitor-reader-token-0123456789",
                json!(["operator"]),
                json!([]),
            ),
            (
                "observer",
                "observer-event-token-0123456789",
                json!(["event"]),
                json!(["roman-alerts"]),
            ),
            (
                "other",
                "other-event-token-0123456789-ab",
                json!(["event"]),
                json!(["roman-anima"]),
            ),
            (
                "runtime",
                "runtime-token-0123456789-abcdef",
                json!(["runtime"]),
                json!(["roman-alerts"]),
            ),
        ];
        let mut config = (*f.events.config).clone();
        for (id, token, roles, sources) in &tokens {
            let file = f.dir.join(format!("{id}.token"));
            std::fs::write(&file, token).unwrap();
            config.principals.push(
                serde_json::from_value(
                    json!({"id":id,"token_file":file,"roles":roles,"sources":sources}),
                )
                .unwrap(),
            );
        }
        let events = Events::new(Arc::new(config), f.events.store.clone()).unwrap();
        let router = crate::server::event_router(events.clone());
        let get = |path: &str, token: &str| {
            axum::http::Request::get(path)
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap()
        };
        let status = |response: Response| response.status();
        // Configured but not yet observed.
        let response = router
            .clone()
            .oneshot(get("/v1/monitors/roman-alerts", tokens[0].1))
            .await
            .unwrap();
        assert_eq!(status(response), StatusCode::NOT_FOUND);
        poll(&events, &f.source).await.unwrap();
        for (token, expected) in [
            (tokens[0].1, StatusCode::OK),
            (tokens[1].1, StatusCode::OK),
            (tokens[2].1, StatusCode::FORBIDDEN),
            (tokens[3].1, StatusCode::FORBIDDEN),
            (
                "unknown-token-0123456789-abcdefgh",
                StatusCode::UNAUTHORIZED,
            ),
        ] {
            let response = router
                .clone()
                .oneshot(get("/v1/monitors/roman-alerts", token))
                .await
                .unwrap();
            assert_eq!(status(response), expected, "{token}");
        }
        let response = router
            .clone()
            .oneshot(get("/v1/monitors/roman-alerts", tokens[1].1))
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let reading: MonitorReading = serde_json::from_slice(&body).unwrap();
        assert_eq!(reading.source, "roman-alerts");
        assert_eq!(reading.feeds.health["core"], HealthStatus::Up);
        let response = router
            .oneshot(get("/v1/monitors/unconfigured", tokens[0].1))
            .await
            .unwrap();
        assert_eq!(status(response), StatusCode::NOT_FOUND);
    }

    fn validate(monitor: Value, endpoints: Value) -> std::result::Result<(), Fault> {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("t");
        std::fs::write(&token, "operator-token-0123456789-abcdef").unwrap();
        let config: Config = serde_json::from_value(json!({"service_id":"events",
            "listen":"127.0.0.1:0","data_dir":dir.path(),
            "principals":[{"id":"operator","token_file":token,"roles":["admin"]}],
            "endpoints":endpoints,"monitors":[monitor.clone(), monitor]}))
        .map_err(|e| Fault::invalid(e.to_string()))?;
        let mut single = config.clone();
        single.monitors.truncate(1);
        single.validate()?;
        config.validate()
    }

    #[test]
    fn monitor_configuration_is_validated_with_diagnostics() {
        let endpoints = json!({"health":{"url":"http://127.0.0.1:1","anonymous":true},
            "grafana":{"url":"https://grafana.example","token_file":"/secrets/grafana-sa.token",
                "ca_file":"/secrets/lab-ca.pem"}});
        let good = json!({"source":"roman-alerts","event_type":"monitoring.changed",
            "health":[{"name":"core","endpoint":"health"}],
            "alertmanager":{"endpoint":"grafana","path":"/api/alertmanager/grafana/api/v2/alerts"},
            "prometheus":{"endpoint":"health","query":"ALERTS"}});
        // Two copies of one source: the single one validates, the pair does not.
        let duplicate = validate(good.clone(), endpoints.clone()).unwrap_err();
        assert!(
            duplicate
                .message
                .contains("akh::events::monitor::duplicate_source"),
            "{}",
            duplicate.message
        );
        let with = |patch: Value| {
            let mut monitor = good.clone();
            monitor
                .as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            monitor
        };
        let single = |monitor: Value| -> std::result::Result<(), Fault> {
            let dir = tempfile::tempdir().unwrap();
            let token = dir.path().join("t");
            std::fs::write(&token, "operator-token-0123456789-abcdef").unwrap();
            let config: Config = serde_json::from_value(json!({"service_id":"events",
                "listen":"127.0.0.1:0","data_dir":dir.path(),
                "principals":[{"id":"operator","token_file":token,"roles":["admin"]}],
                "endpoints":endpoints.clone(),"monitors":[monitor]}))
            .map_err(|e| Fault::invalid(e.to_string()))?;
            config.validate()
        };
        single(good.clone()).unwrap();
        for (patch, code) in [
            (json!({"interval_secs":5}), "interval"),
            (json!({"timeout_secs":60,"interval_secs":60}), "timeout"),
            (
                json!({"health":[],"alertmanager":null,"prometheus":null}),
                "empty",
            ),
            (
                json!({"health":[{"name":"a","endpoint":"health"},{"name":"a","endpoint":"health"}]}),
                "duplicate_health",
            ),
            (
                json!({"health":[{"name":"a","endpoint":"missing"}]}),
                "endpoint",
            ),
            (
                json!({"health":[{"name":"a","endpoint":"health","path":"healthz"}]}),
                "path",
            ),
            (
                json!({"prometheus":{"endpoint":"health","query":" "}}),
                "query",
            ),
            (
                json!({"prometheus":{"endpoint":"health","query":"ALERTS","path":"/api/v1/query?x=1"}}),
                "prometheus_path",
            ),
            (json!({"event_type":""}), "identifier"),
        ] {
            let error = single(with(patch.clone())).unwrap_err();
            assert_eq!(error.code, "invalid_request");
            assert!(
                error
                    .message
                    .contains(&format!("akh::events::monitor::{code}]")),
                "{patch}: {}",
                error.message
            );
            assert!(error.message.contains("help: "), "{}", error.message);
        }
        // Unknown monitor keys are refused by serde.
        assert!(single(with(json!({"extra":1}))).is_err());
        // An anonymous endpoint cannot also name a credential.
        let bad_endpoint = json!({"health":{"url":"http://127.0.0.1:1","anonymous":true,
            "token_file":"/secrets/t"}});
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("t");
        std::fs::write(&token, "operator-token-0123456789-abcdef").unwrap();
        let config: Config = serde_json::from_value(json!({"service_id":"events",
            "listen":"127.0.0.1:0","data_dir":dir.path(),
            "principals":[{"id":"operator","token_file":token,"roles":["admin"]}],
            "endpoints":bad_endpoint}))
        .unwrap();
        assert!(config.validate().is_err());
    }
}
