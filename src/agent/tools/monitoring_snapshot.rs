//! `monitoring_snapshot`: read-only view of an akheventd monitor source
//! (Phase 53g PR 5, ADR 113 decision 10; Anima WI `01a0d569-1cf1`, plan
//! `docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md`).
//!
//! Replaces the Roman observer's `mcp__observer__monitoring_snapshot`. The
//! tool reads `GET {endpoint}/v1/monitors/{source}` from akheventd and returns
//! the same top-level keys the observer's MCP tool returned, so prompts and
//! ledgers written against it keep working:
//!
//! - `monitoring` = `{local_services, grafana, victoriametrics}`, projected
//!   from the structural snapshot (health endpoints, the Alertmanager feed and
//!   the Prometheus `ALERTS` feed). An unreadable feed is `available: false`,
//!   never an empty alert list.
//! - `monitoring_at`, `read_at`, `read_at_utc`.
//! - the observer's own status keys (`anima_status`, `coverage`, `heartbeat`,
//!   `workflow_counts`, `role_activity`, `workflow_failure`,
//!   `provider_availability`, `ingress_status`, `delivery_watch_status`,
//!   `board_watchdog_status`, `laya_status`), read from the file the observer
//!   exports until PR 6 moves those producers into akheventd too.
//!
//! Registered only for akhs whose workspace TOML has a `[monitoring]` table.
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

use akh_session_types::monitor::{Feed, FeedAvailability, MonitorReading};

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolSource,
};
use crate::engine::Engine;

/// Tool name. Kept identical to the observer's MCP tool name.
pub const TOOL_NAME: &str = "monitoring_snapshot";
/// Per-request deadline for the akheventd read.
const READ_TIMEOUT: Duration = Duration::from_secs(5);
/// The observer-owned status keys the old MCP tool returned, in its order.
pub const OBSERVER_STATUS_KEYS: [&str; 11] = [
    "anima_status",
    "coverage",
    "heartbeat",
    "workflow_counts",
    "role_activity",
    "workflow_failure",
    "provider_availability",
    "ingress_status",
    "delivery_watch_status",
    "board_watchdog_status",
    "laya_status",
];

/// A workspace's `[monitoring]` table.
///
/// ```toml
/// [monitoring]
/// endpoint = "http://127.0.0.1:18435"
/// source = "roman-alerts"
/// token_file = "/…/secrets/monitoring-reader.key"
/// observer_status_file = "/…/state/observer-status.json"
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonitoringConfig {
    /// akheventd base URL (`https://…`, or `http://` on loopback only).
    pub endpoint: String,
    /// The akheventd monitor source to read.
    pub source: String,
    /// Bearer credential of an akheventd principal with the `operator` role
    /// (or the `event` role granted `source`).
    pub token_file: PathBuf,
    /// Optional JSON status export of the Roman observer (Phase 53g PR 5
    /// bridge until PR 6); its keys are merged into the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observer_status_file: Option<PathBuf>,
}

/// A `[monitoring]` table that cannot work as written.
#[derive(Debug, Error, Diagnostic)]
pub enum MonitoringConfigError {
    #[error("akh `{akh}` [monitoring] endpoint `{endpoint}` is invalid: {reason}")]
    #[diagnostic(
        code(akh::monitoring::config::endpoint),
        help(
            "Set `endpoint` to akheventd's base URL, e.g. `http://127.0.0.1:18435`, without a path."
        )
    )]
    Endpoint {
        akh: String,
        endpoint: String,
        reason: &'static str,
    },
    #[error("akh `{akh}` [monitoring] endpoint `{endpoint}` sends its token over plain http")]
    #[diagnostic(
        code(akh::monitoring::config::insecure_http),
        help("Use https:// or a loopback host (localhost, 127.0.0.1, [::1]).")
    )]
    InsecureHttp { akh: String, endpoint: String },
    #[error("akh `{akh}` [monitoring] source `{source_id}` is not a valid identifier")]
    #[diagnostic(
        code(akh::monitoring::config::source),
        help(
            "Name an akheventd `monitors[].source`, e.g. `roman-alerts` (1..256 characters, no control characters)."
        )
    )]
    Source { akh: String, source_id: String },
    #[error("akh `{akh}` [monitoring] {field} `{path}` is not absolute")]
    #[diagnostic(
        code(akh::monitoring::config::relative_path),
        help("Use an absolute path; akhomed's working directory is not stable.")
    )]
    RelativePath {
        akh: String,
        field: &'static str,
        path: PathBuf,
    },
}

impl MonitoringConfig {
    /// Static checks run when the workspace TOML loads (no secret is read).
    pub fn validate(&self, akh: &str) -> Result<(), MonitoringConfigError> {
        let invalid = |reason: &'static str| MonitoringConfigError::Endpoint {
            akh: akh.into(),
            endpoint: self.endpoint.clone(),
            reason,
        };
        let rest = self
            .endpoint
            .strip_prefix("https://")
            .or_else(|| self.endpoint.strip_prefix("http://"))
            .ok_or_else(|| invalid("the scheme must be http:// or https://"))?;
        let authority = rest.trim_end_matches('/');
        if authority.is_empty() || authority.starts_with('/') {
            return Err(invalid("it has no host"));
        }
        if authority.contains(['/', '?', '#', '@']) {
            return Err(invalid(
                "it must be a base URL without path, query or credentials",
            ));
        }
        if self.endpoint.starts_with("http://")
            && !super::mcp_config::is_loopback_authority(authority)
        {
            return Err(MonitoringConfigError::InsecureHttp {
                akh: akh.into(),
                endpoint: self.endpoint.clone(),
            });
        }
        if self.source.is_empty()
            || self.source.len() > 256
            || self.source.chars().any(char::is_control)
        {
            return Err(MonitoringConfigError::Source {
                akh: akh.into(),
                source_id: self.source.clone(),
            });
        }
        let relative = |field: &'static str, path: &PathBuf| MonitoringConfigError::RelativePath {
            akh: akh.into(),
            field,
            path: path.clone(),
        };
        if !self.token_file.is_absolute() {
            return Err(relative("token_file", &self.token_file));
        }
        if let Some(path) = &self.observer_status_file
            && !path.is_absolute()
        {
            return Err(relative("observer_status_file", path));
        }
        Ok(())
    }

    fn url(&self) -> String {
        format!(
            "{}/v1/monitors/{}",
            self.endpoint.trim_end_matches('/'),
            percent_segment(&self.source)
        )
    }
}

/// Why one read failed. Rendered into the tool's error output with its help.
#[derive(Debug, Error, Diagnostic)]
pub enum MonitoringSnapshotError {
    #[error("the monitoring credential at {path} could not be read")]
    #[diagnostic(
        code(akh::monitoring::credential),
        help(
            "Create the token file named by [monitoring] token_file (mode 0600) and restart nothing: it is read per call."
        )
    )]
    Credential { path: PathBuf },
    #[error("akheventd at {url} is unreachable: {message}")]
    #[diagnostic(
        code(akh::monitoring::unreachable),
        help(
            "Monitoring state is unknown, not healthy. Check that akheventd is running and listening on the configured endpoint."
        )
    )]
    Unreachable { url: String, message: String },
    #[error("akheventd refused the monitor read with HTTP {status} ({code})")]
    #[diagnostic(
        code(akh::monitoring::refused),
        help(
            "401/403: the token is not an akheventd principal with the operator role (or the event role granted this source). 404: the source is not configured or has not finished its first poll yet."
        )
    )]
    Refused {
        status: u16,
        code: String,
        message: String,
    },
    #[error("akheventd returned an unexpected monitor reading: {message}")]
    #[diagnostic(
        code(akh::monitoring::decode),
        help(
            "akhomed and akheventd disagree on the monitor wire shape; deploy both from the same origin/main commit."
        )
    )]
    Decode { message: String },
}

impl MonitoringSnapshotError {
    fn render(&self) -> String {
        let code = self.code().map(|c| c.to_string()).unwrap_or_default();
        let help = self.help().map(|h| h.to_string()).unwrap_or_default();
        format!("{self} [{code}] help: {help}")
    }
}

/// Percent-encode a single path segment.
fn percent_segment(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

/// RFC 3339 UTC (seconds) without the optional `chrono` dependency.
fn utc(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let rest = seconds % 86_400;
    let (y, m, d) = crate::agent::tasks::civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rest / 3600,
        (rest % 3600) / 60,
        rest % 60
    )
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Project a structural reading onto the observer's `monitoring` value.
///
/// Key names follow the Roman observer (`grafana` = the Alertmanager feed,
/// `victoriametrics` = the Prometheus `ALERTS` feed) because prompts and
/// ledgers were written against them.
pub fn legacy_monitoring(reading: &MonitorReading) -> Value {
    let snapshot = &reading.snapshot;
    let local_services: Map<String, Value> = snapshot
        .health
        .iter()
        .map(|(name, observation)| (name.clone(), json!(observation.status)))
        .collect();
    let unavailable = |configured: bool, error: &str, reason: Value, status: Option<u16>| {
        let mut value = json!({"available": false, "error": error});
        if !configured {
            value["configured"] = json!(false);
        }
        if !reason.is_null() {
            value["reason"] = reason;
        }
        if let Some(status) = status {
            value["http_status"] = json!(status);
        }
        value
    };
    let grafana = match &snapshot.alertmanager {
        Feed::Available { items } => json!({"available": true, "alerts": items}),
        Feed::Unknown {
            reason,
            http_status,
        } => unavailable(
            true,
            "Grafana alert feed unavailable; alert state unknown, not zero",
            json!(reason),
            *http_status,
        ),
        Feed::NotConfigured => unavailable(
            false,
            "no alert feed is configured for this monitor source",
            Value::Null,
            None,
        ),
    };
    let victoriametrics = match &snapshot.prometheus {
        Feed::Available { items } => json!({"available": true, "firing": items}),
        Feed::Unknown {
            reason,
            http_status,
        } => unavailable(
            true,
            "VictoriaMetrics unavailable",
            json!(reason),
            *http_status,
        ),
        Feed::NotConfigured => unavailable(
            false,
            "no metrics feed is configured for this monitor source",
            Value::Null,
            None,
        ),
    };
    json!({
        "local_services": local_services,
        "grafana": grafana,
        "victoriametrics": victoriametrics,
    })
}

/// The full tool result for a reading, the observer export (if any) and the
/// read time.
pub fn snapshot_output(
    reading: &MonitorReading,
    observer: ObserverStatus,
    read_at: u64,
) -> Map<String, Value> {
    let mut data = Map::new();
    data.insert("monitoring".into(), legacy_monitoring(reading));
    data.insert("monitoring_at".into(), json!(reading.observed_at_ms / 1000));
    let observer_status = match observer {
        ObserverStatus::Available { exported_at, state } => {
            for key in OBSERVER_STATUS_KEYS {
                if let Some(value) = state.get(key) {
                    data.insert(key.into(), value.clone());
                }
            }
            json!({"available": true, "exported_at": exported_at})
        }
        ObserverStatus::NotConfigured => json!({"available": false, "configured": false}),
        ObserverStatus::Unavailable => json!({"available": false}),
    };
    data.insert("observer_status".into(), observer_status);
    data.insert("read_at".into(), json!(read_at));
    data.insert("read_at_utc".into(), json!(utc(read_at)));
    let availability = |feed: FeedAvailability| json!(feed);
    data.insert(
        "monitor".into(),
        json!({
            "source": reading.source,
            "signature": reading.signature,
            "transitions": reading.transitions,
            "changed_at_utc": utc(reading.changed_at_ms / 1000),
            "observed_at_utc": utc(reading.observed_at_ms / 1000),
            "alertmanager": availability(reading.feeds.alertmanager),
            "prometheus": availability(reading.feeds.prometheus),
        }),
    );
    data
}

/// The observer's exported status, as read for one call.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserverStatus {
    /// No `observer_status_file` is configured.
    NotConfigured,
    /// Configured but missing or unreadable: those keys are unknown.
    Unavailable,
    Available {
        exported_at: Option<u64>,
        state: Map<String, Value>,
    },
}

#[derive(Deserialize)]
struct ObserverExport {
    #[serde(default)]
    exported_at: Option<u64>,
    state: Map<String, Value>,
}

fn read_observer_status(path: Option<&PathBuf>) -> ObserverStatus {
    let Some(path) = path else {
        return ObserverStatus::NotConfigured;
    };
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<ObserverExport>(&bytes).ok())
        .map_or(ObserverStatus::Unavailable, |export| {
            ObserverStatus::Available {
                exported_at: export.exported_at,
                state: export.state,
            }
        })
}

#[derive(Deserialize)]
struct FaultBody {
    #[serde(default)]
    code: String,
    #[serde(default)]
    message: String,
}

/// Read the monitor reading from akheventd.
pub fn fetch_reading(config: &MonitoringConfig) -> Result<MonitorReading, MonitoringSnapshotError> {
    let token = std::fs::read_to_string(&config.token_file)
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty() && !t.chars().any(char::is_whitespace))
        .ok_or_else(|| MonitoringSnapshotError::Credential {
            path: config.token_file.clone(),
        })?;
    let url = config.url();
    let agent = ureq::AgentBuilder::new()
        .timeout(READ_TIMEOUT)
        .redirects(0)
        .build();
    match agent
        .get(&url)
        .set("Authorization", &format!("Bearer {token}"))
        .call()
    {
        Ok(response) => response.into_json::<MonitorReading>().map_err(|error| {
            MonitoringSnapshotError::Decode {
                message: error.to_string(),
            }
        }),
        Err(ureq::Error::Status(status, response)) => {
            let body = response.into_json::<FaultBody>().unwrap_or(FaultBody {
                code: String::new(),
                message: String::new(),
            });
            Err(MonitoringSnapshotError::Refused {
                status,
                code: body.code,
                message: body.message,
            })
        }
        Err(ureq::Error::Transport(transport)) => Err(MonitoringSnapshotError::Unreachable {
            url,
            message: transport.to_string(),
        }),
    }
}

/// The native read-only `monitoring_snapshot` tool.
pub struct MonitoringSnapshotTool {
    config: MonitoringConfig,
}

impl MonitoringSnapshotTool {
    pub fn new(config: MonitoringConfig) -> Self {
        Self { config }
    }
}

const DESCRIPTION: &str = "Read latest alert, daemon-health, Anima-observer and coverage evidence, \
     including observation times. Unavailable feeds mean unknown state. No filesystem access.";

impl Tool for MonitoringSnapshotTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: TOOL_NAME.into(),
            description: DESCRIPTION.into(),
            parameters: vec![],
        }
    }

    fn execute(&self, _engine: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
        let reading = match fetch_reading(&self.config) {
            Ok(reading) => reading,
            Err(error) => return Ok(ToolOutput::err(error.render())),
        };
        let observer = read_observer_status(self.config.observer_status_file.as_ref());
        let data = snapshot_output(&reading, observer, now_secs());
        Ok(ToolOutput::ok(Value::Object(data).to_string()))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: TOOL_NAME.into(),
            description: DESCRIPTION.into(),
            parameters: vec![],
            danger: DangerInfo {
                effect: ToolEffect::ReadOnly,
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::Network]),
                description: "Reads akheventd's latest monitoring snapshot over loopback; \
                              changes nothing."
                    .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akh_session_types::monitor::{
        Alert, FeedFault, HealthObservation, HealthStatus, LatencyClass, MonitorSnapshot,
    };
    use std::collections::BTreeMap;
    use std::io::{Read, Write};

    fn reading(alertmanager: Feed<Vec<Alert>>) -> MonitorReading {
        let snapshot = MonitorSnapshot {
            observed_at_ms: 1_790_000_000_000,
            health: BTreeMap::from([
                (
                    "core".to_string(),
                    HealthObservation {
                        status: HealthStatus::Up,
                        latency: Some(LatencyClass::Fast),
                    },
                ),
                (
                    "events".to_string(),
                    HealthObservation {
                        status: HealthStatus::Down,
                        latency: None,
                    },
                ),
            ]),
            alertmanager,
            prometheus: Feed::Available {
                items: vec![BTreeMap::from([(
                    "alertname".to_string(),
                    "DiskFull".to_string(),
                )])],
            },
        };
        MonitorReading {
            source: "roman-alerts".into(),
            event_type: "monitoring.changed".into(),
            signature: "ab".repeat(32),
            transitions: 2,
            changed_at_ms: 1_789_999_000_000,
            observed_at_ms: snapshot.observed_at_ms,
            feeds: snapshot.feeds(),
            snapshot,
        }
    }

    fn alert() -> Alert {
        serde_json::from_value(json!({"fingerprint":"f1","labels":{"alertname":"DiskFull"},
            "annotations":{"value":"85"},"startsAt":"2026-09-26T00:00:00Z",
            "status":{"state":"active","silencedBy":[],"inhibitedBy":[]}}))
        .unwrap()
    }

    #[test]
    fn output_keeps_the_observer_tool_keys() {
        let mut state = Map::new();
        for key in OBSERVER_STATUS_KEYS {
            state.insert(key.into(), json!({"ok": true}));
        }
        state.insert("unrelated".into(), json!(1));
        let data = snapshot_output(
            &reading(Feed::Available {
                items: vec![alert()],
            }),
            ObserverStatus::Available {
                exported_at: Some(1),
                state,
            },
            1_790_000_005,
        );
        // Every key the observer's MCP tool returned is present.
        for key in OBSERVER_STATUS_KEYS.iter().chain(&[
            "monitoring",
            "monitoring_at",
            "read_at",
            "read_at_utc",
        ]) {
            assert!(data.contains_key(*key), "missing {key}");
        }
        assert!(!data.contains_key("unrelated"));
        assert_eq!(data["monitoring_at"], json!(1_790_000_000u64));
        assert_eq!(data["read_at_utc"], json!("2026-09-21T14:13:25+00:00"));
        let monitoring = &data["monitoring"];
        assert_eq!(
            monitoring["local_services"],
            json!({"core":"up","events":"down"})
        );
        assert_eq!(monitoring["grafana"]["available"], json!(true));
        let alert = &monitoring["grafana"]["alerts"][0];
        for key in ["fingerprint", "labels", "annotations", "status", "startsAt"] {
            assert!(alert.get(key).is_some(), "alert lacks {key}");
        }
        assert_eq!(alert["status"]["state"], json!("active"));
        assert_eq!(alert["status"]["silencedBy"], json!([]));
        assert_eq!(
            monitoring["victoriametrics"],
            json!({"available": true, "firing": [{"alertname": "DiskFull"}]})
        );
    }

    #[test]
    fn unknown_feed_is_unavailable_not_zero_alerts() {
        let data = snapshot_output(
            &reading(Feed::Unknown {
                reason: FeedFault::Unauthorized,
                http_status: Some(401),
            }),
            ObserverStatus::Unavailable,
            0,
        );
        let grafana = &data["monitoring"]["grafana"];
        assert_eq!(grafana["available"], json!(false));
        assert!(grafana.get("alerts").is_none());
        assert_eq!(grafana["reason"], json!("unauthorized"));
        assert_eq!(grafana["http_status"], json!(401));
        assert_eq!(data["observer_status"], json!({"available": false}));
        let empty = snapshot_output(
            &reading(Feed::Available { items: vec![] }),
            ObserverStatus::NotConfigured,
            0,
        );
        assert_eq!(
            empty["monitoring"]["grafana"],
            json!({"available": true, "alerts": []})
        );
    }

    #[test]
    fn config_validation_names_the_problem() {
        let good = MonitoringConfig {
            endpoint: "http://127.0.0.1:18435".into(),
            source: "roman-alerts".into(),
            token_file: "/secrets/monitoring-reader.key".into(),
            observer_status_file: Some("/state/observer-status.json".into()),
        };
        good.validate("vesta").unwrap();
        assert_eq!(
            good.url(),
            "http://127.0.0.1:18435/v1/monitors/roman-alerts"
        );
        let cases: [(MonitoringConfig, &str); 5] = [
            (
                MonitoringConfig {
                    endpoint: "http://events.lab.home".into(),
                    ..good.clone()
                },
                "akh::monitoring::config::insecure_http",
            ),
            (
                MonitoringConfig {
                    endpoint: "http://127.0.0.1:18435/v1".into(),
                    ..good.clone()
                },
                "akh::monitoring::config::endpoint",
            ),
            (
                MonitoringConfig {
                    source: String::new(),
                    ..good.clone()
                },
                "akh::monitoring::config::source",
            ),
            (
                MonitoringConfig {
                    token_file: "secrets/k".into(),
                    ..good.clone()
                },
                "akh::monitoring::config::relative_path",
            ),
            (
                MonitoringConfig {
                    observer_status_file: Some("state/o.json".into()),
                    ..good.clone()
                },
                "akh::monitoring::config::relative_path",
            ),
        ];
        for (config, code) in cases {
            let error = config.validate("vesta").unwrap_err();
            assert_eq!(error.code().unwrap().to_string(), code);
        }
    }

    /// Serve one canned HTTP response on loopback; return the base URL and
    /// the request line + headers the tool sent.
    fn serve_once(status: &str, body: String) -> (String, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let status = status.to_string();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = stream.read(&mut buffer).unwrap();
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..n]);
            }
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            String::from_utf8_lossy(&request).into_owned()
        });
        (url, handle)
    }

    fn config(dir: &std::path::Path, endpoint: String) -> MonitoringConfig {
        let token = dir.join("monitoring-reader.key");
        std::fs::write(&token, "monitor-reader-token-0123456789\n").unwrap();
        MonitoringConfig {
            endpoint,
            source: "roman-alerts".into(),
            token_file: token,
            observer_status_file: None,
        }
    }

    #[test]
    fn fetch_sends_the_bearer_and_decodes_the_reading() {
        let dir = tempfile::tempdir().unwrap();
        let body = serde_json::to_string(&reading(Feed::Available {
            items: vec![alert()],
        }))
        .unwrap();
        let (url, server) = serve_once("200 OK", body);
        let config = config(dir.path(), url);
        let fetched = fetch_reading(&config).unwrap();
        assert_eq!(fetched.source, "roman-alerts");
        let request = server.join().unwrap();
        assert!(request.starts_with("GET /v1/monitors/roman-alerts HTTP/1.1"));
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer monitor-reader-token-0123456789")
        );
    }

    #[test]
    fn refusals_and_missing_credentials_are_typed_errors() {
        let dir = tempfile::tempdir().unwrap();
        let (url, server) = serve_once(
            "403 Forbidden",
            json!({"code":"forbidden","message":"operator role required"}).to_string(),
        );
        let config = config(dir.path(), url);
        match fetch_reading(&config) {
            Err(MonitoringSnapshotError::Refused { status, code, .. }) => {
                assert_eq!((status, code.as_str()), (403, "forbidden"));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        server.join().unwrap();
        std::fs::remove_file(&config.token_file).unwrap();
        let error = fetch_reading(&config).unwrap_err();
        assert!(matches!(error, MonitoringSnapshotError::Credential { .. }));
        assert!(error.render().contains("akh::monitoring::credential"));
    }

    #[test]
    fn manifest_is_read_only_and_safe() {
        let tool = MonitoringSnapshotTool::new(MonitoringConfig {
            endpoint: "http://127.0.0.1:18435".into(),
            source: "roman-alerts".into(),
            token_file: "/k".into(),
            observer_status_file: None,
        });
        let manifest = tool.manifest();
        assert_eq!(manifest.name, "monitoring_snapshot");
        assert_eq!(manifest.danger.effect, ToolEffect::ReadOnly);
        assert_eq!(manifest.danger.level, DangerLevel::Safe);
        assert_eq!(tool.signature().name, manifest.name);
    }

    #[test]
    fn observer_export_is_read_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observer-status.json");
        assert_eq!(
            read_observer_status(Some(&path)),
            ObserverStatus::Unavailable
        );
        std::fs::write(
            &path,
            json!({"exported_at": 5, "state": {"heartbeat": 4}}).to_string(),
        )
        .unwrap();
        match read_observer_status(Some(&path)) {
            ObserverStatus::Available { exported_at, state } => {
                assert_eq!(exported_at, Some(5));
                assert_eq!(state["heartbeat"], json!(4));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(read_observer_status(None), ObserverStatus::NotConfigured);
    }
}
