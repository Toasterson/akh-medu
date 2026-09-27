//! Structural monitoring snapshots (Phase 53g PR 5, ADR 113 decision 10).
//!
//! akheventd's monitor source observes health endpoints, an Alertmanager
//! alert feed and a Prometheus `ALERTS` query, and records one
//! [`MonitorSnapshot`] per poll. akhomed's native `monitoring_snapshot` tool
//! reads the latest [`MonitorReading`] back over `GET /v1/monitors/{source}`.
//! Both sides share these types so the wire shape cannot drift.
//!
//! Two rules shape every type here:
//!
//! - **An unavailable feed is unknown, never empty.** [`Feed::Unknown`] and
//!   `Feed::Available { items: [] }` are different states, and their
//!   signatures differ, so losing the alert feed is itself a change.
//! - **Change detection is structural.** [`MonitorSnapshot::signature`] keeps
//!   only alert fingerprints, label sets, the Alertmanager status fields
//!   (`state`, `silencedBy`, `inhibitedBy`), Prometheus series label sets,
//!   per-endpoint up/down/unknown and per-feed availability. Annotations,
//!   start times, latency classes and failure reasons are shown to the reader
//!   but never wake anyone. Nothing is matched against text.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A Prometheus / Alertmanager label set. `BTreeMap` makes the order of
/// labels irrelevant to equality and to the canonical signature bytes.
pub type LabelSet = BTreeMap<String, String>;

/// What one health probe found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// The endpoint answered with a 2xx status.
    Up,
    /// The endpoint refused the connection or answered with a non-2xx status.
    Down,
    /// No answer could be obtained (timeout, missing credential, transport
    /// failure): the state is not known, which is not the same as down.
    Unknown,
}

/// Coarse response-time bucket of a probe that got an HTTP answer. Shown to
/// the reader only; it is not part of the signature, so jitter never wakes
/// an akh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyClass {
    /// Below [`LatencyClass::FAST_BELOW_MS`].
    Fast,
    /// Below [`LatencyClass::SLOW_BELOW_MS`].
    Slow,
    /// At or above [`LatencyClass::SLOW_BELOW_MS`].
    Degraded,
}

impl LatencyClass {
    /// Upper bound (exclusive) of [`LatencyClass::Fast`], in milliseconds.
    pub const FAST_BELOW_MS: u64 = 250;
    /// Upper bound (exclusive) of [`LatencyClass::Slow`], in milliseconds.
    pub const SLOW_BELOW_MS: u64 = 1_000;

    /// Bucket an elapsed time.
    pub fn classify(elapsed_ms: u64) -> Self {
        if elapsed_ms < Self::FAST_BELOW_MS {
            Self::Fast
        } else if elapsed_ms < Self::SLOW_BELOW_MS {
            Self::Slow
        } else {
            Self::Degraded
        }
    }
}

/// One health endpoint's latest observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthObservation {
    pub status: HealthStatus,
    /// Present only when an HTTP answer arrived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency: Option<LatencyClass>,
}

/// Alertmanager's `status.state` (API v2 `alertStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertState {
    Active,
    Suppressed,
    Unprocessed,
    /// A state this build does not know. Kept (rather than rejected) so a
    /// newer Alertmanager cannot make the whole feed unknown.
    #[serde(other)]
    Unrecognized,
}

/// Alertmanager's `status` object: the fields that decide whether an alert
/// is firing, silenced or inhibited.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlertStatus {
    pub state: AlertState,
    #[serde(default)]
    pub silenced_by: Vec<String>,
    #[serde(default)]
    pub inhibited_by: Vec<String>,
}

/// One Alertmanager alert, in the key shape the Roman observer exposed
/// (`fingerprint`, `labels`, `annotations`, `status`, `startsAt`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Alert {
    pub fingerprint: String,
    #[serde(default)]
    pub labels: LabelSet,
    /// Display only; not part of the signature.
    #[serde(default)]
    pub annotations: LabelSet,
    pub status: AlertStatus,
    /// Display only; not part of the signature.
    #[serde(default, rename = "startsAt", skip_serializing_if = "Option::is_none")]
    pub starts_at: Option<String>,
}

/// Why a configured feed could not be read. Display only: the signature
/// records only that the feed is unknown, so a timeout flapping into a
/// refused connection does not wake anyone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedFault {
    /// The request did not finish within the monitor's timeout.
    Timeout,
    /// The connection could not be established.
    Unreachable,
    /// The feed answered 401 or 403 (for example an expired service-account token).
    Unauthorized,
    /// The feed answered with another non-2xx status.
    HttpStatus,
    /// The body was not the expected JSON shape.
    InvalidResponse,
    /// The body exceeded the size bound.
    TooLarge,
    /// The query itself failed (Prometheus `status` other than `success`).
    QueryFailed,
    /// The endpoint's credential could not be read.
    CredentialUnavailable,
}

/// A feed's latest state. `Unknown` is never "zero items".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Feed<T> {
    /// The monitor source does not configure this feed.
    NotConfigured,
    /// The feed is configured but could not be read this poll.
    Unknown {
        reason: FeedFault,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        http_status: Option<u16>,
    },
    /// The feed answered; `items` may legitimately be empty.
    Available { items: T },
}

/// Availability of a feed without its contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedAvailability {
    NotConfigured,
    Unknown,
    Available,
}

impl<T> Feed<T> {
    pub fn availability(&self) -> FeedAvailability {
        match self {
            Self::NotConfigured => FeedAvailability::NotConfigured,
            Self::Unknown { .. } => FeedAvailability::Unknown,
            Self::Available { .. } => FeedAvailability::Available,
        }
    }
    /// The items, only when the feed answered.
    pub fn items(&self) -> Option<&T> {
        match self {
            Self::Available { items } => Some(items),
            _ => None,
        }
    }
}

/// One poll's full observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonitorSnapshot {
    pub observed_at_ms: u64,
    /// Health endpoint name -> observation.
    pub health: BTreeMap<String, HealthObservation>,
    /// Alerts sorted by fingerprint.
    pub alertmanager: Feed<Vec<Alert>>,
    /// Firing series label sets, sorted and deduplicated.
    pub prometheus: Feed<Vec<LabelSet>>,
}

/// The structural projection that decides whether monitoring changed.
///
/// Serialized with `serde_json` it is canonical: struct fields keep their
/// declaration order, maps are `BTreeMap`s, and every list is sorted, so the
/// bytes depend only on the monitored state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MonitorSignature {
    pub health: BTreeMap<String, HealthStatus>,
    pub alertmanager: FeedSignature<Vec<AlertSignature>>,
    pub prometheus: FeedSignature<Vec<LabelSet>>,
}

/// A feed's contribution to the signature: availability plus items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum FeedSignature<T> {
    NotConfigured,
    Unknown,
    Available { items: T },
}

/// An alert's contribution to the signature.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct AlertSignature {
    pub fingerprint: String,
    pub labels: LabelSet,
    pub state: AlertState,
    pub silenced_by: Vec<String>,
    pub inhibited_by: Vec<String>,
}

fn sorted(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    values
}

fn project<T, S>(feed: &Feed<T>, items: impl FnOnce(&T) -> S) -> FeedSignature<S> {
    match feed {
        Feed::NotConfigured => FeedSignature::NotConfigured,
        Feed::Unknown { .. } => FeedSignature::Unknown,
        Feed::Available { items: value } => FeedSignature::Available {
            items: items(value),
        },
    }
}

impl MonitorSnapshot {
    /// Sort every list so equal states serialize identically.
    pub fn normalized(mut self) -> Self {
        if let Feed::Available { items } = &mut self.alertmanager {
            for alert in items.iter_mut() {
                alert.status.silenced_by = sorted(std::mem::take(&mut alert.status.silenced_by));
                alert.status.inhibited_by = sorted(std::mem::take(&mut alert.status.inhibited_by));
            }
            items.sort_by(|a, b| (&a.fingerprint, &a.labels).cmp(&(&b.fingerprint, &b.labels)));
        }
        if let Feed::Available { items } = &mut self.prometheus {
            items.sort();
            items.dedup();
        }
        self
    }

    /// The structural projection (see [`MonitorSignature`]). Independent of
    /// the input order of alerts, labels, silences and series.
    pub fn signature(&self) -> MonitorSignature {
        MonitorSignature {
            health: self
                .health
                .iter()
                .map(|(name, observation)| (name.clone(), observation.status))
                .collect(),
            alertmanager: project(&self.alertmanager, |alerts| {
                let mut items: Vec<AlertSignature> = alerts
                    .iter()
                    .map(|alert| AlertSignature {
                        fingerprint: alert.fingerprint.clone(),
                        labels: alert.labels.clone(),
                        state: alert.status.state,
                        silenced_by: sorted(alert.status.silenced_by.clone()),
                        inhibited_by: sorted(alert.status.inhibited_by.clone()),
                    })
                    .collect();
                items.sort();
                items
            }),
            prometheus: project(&self.prometheus, |series| {
                let mut items = series.clone();
                items.sort();
                items.dedup();
                items
            }),
        }
    }

    /// Per-feed availability, without contents.
    pub fn feeds(&self) -> FeedsAvailability {
        FeedsAvailability {
            health: self
                .health
                .iter()
                .map(|(name, observation)| (name.clone(), observation.status))
                .collect(),
            alertmanager: self.alertmanager.availability(),
            prometheus: self.prometheus.availability(),
        }
    }
}

impl MonitorSignature {
    /// The canonical bytes hashed into the signature digest.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        // Serializing plain structs, enums, strings and BTreeMaps cannot fail.
        serde_json::to_vec(self).unwrap_or_default()
    }
}

/// Availability summary for `GET /v1/monitors/{source}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedsAvailability {
    pub health: BTreeMap<String, HealthStatus>,
    pub alertmanager: FeedAvailability,
    pub prometheus: FeedAvailability,
}

/// The persisted latest observation of one monitor source, as served by
/// `GET /v1/monitors/{source}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonitorReading {
    pub source: String,
    pub event_type: String,
    /// Hex digest of the snapshot's [`MonitorSignature`].
    pub signature: String,
    /// Signature changes recorded since the first observation (the baseline).
    pub transitions: u64,
    /// When the current signature was first observed.
    pub changed_at_ms: u64,
    pub observed_at_ms: u64,
    pub feeds: FeedsAvailability,
    pub snapshot: MonitorSnapshot,
}

/// The `data` of an emitted monitoring change event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonitorChange {
    /// The state before the change.
    pub previous: MonitorSnapshot,
    /// The state that triggered the event.
    pub current: MonitorSnapshot,
    pub previous_signature: String,
    pub signature: String,
    /// Transition number (1 for the first change after the baseline).
    pub transition: u64,
    pub observed_at_utc: String,
    /// True when alert annotations were dropped to keep the event under the
    /// ingress size bound; read `GET /v1/monitors/{source}` for them.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub annotations_elided: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn alert(value: serde_json::Value) -> Alert {
        serde_json::from_value(value).unwrap()
    }

    fn snapshot(alerts: Feed<Vec<Alert>>) -> MonitorSnapshot {
        MonitorSnapshot {
            observed_at_ms: 1,
            health: BTreeMap::from([(
                "core".to_string(),
                HealthObservation {
                    status: HealthStatus::Up,
                    latency: Some(LatencyClass::Fast),
                },
            )]),
            alertmanager: alerts,
            prometheus: Feed::Available { items: vec![] },
        }
    }

    #[test]
    fn signature_ignores_label_and_list_order() {
        let a = alert(json!({"fingerprint":"f1","labels":{"a":"1","b":"2"},
            "status":{"state":"suppressed","silencedBy":["s2","s1"],"inhibitedBy":[]}}));
        let b = alert(json!({"fingerprint":"f2","labels":{"b":"2","a":"1"},
            "status":{"state":"active"}}));
        let reordered_a = alert(json!({"labels":{"b":"2","a":"1"},"fingerprint":"f1",
            "status":{"inhibitedBy":[],"silencedBy":["s1","s2"],"state":"suppressed"}}));
        let one = snapshot(Feed::Available {
            items: vec![a, b.clone()],
        });
        let two = snapshot(Feed::Available {
            items: vec![b, reordered_a],
        });
        assert_eq!(one.signature(), two.signature());
        assert_eq!(
            one.signature().canonical_bytes(),
            two.signature().canonical_bytes()
        );
        assert_eq!(one.clone().normalized(), two.clone().normalized());
    }

    #[test]
    fn silencing_changes_the_signature_but_annotations_and_latency_do_not() {
        let active = snapshot(Feed::Available {
            items: vec![alert(json!({"fingerprint":"f","labels":{},
                "annotations":{"value":"85"},"status":{"state":"active"}}))],
        });
        let mut annotated = active.clone();
        if let Feed::Available { items } = &mut annotated.alertmanager {
            items[0].annotations.insert("value".into(), "86".into());
            items[0].starts_at = Some("2026-09-26T00:00:00Z".into());
        }
        annotated.health.get_mut("core").unwrap().latency = Some(LatencyClass::Degraded);
        annotated.observed_at_ms = 99;
        assert_eq!(active.signature(), annotated.signature());

        let mut silenced = active.clone();
        if let Feed::Available { items } = &mut silenced.alertmanager {
            items[0].status.state = AlertState::Suppressed;
            items[0].status.silenced_by = vec!["silence-1".into()];
        }
        assert_ne!(active.signature(), silenced.signature());
    }

    #[test]
    fn unknown_feed_is_not_an_empty_feed() {
        let empty = snapshot(Feed::Available { items: vec![] });
        let unknown = snapshot(Feed::Unknown {
            reason: FeedFault::Timeout,
            http_status: None,
        });
        let other_reason = snapshot(Feed::Unknown {
            reason: FeedFault::Unauthorized,
            http_status: Some(401),
        });
        assert_ne!(empty.signature(), unknown.signature());
        assert_eq!(unknown.signature(), other_reason.signature());
        assert_ne!(
            unknown.signature(),
            snapshot(Feed::NotConfigured).signature()
        );
        let wire = serde_json::to_value(&unknown.alertmanager).unwrap();
        assert_eq!(wire, json!({"state":"unknown","reason":"timeout"}));
        assert_eq!(
            serde_json::to_value(&empty.alertmanager).unwrap(),
            json!({"state":"available","items":[]})
        );
    }

    #[test]
    fn unrecognized_alert_state_does_not_fail_the_feed() {
        let a = alert(json!({"fingerprint":"f","status":{"state":"future-state"}}));
        assert_eq!(a.status.state, AlertState::Unrecognized);
    }

    #[test]
    fn latency_buckets() {
        assert_eq!(LatencyClass::classify(0), LatencyClass::Fast);
        assert_eq!(LatencyClass::classify(250), LatencyClass::Slow);
        assert_eq!(LatencyClass::classify(1_000), LatencyClass::Degraded);
    }
}
