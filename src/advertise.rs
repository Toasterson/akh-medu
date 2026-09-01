//! AkhAdvertisement: the daemon-owned contract for advertising akhs.
//!
//! Wave R3 of the akh-medu bring-up UX HANDOFF.
//!
//! `_akh/daemon/akhs` serves a proto3-JSON versioned envelope — the wire shape
//! is `{"v": 1, "akhs": [...]}` (spec seshat ce45c8f8 §b.2). akh-medu has no
//! proto tooling, so the hand-written serde here is pinned by vendored golden
//! fixtures (`tests/fixtures/akhs_envelope_*.json`) — the fixtures are the
//! conformance oracle; the envelope is what the gateway pipes field-for-field.
//!
//! `AkhAdvertisement` is a routing-inert `RunnerCapabilities.akhs` field in
//! Anima's `runner.proto` — it never appears in `executor_names`. The gateway
//! uses it for roster display (nickname/avatar) and power-state awareness,
//! not for ACP routing.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PowerState
// ---------------------------------------------------------------------------

/// The power state of an akh as advertised by the daemon.
///
/// `Errored` is distinct from `Asleep` — a fail-closed akh (config parse
/// failure, engine open failure, etc.) surfaces as `Errored`, not silently
/// `Asleep`. Confirmed by ticket 10.
///
/// Serde emits the proto3-JSON symbolic enum names (`POWER_STATE_*`) exactly —
/// the gateway pipes them through verbatim, so these renames are wire-locked
/// and guarded by the golden fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerState {
    /// The akh is running (daemon active, agent loaded).
    #[serde(rename = "POWER_STATE_AWAKE")]
    Awake,
    /// The akh is configured but not running (cold workspace).
    #[serde(rename = "POWER_STATE_ASLEEP")]
    Asleep,
    /// The akh failed to start — config parse error, engine open failure, etc.
    /// Fail-closed: the workspace is listed but marked errored, with `error`.
    #[serde(rename = "POWER_STATE_ERRORED")]
    Errored,
}

impl PowerState {
    /// String representation matching the proto3-JSON enum name.
    pub fn as_str(&self) -> &'static str {
        match self {
            PowerState::Awake => "POWER_STATE_AWAKE",
            PowerState::Asleep => "POWER_STATE_ASLEEP",
            PowerState::Errored => "POWER_STATE_ERRORED",
        }
    }
}

impl Default for PowerState {
    fn default() -> Self {
        PowerState::Asleep
    }
}

// ---------------------------------------------------------------------------
// AkhAdvertisement
// ---------------------------------------------------------------------------

/// A single akh advertisement entry.
///
/// This is the per-akh field in `RunnerCapabilities.akhs` in Anima's
/// `runner.proto`. It is routing-inert — the gateway uses it for display
/// and power-state awareness, never for ACP executor routing.
///
/// Unset optionals are omitted (proto3-JSON convention) via
/// `skip_serializing_if` — the golden fixtures pin the exact wire shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkhAdvertisement {
    /// Workspace name (unique within this daemon).
    pub name: String,
    /// Pinned model (`model` field of `akh.toml`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Power state: awake, asleep, or errored.
    pub power_state: PowerState,
    /// Whether the akh is enabled (disabled akhs are advertised but marked).
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Display nickname (mutable, operator-set, display-only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nickname: Option<String>,
    /// Avatar reference — the BLAKE3 hash of the avatar bytes
    /// (`Psyche.avatar_hash`), bytes-on-change (ticket 12). `None` when no
    /// avatar has been set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avatar_ref: Option<String>,
    /// Failure detail — set only when `power_state == Errored` (ticket 10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn default_enabled() -> bool {
    true
}

impl AkhAdvertisement {
    /// Construct an advertisement with only the required fields.
    pub fn new(name: impl Into<String>, power_state: PowerState, enabled: bool) -> Self {
        Self {
            name: name.into(),
            model: None,
            power_state,
            enabled,
            nickname: None,
            avatar_ref: None,
            error: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Versioned envelope — the wire contract (spec §b.2)
// ---------------------------------------------------------------------------

/// The proto3-JSON versioned envelope for `_akh/daemon/akhs`.
///
/// Wire shape is exactly `{"v": 1, "akhs": [...]}` — nothing else. The
/// gateway is a dumb field-for-field pipe against this shape, so the struct
/// intentionally carries no daemon_version / include_unowned / schema_version
/// extras; `v` is the only versioning lever and bumping it is a breaking
/// change requiring coordinated gateway deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkhsEnvelope {
    /// Schema version of the envelope. Currently `1`.
    pub v: u32,
    /// The advertised akhs, sorted by name.
    pub akhs: Vec<AkhAdvertisement>,
}

impl AkhsEnvelope {
    /// Current envelope version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Build the envelope for a set of advertisements (sorted by name).
    pub fn new(mut akhs: Vec<AkhAdvertisement>) -> Self {
        akhs.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            v: Self::CURRENT_VERSION,
            akhs,
        }
    }
}

// ---------------------------------------------------------------------------
// Request parameters
// ---------------------------------------------------------------------------

/// Query parameters for `_akh/daemon/akhs`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AkhsQuery {
    /// If true, include unowned (cold/configured) akhs in the response.
    /// Default: false (owned, awake akhs only).
    #[serde(default)]
    pub include_unowned: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_state_serde_proto3_names() {
        assert_eq!(
            serde_json::to_string(&PowerState::Awake).unwrap(),
            "\"POWER_STATE_AWAKE\""
        );
        assert_eq!(
            serde_json::to_string(&PowerState::Asleep).unwrap(),
            "\"POWER_STATE_ASLEEP\""
        );
        assert_eq!(
            serde_json::to_string(&PowerState::Errored).unwrap(),
            "\"POWER_STATE_ERRORED\""
        );
    }

    #[test]
    fn power_state_deserialize_proto3_names() {
        let awake: PowerState = serde_json::from_str("\"POWER_STATE_AWAKE\"").unwrap();
        assert_eq!(awake, PowerState::Awake);
        let errored: PowerState = serde_json::from_str("\"POWER_STATE_ERRORED\"").unwrap();
        assert_eq!(errored, PowerState::Errored);
    }

    #[test]
    fn advertisement_serialization_minimal() {
        let ad = AkhAdvertisement::new("sherlock", PowerState::Asleep, true);
        let json = serde_json::to_value(&ad).unwrap();
        assert_eq!(json["name"], "sherlock");
        assert_eq!(json["power_state"], "POWER_STATE_ASLEEP");
        assert_eq!(json["enabled"], true);
        // Unset optionals are omitted (proto3-JSON convention).
        assert!(json.get("model").is_none());
        assert!(json.get("nickname").is_none());
        assert!(json.get("avatar_ref").is_none());
        assert!(json.get("error").is_none());
    }

    #[test]
    fn advertisement_serialization_full() {
        let mut ad = AkhAdvertisement::new("ptah", PowerState::Awake, true);
        ad.model = Some("qwen3:8b".into());
        ad.nickname = Some("The Builder".into());
        ad.avatar_ref = Some("blake3:abc123".into());
        let json = serde_json::to_value(&ad).unwrap();
        assert_eq!(json["model"], "qwen3:8b");
        assert_eq!(json["nickname"], "The Builder");
        assert_eq!(json["avatar_ref"], "blake3:abc123");
        assert_eq!(json["power_state"], "POWER_STATE_AWAKE");
    }

    #[test]
    fn errored_advertisement_carries_error() {
        let mut ad = AkhAdvertisement::new("broken", PowerState::Errored, true);
        ad.error = Some("config parse failed: missing field `dimension`".into());
        let json = serde_json::to_value(&ad).unwrap();
        assert_eq!(json["power_state"], "POWER_STATE_ERRORED");
        assert_eq!(
            json["error"],
            "config parse failed: missing field `dimension`"
        );
    }

    #[test]
    fn envelope_wire_shape_is_v_and_akhs() {
        let env = AkhsEnvelope::new(vec![
            AkhAdvertisement::new("zeta", PowerState::Awake, true),
            AkhAdvertisement::new("alpha", PowerState::Asleep, true),
        ]);
        let json = serde_json::to_value(&env).unwrap();
        // Exactly the contract shape: {"v": 1, "akhs": [...]}, no extras.
        let obj = json.as_object().unwrap();
        assert_eq!(obj.len(), 2, "envelope must carry only v + akhs: {json}");
        assert_eq!(json["v"], 1);
        let names: Vec<&str> = json["akhs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["alpha", "zeta"], "envelope sorts by name");
    }

    #[test]
    fn envelope_round_trip() {
        let env = AkhsEnvelope::new(vec![AkhAdvertisement::new(
            "tecton",
            PowerState::Awake,
            true,
        )]);
        let s = serde_json::to_string(&env).unwrap();
        let back: AkhsEnvelope = serde_json::from_str(&s).unwrap();
        assert_eq!(back.v, 1);
        assert_eq!(back.akhs.len(), 1);
        assert_eq!(back.akhs[0].name, "tecton");
        assert_eq!(back.akhs[0].power_state, PowerState::Awake);
    }

    #[test]
    fn query_default_omits_unowned() {
        let q: AkhsQuery = serde_json::from_str("{}").unwrap();
        assert!(!q.include_unowned);
        let q: AkhsQuery = serde_json::from_str("{\"include_unowned\": true}").unwrap();
        assert!(q.include_unowned);
    }
}
