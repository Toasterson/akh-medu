//! AkhAdvertisement: the daemon-owned contract for advertising akhs.
//!
//! Wave R3 of the akh-medu bring-up UX HANDOFF.
//!
//! `_akh/daemon/akhs` is a proto3-JSON versioned envelope served by akhomed.
//! It advertises owned (and optionally unowned) akhs with their power state,
//! enabled flag, and avatar hash. The Anima gateway forwards this via
//! ADR 0022 (akh registration via gateway runners).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PowerState {
    /// The akh is running (daemon active, agent loaded).
    Awake,
    /// The akh is not running (daemon stopped, agent may or may not be loaded).
    Asleep,
    /// The akh failed to start — config parse error, engine open failure, etc.
    /// Fail-closed: the workspace is listed but marked errored.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkhAdvertisement {
    /// Workspace name (unique within this daemon).
    pub name: String,
    /// Power state: awake, asleep, or errored.
    pub power_state: PowerState,
    /// Whether the akh is enabled (disabled akhs are advertised but marked).
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// BLAKE3 hash of the akh's avatar bytes, if any.
    /// Bytes-on-change: updated whenever the avatar image changes.
    /// `None` when no avatar has been set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avatar_hash: Option<String>,
    /// Display nickname (mutable, operator-set, display-only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nickname: Option<String>,
    /// Ennead role assigned during ritual (e.g. "Digital Architect").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ennead_role: Option<String>,
}

fn default_enabled() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Versioned envelope
// ---------------------------------------------------------------------------

/// The proto3-JSON versioned envelope for `_akh/daemon/akhs`.
///
/// Versioned for forward compatibility — the gateway checks `schema_version`
/// before consuming. Bumping the version is a breaking change that requires
/// coordinated gateway deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkhsEnvelope {
    /// Schema version of the envelope. Currently `1`.
    pub schema_version: u32,
    /// Daemon version (from `CARGO_PKG_VERSION`).
    pub daemon_version: String,
    /// Whether this response includes unowned akhs (the `include_unowned`
    /// arg was true).
    pub include_unowned: bool,
    /// The advertised akhs.
    pub akhs: Vec<AkhAdvertisement>,
}

impl AkhsEnvelope {
    /// Current schema version.
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    /// Build an envelope with owned akhs only (default).
    pub fn owned_only(daemon_version: &str, akhs: Vec<AkhAdvertisement>) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            daemon_version: daemon_version.to_string(),
            include_unowned: false,
            akhs,
        }
    }

    /// Build an envelope including unowned akhs.
    pub fn including_unowned(daemon_version: &str, akhs: Vec<AkhAdvertisement>) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            daemon_version: daemon_version.to_string(),
            include_unowned: true,
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
    /// Default: false (owned akhs only).
    #[serde(default)]
    pub include_unowned: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_state_serde_snake_case() {
        let awake = serde_json::to_string(&PowerState::Awake).unwrap();
        assert_eq!(awake, "\"POWER_STATE_AWAKE\"");
        let asleep = serde_json::to_string(&PowerState::Asleep).unwrap();
        assert_eq!(asleep, "\"POWER_STATE_ASLEEP\"");
        let errored = serde_json::to_string(&PowerState::Errored).unwrap();
        assert_eq!(errored, "\"POWER_STATE_ERRORED\"");
    }

    #[test]
    fn power_state_deserialize_from_snake_case() {
        let awake: PowerState = serde_json::from_str("\"POWER_STATE_AWAKE\"").unwrap();
        assert_eq!(awake, PowerState::Awake);
        let errored: PowerState = serde_json::from_str("\"POWER_STATE_ERRORED\"").unwrap();
        assert_eq!(errored, PowerState::Errored);
    }

    #[test]
    fn advertisement_serialization_minimal() {
        let ad = AkhAdvertisement {
            name: "sherlock".into(),
            power_state: PowerState::Asleep,
            enabled: true,
            avatar_hash: None,
            nickname: None,
            ennead_role: None,
        };
        let json = serde_json::to_value(&ad).unwrap();
        assert_eq!(json["name"], "sherlock");
        assert_eq!(json["power_state"], "POWER_STATE_ASLEEP");
        assert_eq!(json["enabled"], true);
        assert!(json.get("avatar_hash").is_none());
        assert!(json.get("nickname").is_none());
        assert!(json.get("ennead_role").is_none());
    }

    #[test]
    fn advertisement_serialization_full() {
        let ad = AkhAdvertisement {
            name: "ptah".into(),
            power_state: PowerState::Awake,
            enabled: true,
            avatar_hash: Some("abc123".into()),
            nickname: Some("The Builder".into()),
            ennead_role: Some("Digital Architect".into()),
        };
        let json = serde_json::to_value(&ad).unwrap();
        assert_eq!(json["avatar_hash"], "abc123");
        assert_eq!(json["nickname"], "The Builder");
        assert_eq!(json["ennead_role"], "Digital Architect");
    }

    #[test]
    fn envelope_owned_only() {
        let env = AkhsEnvelope::owned_only("0.1.0", vec![]);
        assert_eq!(env.schema_version, 1);
        assert!(!env.include_unowned);
        assert_eq!(env.daemon_version, "0.1.0");
    }

    #[test]
    fn envelope_including_unowned() {
        let env = AkhsEnvelope::including_unowned("0.1.0", vec![]);
        assert!(env.include_unowned);
    }

    #[test]
    fn envelope_round_trip() {
        let env = AkhsEnvelope::owned_only("0.1.0", vec![
            AkhAdvertisement {
                name: "sherlock".into(),
                power_state: PowerState::Awake,
                enabled: true,
                avatar_hash: Some("deadbeef".into()),
                nickname: None,
                ennead_role: Some("Investigator".into()),
            },
        ]);
        let json = serde_json::to_string(&env).unwrap();
        let back: AkhsEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back.akhs.len(), 1);
        assert_eq!(back.akhs[0].name, "sherlock");
        assert_eq!(back.akhs[0].power_state, PowerState::Awake);
    }

    #[test]
    fn akhs_query_default_excludes_unowned() {
        let q = AkhsQuery::default();
        assert!(!q.include_unowned);
    }
}
