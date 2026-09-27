//! Durable incident ledger for escalation and merge-repair delegations
//! (Phase 53g PR 7).
//!
//! Plan: `docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md` ("PR 7");
//! Anima WI `01a0d569-1cf1` item 5. This replaces the SQLite incident table of
//! `deploy/roman-mini/janus_escalation.py`.
//!
//! One record per [`IncidentKey`]: which delegation handles it, the akhsessd
//! session and turn, and where it stands ([`IncidentState`]). The ledger is
//! what makes escalation safe to call from every wake:
//!
//! - **Dedup.** Admitting a key that already has a record returns that record
//!   with `duplicate: true`; nothing new is created, whatever its state.
//! - **Rate limit.** At most `max_new_per_day` new keys per UTC day
//!   ([`UtcDay`]); a duplicate never counts.
//! - **Restart safety.** Every record carries an akhsessd create request id
//!   derived from the akh, the kind and the key ([`request_id_for`]). Before a
//!   create is sent the record notes the attempt and the grant id; after a
//!   restart, reconciliation looks the session up by that request id instead
//!   of creating a second one.
//!
//! The ledger is persisted in the engine's durable metadata under
//! [`INCIDENT_LEDGER_KEY`] and carried in the snapshot extra
//! (`extra.delegation.incidents`), like the provider gate and the autonomy
//! budget ledger. Every mutation is written through before it is visible, so
//! an admission the store could not record is refused rather than forgotten.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::agent::autonomy::UnixSecs;
use crate::engine::Engine;

/// Durable-metadata key of the incident ledger. The `v1` suffix versions the
/// JSON shape of [`LedgerState`].
pub const INCIDENT_LEDGER_KEY: &[u8] = b"delegation:incidents:v1";

/// Settled records older than this are pruned on the next admission.
pub const RETAIN_SETTLED_SECS: u64 = 30 * 24 * 60 * 60;

/// Longest accepted incident key (bytes).
pub const MAX_INCIDENT_KEY_LEN: usize = 128;

/// Prefix reserved for keys the merge-repair tool derives itself.
pub const MERGE_REPAIR_KEY_PREFIX: &str = "merge-repair-";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors of the incident ledger.
#[derive(Debug, Error, Diagnostic)]
pub enum LedgerError {
    /// The incident key is not a stable ASCII identifier.
    #[error("incident key `{key}` is invalid: {reason}")]
    #[diagnostic(
        code(akh::delegation::incident_key),
        help(
            "Use a stable ASCII key of 1..=128 characters that starts with a letter or digit \
             and continues with letters, digits, `.`, `_`, `:` or `-`. Reuse the same key for \
             the same blocker across wakes; never add a date or wake counter to it."
        )
    )]
    InvalidKey {
        /// The rejected key (truncated).
        key: String,
        /// Which rule it broke.
        reason: &'static str,
    },

    /// The key is already recorded for the other kind of delegation.
    #[error("incident `{key}` is already a {existing} incident, not a {requested} one")]
    #[diagnostic(
        code(akh::delegation::incident_kind),
        help(
            "An incident key names exactly one delegation. Check the existing one with \
             `escalation_status`; a different blocker needs its own key."
        )
    )]
    KindMismatch {
        /// The key.
        key: String,
        /// The kind already recorded.
        existing: IncidentKind,
        /// The kind that was requested.
        requested: IncidentKind,
    },

    /// The daily limit of new incidents is reached.
    #[error(
        "{admitted} new incidents were already admitted on UTC day {day}; the limit is {limit}"
    )]
    #[diagnostic(
        code(akh::delegation::daily_limit),
        help(
            "Inspect the existing incidents with `escalation_status` instead of opening new \
             ones, and report the limit. Repeating an existing incident key is always allowed. \
             The operator can raise `[delegation.native] max_new_per_day`; the count resets at \
             00:00 UTC."
        )
    )]
    DailyLimit {
        /// The UTC day (days since the epoch).
        day: u64,
        /// Incidents admitted that day.
        admitted: usize,
        /// The configured limit.
        limit: u32,
    },

    /// No record exists for the key.
    #[error("no incident is recorded under `{key}`")]
    #[diagnostic(
        code(akh::delegation::unknown_incident),
        help(
            "Use the exact incident key returned by `escalate` or `dispatch_merge_repair`. \
             Settled incidents are kept for 30 days."
        )
    )]
    UnknownIncident {
        /// The key.
        key: String,
    },

    /// The persisted ledger could not be decoded.
    #[error("the incident ledger at `{key}` is corrupt: {source}")]
    #[diagnostic(
        code(akh::delegation::ledger_corrupt),
        help(
            "No escalation is admitted while the ledger is unreadable, so deduplication cannot \
             silently break. Stop the akh, inspect the durable meta key `{key}` and repair or \
             remove it (removing it forgets every open incident; check akhsessd for sessions \
             whose request ids start with `akh-incident-` first)."
        )
    )]
    Corrupt {
        /// The metadata key.
        key: String,
        /// The decode error.
        #[source]
        source: serde_json::Error,
    },

    /// Reading the ledger from the durable store failed.
    #[error("failed to read the incident ledger at `{key}`")]
    #[diagnostic(
        code(akh::delegation::ledger_read),
        help(
            "The engine's durable store did not answer. Check the akh's data directory (disk \
             full, permissions, or a lock held by another process)."
        )
    )]
    Read {
        /// The metadata key.
        key: String,
        /// The store error.
        #[source]
        source: crate::error::StoreError,
    },

    /// Writing the ledger to the durable store failed.
    #[error("failed to persist the incident ledger at `{key}`")]
    #[diagnostic(
        code(akh::delegation::ledger_write),
        help(
            "Nothing was admitted or changed: an incident the ledger cannot record could be \
             duplicated after a restart. Check free disk space and the data directory's \
             permissions, then retry."
        )
    )]
    Write {
        /// The metadata key.
        key: String,
        /// The store error.
        #[source]
        source: crate::error::StoreError,
    },

    /// The ledger could not be encoded (a bug: the state is plain data).
    #[error("failed to encode the incident ledger")]
    #[diagnostic(
        code(akh::delegation::ledger_encode),
        help("This is a bug in akh-medu; please report it with the daemon log.")
    )]
    Encode {
        /// The encode error.
        #[source]
        source: serde_json::Error,
    },
}

/// Result alias for the incident ledger.
pub type LedgerResult<T> = Result<T, LedgerError>;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A validated incident key: 1..=128 ASCII bytes, `[A-Za-z0-9]` first, then
/// `[A-Za-z0-9._:-]`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct IncidentKey(String);

impl IncidentKey {
    /// Validate `key`.
    ///
    /// # Errors
    ///
    /// [`LedgerError::InvalidKey`] naming the rule it broke.
    pub fn parse(key: &str) -> LedgerResult<Self> {
        let invalid = |reason| LedgerError::InvalidKey {
            key: key.chars().take(MAX_INCIDENT_KEY_LEN).collect(),
            reason,
        };
        if key.is_empty() {
            return Err(invalid("it is empty"));
        }
        if key.len() > MAX_INCIDENT_KEY_LEN {
            return Err(invalid("it is longer than 128 bytes"));
        }
        let mut bytes = key.bytes();
        if !bytes.next().is_some_and(|b| b.is_ascii_alphanumeric()) {
            return Err(invalid("it must start with an ASCII letter or digit"));
        }
        if !bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-')) {
            return Err(invalid(
                "it may only contain ASCII letters, digits, `.`, `_`, `:` and `-`",
            ));
        }
        Ok(Self(key.to_owned()))
    }

    /// The key.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether the key is in the namespace the merge-repair tool derives.
    pub fn is_merge_repair(&self) -> bool {
        self.0.starts_with(MERGE_REPAIR_KEY_PREFIX)
    }
}

impl TryFrom<String> for IncidentKey {
    type Error = LedgerError;

    fn try_from(key: String) -> Result<Self, Self::Error> {
        Self::parse(&key)
    }
}

impl From<IncidentKey> for String {
    fn from(key: IncidentKey) -> Self {
        key.0
    }
}

impl std::fmt::Display for IncidentKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a delegation does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncidentKind {
    /// A bounded blocker escalated to a native worker (`escalate`).
    Escalation,
    /// A merge-conflict repair job (`dispatch_merge_repair`).
    MergeRepair,
}

impl IncidentKind {
    /// Stable label for request ids, logs and output.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Escalation => "escalation",
            Self::MergeRepair => "merge_repair",
        }
    }
}

impl std::fmt::Display for IncidentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Where an incident's delegation stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncidentState {
    /// Admitted; the worker session or its first turn is not confirmed yet.
    Pending,
    /// The worker's turn was admitted by akhsessd and has not ended.
    Running,
    /// The worker's turn completed. Not proof the incident is resolved.
    Succeeded,
    /// The worker's turn (or the delegation itself) failed.
    Failed,
    /// The worker's turn was cancelled.
    Cancelled,
}

impl IncidentState {
    /// Whether nothing more will happen to this incident.
    pub const fn is_settled(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

/// Why a delegation failed, in machine terms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentFailure {
    /// A stable code: akhsessd's fault code for a failed turn, or the
    /// delegation's own (`runtime_mismatch`, `session_missing`, ...).
    pub code: String,
    /// The detail, bounded.
    pub message: String,
    /// The provider-failure class (53g PR 4 taxonomy) when the code is listed
    /// in the akh's `[provider]` lexicon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_class: Option<String>,
}

/// One incident.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentRecord {
    /// The dedup key.
    pub incident_key: IncidentKey,
    /// What the delegation does.
    pub kind: IncidentKind,
    /// The durable delegation id (UUID v7).
    pub delegation_id: Uuid,
    /// The registered project.
    pub project: String,
    /// The akhsessd workspace profile the worker runs in.
    pub workspace_profile: String,
    /// The akhsessd create request id (derived from the key; stable).
    pub request_id: String,
    /// The exact commit the worker's repository checkout is pinned to (53g
    /// PR bootstrap fix). Set whenever the request carried a `head_sha`;
    /// required for a project whose profile declares a `repository`, and
    /// used to prepare that session's bootstrap intent on every dispatch
    /// while the incident stays pending.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_sha: Option<String>,
    /// Creates sent (or possibly sent) to akhsessd. Non-zero means a lookup
    /// by [`Self::request_id`] must precede any new create.
    #[serde(default)]
    pub create_attempts: u32,
    /// The grant minted for the last create (never the secret).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant_id: Option<String>,
    /// The akhsessd session, once known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The worker's turn, once admitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Where it stands.
    pub state: IncidentState,
    /// When it was admitted.
    pub created_at: UnixSecs,
    /// When it last changed.
    pub updated_at: UnixSecs,
    /// The validated job the worker is briefed with. Dropped once settled so
    /// the ledger stays small; the worker's session keeps the brief.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job: Option<serde_json::Value>,
    /// The last failure (or the last dispatch error while pending).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<IncidentFailure>,
}

/// The persisted ledger (`delegation:incidents:v1`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerState {
    /// Records by incident key.
    #[serde(default)]
    pub records: BTreeMap<String, IncidentRecord>,
}

/// A UTC calendar day (days since the Unix epoch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UtcDay(u64);

impl UtcDay {
    /// The UTC day containing `at`.
    pub const fn of(at: UnixSecs) -> Self {
        Self(at.as_secs() / 86_400)
    }

    /// Days since the epoch.
    pub const fn index(self) -> u64 {
        self.0
    }
}

/// An incident to admit.
#[derive(Debug, Clone)]
pub struct NewIncident {
    /// The dedup key.
    pub key: IncidentKey,
    /// What the delegation does.
    pub kind: IncidentKind,
    /// The registered project.
    pub project: String,
    /// The akhsessd workspace profile.
    pub workspace_profile: String,
    /// The akhsessd create request id ([`request_id_for`]).
    pub request_id: String,
    /// The exact commit to bootstrap the worker's repository checkout at,
    /// when the request carried one (53g PR bootstrap fix).
    pub head_sha: Option<String>,
    /// The validated job.
    pub job: serde_json::Value,
}

/// The result of an admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admitted {
    /// The record (new, or the one already under the key).
    pub record: IncidentRecord,
    /// Whether the key already had a record.
    pub duplicate: bool,
}

/// The akhsessd create request id for an incident: stable for the same akh,
/// kind and key, so a create replayed after a restart is recognised.
pub fn request_id_for(akh: &str, kind: IncidentKind, key: &IncidentKey) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in [akh, kind.label(), key.as_str()] {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    let digest = hasher.finalize().to_hex();
    format!("akh-incident-{}", &digest[..32])
}

// ---------------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum LedgerStore {
    Durable(Arc<Engine>),
    Memory,
}

/// One akh's incident ledger. All mutations go through one mutex and are
/// written through to the durable store before they become visible.
#[derive(Debug)]
pub struct IncidentLedger {
    store: LedgerStore,
    state: Mutex<LedgerState>,
}

impl IncidentLedger {
    /// A ledger that lives only in memory (tests, in-memory engines).
    pub fn in_memory() -> Self {
        Self {
            store: LedgerStore::Memory,
            state: Mutex::new(LedgerState::default()),
        }
    }

    /// Open the ledger for `engine`, loading the persisted state when the
    /// engine has a durable tier.
    ///
    /// # Errors
    ///
    /// [`LedgerError::Read`] or [`LedgerError::Corrupt`]: the ledger is
    /// never silently reset, since that would break deduplication.
    pub fn open(engine: &Arc<Engine>) -> LedgerResult<Self> {
        if engine.store().durable.is_none() {
            return Ok(Self::in_memory());
        }
        let state = load_persisted(engine)?.unwrap_or_default();
        Ok(Self {
            store: LedgerStore::Durable(Arc::clone(engine)),
            state: Mutex::new(state),
        })
    }

    /// Whether the ledger survives a restart.
    pub fn is_durable(&self) -> bool {
        matches!(self.store, LedgerStore::Durable(_))
    }

    /// Admit `incident` at `now`: the existing record for its key
    /// (`duplicate: true`), or a new `Pending` record if fewer than `limit`
    /// new incidents were admitted on the same UTC day.
    ///
    /// # Errors
    ///
    /// [`LedgerError::KindMismatch`], [`LedgerError::DailyLimit`], or a
    /// store error (nothing is admitted then).
    pub fn admit(
        &self,
        incident: NewIncident,
        limit: u32,
        now: UnixSecs,
    ) -> LedgerResult<Admitted> {
        let mut state = self.lock();
        if let Some(existing) = state.records.get(incident.key.as_str()) {
            if existing.kind != incident.kind {
                return Err(LedgerError::KindMismatch {
                    key: incident.key.to_string(),
                    existing: existing.kind,
                    requested: incident.kind,
                });
            }
            return Ok(Admitted {
                record: existing.clone(),
                duplicate: true,
            });
        }
        let day = UtcDay::of(now);
        let admitted = state
            .records
            .values()
            .filter(|r| UtcDay::of(r.created_at) == day)
            .count();
        if admitted >= usize::try_from(limit).unwrap_or(usize::MAX) {
            return Err(LedgerError::DailyLimit {
                day: day.index(),
                admitted,
                limit,
            });
        }
        let record = IncidentRecord {
            incident_key: incident.key.clone(),
            kind: incident.kind,
            delegation_id: Uuid::now_v7(),
            project: incident.project,
            workspace_profile: incident.workspace_profile,
            request_id: incident.request_id,
            head_sha: incident.head_sha,
            create_attempts: 0,
            grant_id: None,
            session_id: None,
            turn_id: None,
            state: IncidentState::Pending,
            created_at: now,
            updated_at: now,
            job: Some(incident.job),
            last_error: None,
        };
        let mut next = state.clone();
        next.records
            .retain(|_, r| !(r.state.is_settled() && retention_expired(r.updated_at, now)));
        next.records
            .insert(incident.key.to_string(), record.clone());
        self.persist(&next)?;
        *state = next;
        Ok(Admitted {
            record,
            duplicate: false,
        })
    }

    /// The record under `key`.
    pub fn get(&self, key: &IncidentKey) -> Option<IncidentRecord> {
        self.lock().records.get(key.as_str()).cloned()
    }

    /// Change the record under `key` with `change`, stamp `updated_at` and
    /// persist it. A record that becomes settled drops its job.
    ///
    /// # Errors
    ///
    /// [`LedgerError::UnknownIncident`] or a store error (the record is then
    /// unchanged).
    pub fn update(
        &self,
        key: &IncidentKey,
        now: UnixSecs,
        change: impl FnOnce(&mut IncidentRecord),
    ) -> LedgerResult<IncidentRecord> {
        let mut state = self.lock();
        let mut record = state.records.get(key.as_str()).cloned().ok_or_else(|| {
            LedgerError::UnknownIncident {
                key: key.to_string(),
            }
        })?;
        change(&mut record);
        record.updated_at = now;
        if record.state.is_settled() {
            record.job = None;
        }
        let mut next = state.clone();
        next.records.insert(key.to_string(), record.clone());
        self.persist(&next)?;
        *state = next;
        Ok(record)
    }

    /// Records that are not settled, oldest first (the reconciliation set).
    pub fn unsettled(&self) -> Vec<IncidentRecord> {
        let mut records: Vec<_> = self
            .lock()
            .records
            .values()
            .filter(|r| !r.state.is_settled())
            .cloned()
            .collect();
        records.sort_by_key(|r| (r.created_at, r.delegation_id));
        records
    }

    /// New incidents admitted on `day`.
    pub fn admitted_on(&self, day: UtcDay) -> usize {
        self.lock()
            .records
            .values()
            .filter(|r| UtcDay::of(r.created_at) == day)
            .count()
    }

    fn persist(&self, state: &LedgerState) -> LedgerResult<()> {
        match &self.store {
            LedgerStore::Memory => Ok(()),
            LedgerStore::Durable(engine) => persist(engine, state),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LedgerState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn retention_expired(updated_at: UnixSecs, now: UnixSecs) -> bool {
    now.as_secs().saturating_sub(updated_at.as_secs()) > RETAIN_SETTLED_SECS
}

fn key_display() -> String {
    String::from_utf8_lossy(INCIDENT_LEDGER_KEY).into_owned()
}

fn persist(engine: &Engine, state: &LedgerState) -> LedgerResult<()> {
    let bytes = serde_json::to_vec(state).map_err(|source| LedgerError::Encode { source })?;
    engine
        .store()
        .put_meta(INCIDENT_LEDGER_KEY, &bytes)
        .map_err(|source| LedgerError::Write {
            key: key_display(),
            source,
        })
}

/// Read the persisted ledger. `Ok(None)` when none was ever written (or the
/// engine has no durable tier).
///
/// # Errors
///
/// [`LedgerError::Read`] or [`LedgerError::Corrupt`].
pub fn load_persisted(engine: &Engine) -> LedgerResult<Option<LedgerState>> {
    let bytes = engine
        .store()
        .get_meta(INCIDENT_LEDGER_KEY)
        .map_err(|source| LedgerError::Read {
            key: key_display(),
            source,
        })?;
    bytes
        .map(|bytes| {
            serde_json::from_slice(&bytes).map_err(|source| LedgerError::Corrupt {
                key: key_display(),
                source,
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOON: UnixSecs = UnixSecs::from_secs(1_790_251_200 + 12 * 3_600);

    fn key(value: &str) -> IncidentKey {
        IncidentKey::parse(value).unwrap()
    }

    fn incident(value: &str, kind: IncidentKind) -> NewIncident {
        let key = key(value);
        NewIncident {
            request_id: request_id_for("janus", kind, &key),
            key,
            kind,
            project: "akh-medu".into(),
            workspace_profile: "janus-worker-akh-medu".into(),
            head_sha: None,
            job: serde_json::json!({"summary": value}),
        }
    }

    fn durable_engine(dir: &std::path::Path) -> Arc<Engine> {
        Arc::new(
            Engine::new(crate::engine::EngineConfig {
                data_dir: Some(dir.to_path_buf()),
                ..crate::engine::EngineConfig::default()
            })
            .unwrap(),
        )
    }

    #[test]
    fn incident_keys_are_validated_structurally() {
        for good in ["WI-403-owner", "a", "x:y.z_1-2", &"k".repeat(128)] {
            assert!(IncidentKey::parse(good).is_ok(), "{good}");
        }
        for bad in [
            "",
            "-leading",
            "bad key",
            "slash/key",
            "ünicode",
            &"k".repeat(129),
        ] {
            assert!(
                matches!(IncidentKey::parse(bad), Err(LedgerError::InvalidKey { .. })),
                "{bad}"
            );
        }
        assert!(key("merge-repair-abc").is_merge_repair());
        assert!(!key("merge-abc").is_merge_repair());
        // The serde form validates too.
        assert!(serde_json::from_str::<IncidentKey>("\"bad key\"").is_err());
    }

    #[test]
    fn request_ids_are_stable_and_distinct() {
        let a = request_id_for("janus", IncidentKind::Escalation, &key("x"));
        assert_eq!(
            a,
            request_id_for("janus", IncidentKind::Escalation, &key("x"))
        );
        assert!(a.starts_with("akh-incident-") && a.len() == 13 + 32, "{a}");
        assert_ne!(
            a,
            request_id_for("janus", IncidentKind::MergeRepair, &key("x"))
        );
        assert_ne!(
            a,
            request_id_for("vesta", IncidentKind::Escalation, &key("x"))
        );
        assert_ne!(
            a,
            request_id_for("janus", IncidentKind::Escalation, &key("y"))
        );
    }

    #[test]
    fn repeats_are_duplicates_and_never_count() {
        let ledger = IncidentLedger::in_memory();
        let first = ledger
            .admit(incident("WI-1", IncidentKind::Escalation), 4, NOON)
            .unwrap();
        assert!(!first.duplicate);
        assert_eq!(first.record.state, IncidentState::Pending);
        for _ in 0..10 {
            let again = ledger
                .admit(incident("WI-1", IncidentKind::Escalation), 4, NOON)
                .unwrap();
            assert!(again.duplicate);
            assert_eq!(again.record.delegation_id, first.record.delegation_id);
        }
        assert_eq!(ledger.admitted_on(UtcDay::of(NOON)), 1);
        assert!(matches!(
            ledger.admit(incident("WI-1", IncidentKind::MergeRepair), 4, NOON),
            Err(LedgerError::KindMismatch { .. })
        ));
    }

    #[test]
    fn the_fifth_new_incident_of_a_utc_day_is_refused_until_midnight() {
        let ledger = IncidentLedger::in_memory();
        for i in 0..4 {
            ledger
                .admit(
                    incident(&format!("incident-{i}"), IncidentKind::Escalation),
                    4,
                    NOON,
                )
                .unwrap();
        }
        let refused = ledger
            .admit(incident("incident-4", IncidentKind::Escalation), 4, NOON)
            .unwrap_err();
        assert!(
            matches!(
                refused,
                LedgerError::DailyLimit {
                    admitted: 4,
                    limit: 4,
                    ..
                }
            ),
            "{refused:?}"
        );
        // Still refused one second before midnight UTC...
        let last_second = UnixSecs::from_secs((UtcDay::of(NOON).index() + 1) * 86_400 - 1);
        assert!(
            ledger
                .admit(
                    incident("incident-4", IncidentKind::Escalation),
                    4,
                    last_second
                )
                .is_err()
        );
        // ...a duplicate is still answered...
        assert!(
            ledger
                .admit(
                    incident("incident-0", IncidentKind::Escalation),
                    4,
                    last_second
                )
                .unwrap()
                .duplicate
        );
        // ...and the next UTC day admits again.
        let next_day = last_second.saturating_add(std::time::Duration::from_secs(1));
        assert!(
            !ledger
                .admit(
                    incident("incident-4", IncidentKind::Escalation),
                    4,
                    next_day
                )
                .unwrap()
                .duplicate
        );
        // A zero limit admits nothing new.
        assert!(matches!(
            IncidentLedger::in_memory().admit(incident("x", IncidentKind::Escalation), 0, NOON),
            Err(LedgerError::DailyLimit { limit: 0, .. })
        ));
    }

    #[test]
    fn updates_settle_drop_the_job_and_old_settled_records_are_pruned() {
        let ledger = IncidentLedger::in_memory();
        let admitted = ledger
            .admit(incident("old", IncidentKind::Escalation), 4, NOON)
            .unwrap()
            .record;
        assert!(admitted.job.is_some());
        let settled = ledger
            .update(&key("old"), NOON, |r| r.state = IncidentState::Succeeded)
            .unwrap();
        assert_eq!(settled.job, None);
        assert!(ledger.unsettled().is_empty());
        assert!(matches!(
            ledger.update(&key("missing"), NOON, |_| {}),
            Err(LedgerError::UnknownIncident { .. })
        ));
        let later = NOON.saturating_add(std::time::Duration::from_secs(RETAIN_SETTLED_SECS + 1));
        ledger
            .admit(incident("new", IncidentKind::Escalation), 4, later)
            .unwrap();
        assert_eq!(ledger.get(&key("old")), None, "pruned");
        assert_eq!(ledger.unsettled().len(), 1);
    }

    #[test]
    fn the_ledger_survives_an_engine_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let first = {
            let engine = durable_engine(dir.path());
            let ledger = IncidentLedger::open(&engine).unwrap();
            assert!(ledger.is_durable());
            let first = ledger
                .admit(incident("WI-9", IncidentKind::MergeRepair), 4, NOON)
                .unwrap()
                .record;
            ledger
                .update(&key("WI-9"), NOON, |r| {
                    r.create_attempts = 1;
                    r.grant_id = Some("grant-1".into());
                })
                .unwrap();
            engine.persist().unwrap();
            first
        };
        let engine = durable_engine(dir.path());
        let ledger = IncidentLedger::open(&engine).unwrap();
        let reopened = ledger.get(&key("WI-9")).unwrap();
        assert_eq!(reopened.delegation_id, first.delegation_id);
        assert_eq!(reopened.create_attempts, 1);
        assert_eq!(reopened.grant_id.as_deref(), Some("grant-1"));
        assert!(
            ledger
                .admit(incident("WI-9", IncidentKind::MergeRepair), 4, NOON)
                .unwrap()
                .duplicate
        );
    }

    #[test]
    fn a_corrupt_ledger_is_an_error_not_a_reset() {
        let dir = tempfile::tempdir().unwrap();
        let engine = durable_engine(dir.path());
        engine
            .store()
            .put_meta(INCIDENT_LEDGER_KEY, b"not json")
            .unwrap();
        assert!(matches!(
            IncidentLedger::open(&engine),
            Err(LedgerError::Corrupt { .. })
        ));
    }
}
