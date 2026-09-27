//! Escalation and merge-repair delegation to contained native workers
//! (Phase 53g PR 7, decisions D3 and D4).
//!
//! Plan: `docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md` ("PR 7");
//! Anima WI `01a0d569-1cf1` item 5. This replaces
//! `deploy/roman-mini/janus_escalation.py` and `mergiraf_worker.py`, which
//! turned MCP tool calls into T3/Astra threads.
//!
//! An akh (Janus) escalates a bounded blocker, or dispatches a merge-conflict
//! repair, to a native worker session brokered by akhsessd:
//!
//! 1. the request is validated structurally ([`EscalationRequest`],
//!    [`MergeRepairRequest`]): UUIDs, 40-hex commit ids, PR numbers ≥ 1,
//!    bounded strings and arrays;
//! 2. the [`IncidentLedger`] admits it (dedup by incident key, at most
//!    `max_new_per_day` new incidents per UTC day);
//! 3. the service mints a per-delegation grant (53e), creates the akhsessd
//!    session in the project's worker profile with the incident's stable
//!    request id, and submits the brief: the skillpack's procedures, loaded
//!    by path now, followed by the job as an `akh-delegation-job` block;
//! 4. [`EscalationService::status`] later reads the turn, settles the
//!    incident when it ended, revokes the grant and closes the session, and
//!    explains a failed turn with the provider-failure taxonomy (53g PR 4).
//!
//! The worker's own policy text lives only in the skillpacks
//! (`deploy/roman-mini/skills/astra-escalation`, `.../mergiraf`); nothing in
//! this module is natural-language instruction.
//!
//! # Restart reconciliation
//!
//! The ledger records the create attempt and the grant id before a create is
//! sent. A create whose answer is lost is replayed in-process with the same
//! request id and grant, which akhsessd deduplicates. After a restart the
//! grant secret is gone, so a replay would carry another grant fingerprint and
//! be refused; [`EscalationService::reconcile`] instead looks the session up
//! by request id (`GET /v1/session-requests/{id}`) and adopts it, or creates
//! it when no create ever landed.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use akh_session_types::{SubmitInput, TurnStatus};
use miette::Diagnostic;
use serde::Serialize;
use serde_json::{Map, Value, json};
use thiserror::Error;
use uuid::Uuid;

use super::error::DelegationError;
use super::ledger::{
    IncidentFailure, IncidentKey, IncidentKind, IncidentLedger, IncidentRecord, IncidentState,
    LedgerError, MERGE_REPAIR_KEY_PREFIX, NewIncident, request_id_for,
};
use super::native_session::{
    DelegationGrantIssuer, GrantRequest, NativeSessionClient, NativeSessionError, SessionView,
    TurnView, delegate_principal_for, new_session_request, parse_structured_output,
    recover_created_session,
};
use super::types::{
    DelegationOutput, GrantTools, NativeDelegationConfig, NativeEscalationConfig,
    NativeProjectProfile,
};
use crate::agent::autonomy::UnixSecs;
use crate::agent::provider_failure::{ProviderErrorLexicon, classify_identifier};
use crate::skills::procedures::SkillProcedures;

/// Most evidence or attempt entries one request may carry.
pub const MAX_LIST_ENTRIES: usize = 12;
/// Longest evidence, attempt or requested-action text (characters).
pub const MAX_TEXT_CHARS: usize = 4_000;
/// Longest summary (characters).
pub const MAX_SUMMARY_CHARS: usize = 160;
/// Longest project or repository reference (characters).
pub const MAX_REFERENCE_CHARS: usize = 256;
/// How much of a worker's final output a status answer carries (characters).
pub const OUTPUT_TAIL_CHARS: usize = 16_000;
/// Longest failure detail kept in the ledger (characters).
const MAX_FAILURE_CHARS: usize = 2_000;
/// The longest grant a worker may hold (the core's maximum).
const MAX_GRANT_TTL: Duration = Duration::from_secs(24 * 60 * 60);
/// Margin a grant outlives the worker timeout by, for final tool calls.
const GRANT_MARGIN: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors of the escalation tools.
#[derive(Debug, Error, Diagnostic)]
pub enum EscalationError {
    /// A field failed structural validation.
    #[error("`{field}` is invalid: {reason}")]
    #[diagnostic(
        code(akh::delegation::escalation::input),
        help(
            "Send exactly the documented fields. Work items and sessions are full UUIDs, \
             commits are full 40-hex SHAs, PR numbers are integers ≥ 1, lists hold 1..=12 \
             non-empty entries of at most 4000 characters, the summary at most 160."
        )
    )]
    InvalidInput {
        /// The offending field.
        field: String,
        /// What is wrong with it.
        reason: String,
    },

    /// The project is not registered for delegation.
    #[error("project `{project}` is not registered for delegation")]
    #[diagnostic(
        code(akh::delegation::escalation::unknown_project),
        help(
            "Use one of the registered projects: {registered}. The operator registers a \
             project under `[delegation.native.profiles]` with its akhsessd worker profile."
        )
    )]
    UnknownProject {
        /// The requested project.
        project: String,
        /// The registered projects, comma-separated.
        registered: String,
    },

    /// An escalation used a key from the merge-repair namespace.
    #[error("incident key `{key}` is reserved for merge-repair jobs")]
    #[diagnostic(
        code(akh::delegation::escalation::reserved_key),
        help(
            "Keys starting with `merge-repair-` are derived by `dispatch_merge_repair`. Use \
             that tool for merge conflicts, or choose another key for this blocker."
        )
    )]
    ReservedKey {
        /// The key.
        key: String,
    },

    /// The escalation configuration is unusable.
    #[error("the escalation tools are misconfigured: {reason}")]
    #[diagnostic(
        code(akh::delegation::escalation::config),
        help(
            "Fix `[delegation.native]` in the akh's workspace TOML: `enabled = true`, a readable \
             `token_file`, at least one project under `profiles`, and under `escalation` both \
             skill directories (each with `skill.json` and its procedures) and, if set, a UUID \
             `report_thread`. The tools are not registered until this is fixed."
        )
    )]
    Config {
        /// What is wrong.
        reason: String,
    },

    /// A worker skillpack could not be loaded.
    #[error("the {kind} worker skillpack at {path} could not be loaded")]
    #[diagnostic(
        code(akh::delegation::escalation::skill),
        help(
            "The worker's instructions are the skill's procedures, read from disk on every \
             dispatch. Restore the skill directory (skill.json plus its procedures file); the \
             incident stays pending and is dispatched on the next status check."
        )
    )]
    Skill {
        /// Which worker.
        kind: IncidentKind,
        /// The skill directory.
        path: String,
        /// The load error.
        #[source]
        source: crate::error::SkillError,
    },

    /// The session runs on another runtime than the project profile requires.
    #[error(
        "session {session_id} runs on runtime profile `{actual}`, not the required `{expected}`"
    )]
    #[diagnostic(
        code(akh::delegation::escalation::runtime_mismatch),
        help(
            "The worker would not have run contained. The session was retired and its grant \
             revoked. In akhsessd's config give the workspace profile a `session_grants` entry \
             whose `runtime` names the required profile, then open a new incident."
        )
    )]
    RuntimeMismatch {
        /// The session.
        session_id: String,
        /// The required runtime profile.
        expected: String,
        /// The profile the session is bound to (`none` when unreported).
        actual: String,
    },

    /// The session's repository-bootstrap intent conflicts with one akhsessd
    /// already recorded for it (53g PR bootstrap fix).
    #[error(
        "session {session_id} already has another repository-bootstrap intent for project \
         {project}"
    )]
    #[diagnostic(
        code(akh::delegation::escalation::bootstrap_conflict),
        help(
            "akhsessd accepts exactly one repository-checkout intent per session. This usually \
             means the incident's `head_sha` changed between retries under the same key, or \
             this session was recovered from a create that another principal's intent already \
             reached. The session was retired and its grant revoked. Open a new incident key \
             with the current commit."
        )
    )]
    RepositoryBootstrapConflict {
        /// The session.
        session_id: String,
        /// The project.
        project: String,
    },

    /// Ledger failure (invalid key, limit, unknown incident, store).
    #[error(transparent)]
    #[diagnostic(transparent)]
    Ledger(#[from] LedgerError),

    /// Grant minting failed.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Delegation(#[from] DelegationError),

    /// akhsessd failed or refused.
    #[error(transparent)]
    #[diagnostic(transparent)]
    NativeSession(#[from] NativeSessionError),
}

impl EscalationError {
    fn input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// The diagnostic code, for machine-readable output.
    pub fn code_str(&self) -> String {
        self.code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "akh::delegation::escalation".into())
    }

    /// This error as a ledger failure record.
    fn as_failure(&self) -> IncidentFailure {
        IncidentFailure {
            code: self.code_str(),
            message: bounded(&self.to_string(), MAX_FAILURE_CHARS),
            provider_class: None,
        }
    }
}

/// Result alias for the escalation tools.
pub type EscalationResult<T> = Result<T, EscalationError>;

// ---------------------------------------------------------------------------
// Validated requests
// ---------------------------------------------------------------------------

/// A full commit id: 40 hex digits, stored lowercase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct CommitSha(String);

impl CommitSha {
    /// Validate a full 40-hex SHA-1 commit id.
    pub fn parse(field: &str, value: &str) -> EscalationResult<Self> {
        if value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
            Ok(Self(value.to_ascii_lowercase()))
        } else {
            Err(EscalationError::input(
                field,
                "expected a full 40-hex commit SHA",
            ))
        }
    }

    /// The SHA, lowercase.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A PR number (≥ 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct PrNumber(u64);

impl PrNumber {
    /// Validate a PR number.
    pub fn new(field: &str, value: u64) -> EscalationResult<Self> {
        if value >= 1 {
            Ok(Self(value))
        } else {
            Err(EscalationError::input(field, "a PR number is at least 1"))
        }
    }

    /// The number.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// A repository reference: `owner/name`, or an `https://` / `ssh://` URL
/// without credentials, query or fragment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct RepositoryRef(String);

impl RepositoryRef {
    /// Validate a repository reference.
    pub fn parse(field: &str, value: &str) -> EscalationResult<Self> {
        let slug_part = |part: &str| {
            !part.is_empty()
                && !part.starts_with('.')
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        };
        let url = |rest: &str, userinfo_allowed: bool| {
            let Some((authority, path)) = rest.split_once('/') else {
                return false;
            };
            let host = match authority.rsplit_once('@') {
                // A user name is fine for SSH; a password never is.
                Some((user, host)) => {
                    if !userinfo_allowed || user.is_empty() || user.contains(':') {
                        return false;
                    }
                    host
                }
                None => authority,
            };
            !host.is_empty()
                && !path.is_empty()
                && rest.bytes().all(|b| {
                    b.is_ascii_alphanumeric()
                        || matches!(b, b'.' | b'_' | b'~' | b'/' | b':' | b'@' | b'-')
                })
                && !path.split('/').any(|segment| segment == "..")
        };
        let valid = if let Some(rest) = value.strip_prefix("https://") {
            url(rest, false)
        } else if let Some(rest) = value.strip_prefix("ssh://") {
            url(rest, true)
        } else if value.contains("://") {
            false
        } else {
            value
                .split_once('/')
                .is_some_and(|(owner, name)| slug_part(owner) && slug_part(name))
        };
        if valid && value.len() <= MAX_REFERENCE_CHARS {
            Ok(Self(value.to_owned()))
        } else {
            Err(EscalationError::input(
                field,
                "expected `owner/name` or an https:// or ssh:// repository URL without credentials",
            ))
        }
    }
}

/// A validated `escalate` request (mirrors the retired `escalate_to_astra`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EscalationRequest {
    /// Stable key for this blocker across wakes.
    pub incident_key: IncidentKey,
    /// The registered project.
    pub project: String,
    /// One-line summary (≤ 160 characters).
    pub summary: String,
    /// 1..=12 evidence entries.
    pub evidence: Vec<String>,
    /// 1..=12 attempts already made.
    pub attempted: Vec<String>,
    /// The bounded action the worker may take.
    pub requested_action: String,
    /// The Anima work item, if one is involved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub work_item_id: Option<Uuid>,
    /// The PR, if one is involved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr_number: Option<PrNumber>,
    /// The PR head commit, if one is involved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_sha: Option<CommitSha>,
    /// The Anima session, if one is involved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anima_session_id: Option<Uuid>,
}

/// The `escalate` fields.
const ESCALATION_FIELDS: [&str; 10] = [
    "incident_key",
    "project",
    "summary",
    "evidence",
    "attempted",
    "requested_action",
    "work_item_id",
    "pr_number",
    "head_sha",
    "anima_session_id",
];

impl EscalationRequest {
    /// Validate tool parameters.
    ///
    /// # Errors
    ///
    /// [`EscalationError::InvalidInput`] naming the first bad field,
    /// [`EscalationError::ReservedKey`] for a merge-repair key, or
    /// [`LedgerError::InvalidKey`].
    pub fn from_params(params: &Map<String, Value>) -> EscalationResult<Self> {
        reject_unknown(params, &ESCALATION_FIELDS)?;
        let incident_key = IncidentKey::parse(&text(params, "incident_key", MAX_REFERENCE_CHARS)?)?;
        if incident_key.is_merge_repair() {
            return Err(EscalationError::ReservedKey {
                key: incident_key.to_string(),
            });
        }
        Ok(Self {
            incident_key,
            project: text(params, "project", MAX_REFERENCE_CHARS)?,
            summary: text(params, "summary", MAX_SUMMARY_CHARS)?,
            evidence: list(params, "evidence")?,
            attempted: list(params, "attempted")?,
            requested_action: text(params, "requested_action", MAX_TEXT_CHARS)?,
            work_item_id: optional(params, "work_item_id", uuid)?,
            pr_number: optional(params, "pr_number", pr_number)?,
            head_sha: optional(params, "head_sha", |field, value| {
                CommitSha::parse(field, &string(field, value)?)
            })?,
            anima_session_id: optional(params, "anima_session_id", uuid)?,
        })
    }
}

/// A validated `dispatch_merge_repair` request (mirrors the retired
/// `dispatch_mergiraf`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MergeRepairRequest {
    /// The registered project.
    pub project: String,
    /// The repository the PR lives in.
    pub repository: RepositoryRef,
    /// The Anima work item.
    pub work_item_id: Uuid,
    /// The PR.
    pub pr_number: PrNumber,
    /// The PR source-branch head the worker starts from.
    pub head_sha: CommitSha,
    /// The base commit to merge into the PR branch.
    pub base_sha: CommitSha,
    /// 1..=12 conflict and ownership evidence entries.
    pub evidence: Vec<String>,
}

/// The `dispatch_merge_repair` fields.
const MERGE_REPAIR_FIELDS: [&str; 7] = [
    "project",
    "repository",
    "work_item_id",
    "pr_number",
    "head_sha",
    "base_sha",
    "evidence",
];

impl MergeRepairRequest {
    /// Validate tool parameters.
    ///
    /// # Errors
    ///
    /// [`EscalationError::InvalidInput`] naming the first bad field.
    pub fn from_params(params: &Map<String, Value>) -> EscalationResult<Self> {
        reject_unknown(params, &MERGE_REPAIR_FIELDS)?;
        let required = |field: &'static str| {
            params
                .get(field)
                .filter(|v| !v.is_null())
                .ok_or_else(|| EscalationError::input(field, "is required"))
        };
        Ok(Self {
            project: text(params, "project", MAX_REFERENCE_CHARS)?,
            repository: RepositoryRef::parse(
                "repository",
                &text(params, "repository", MAX_REFERENCE_CHARS)?,
            )?,
            work_item_id: uuid("work_item_id", required("work_item_id")?)?,
            pr_number: pr_number("pr_number", required("pr_number")?)?,
            head_sha: CommitSha::parse("head_sha", &string("head_sha", required("head_sha")?)?)?,
            base_sha: CommitSha::parse("base_sha", &string("base_sha", required("base_sha")?)?)?,
            evidence: list(params, "evidence")?,
        })
    }

    /// The derived incident key: the same project, repository, PR and exact
    /// heads always name the same job.
    pub fn incident_key(&self) -> IncidentKey {
        let mut hasher = blake3::Hasher::new();
        for part in [
            self.project.as_str(),
            self.repository.0.as_str(),
            &self.pr_number.get().to_string(),
            self.head_sha.as_str(),
            self.base_sha.as_str(),
        ] {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part.as_bytes());
        }
        let digest = hasher.finalize().to_hex();
        IncidentKey::parse(&format!("{MERGE_REPAIR_KEY_PREFIX}{}", &digest[..40]))
            .expect("a hex digest after the reserved prefix is a valid key")
    }
}

fn reject_unknown(params: &Map<String, Value>, allowed: &[&str]) -> EscalationResult<()> {
    match params.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => Err(EscalationError::input(
            bounded(key, 64),
            format!(
                "is not a field of this tool (fields: {})",
                allowed.join(", ")
            ),
        )),
        None => Ok(()),
    }
}

fn string(field: &str, value: &Value) -> EscalationResult<String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| EscalationError::input(field, "expected a string"))
}

/// A required, trimmed, non-empty string of at most `max` characters.
fn text(params: &Map<String, Value>, field: &str, max: usize) -> EscalationResult<String> {
    let value = params
        .get(field)
        .ok_or_else(|| EscalationError::input(field, "is required"))?;
    bounded_text(field, value, max)
}

fn bounded_text(field: &str, value: &Value, max: usize) -> EscalationResult<String> {
    let value = string(field, value)?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(EscalationError::input(field, "must not be empty"));
    }
    if trimmed.chars().count() > max {
        return Err(EscalationError::input(
            field,
            format!("must be at most {max} characters"),
        ));
    }
    Ok(trimmed.to_owned())
}

/// A required list of 1..=12 bounded strings.
fn list(params: &Map<String, Value>, field: &str) -> EscalationResult<Vec<String>> {
    let items = params
        .get(field)
        .ok_or_else(|| EscalationError::input(field, "is required"))?
        .as_array()
        .ok_or_else(|| EscalationError::input(field, "expected an array of strings"))?;
    if items.is_empty() || items.len() > MAX_LIST_ENTRIES {
        return Err(EscalationError::input(
            field,
            format!("needs 1..={MAX_LIST_ENTRIES} entries"),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(index, item)| bounded_text(&format!("{field}[{index}]"), item, MAX_TEXT_CHARS))
        .collect()
}

fn optional<T>(
    params: &Map<String, Value>,
    field: &str,
    parse: impl FnOnce(&str, &Value) -> EscalationResult<T>,
) -> EscalationResult<Option<T>> {
    match params.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => parse(field, value).map(Some),
    }
}

fn uuid(field: &str, value: &Value) -> EscalationResult<Uuid> {
    let raw = string(field, value)?;
    // Only the hyphenated 36-character form Anima prints is accepted.
    if raw.len() != 36 {
        return Err(EscalationError::input(
            field,
            "expected a full hyphenated UUID",
        ));
    }
    Uuid::parse_str(&raw)
        .map_err(|_| EscalationError::input(field, "expected a full hyphenated UUID"))
}

/// A PR number from an integer or a decimal string (models send both).
fn pr_number(field: &str, value: &Value) -> EscalationResult<PrNumber> {
    let number = match value {
        Value::Number(n) => n.as_u64(),
        Value::String(s) if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => {
            s.parse().ok()
        }
        _ => None,
    }
    .ok_or_else(|| EscalationError::input(field, "expected a positive integer"))?;
    PrNumber::new(field, number)
}

/// The last `max` characters of `text` (char-boundary safe).
fn tail(text: &str, max: usize) -> String {
    let count = text.chars().count();
    text.chars().skip(count.saturating_sub(max)).collect()
}

/// The first `max` characters of `text`.
fn bounded(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// Who acts next on an incident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NextOwner {
    /// The worker owns it (its turn is running).
    Worker,
    /// The delegating akh: verify a result, retry a dispatch, or decide.
    Akh,
    /// The operator: provider credentials or quota, or a configuration fix.
    Operator,
}

/// What `escalate` / `dispatch_merge_repair` answer.
#[derive(Debug, Clone, Serialize)]
pub struct EscalationReceipt {
    /// The incident key (derived for merge repair).
    pub incident_key: IncidentKey,
    /// What the delegation does.
    pub kind: IncidentKind,
    /// The durable delegation id.
    pub delegation_id: Uuid,
    /// Where it stands. `pending` is admitted but not yet started.
    pub state: IncidentState,
    /// Whether the key already had an incident (nothing new was created).
    pub duplicate: bool,
    /// The registered project.
    pub project: String,
    /// The worker profile.
    pub workspace_profile: String,
    /// The akhsessd session, once created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Why dispatch has not completed yet (the incident stays admitted and is
    /// retried on the next status check).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dispatch_error: Option<IncidentFailure>,
}

/// The worker's turn, as a status answer shows it.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerView {
    /// The turn.
    pub turn_id: String,
    /// akhsessd's turn status.
    pub status: TurnStatus,
    /// The last characters of the worker's final output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tail: Option<String>,
    /// The worker's structured `akh-result` block, if it wrote one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<DelegationOutput>,
}

/// What `escalation_status` answers.
#[derive(Debug, Clone, Serialize)]
pub struct IncidentStatus {
    /// The ledger record (without the job once settled).
    pub incident: IncidentRecord,
    /// The worker's turn, when akhsessd answered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerView>,
    /// Why a pending incident still did not dispatch on this check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dispatch_error: Option<IncidentFailure>,
    /// akhsessd could not be read; the incident is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_unavailable: Option<String>,
    /// Whether the failure's provider class closes a provider gate (the
    /// worker's credential, quota or rate limit needs the operator).
    pub provider_blocked: bool,
    /// Who acts next.
    pub next_owner: NextOwner,
}

/// What a reconciliation pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ReconcileReport {
    /// Unsettled incidents examined.
    pub examined: usize,
    /// Pending incidents now running.
    pub dispatched: usize,
    /// Incidents that settled.
    pub settled: usize,
    /// Incidents still pending after a failed dispatch.
    pub still_pending: usize,
}

/// A clock, injectable for tests.
pub type Clock = Arc<dyn Fn() -> UnixSecs + Send + Sync>;

/// The escalation and merge-repair delegations of one akh.
pub struct EscalationService {
    akh: String,
    native: NativeDelegationConfig,
    escalation_skill: std::path::PathBuf,
    merge_repair_skill: std::path::PathBuf,
    report_thread: Option<Uuid>,
    grant_tools: GrantTools,
    worker_ttl: Duration,
    client: NativeSessionClient,
    issuer: Arc<dyn DelegationGrantIssuer>,
    ledger: IncidentLedger,
    lexicon: ProviderErrorLexicon,
    clock: Clock,
    /// Serialises dispatch so one incident is never driven twice at once.
    dispatch: Mutex<()>,
}

impl std::fmt::Debug for EscalationService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EscalationService")
            .field("akh", &self.akh)
            .field("client", &self.client)
            .field("projects", &self.native.profiles.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl EscalationService {
    /// Build the service from `[delegation.native]`, reading the akhsessd
    /// token file.
    ///
    /// # Errors
    ///
    /// [`EscalationError::Config`] (the tools must then not be registered),
    /// [`EscalationError::Skill`], or a token error.
    pub fn from_config(
        akh: impl Into<String>,
        native: &NativeDelegationConfig,
        lexicon: ProviderErrorLexicon,
        issuer: Arc<dyn DelegationGrantIssuer>,
        ledger: IncidentLedger,
    ) -> EscalationResult<Self> {
        let client = NativeSessionClient::from_config(native)?;
        Self::with_client(akh, native, lexicon, issuer, ledger, client)
    }

    /// Build the service with an explicit akhsessd client.
    ///
    /// # Errors
    ///
    /// See [`Self::from_config`].
    pub fn with_client(
        akh: impl Into<String>,
        native: &NativeDelegationConfig,
        lexicon: ProviderErrorLexicon,
        issuer: Arc<dyn DelegationGrantIssuer>,
        ledger: IncidentLedger,
        client: NativeSessionClient,
    ) -> EscalationResult<Self> {
        let config = |reason: &str| EscalationError::Config {
            reason: reason.into(),
        };
        let escalation: &NativeEscalationConfig = &native.escalation;
        if !native.enabled || !escalation.enabled {
            return Err(config(
                "both `[delegation.native] enabled` and `[delegation.native.escalation] enabled` must be true",
            ));
        }
        if native.profiles.is_empty() {
            return Err(config("no project is registered under `profiles`"));
        }
        if native.profiles.iter().any(|(project, profile)| {
            project.trim().is_empty() || profile.workspace_profile.trim().is_empty()
        }) {
            return Err(config(
                "a registered project or its workspace profile is empty",
            ));
        }
        if native
            .profiles
            .values()
            .any(|profile| profile.repository.as_deref().is_some_and(str::is_empty))
        {
            return Err(config(
                "a registered project's `repository` is set but empty; omit it instead to \
                 leave that project's sessions unbootstrapped",
            ));
        }
        let escalation_skill = escalation
            .escalation_skill
            .clone()
            .ok_or_else(|| config("`escalation_skill` is not set"))?;
        let merge_repair_skill = escalation
            .merge_repair_skill
            .clone()
            .ok_or_else(|| config("`merge_repair_skill` is not set"))?;
        let report_thread = escalation
            .report_thread
            .as_deref()
            .map(|thread| {
                Uuid::parse_str(thread).map_err(|_| config("`report_thread` is not a UUID"))
            })
            .transpose()?;
        if escalation.worker_timeout_seconds == 0 {
            return Err(config("`worker_timeout_seconds` must be at least 1"));
        }
        // Fail registration on a broken skillpack instead of the first call.
        for (kind, path) in [
            (IncidentKind::Escalation, &escalation_skill),
            (IncidentKind::MergeRepair, &merge_repair_skill),
        ] {
            load_skill(kind, path)?;
        }
        let worker_ttl = (Duration::from_secs(escalation.worker_timeout_seconds) + GRANT_MARGIN)
            .min(MAX_GRANT_TTL);
        Ok(Self {
            akh: akh.into(),
            native: native.clone(),
            escalation_skill,
            merge_repair_skill,
            report_thread,
            grant_tools: escalation.grant_tools.clone(),
            worker_ttl,
            client,
            issuer,
            ledger,
            lexicon,
            clock: Arc::new(UnixSecs::now),
            dispatch: Mutex::new(()),
        })
    }

    /// Replace the clock (tests).
    pub fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    /// The ledger.
    pub fn ledger(&self) -> &IncidentLedger {
        &self.ledger
    }

    /// The registered projects.
    pub fn projects(&self) -> Vec<String> {
        self.native.profiles.keys().cloned().collect()
    }

    fn now(&self) -> UnixSecs {
        (self.clock)()
    }

    fn profile(&self, project: &str) -> EscalationResult<&NativeProjectProfile> {
        self.native
            .project_profile(project)
            .ok_or_else(|| EscalationError::UnknownProject {
                project: project.to_owned(),
                registered: self.projects().join(", "),
            })
    }

    /// Escalate a bounded blocker to a worker.
    ///
    /// # Errors
    ///
    /// Validation, an unknown project, the daily limit or a ledger store
    /// failure. A dispatch failure after admission is not an error: the
    /// receipt carries it and the incident is retried later.
    pub fn escalate(&self, request: &EscalationRequest) -> EscalationResult<EscalationReceipt> {
        let job = json!({
            "kind": IncidentKind::Escalation,
            "incident_key": request.incident_key,
            "request": request,
        });
        self.admit_and_dispatch(
            request.incident_key.clone(),
            IncidentKind::Escalation,
            &request.project,
            request.head_sha.as_ref().map(CommitSha::as_str),
            job,
        )
    }

    /// Dispatch a merge-conflict repair job to a worker.
    ///
    /// # Errors
    ///
    /// As [`Self::escalate`].
    pub fn dispatch_merge_repair(
        &self,
        request: &MergeRepairRequest,
    ) -> EscalationResult<EscalationReceipt> {
        let key = request.incident_key();
        let job = json!({
            "kind": IncidentKind::MergeRepair,
            "incident_key": key,
            "request": request,
        });
        self.admit_and_dispatch(
            key,
            IncidentKind::MergeRepair,
            &request.project,
            Some(request.head_sha.as_str()),
            job,
        )
    }

    fn admit_and_dispatch(
        &self,
        key: IncidentKey,
        kind: IncidentKind,
        project: &str,
        head_sha: Option<&str>,
        job: Value,
    ) -> EscalationResult<EscalationReceipt> {
        let profile = self.profile(project)?;
        if profile.repository.is_some() && head_sha.is_none() {
            // The worker session would stall in `bootstrap_pending` and be
            // force-retired without a commit to check the repository out at
            // (the 2026-09-27 incident); fail the call itself rather than
            // admitting an incident that can never make progress.
            return Err(EscalationError::input(
                "head_sha",
                format!(
                    "is required: project `{project}`'s delegation profile declares a \
                     repository to bootstrap, so the worker session needs the exact commit \
                     (the current base/head from Anima) to check out"
                ),
            ));
        }
        let admitted = self.ledger.admit(
            NewIncident {
                request_id: request_id_for(&self.akh, kind, &key),
                key: key.clone(),
                kind,
                project: project.to_owned(),
                workspace_profile: profile.workspace_profile.clone(),
                head_sha: head_sha.map(str::to_owned),
                job,
            },
            self.native.max_new_per_day,
            self.now(),
        )?;
        let (record, dispatch_error) = if admitted.record.state == IncidentState::Pending {
            self.drive_recording(&key)?
        } else {
            (admitted.record, None)
        };
        Ok(EscalationReceipt {
            incident_key: record.incident_key.clone(),
            kind: record.kind,
            delegation_id: record.delegation_id,
            state: record.state,
            duplicate: admitted.duplicate,
            project: record.project.clone(),
            workspace_profile: record.workspace_profile.clone(),
            session_id: record.session_id.clone(),
            dispatch_error,
        })
    }

    /// Drive a pending incident; a failure is recorded on the incident and
    /// returned alongside it instead of failing the call.
    fn drive_recording(
        &self,
        key: &IncidentKey,
    ) -> EscalationResult<(IncidentRecord, Option<IncidentFailure>)> {
        match self.drive(key) {
            Ok(record) => Ok((record, None)),
            Err(error) => {
                let failure = error.as_failure();
                let settled = matches!(
                    error,
                    EscalationError::RuntimeMismatch { .. }
                        | EscalationError::RepositoryBootstrapConflict { .. }
                );
                tracing::warn!(
                    target: "akh::delegation",
                    akh = %self.akh, incident = %key, error = ?miette::Report::new(error),
                    "incident dispatch did not complete"
                );
                let record = self.ledger.update(key, self.now(), |r| {
                    if !settled {
                        r.last_error = Some(failure.clone());
                    }
                })?;
                Ok((record, Some(failure)))
            }
        }
    }

    /// Create the worker session and submit the brief, resuming from
    /// whatever the ledger already recorded.
    fn drive(&self, key: &IncidentKey) -> EscalationResult<IncidentRecord> {
        let _dispatch = self.dispatch.lock().unwrap_or_else(|e| e.into_inner());
        let record = self
            .ledger
            .get(key)
            .ok_or_else(|| LedgerError::UnknownIncident {
                key: key.to_string(),
            })?;
        if record.state != IncidentState::Pending {
            return Ok(record);
        }
        let profile = self.profile(&record.project)?.clone();
        let session = match &record.session_id {
            Some(id) => self.client.session(id)?,
            None => self.find_or_create_session(&record)?,
        };
        let record = self.ledger.get(key).unwrap_or(record);
        // Prepare the session's repository-checkout intent before its
        // workspace is needed (53g PR bootstrap fix, 2026-09-27 incident):
        // akhsessd's allocator otherwise stalls the session on
        // `bootstrap_pending` and force-retires it as stalled. Safe to repeat
        // every dispatch while the incident stays pending; akhsessd answers
        // the same receipt for a retry with the same commit.
        if let Some(repository) = &profile.repository {
            if let Err(error) = self.prepare_repository(&record, &session.id, repository) {
                if matches!(error, EscalationError::RepositoryBootstrapConflict { .. }) {
                    self.release(&record, &session.id);
                    let failure = error.as_failure();
                    self.ledger.update(key, self.now(), |r| {
                        r.state = IncidentState::Failed;
                        r.last_error = Some(failure);
                    })?;
                }
                return Err(error);
            }
        }
        if let Some(expected) = &profile.runtime_profile {
            let actual = session.runtime.as_ref().map(|r| r.profile.as_str());
            if actual != Some(expected.as_str()) {
                let error = EscalationError::RuntimeMismatch {
                    session_id: session.id.clone(),
                    expected: expected.clone(),
                    actual: actual.unwrap_or("none").to_owned(),
                };
                self.release(&record, &session.id);
                let failure = error.as_failure();
                self.ledger.update(key, self.now(), |r| {
                    r.state = IncidentState::Failed;
                    r.last_error = Some(failure);
                })?;
                return Err(error);
            }
        }
        let turn_id = match (&record.turn_id, session.turns.first()) {
            (Some(turn), _) => turn.clone(),
            (None, Some(turn)) => turn.id.clone(),
            (None, None) => self.submit_brief(&record, &session.id)?,
        };
        Ok(self.ledger.update(key, self.now(), |r| {
            r.turn_id = Some(turn_id);
            r.state = IncidentState::Running;
            r.last_error = None;
        })?)
    }

    fn find_or_create_session(&self, record: &IncidentRecord) -> EscalationResult<SessionView> {
        let key = &record.incident_key;
        if record.create_attempts > 0
            && let Some(session) = self.lookup(record)?
        {
            return Ok(session);
        }
        if let Some(old) = &record.grant_id
            && let Err(error) = self.issuer.revoke(old)
        {
            tracing::debug!(target: "akh::delegation", grant_id = %old, %error, "stale incident grant not revoked; it expires on its TTL");
        }
        let grant = self.issuer.issue(&GrantRequest {
            principal: delegate_principal_for(&self.native.service, &record.workspace_profile),
            service_principal: self.native.core_principal.clone(),
            tools: self.grant_tools.clone(),
            ttl: self.worker_ttl,
        })?;
        // Recorded before the create leaves: a crash after sending is then
        // reconciled by lookup, never by a second create.
        self.ledger.update(key, self.now(), |r| {
            r.create_attempts = r.create_attempts.saturating_add(1);
            r.grant_id = Some(grant.id.clone());
        })?;
        let create = new_session_request(
            &grant.akh,
            &record.workspace_profile,
            &record.request_id,
            grant.token.clone(),
        );
        let session = match self.client.create_session(&create) {
            Ok(session) => session,
            Err(error) if error.may_have_been_applied() => {
                let backoff = Duration::from_millis(self.native.poll_interval_ms.max(10));
                match recover_created_session(&self.client, &create, backoff) {
                    Some(id) => self.client.session(&id)?,
                    None => return Err(error.into()),
                }
            }
            Err(error) if error.is_conflict() => {
                // The request id already made a session, with the grant the
                // record held before this attempt: adopt it and drop the
                // grant minted for this attempt.
                if let Err(revoke) = self.issuer.revoke(&grant.id) {
                    tracing::debug!(target: "akh::delegation", grant_id = %grant.id, error = %revoke, "unused incident grant not revoked");
                }
                let previous = record.grant_id.clone();
                self.ledger
                    .update(key, self.now(), |r| r.grant_id = previous)?;
                match self.lookup(record)? {
                    Some(session) => return Ok(session),
                    None => return Err(error.into()),
                }
            }
            Err(error) => {
                // Refused before anything was created: the grant is unused.
                if let Err(revoke) = self.issuer.revoke(&grant.id) {
                    tracing::debug!(target: "akh::delegation", grant_id = %grant.id, error = %revoke, "unused incident grant not revoked");
                }
                self.ledger.update(key, self.now(), |r| r.grant_id = None)?;
                return Err(error.into());
            }
        };
        self.adopt(record, &session, &grant.id)?;
        Ok(session)
    }

    /// The session an earlier create with the incident's request id made.
    fn lookup(&self, record: &IncidentRecord) -> EscalationResult<Option<SessionView>> {
        match self.client.session_by_request(&record.request_id) {
            Ok(session) => {
                tracing::info!(
                    target: "akh::delegation",
                    akh = %self.akh, incident = %record.incident_key, session_id = %session.id,
                    "adopted the session of an earlier create by its request id"
                );
                let grant = record.grant_id.clone().unwrap_or_default();
                self.adopt(record, &session, &grant)?;
                Ok(Some(session))
            }
            Err(error) if error.is_not_found() => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn adopt(
        &self,
        record: &IncidentRecord,
        session: &SessionView,
        grant_id: &str,
    ) -> EscalationResult<()> {
        self.ledger.update(&record.incident_key, self.now(), |r| {
            r.session_id = Some(session.id.clone());
        })?;
        if !grant_id.is_empty()
            && let Err(error) = self.issuer.bind_session(grant_id, &session.id)
        {
            // The core binds an unbound grant to the first session that
            // presents it, so this is belt and braces.
            tracing::debug!(target: "akh::delegation", grant_id, session_id = %session.id, %error, "grant not bound explicitly");
        }
        Ok(())
    }

    fn skill_path(&self, kind: IncidentKind) -> &std::path::Path {
        match kind {
            IncidentKind::Escalation => &self.escalation_skill,
            IncidentKind::MergeRepair => &self.merge_repair_skill,
        }
    }

    /// The worker's first input: the skill's procedures, then the job.
    fn brief(&self, record: &IncidentRecord) -> EscalationResult<String> {
        let skill = load_skill(record.kind, self.skill_path(record.kind))?;
        let mut job = record.job.clone().unwrap_or(Value::Null);
        if let Value::Object(map) = &mut job {
            map.insert("delegation_id".into(), json!(record.delegation_id));
            map.insert("delegating_akh".into(), json!(self.akh));
            map.insert("skill".into(), json!(skill.label()));
            if let Some(thread) = self.report_thread {
                map.insert("report_thread".into(), json!(thread));
            }
        }
        let job = serde_json::to_string_pretty(&job).map_err(|error| EscalationError::Config {
            reason: format!("the job could not be encoded: {error}"),
        })?;
        Ok(format!(
            "{}\n\n```akh-delegation-job\n{job}\n```\n",
            skill.text.trim_end()
        ))
    }

    fn submit_brief(&self, record: &IncidentRecord, session_id: &str) -> EscalationResult<String> {
        let input = SubmitInput {
            request_id: format!("{}-brief", record.request_id),
            text: self.brief(record)?,
            artifacts: vec![],
        };
        match self.client.submit(session_id, &input) {
            Ok(turn) => Ok(turn.id),
            Err(error) if error.is_conflict() || error.may_have_been_applied() => self
                .client
                .session(session_id)?
                .turns
                .first()
                .map(|turn| turn.id.clone())
                .ok_or_else(|| error.into()),
            Err(error) => Err(error.into()),
        }
    }

    /// Prepare the session's repository-checkout intent (53g PR bootstrap
    /// fix): `POST /v1/sessions/{id}/repository` with the incident's exact
    /// commit. A transport failure or other rejection is returned as-is and
    /// retried on the incident's next dispatch (the ledger keeps it
    /// pending); a conflicting intent already on the session is
    /// [`EscalationError::RepositoryBootstrapConflict`], which the caller
    /// settles as failed.
    ///
    /// # Errors
    ///
    /// [`EscalationError::Config`] if the incident has no `head_sha` (should
    /// not happen: `admit_and_dispatch` requires one whenever the project's
    /// profile carries a `repository`), or the akhsessd call's own error.
    fn prepare_repository(
        &self,
        record: &IncidentRecord,
        session_id: &str,
        repository: &str,
    ) -> EscalationResult<()> {
        let Some(head_sha) = record.head_sha.as_deref() else {
            return Err(EscalationError::Config {
                reason: format!(
                    "incident `{}` has no head_sha but project `{}`'s delegation profile \
                     declares a repository to bootstrap",
                    record.incident_key, record.project
                ),
            });
        };
        let request = akh_session_types::bootstrap::BootstrapRepository {
            request_id: format!("{}-repository", record.request_id),
            checkout: akh_session_types::bootstrap::RepositoryCheckout {
                repository: repository.to_owned(),
                revision: head_sha.to_owned(),
            },
        };
        match self.client.prepare_repository(session_id, &request) {
            Ok(_receipt) => Ok(()),
            Err(error) if error.is_conflict() => Err(EscalationError::RepositoryBootstrapConflict {
                session_id: session_id.to_owned(),
                project: record.project.clone(),
            }),
            Err(error) => Err(error.into()),
        }
    }

    /// Revoke the incident's grant and close its session, best effort.
    fn release(&self, record: &IncidentRecord, session_id: &str) {
        if let Some(grant) = &record.grant_id
            && let Err(error) = self.issuer.revoke(grant)
        {
            tracing::warn!(target: "akh::delegation", grant_id = %grant, %error, "incident grant not revoked; it expires on its TTL");
        }
        if let Err(error) = self.client.close_session(session_id) {
            tracing::warn!(
                target: "akh::delegation",
                session_id, error = ?miette::Report::new(error),
                "incident session not archived or retired; akhsessd's sweeps close it"
            );
        }
    }

    /// Read an incident: dispatch it if still pending, settle it if its turn
    /// ended, and explain a failure.
    ///
    /// # Errors
    ///
    /// An invalid or unknown key, or a ledger store failure. akhsessd being
    /// unreachable is not an error (`status_unavailable`).
    pub fn status(&self, key: &IncidentKey) -> EscalationResult<IncidentStatus> {
        let mut record = self
            .ledger
            .get(key)
            .ok_or_else(|| LedgerError::UnknownIncident {
                key: key.to_string(),
            })?;
        let mut dispatch_error = None;
        if record.state == IncidentState::Pending {
            let (driven, error) = self.drive_recording(key)?;
            record = driven;
            dispatch_error = error;
        }
        let mut worker = None;
        let mut status_unavailable = None;
        if let Some(session_id) = record.session_id.clone() {
            match self.client.session(&session_id) {
                Ok(view) => {
                    let turn = record
                        .turn_id
                        .as_ref()
                        .and_then(|id| view.turns.iter().find(|t| &t.id == id))
                        .cloned();
                    if let Some(turn) = &turn {
                        worker = Some(worker_view(turn));
                        if record.state == IncidentState::Running && turn.status.terminal() {
                            record = self.settle(&record, &session_id, turn)?;
                        }
                    }
                }
                Err(error) if error.is_not_found() && record.state == IncidentState::Running => {
                    let failure = IncidentFailure {
                        code: "session_missing".into(),
                        message: bounded(&error.to_string(), MAX_FAILURE_CHARS),
                        provider_class: None,
                    };
                    self.release(&record, &session_id);
                    record = self.ledger.update(key, self.now(), |r| {
                        r.state = IncidentState::Failed;
                        r.last_error = Some(failure);
                    })?;
                }
                Err(error) => status_unavailable = Some(error.to_string()),
            }
        }
        let provider_blocked = record
            .last_error
            .as_ref()
            .and_then(|failure| failure.provider_class.as_deref())
            .is_some_and(|class| matches!(class, "authentication" | "quota" | "rate_limited"));
        let next_owner = match record.state {
            IncidentState::Running => NextOwner::Worker,
            IncidentState::Failed
                if provider_blocked
                    || record.last_error.as_ref().is_some_and(|f| {
                        f.code.ends_with("runtime_mismatch") || f.code.ends_with("bootstrap_conflict")
                    }) =>
            {
                NextOwner::Operator
            }
            IncidentState::Pending
                if dispatch_error.as_ref().is_some_and(|f| {
                    f.code.ends_with("::config") || f.code.ends_with("::skill")
                }) =>
            {
                NextOwner::Operator
            }
            _ => NextOwner::Akh,
        };
        Ok(IncidentStatus {
            incident: record,
            worker,
            dispatch_error,
            status_unavailable,
            provider_blocked,
            next_owner,
        })
    }

    fn settle(
        &self,
        record: &IncidentRecord,
        session_id: &str,
        turn: &TurnView,
    ) -> EscalationResult<IncidentRecord> {
        let (state, failure) = match turn.status {
            TurnStatus::Completed => (IncidentState::Succeeded, None),
            TurnStatus::Cancelled => (
                IncidentState::Cancelled,
                turn.error
                    .as_ref()
                    .map(|f| self.failure_of(&f.code, &f.message)),
            ),
            _ => (
                IncidentState::Failed,
                Some(match &turn.error {
                    Some(fault) => self.failure_of(&fault.code, &fault.message),
                    None => IncidentFailure {
                        code: "turn_failed".into(),
                        message: String::new(),
                        provider_class: None,
                    },
                }),
            ),
        };
        self.release(record, session_id);
        Ok(self.ledger.update(&record.incident_key, self.now(), |r| {
            r.state = state;
            r.last_error = failure;
        })?)
    }

    /// A failed turn's fault, classified by the akh's `[provider]` lexicon.
    fn failure_of(&self, code: &str, message: &str) -> IncidentFailure {
        IncidentFailure {
            code: code.to_owned(),
            message: bounded(message, MAX_FAILURE_CHARS),
            provider_class: classify_identifier(code, &self.lexicon)
                .map(|class| class.label().to_owned()),
        }
    }

    /// Reconcile every unsettled incident: dispatch pending ones (adopting a
    /// session an earlier create made) and settle running ones whose turn
    /// ended. Run at startup.
    pub fn reconcile(&self) -> ReconcileReport {
        let mut report = ReconcileReport::default();
        for record in self.ledger.unsettled() {
            report.examined += 1;
            let before = record.state;
            match self.status(&record.incident_key) {
                Ok(status) => {
                    let after = status.incident.state;
                    if before == IncidentState::Pending && after == IncidentState::Running {
                        report.dispatched += 1;
                    }
                    if after.is_settled() {
                        report.settled += 1;
                    } else if after == IncidentState::Pending {
                        report.still_pending += 1;
                    }
                }
                Err(error) => {
                    report.still_pending += usize::from(before == IncidentState::Pending);
                    tracing::warn!(
                        target: "akh::delegation",
                        akh = %self.akh, incident = %record.incident_key,
                        error = ?miette::Report::new(error),
                        "incident not reconciled"
                    );
                }
            }
        }
        report
    }
}

fn load_skill(kind: IncidentKind, path: &std::path::Path) -> EscalationResult<SkillProcedures> {
    SkillProcedures::load(path).map_err(|source| EscalationError::Skill {
        kind,
        path: path.display().to_string(),
        source,
    })
}

fn worker_view(turn: &TurnView) -> WorkerView {
    let output = turn.output.as_deref().filter(|o| !o.is_empty());
    WorkerView {
        turn_id: turn.id.clone(),
        status: turn.status,
        output_tail: output.map(|o| tail(o, OUTPUT_TAIL_CHARS)),
        result: output.and_then(parse_structured_output),
    }
}

#[cfg(test)]
mod tests;
