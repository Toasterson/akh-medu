//! Provider-failure taxonomy and the per-akh provider gate (Phase 53g PR 4).
//!
//! Plan: `docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md` ("PR 4").
//! This replaces `deploy/roman-mini/provider_failure.py`, which classified
//! provider errors by searching English error prose. Here nothing reads prose:
//!
//! - [`classify`] maps a [`ProviderFailureEvidence`] to a
//!   [`ProviderFailureClass`] from the HTTP status, the `Retry-After` header and
//!   the *exact* structured `error.code` / `error.type` of the response body,
//!   matched against the [`ProviderErrorLexicon`] from the akh's `[provider]`
//!   table. The lexicon is empty by default, so an unconfigured akh classifies
//!   by status alone.
//! - [`ProviderGate`] is the per-akh state machine the daemon consults before
//!   every autonomous model turn (`LlmTurn`, `DjehutyReply`). Quota and rate
//!   limits cool the gate on a ladder (default 30 min, 1 h, 2 h, 4 h);
//!   authentication failures block it until a model turn succeeds or an
//!   operator resets it (`_akh/provider/gate/reset`). The gate is persisted in
//!   the engine's durable metadata under [`PROVIDER_GATE_KEY`] and carried in
//!   the snapshot extra (`extra.provider.gate`), like the autonomy budget
//!   ledger.
//! - [`ProviderObservationScope`] is a thread-local record of what the akh's
//!   own model calls did during one reactor occurrence. The agentic loop
//!   swallows backend errors into a refused turn, so the call sites report
//!   each backend result here ([`observe_chat_result`]) and the daemon reads
//!   the last one.
//!
//! # Which failures close the gate
//!
//! Authentication, quota and rate-limit failures close it: retrying them
//! cannot succeed before the provider (or the operator) changes something.
//! Transient failures (5xx, timeouts, transport errors) and unknown ones do
//! not. They keep the reactor's bounded retry, which never feeds the
//! no-progress trip-wire when the model was not reached.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::autonomy::UnixSecs;
use super::chat_backend::ChatBackendError;
use crate::engine::Engine;

/// Durable-metadata key of the per-akh provider gate. The `v1` suffix
/// versions the JSON shape of [`ProviderGateState`].
pub const PROVIDER_GATE_KEY: &[u8] = b"provider:gate:v1";

/// How long a reactor occurrence waits before re-checking an
/// authentication-blocked gate. The check is free (no model call); the
/// interval only bounds how long a claim waits after an operator reset.
pub const AUTH_BLOCKED_RECHECK: Duration = Duration::from_secs(15 * 60);

/// The default cooldown ladder: 30 min, 1 h, 2 h, 4 h.
pub const DEFAULT_COOLDOWN_LADDER_SECS: [u64; 4] = [1_800, 3_600, 7_200, 14_400];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the provider gate's persistence.
#[derive(Debug, Error, Diagnostic)]
pub enum ProviderGateError {
    /// The persisted gate could not be decoded.
    #[error("the provider gate at `{key}` is corrupt: {source}")]
    #[diagnostic(
        code(akh::provider::corrupt_gate),
        help(
            "The daemon starts with an open gate and overwrites the record on the next \
             provider failure. To inspect it first, stop the akh and read the durable meta \
             key `{key}`. An open gate costs at most one failed model call per occurrence."
        )
    )]
    CorruptGate {
        /// The metadata key that failed to decode.
        key: String,
        /// The decode error.
        #[source]
        source: serde_json::Error,
    },

    /// Reading the gate from the durable store failed.
    #[error("failed to read the provider gate at `{key}`")]
    #[diagnostic(
        code(akh::provider::gate_read),
        help(
            "The engine's durable store did not answer. Check the akh's data directory \
             (disk full, permissions, or a redb lock held by another process). The daemon \
             keeps using its in-memory gate meanwhile."
        )
    )]
    GateRead {
        /// The metadata key.
        key: String,
        /// The store error.
        #[source]
        source: crate::error::StoreError,
    },

    /// Writing the gate to the durable store failed.
    #[error("failed to persist the provider gate at `{key}`")]
    #[diagnostic(
        code(akh::provider::gate_write),
        help(
            "The gate is still enforced in memory until the daemon restarts, but a restart \
             forgets it. Check free disk space and the data directory's permissions."
        )
    )]
    GateWrite {
        /// The metadata key.
        key: String,
        /// The store error.
        #[source]
        source: crate::error::StoreError,
    },

    /// The gate could not be encoded (a bug: the state is plain data).
    #[error("failed to encode the provider gate")]
    #[diagnostic(
        code(akh::provider::gate_encode),
        help("This is a bug in akh-medu; please report it with the daemon log.")
    )]
    GateEncode {
        /// The encode error.
        #[source]
        source: serde_json::Error,
    },

    /// The engine has no durable tier, so there is no persisted gate to
    /// inspect or reset.
    #[error("this akh has no durable store, so its provider gate is not persisted")]
    #[diagnostic(
        code(akh::provider::not_durable),
        help(
            "An in-memory akh keeps its provider gate inside the running daemon only. \
             Restart the daemon to clear it, or give the workspace a data directory."
        )
    )]
    NotDurable,
}

/// Result alias for the provider gate.
pub type ProviderGateResult<T> = Result<T, ProviderGateError>;

/// An invalid `[provider]` table.
#[derive(Debug, Error, Diagnostic)]
pub enum ProviderConfigError {
    /// `cooldown_ladder_secs` was empty.
    #[error("`[provider] cooldown_ladder_secs` must list at least one cooldown")]
    #[diagnostic(
        code(akh::provider::empty_ladder),
        help(
            "Remove the key to use the default [1800, 3600, 7200, 14400], or list the \
             cooldowns in seconds, shortest first."
        )
    )]
    EmptyLadder,

    /// A ladder step was zero seconds, which would not cool anything.
    #[error("`[provider] cooldown_ladder_secs` step {index} is 0 seconds")]
    #[diagnostic(
        code(akh::provider::zero_cooldown),
        help(
            "Every step must be at least 1 second. A zero step would retry a failing \
             provider immediately, which is what the gate exists to prevent."
        )
    )]
    ZeroCooldown {
        /// The 0-based index of the offending step.
        index: usize,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The cooldown ladder: step `n` of consecutive quota / rate-limit failures
/// cools the gate for `ladder[n]` (the last step repeats). Never empty and no
/// step is zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<u64>", into = "Vec<u64>")]
pub struct CooldownLadder(Vec<Duration>);

impl CooldownLadder {
    /// Build a ladder from seconds, rejecting an empty ladder or a zero step.
    pub fn from_secs(secs: &[u64]) -> Result<Self, ProviderConfigError> {
        if secs.is_empty() {
            return Err(ProviderConfigError::EmptyLadder);
        }
        if let Some(index) = secs.iter().position(|&s| s == 0) {
            return Err(ProviderConfigError::ZeroCooldown { index });
        }
        Ok(Self(
            secs.iter().copied().map(Duration::from_secs).collect(),
        ))
    }

    /// The cooldown for `step` (clamped to the last step).
    pub fn cooldown(&self, step: u32) -> Duration {
        let index = usize::try_from(step)
            .unwrap_or(usize::MAX)
            .min(self.0.len() - 1);
        self.0[index]
    }

    /// The highest step index that has its own cooldown.
    pub fn last_step(&self) -> u32 {
        u32::try_from(self.0.len() - 1).unwrap_or(u32::MAX)
    }
}

impl Default for CooldownLadder {
    fn default() -> Self {
        Self(
            DEFAULT_COOLDOWN_LADDER_SECS
                .iter()
                .copied()
                .map(Duration::from_secs)
                .collect(),
        )
    }
}

impl TryFrom<Vec<u64>> for CooldownLadder {
    type Error = ProviderConfigError;

    fn try_from(secs: Vec<u64>) -> Result<Self, Self::Error> {
        Self::from_secs(&secs)
    }
}

impl From<CooldownLadder> for Vec<u64> {
    fn from(ladder: CooldownLadder) -> Self {
        ladder.0.iter().map(Duration::as_secs).collect()
    }
}

/// Exact structured error identifiers (`error.code` or `error.type` in a
/// provider's JSON error body) that name a failure class. Matching is exact
/// and case-sensitive: these are protocol tokens, never prose. All lists are
/// empty by default.
///
/// ```toml
/// [provider]
/// quota_error_codes = ["insufficient_quota"]
/// auth_error_codes = ["invalid_api_key"]
/// rate_limit_error_codes = ["rate_limit_exceeded"]
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderErrorLexicon {
    /// Codes that mean the credential is invalid, expired or revoked.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub auth_error_codes: BTreeSet<String>,
    /// Codes that mean the account's quota or credit is spent.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub quota_error_codes: BTreeSet<String>,
    /// Codes that mean the request rate is limited.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub rate_limit_error_codes: BTreeSet<String>,
}

impl ProviderErrorLexicon {
    /// The class a structured identifier names, if any. Authentication wins
    /// over quota, and quota over rate limiting, when an identifier is
    /// (mis)listed under several classes.
    fn class_of(&self, identifier: &str) -> Option<LexiconClass> {
        if self.auth_error_codes.contains(identifier) {
            Some(LexiconClass::Authentication)
        } else if self.quota_error_codes.contains(identifier) {
            Some(LexiconClass::Quota)
        } else if self.rate_limit_error_codes.contains(identifier) {
            Some(LexiconClass::RateLimited)
        } else {
            None
        }
    }
}

/// The per-akh `[provider]` table.
///
/// ```toml
/// [provider]
/// cooldown_ladder_secs = [1800, 3600, 7200, 14400]
/// quota_error_codes = ["insufficient_quota"]
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Cooldowns in seconds for consecutive quota / rate-limit failures.
    #[serde(default)]
    pub cooldown_ladder_secs: CooldownLadder,
    /// Structured error identifiers per class.
    #[serde(flatten)]
    pub lexicon: ProviderErrorLexicon,
}

impl ProviderConfig {
    /// Whether this is the default table (so `akh.toml` can omit it).
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

// ---------------------------------------------------------------------------
// Evidence captured by the backends
// ---------------------------------------------------------------------------

/// A parsed `Retry-After` header (RFC 9110 §10.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryAfter {
    /// `Retry-After: <seconds>`.
    Delay(Duration),
    /// `Retry-After: <HTTP-date>`.
    At(UnixSecs),
}

impl RetryAfter {
    /// Parse a header value. `None` for a value that is neither a
    /// non-negative integer nor an HTTP-date.
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        if let Ok(secs) = value.parse::<u64>() {
            return Some(Self::Delay(Duration::from_secs(secs)));
        }
        let at = httpdate::parse_http_date(value).ok()?;
        let secs = at
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Some(Self::At(UnixSecs::from_secs(secs)))
    }

    /// How long to wait from `now` (zero for a date in the past).
    pub fn delay_from(self, now: UnixSecs) -> Duration {
        match self {
            Self::Delay(delay) => delay,
            Self::At(at) => Duration::from_secs(at.as_secs().saturating_sub(now.as_secs())),
        }
    }
}

/// The machine-readable parts of a provider's error response: the
/// `Retry-After` header and the structured `error.code` / `error.type` of a
/// JSON body. The human-readable message is deliberately not captured here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderErrorDetail {
    /// The parsed `Retry-After` header.
    pub retry_after: Option<RetryAfter>,
    /// `error.code` of a JSON error body (a number is kept as its digits).
    pub error_code: Option<String>,
    /// `error.type` of a JSON error body.
    pub error_type: Option<String>,
}

impl ProviderErrorDetail {
    /// Extract the detail from a `Retry-After` header value and a response
    /// body. A body that is not JSON, or has no `error` object, yields no
    /// structured identifiers.
    pub fn from_parts(retry_after: Option<&str>, body: &str) -> Self {
        let parsed = serde_json::from_str::<serde_json::Value>(body).ok();
        let error = parsed.as_ref().and_then(|v| v.get("error"));
        let identifier = |key: &str| {
            error
                .and_then(|e| e.get(key))
                .and_then(|value| match value {
                    serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
                    serde_json::Value::Number(n) => Some(n.to_string()),
                    _ => None,
                })
        };
        Self {
            retry_after: retry_after.and_then(RetryAfter::parse),
            error_code: identifier("code"),
            error_type: identifier("type"),
        }
    }

    /// Consume a non-2xx `ureq` response: its body text and its detail.
    pub fn from_response(response: ureq::Response) -> (String, Self) {
        let retry_after = response.header("Retry-After").map(str::to_owned);
        let body = response.into_string().unwrap_or_default();
        let detail = Self::from_parts(retry_after.as_deref(), &body);
        (body, detail)
    }
}

/// What a failed provider call left behind, as captured by the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderFailureEvidence {
    /// The provider answered with a non-2xx status.
    Http {
        /// The HTTP status code.
        status: u16,
        /// Header and structured-body detail.
        detail: ProviderErrorDetail,
    },
    /// No usable answer: DNS, TLS, connection refused or reset, timeout, or a
    /// stream that broke mid-response.
    Transport,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Why a provider call failed, as far as the akh's scheduling is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderFailureClass {
    /// The credential is missing, invalid, expired or revoked (401/403).
    /// Retrying cannot help until an operator repairs it.
    Authentication,
    /// The account's quota or credit is spent (402, or a lexicon code).
    Quota,
    /// Too many requests (429, or a lexicon code).
    RateLimited {
        /// The provider's `Retry-After`, resolved against the failure time.
        retry_after: Option<Duration>,
    },
    /// A server error (5xx, 408) or a transport failure.
    Transient,
    /// Anything else (a 4xx the akh caused, for example).
    Unknown,
}

impl ProviderFailureClass {
    /// Whether this failure closes the provider gate (see the module docs).
    pub const fn closes_gate(self) -> bool {
        matches!(
            self,
            Self::Authentication | Self::Quota | Self::RateLimited { .. }
        )
    }

    /// Stable label for logs and episodes.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::Quota => "quota",
            Self::RateLimited { .. } => "rate_limited",
            Self::Transient => "transient",
            Self::Unknown => "unknown",
        }
    }
}

/// The classes a lexicon identifier can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LexiconClass {
    Authentication,
    Quota,
    RateLimited,
}

/// The class a structured failure identifier names in `lexicon`, if any.
///
/// Used where no HTTP status is available, for example a delegated worker's
/// failed turn, which akhsessd reports as a `Fault` with a machine `code`
/// (53g PR 7). The match is exact, like [`classify`]'s: a code the lexicon
/// does not list is `None`, never a guess.
pub fn classify_identifier(
    identifier: &str,
    lexicon: &ProviderErrorLexicon,
) -> Option<ProviderFailureClass> {
    lexicon.class_of(identifier).map(|class| match class {
        LexiconClass::Authentication => ProviderFailureClass::Authentication,
        LexiconClass::Quota => ProviderFailureClass::Quota,
        LexiconClass::RateLimited => ProviderFailureClass::RateLimited { retry_after: None },
    })
}

/// Classify a provider failure.
///
/// Precedence: an exact lexicon match on `error.code`, then on `error.type`,
/// then the HTTP status (401/403 authentication, 402 quota, 429 rate limited,
/// 408/5xx transient, anything else unknown). A lexicon match wins over the
/// status because it is more specific: some providers answer an exhausted
/// quota with 429. A transport failure is always transient. `now` resolves an
/// HTTP-date `Retry-After`.
pub fn classify(
    evidence: &ProviderFailureEvidence,
    lexicon: &ProviderErrorLexicon,
    now: UnixSecs,
) -> ProviderFailureClass {
    let ProviderFailureEvidence::Http { status, detail } = evidence else {
        return ProviderFailureClass::Transient;
    };
    let retry_after = detail.retry_after.map(|r| r.delay_from(now));
    let named = [&detail.error_code, &detail.error_type]
        .into_iter()
        .flatten()
        .find_map(|identifier| lexicon.class_of(identifier));
    match named {
        Some(LexiconClass::Authentication) => return ProviderFailureClass::Authentication,
        Some(LexiconClass::Quota) => return ProviderFailureClass::Quota,
        Some(LexiconClass::RateLimited) => {
            return ProviderFailureClass::RateLimited { retry_after };
        }
        None => {}
    }
    match *status {
        401 | 403 => ProviderFailureClass::Authentication,
        402 => ProviderFailureClass::Quota,
        429 => ProviderFailureClass::RateLimited { retry_after },
        408 | 500..=599 => ProviderFailureClass::Transient,
        _ => ProviderFailureClass::Unknown,
    }
}

// ---------------------------------------------------------------------------
// Observation scope (what the akh's own model calls did)
// ---------------------------------------------------------------------------

/// One backend result, as reported by a model call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderObservation {
    /// The provider answered.
    Responded,
    /// The provider failed with this evidence.
    Failed(ProviderFailureEvidence),
}

thread_local! {
    /// Stack of open [`ProviderObservationScope`]s on this thread; each keeps
    /// the last observation reported while it was open.
    static OBSERVATION_SCOPES: RefCell<Vec<Option<ProviderObservation>>> =
        const { RefCell::new(Vec::new()) };
}

/// A thread-local record of the last provider result reported on this
/// thread while the scope is open. `!Send` on purpose, like
/// [`UsageScope`](super::llm_metrics::UsageScope): the model calls it records
/// are synchronous. Scopes nest; every open scope sees every observation.
#[must_use = "a provider observation scope records only while it is alive"]
pub struct ProviderObservationScope {
    depth: usize,
    closed: bool,
    _not_send: PhantomData<*const ()>,
}

impl ProviderObservationScope {
    /// Open a new scope on the current thread.
    pub fn open() -> Self {
        let depth = OBSERVATION_SCOPES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            scopes.push(None);
            scopes.len()
        });
        Self {
            depth,
            closed: false,
            _not_send: PhantomData,
        }
    }

    /// Close the scope and return the last observation, if any.
    pub fn finish(mut self) -> Option<ProviderObservation> {
        self.close()
    }

    fn close(&mut self) -> Option<ProviderObservation> {
        if self.closed {
            return None;
        }
        self.closed = true;
        OBSERVATION_SCOPES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            // Tolerate out-of-order drops: truncate to below this scope.
            let last = scopes.get_mut(self.depth - 1).and_then(Option::take);
            scopes.truncate(self.depth - 1);
            last
        })
    }
}

impl Drop for ProviderObservationScope {
    fn drop(&mut self) {
        self.close();
    }
}

/// Report one provider result to every open scope on this thread. A no-op
/// when none is open.
pub fn observe(observation: ProviderObservation) {
    OBSERVATION_SCOPES.with(|scopes| {
        for slot in scopes.borrow_mut().iter_mut() {
            *slot = Some(observation.clone());
        }
    });
}

/// Report a backend call's result: a success is [`ProviderObservation::Responded`],
/// a [`ChatBackendError::ProviderFailed`] its evidence. Other errors (an
/// unavailable backend, a malformed answer) say nothing about the provider
/// and are not reported.
pub fn observe_chat_result<T>(result: &Result<T, ChatBackendError>) {
    match result {
        Ok(_) => observe(ProviderObservation::Responded),
        Err(error) => {
            if let Some(evidence) = error.provider_evidence() {
                observe(ProviderObservation::Failed(evidence.clone()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

/// Why a cooling gate is cooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoolingCause {
    /// Quota or credit exhausted.
    Quota,
    /// Rate limited.
    RateLimited,
}

/// The persisted provider gate (`provider:gate:v1`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProviderGateState {
    /// Model turns may run.
    #[default]
    Open,
    /// Closed until `until`. An expired cooling state admits turns but keeps
    /// `step`, so the next consecutive failure climbs the ladder.
    Cooling {
        /// When the gate reopens.
        until: UnixSecs,
        /// What closed it.
        cause: CoolingCause,
        /// The ladder step this cooldown used (0-based).
        step: u32,
    },
    /// Closed since `since` by an authentication failure, until a model turn
    /// succeeds or an operator resets the gate.
    AuthBlocked {
        /// When the first authentication failure was recorded.
        since: UnixSecs,
    },
}

impl ProviderGateState {
    /// Whether the gate refuses a model turn at `now`, and why.
    pub fn closed_at(self, now: UnixSecs) -> Option<GateClosed> {
        match self {
            Self::Open => None,
            Self::Cooling { until, .. } if now >= until => None,
            Self::Cooling { until, cause, step } => {
                Some(GateClosed::Cooling { until, cause, step })
            }
            Self::AuthBlocked { since } => Some(GateClosed::AuthBlocked { since }),
        }
    }

    /// The state after `class` failed at `now`. Classes that do not close
    /// the gate leave it unchanged.
    fn after_failure(
        self,
        class: ProviderFailureClass,
        ladder: &CooldownLadder,
        now: UnixSecs,
    ) -> Self {
        let cooling = |cause, retry_after: Option<Duration>| {
            let step = match self {
                Self::Cooling { step, .. } => step.saturating_add(1).min(ladder.last_step()),
                Self::Open | Self::AuthBlocked { .. } => 0,
            };
            let cooldown = ladder.cooldown(step).max(retry_after.unwrap_or_default());
            Self::Cooling {
                until: now.saturating_add(cooldown),
                cause,
                step,
            }
        };
        match class {
            ProviderFailureClass::Authentication => match self {
                Self::AuthBlocked { since } => Self::AuthBlocked { since },
                Self::Open | Self::Cooling { .. } => Self::AuthBlocked { since: now },
            },
            ProviderFailureClass::Quota => cooling(CoolingCause::Quota, None),
            ProviderFailureClass::RateLimited { retry_after } => {
                cooling(CoolingCause::RateLimited, retry_after)
            }
            ProviderFailureClass::Transient | ProviderFailureClass::Unknown => self,
        }
    }
}

/// A closed gate, as the daemon sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateClosed {
    /// Cooling down after a quota or rate-limit failure.
    Cooling {
        /// When the gate reopens.
        until: UnixSecs,
        /// What closed it.
        cause: CoolingCause,
        /// The ladder step (0-based).
        step: u32,
    },
    /// Blocked by an authentication failure.
    AuthBlocked {
        /// When it was blocked.
        since: UnixSecs,
    },
}

impl GateClosed {
    /// How long a deferred occurrence should wait from `now` before it asks
    /// the gate again.
    pub fn recheck_after(self, now: UnixSecs) -> Duration {
        match self {
            Self::Cooling { until, .. } => {
                Duration::from_secs(until.as_secs().saturating_sub(now.as_secs()).max(1))
            }
            Self::AuthBlocked { .. } => AUTH_BLOCKED_RECHECK,
        }
    }
}

impl std::fmt::Display for GateClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cooling { until, cause, step } => write!(
                f,
                "provider gate cooling ({}) until {} (ladder step {})",
                match cause {
                    CoolingCause::Quota => "quota",
                    CoolingCause::RateLimited => "rate_limited",
                },
                until.as_secs(),
                step + 1
            ),
            Self::AuthBlocked { since } => write!(
                f,
                "provider gate blocked by an authentication failure since {}",
                since.as_secs()
            ),
        }
    }
}

/// Why an operator must act on an authentication-blocked gate. Logged once
/// per blocked episode with its miette help.
#[derive(Debug, Error, Diagnostic)]
#[error("the akh's model provider rejected its credential; autonomous model turns are paused")]
#[diagnostic(
    code(akh::provider::auth_blocked),
    help(
        "Repair the credential this akh uses (`[ollama_endpoint] api_key` or \
         `[openai_endpoint] api_key`, and the environment variable it names), then either \
         run one ACP turn against the akh (a successful turn reopens the gate) or clear it \
         directly with the `_akh/provider/gate/reset` ACP method. The gate is persisted, so \
         restarting the daemon does not clear it."
    )
)]
pub struct AuthBlockedNotice {
    /// When the gate was blocked.
    pub since: UnixSecs,
}

/// Where the gate lives.
#[derive(Debug)]
enum GateStore {
    /// The engine's durable metadata tier.
    Durable(Arc<Engine>),
    /// Process memory only; the engine has no durable tier.
    Memory,
}

#[derive(Debug)]
struct GateInner {
    state: ProviderGateState,
    /// The last write failed, so memory is ahead of the durable copy and a
    /// refresh must not overwrite it.
    unsynced: bool,
}

/// One akh's provider gate. The daemon is its main writer; the agentic turn
/// and the `_akh/provider/gate/reset` ACP method also write the durable copy
/// (see [`reset_persisted`], [`note_model_success`]), so [`Self::closed`]
/// re-reads it before deciding.
#[derive(Debug)]
pub struct ProviderGate {
    ladder: CooldownLadder,
    inner: Mutex<GateInner>,
    store: GateStore,
}

impl ProviderGate {
    /// A gate that lives only in memory.
    pub fn in_memory(ladder: CooldownLadder) -> Self {
        Self {
            ladder,
            inner: Mutex::new(GateInner {
                state: ProviderGateState::Open,
                unsynced: false,
            }),
            store: GateStore::Memory,
        }
    }

    /// Open the gate for `engine`, loading the persisted state when the
    /// engine has a durable tier.
    pub fn open(engine: &Arc<Engine>, ladder: CooldownLadder) -> ProviderGateResult<Self> {
        if engine.store().durable.is_none() {
            return Ok(Self::in_memory(ladder));
        }
        let state = load_persisted(engine)?.unwrap_or_default();
        Ok(Self {
            ladder,
            inner: Mutex::new(GateInner {
                state,
                unsynced: false,
            }),
            store: GateStore::Durable(Arc::clone(engine)),
        })
    }

    /// Like [`Self::open`], but an unreadable or corrupt gate is logged and
    /// replaced by an open one (failing closed would stop every autonomous
    /// model turn until an operator repaired the store).
    pub fn open_or_reset(engine: &Arc<Engine>, ladder: CooldownLadder) -> Self {
        match Self::open(engine, ladder.clone()) {
            Ok(gate) => gate,
            Err(error) => {
                tracing::warn!(
                    target: "akh::provider",
                    error = ?miette::Report::new(error),
                    "provider gate unreadable; starting open"
                );
                let mut gate = Self::in_memory(ladder);
                if engine.store().durable.is_some() {
                    gate.store = GateStore::Durable(Arc::clone(engine));
                }
                gate
            }
        }
    }

    /// Replace the cooldown ladder. The current state is kept.
    pub fn set_ladder(&mut self, ladder: CooldownLadder) {
        self.ladder = ladder;
    }

    /// Whether the gate survives a restart.
    pub fn is_durable(&self) -> bool {
        matches!(self.store, GateStore::Durable(_))
    }

    /// The current state, refreshed from the durable copy.
    pub fn state(&self) -> ProviderGateState {
        self.refresh();
        self.lock().state
    }

    /// Whether the gate refuses a model turn at `now`, and why.
    pub fn closed(&self, now: UnixSecs) -> Option<GateClosed> {
        self.state().closed_at(now)
    }

    /// Record a failed model call of `class` at `now` and return the new
    /// state. Memory is updated before the write, so a failed write still
    /// enforces the gate until the daemon restarts.
    pub fn record_failure(
        &self,
        class: ProviderFailureClass,
        now: UnixSecs,
    ) -> ProviderGateResult<ProviderGateState> {
        self.refresh();
        self.transition(|state| state.after_failure(class, &self.ladder, now))
    }

    /// Record a model response: the gate opens and the ladder restarts.
    /// Returns the previous state.
    pub fn record_success(&self) -> ProviderGateResult<ProviderGateState> {
        self.refresh();
        let before = self.lock().state;
        self.transition(|_| ProviderGateState::Open)?;
        Ok(before)
    }

    fn transition(
        &self,
        next: impl FnOnce(ProviderGateState) -> ProviderGateState,
    ) -> ProviderGateResult<ProviderGateState> {
        let (state, changed) = {
            let mut inner = self.lock();
            let before = inner.state;
            inner.state = next(before);
            (inner.state, inner.state != before || inner.unsynced)
        };
        if changed && let GateStore::Durable(engine) = &self.store {
            let written = persist(engine, state);
            self.lock().unsynced = written.is_err();
            written?;
        }
        Ok(state)
    }

    /// Adopt the durable copy unless memory is ahead of it.
    fn refresh(&self) {
        let GateStore::Durable(engine) = &self.store else {
            return;
        };
        if self.lock().unsynced {
            return;
        }
        match load_persisted(engine) {
            Ok(persisted) => {
                let mut inner = self.lock();
                if !inner.unsynced {
                    inner.state = persisted.unwrap_or_default();
                }
            }
            Err(error) => tracing::debug!(
                target: "akh::provider",
                error = %error,
                "provider gate not refreshed; using the in-memory state"
            ),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GateInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn gate_key_display() -> String {
    String::from_utf8_lossy(PROVIDER_GATE_KEY).into_owned()
}

fn persist(engine: &Engine, state: ProviderGateState) -> ProviderGateResult<()> {
    let bytes =
        serde_json::to_vec(&state).map_err(|source| ProviderGateError::GateEncode { source })?;
    engine
        .store()
        .put_meta(PROVIDER_GATE_KEY, &bytes)
        .map_err(|source| ProviderGateError::GateWrite {
            key: gate_key_display(),
            source,
        })
}

/// Read the persisted gate. `Ok(None)` when none was ever written (or the
/// engine has no durable tier).
pub fn load_persisted(engine: &Engine) -> ProviderGateResult<Option<ProviderGateState>> {
    let bytes = engine
        .store()
        .get_meta(PROVIDER_GATE_KEY)
        .map_err(|source| ProviderGateError::GateRead {
            key: gate_key_display(),
            source,
        })?;
    bytes
        .map(|bytes| {
            serde_json::from_slice(&bytes).map_err(|source| ProviderGateError::CorruptGate {
                key: gate_key_display(),
                source,
            })
        })
        .transpose()
}

/// Open the persisted gate (the operator's reset). Returns the state it
/// replaced. A corrupt record is overwritten.
///
/// # Errors
///
/// [`ProviderGateError::NotDurable`] for an engine without a durable tier
/// (restart the daemon instead), or a store read/write failure.
pub fn reset_persisted(engine: &Engine) -> ProviderGateResult<ProviderGateState> {
    if engine.store().durable.is_none() {
        return Err(ProviderGateError::NotDurable);
    }
    let before = match load_persisted(engine) {
        Ok(state) => state.unwrap_or_default(),
        Err(ProviderGateError::CorruptGate { .. }) => ProviderGateState::Open,
        Err(error) => return Err(error),
    };
    persist(engine, ProviderGateState::Open)?;
    tracing::info!(
        target: "akh::provider",
        previous = ?before,
        "provider gate reset by the operator"
    );
    Ok(before)
}

/// A model turn outside the daemon (an ACP turn) succeeded: open a closed
/// persisted gate. Cheap when the gate is already open (one metadata read).
/// Failures are logged, never propagated: the turn itself succeeded.
pub fn note_model_success(engine: &Engine) {
    let previous = match load_persisted(engine) {
        Ok(None | Some(ProviderGateState::Open)) => return,
        Ok(Some(previous)) => Some(previous),
        // A corrupt record is replaced by the open gate the turn just proved.
        Err(ProviderGateError::CorruptGate { .. }) => None,
        Err(error) => {
            tracing::debug!(
                target: "akh::provider",
                error = %error,
                "provider gate not checked after a successful turn"
            );
            return;
        }
    };
    match persist(engine, ProviderGateState::Open) {
        Ok(()) => tracing::info!(
            target: "akh::provider",
            previous = ?previous,
            "provider gate reopened: a model turn succeeded"
        ),
        Err(error) => tracing::warn!(
            target: "akh::provider",
            error = %error,
            "provider gate not reopened after a successful turn"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn http(status: u16, retry_after: Option<&str>, body: &str) -> ProviderFailureEvidence {
        ProviderFailureEvidence::Http {
            status,
            detail: ProviderErrorDetail::from_parts(retry_after, body),
        }
    }

    const NOW: UnixSecs = UnixSecs::from_secs(1_790_251_200);

    fn class(evidence: &ProviderFailureEvidence) -> ProviderFailureClass {
        classify(evidence, &ProviderErrorLexicon::default(), NOW)
    }

    #[test]
    fn status_codes_map_to_classes() {
        assert_eq!(
            class(&http(401, None, "")),
            ProviderFailureClass::Authentication
        );
        assert_eq!(
            class(&http(403, None, "")),
            ProviderFailureClass::Authentication
        );
        assert_eq!(class(&http(402, None, "")), ProviderFailureClass::Quota);
        assert_eq!(
            class(&http(429, None, "")),
            ProviderFailureClass::RateLimited { retry_after: None }
        );
        for status in [408, 500, 502, 503, 504, 599] {
            assert_eq!(
                class(&http(status, None, "")),
                ProviderFailureClass::Transient
            );
        }
        for status in [400, 404, 409, 422] {
            assert_eq!(
                class(&http(status, None, "")),
                ProviderFailureClass::Unknown
            );
        }
        assert_eq!(
            class(&ProviderFailureEvidence::Transport),
            ProviderFailureClass::Transient
        );
    }

    #[test]
    fn retry_after_seconds_and_http_date() {
        assert_eq!(
            class(&http(429, Some("120"), "")),
            ProviderFailureClass::RateLimited {
                retry_after: Some(Duration::from_secs(120))
            }
        );
        // NOW + 90 s as an IMF-fixdate.
        let at = SystemTime::UNIX_EPOCH + Duration::from_secs(NOW.as_secs() + 90);
        let date = httpdate::fmt_http_date(at);
        assert_eq!(
            class(&http(429, Some(&date), "")),
            ProviderFailureClass::RateLimited {
                retry_after: Some(Duration::from_secs(90))
            }
        );
        // A date in the past means "now".
        let past = httpdate::fmt_http_date(SystemTime::UNIX_EPOCH + Duration::from_secs(1));
        assert_eq!(
            class(&http(429, Some(&past), "")),
            ProviderFailureClass::RateLimited {
                retry_after: Some(Duration::ZERO)
            }
        );
        // Garbage is ignored, not guessed.
        assert_eq!(
            class(&http(429, Some("soon"), "")),
            ProviderFailureClass::RateLimited { retry_after: None }
        );
    }

    #[test]
    fn structured_codes_match_the_lexicon_exactly() {
        let lexicon = ProviderErrorLexicon {
            auth_error_codes: ["invalid_api_key".to_owned()].into(),
            quota_error_codes: ["insufficient_quota".to_owned()].into(),
            rate_limit_error_codes: ["rate_limit_exceeded".to_owned(), "1015".to_owned()].into(),
        };
        let quota_429 = http(
            429,
            Some("30"),
            r#"{"error":{"message":"x","type":"insufficient_quota","code":null}}"#,
        );
        assert_eq!(
            classify(&quota_429, &lexicon, NOW),
            ProviderFailureClass::Quota,
            "a lexicon code beats the 429 status"
        );
        let auth_400 = http(400, None, r#"{"error":{"code":"invalid_api_key"}}"#);
        assert_eq!(
            classify(&auth_400, &lexicon, NOW),
            ProviderFailureClass::Authentication
        );
        let numeric = http(503, Some("7"), r#"{"error":{"code":1015}}"#);
        assert_eq!(
            classify(&numeric, &lexicon, NOW),
            ProviderFailureClass::RateLimited {
                retry_after: Some(Duration::from_secs(7))
            }
        );
        // Exact, case-sensitive: a near miss falls back to the status.
        let near = http(400, None, r#"{"error":{"code":"Invalid_API_Key"}}"#);
        assert_eq!(
            classify(&near, &lexicon, NOW),
            ProviderFailureClass::Unknown
        );
        // Without a lexicon the same bodies classify by status alone.
        assert_eq!(
            class(&quota_429),
            ProviderFailureClass::RateLimited {
                retry_after: Some(Duration::from_secs(30))
            }
        );
        assert_eq!(class(&auth_400), ProviderFailureClass::Unknown);
    }

    #[test]
    fn structured_identifiers_classify_without_a_status() {
        let lexicon = ProviderErrorLexicon {
            auth_error_codes: ["login_failed".to_owned()].into(),
            quota_error_codes: ["usage_limit_exceeded".to_owned()].into(),
            rate_limit_error_codes: ["rate_limited".to_owned()].into(),
        };
        assert_eq!(
            classify_identifier("login_failed", &lexicon),
            Some(ProviderFailureClass::Authentication)
        );
        assert_eq!(
            classify_identifier("usage_limit_exceeded", &lexicon),
            Some(ProviderFailureClass::Quota)
        );
        assert_eq!(
            classify_identifier("rate_limited", &lexicon),
            Some(ProviderFailureClass::RateLimited { retry_after: None })
        );
        assert_eq!(classify_identifier("unknown_outcome", &lexicon), None);
        assert_eq!(classify_identifier("Login_Failed", &lexicon), None);
    }

    #[test]
    fn english_prose_never_changes_the_class() {
        let lexicon = ProviderErrorLexicon {
            quota_error_codes: ["insufficient_quota".to_owned()].into(),
            ..ProviderErrorLexicon::default()
        };
        for body in [
            "You exceeded your current quota, please check your plan and billing details.",
            r#"{"error":"unauthorized: invalid api key, usage limit reached, too many requests"}"#,
            r#"{"error":{"message":"insufficient_quota rate limit HTTP 401 unauthorized"}}"#,
            "refresh token was revoked",
        ] {
            assert_eq!(
                classify(&http(500, None, body), &lexicon, NOW),
                ProviderFailureClass::Transient,
                "{body}"
            );
            assert_eq!(
                classify(&http(418, None, body), &lexicon, NOW),
                ProviderFailureClass::Unknown,
                "{body}"
            );
        }
    }

    #[test]
    fn detail_extracts_only_structured_identifiers() {
        let detail = ProviderErrorDetail::from_parts(
            Some(" 42 "),
            r#"{"error":{"message":"prose","type":"server_error","code":"overloaded"}}"#,
        );
        assert_eq!(
            detail,
            ProviderErrorDetail {
                retry_after: Some(RetryAfter::Delay(Duration::from_secs(42))),
                error_code: Some("overloaded".into()),
                error_type: Some("server_error".into()),
            }
        );
        // Ollama's `{"error": "<string>"}` carries no identifier.
        assert_eq!(
            ProviderErrorDetail::from_parts(None, r#"{"error":"model not found"}"#),
            ProviderErrorDetail::default()
        );
    }

    #[test]
    fn ladder_rejects_empty_and_zero_steps() {
        assert!(matches!(
            CooldownLadder::from_secs(&[]),
            Err(ProviderConfigError::EmptyLadder)
        ));
        assert!(matches!(
            CooldownLadder::from_secs(&[60, 0]),
            Err(ProviderConfigError::ZeroCooldown { index: 1 })
        ));
        let ladder = CooldownLadder::default();
        assert_eq!(ladder.cooldown(0), Duration::from_secs(1_800));
        assert_eq!(ladder.cooldown(3), Duration::from_secs(14_400));
        assert_eq!(ladder.cooldown(99), Duration::from_secs(14_400));
    }

    #[test]
    fn config_parses_from_toml_with_defaults() {
        let config: ProviderConfig = toml::from_str("").unwrap();
        assert!(config.is_default());
        let config: ProviderConfig = toml::from_str(
            "cooldown_ladder_secs = [60, 120]\nquota_error_codes = [\"insufficient_quota\"]\n",
        )
        .unwrap();
        assert_eq!(
            config.cooldown_ladder_secs.cooldown(1),
            Duration::from_secs(120)
        );
        assert!(
            config
                .lexicon
                .quota_error_codes
                .contains("insufficient_quota")
        );
        let error = toml::from_str::<ProviderConfig>("cooldown_ladder_secs = []").unwrap_err();
        assert!(
            error.to_string().contains("cooldown_ladder_secs"),
            "{error}"
        );
    }

    #[test]
    fn gate_climbs_the_ladder_and_resets_on_success() {
        let gate = ProviderGate::in_memory(CooldownLadder::default());
        assert_eq!(gate.closed(NOW), None);

        let state = gate
            .record_failure(ProviderFailureClass::Quota, NOW)
            .unwrap();
        assert_eq!(
            state,
            ProviderGateState::Cooling {
                until: NOW.saturating_add(Duration::from_secs(1_800)),
                cause: CoolingCause::Quota,
                step: 0
            }
        );
        assert!(gate.closed(NOW).is_some());
        let reopen = NOW.saturating_add(Duration::from_secs(1_800));
        assert_eq!(gate.closed(reopen), None, "an expired cooldown admits");

        // Consecutive failures climb, and the last step repeats.
        let mut now = reopen;
        for (step, secs) in [(1, 3_600), (2, 7_200), (3, 14_400), (3, 14_400)] {
            let state = gate
                .record_failure(ProviderFailureClass::RateLimited { retry_after: None }, now)
                .unwrap();
            assert_eq!(
                state,
                ProviderGateState::Cooling {
                    until: now.saturating_add(Duration::from_secs(secs)),
                    cause: CoolingCause::RateLimited,
                    step
                }
            );
            now = now.saturating_add(Duration::from_secs(secs));
        }

        // Transient and unknown failures leave the gate alone.
        let before = gate.state();
        gate.record_failure(ProviderFailureClass::Transient, now)
            .unwrap();
        gate.record_failure(ProviderFailureClass::Unknown, now)
            .unwrap();
        assert_eq!(gate.state(), before);

        // A success restarts the ladder.
        gate.record_success().unwrap();
        assert_eq!(gate.state(), ProviderGateState::Open);
        let state = gate
            .record_failure(ProviderFailureClass::Quota, now)
            .unwrap();
        assert!(matches!(state, ProviderGateState::Cooling { step: 0, .. }));
    }

    #[test]
    fn rate_limit_waits_for_the_longer_of_retry_after_and_the_ladder() {
        let gate = ProviderGate::in_memory(CooldownLadder::from_secs(&[60]).unwrap());
        let state = gate
            .record_failure(
                ProviderFailureClass::RateLimited {
                    retry_after: Some(Duration::from_secs(600)),
                },
                NOW,
            )
            .unwrap();
        assert!(
            matches!(state, ProviderGateState::Cooling { until, .. } if until == NOW.saturating_add(Duration::from_secs(600)))
        );
        gate.record_success().unwrap();
        let state = gate
            .record_failure(
                ProviderFailureClass::RateLimited {
                    retry_after: Some(Duration::from_secs(5)),
                },
                NOW,
            )
            .unwrap();
        assert!(
            matches!(state, ProviderGateState::Cooling { until, .. } if until == NOW.saturating_add(Duration::from_secs(60)))
        );
    }

    #[test]
    fn authentication_blocks_until_success() {
        let gate = ProviderGate::in_memory(CooldownLadder::default());
        gate.record_failure(ProviderFailureClass::Authentication, NOW)
            .unwrap();
        let later = NOW.saturating_add(Duration::from_secs(86_400 * 7));
        gate.record_failure(ProviderFailureClass::Authentication, later)
            .unwrap();
        assert_eq!(
            gate.closed(later),
            Some(GateClosed::AuthBlocked { since: NOW }),
            "time alone never reopens it; `since` keeps the first failure"
        );
        assert_eq!(
            GateClosed::AuthBlocked { since: NOW }.recheck_after(later),
            AUTH_BLOCKED_RECHECK
        );
        gate.record_success().unwrap();
        assert_eq!(gate.closed(later), None);
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
    fn gate_persists_across_engine_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = durable_engine(dir.path());
            let gate = ProviderGate::open(&engine, CooldownLadder::default()).unwrap();
            assert!(gate.is_durable());
            gate.record_failure(ProviderFailureClass::Quota, NOW)
                .unwrap();
            gate.record_failure(ProviderFailureClass::Quota, NOW)
                .unwrap();
            engine.persist().unwrap();
        }
        let engine = durable_engine(dir.path());
        let gate = ProviderGate::open(&engine, CooldownLadder::default()).unwrap();
        assert!(matches!(
            gate.state(),
            ProviderGateState::Cooling {
                step: 1,
                cause: CoolingCause::Quota,
                ..
            }
        ));
    }

    #[test]
    fn external_reset_and_turn_success_are_seen_by_the_daemon_gate() {
        let dir = tempfile::tempdir().unwrap();
        let engine = durable_engine(dir.path());
        let gate = ProviderGate::open(&engine, CooldownLadder::default()).unwrap();
        gate.record_failure(ProviderFailureClass::Authentication, NOW)
            .unwrap();
        assert_eq!(
            reset_persisted(&engine).unwrap(),
            ProviderGateState::AuthBlocked { since: NOW }
        );
        assert_eq!(gate.closed(NOW), None, "the operator reset is re-read");

        gate.record_failure(ProviderFailureClass::Authentication, NOW)
            .unwrap();
        note_model_success(&engine);
        assert_eq!(gate.closed(NOW), None, "an ACP turn's success is re-read");
    }

    #[test]
    fn corrupt_gate_opens_and_reset_overwrites_it() {
        let dir = tempfile::tempdir().unwrap();
        let engine = durable_engine(dir.path());
        engine
            .store()
            .put_meta(PROVIDER_GATE_KEY, b"not json")
            .unwrap();
        assert!(matches!(
            ProviderGate::open(&engine, CooldownLadder::default()),
            Err(ProviderGateError::CorruptGate { .. })
        ));
        let gate = ProviderGate::open_or_reset(&engine, CooldownLadder::default());
        assert!(gate.is_durable());
        assert_eq!(gate.closed(NOW), None);
        assert_eq!(reset_persisted(&engine).unwrap(), ProviderGateState::Open);
        assert_eq!(
            load_persisted(&engine).unwrap(),
            Some(ProviderGateState::Open)
        );
    }

    #[test]
    fn in_memory_engine_cannot_reset_a_persisted_gate() {
        let engine = Arc::new(Engine::new(crate::engine::EngineConfig::default()).unwrap());
        assert!(
            !ProviderGate::open(&engine, CooldownLadder::default())
                .unwrap()
                .is_durable()
        );
        assert!(matches!(
            reset_persisted(&engine),
            Err(ProviderGateError::NotDurable)
        ));
    }

    #[test]
    fn observation_scope_keeps_the_last_result() {
        let outer = ProviderObservationScope::open();
        observe_chat_result::<()>(&Err(ChatBackendError::ProviderFailed {
            backend: "test".into(),
            reason: "HTTP 429".into(),
            evidence: http(429, None, ""),
        }));
        let inner = ProviderObservationScope::open();
        observe_chat_result::<()>(&Ok(()));
        assert_eq!(inner.finish(), Some(ProviderObservation::Responded));
        // A non-provider error says nothing.
        observe_chat_result::<()>(&Err(ChatBackendError::TurnFailed {
            backend: "test".into(),
            reason: "malformed".into(),
        }));
        assert_eq!(outer.finish(), Some(ProviderObservation::Responded));
        // Nothing open: a no-op.
        observe(ProviderObservation::Responded);
        let fresh = ProviderObservationScope::open();
        assert_eq!(fresh.finish(), None);
    }

    #[test]
    fn gate_state_serializes_with_a_state_tag() {
        let state = ProviderGateState::Cooling {
            until: NOW,
            cause: CoolingCause::RateLimited,
            step: 2,
        };
        let json = serde_json::to_value(state).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"state": "cooling", "until": NOW.as_secs(), "cause": "rate_limited", "step": 2})
        );
        assert_eq!(
            serde_json::from_value::<ProviderGateState>(json).unwrap(),
            state
        );
    }
}
