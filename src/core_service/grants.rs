//! Per-delegation tool grants (Phase 53e, ADR 113 decision 9).
//!
//! When an akh delegates to a contained native runtime it mints a
//! [`DelegationGrant`]: the cognitive tools the worker may call (read-only by
//! default, plus `memory_write`), bound to one akh, one delegate principal and
//! the service principal that presents it (akhsessd), with an expiry. The
//! worker's calls then run under the grant and are recorded as the delegate's
//! provenance (`DerivationKind::DelegatedCall`), not the akh's.
//!
//! Only the blake3 hash of a [`GrantSecret`] is stored: grants live in the
//! core API's operations [`Store`] under `key("delegation_grant", [hash])`.
//!
//! Anima WI `01a0d46a-0381`; brief: `docs/ai/phases/phase-53/README.md` (53e).
use super::is_cognitive_read_only;
use crate::agent::tool::ToolRegistry;
use akh_service_store::{Store, key};
use akh_session_types::{Fault, GrantToken, ToolResult};
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc, time::Duration};
use thiserror::Error;
use uuid::Uuid;

/// The namespace grants are stored under in the core operations store.
pub const GRANT_NAMESPACE: &str = "delegation_grant";
/// Grant lifetime when the minting caller does not choose one.
pub const DEFAULT_GRANT_TTL: Duration = Duration::from_secs(60 * 60);
/// The longest lifetime a grant may be minted with.
pub const MAX_GRANT_TTL: Duration = Duration::from_secs(24 * 60 * 60);
/// Expired or revoked grants older than this are pruned on the next mint.
const RETAIN_ENDED_GRANTS_MS: u64 = 24 * 60 * 60 * 1000;

/// Whether a core principal must present a grant on every tool call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantPolicy {
    /// A grant is honoured when presented; without one the principal's
    /// [`super::ToolScope`] applies (the pre-53e behaviour).
    #[default]
    Optional,
    /// Every tool call must present a valid grant.
    Required,
}

/// A grant secret: 32 random bytes, base64url without padding. Never stored,
/// never logged (`Debug` is redacted); only its blake3 hash is persisted.
#[derive(Clone, PartialEq, Eq)]
pub struct GrantSecret(String);

impl GrantSecret {
    /// Draw a fresh secret from the OS RNG.
    pub fn generate() -> Self {
        use base64::Engine as _;
        use rand::RngCore as _;
        let mut bytes = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        Self(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
    }

    /// Wrap a presented wire token.
    pub fn from_token(token: &GrantToken) -> Self {
        Self(token.expose().to_string())
    }

    /// The wire token to hand to the session service.
    pub fn token(&self) -> GrantToken {
        GrantToken::new(self.0.clone())
    }

    /// The raw secret, for handing to the delegate exactly once.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// The storage key derived from the secret's blake3 hash.
    fn storage_key(&self) -> String {
        key(
            GRANT_NAMESPACE,
            &[blake3::hash(self.0.as_bytes()).to_hex().as_str()],
        )
    }
}

impl std::fmt::Debug for GrantSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GrantSecret(<redacted>)")
    }
}

/// Which tools a grant names, as requested at mint time. Defined next to the
/// delegation config (which selects it per akh); resolved here.
pub use crate::agent::delegation::GrantTools;

impl GrantTools {
    /// Resolve against the akh's live registry into the concrete tool set the
    /// grant carries.
    pub fn resolve(&self, registry: &ToolRegistry) -> BTreeSet<String> {
        match self {
            Self::ReadOnlyPlusMemoryWrite => registry
                .list_advertised_manifests()
                .iter()
                .filter(|manifest| is_cognitive_read_only(manifest))
                .map(|manifest| manifest.name.clone())
                .chain(std::iter::once("memory_write".to_string()))
                .collect(),
            Self::Explicit { tools } => tools.clone(),
        }
    }
}

/// A minted grant as stored (secret-free).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationGrant {
    /// UUID v7 of the grant (the handle for revoke/list and provenance).
    pub id: Uuid,
    /// The akh whose tools the grant opens.
    pub akh: String,
    /// The delegate the calls are attributed to.
    pub principal: String,
    /// The core principal allowed to present the grant (e.g. `sessions`).
    pub service_principal: String,
    /// The tools the delegate may call.
    pub tools: BTreeSet<String>,
    /// The session the grant is bound to; `None` = any session of the
    /// presenter until bound.
    pub session_id: Option<String>,
    /// Unix milliseconds.
    pub issued_at: u64,
    /// Unix milliseconds; the grant is dead from this instant.
    pub expires_at: u64,
    pub revoked: bool,
}

impl DelegationGrant {
    /// Whether the grant names `tool`.
    pub fn permits(&self, tool: &str) -> bool {
        self.tools.contains(tool)
    }

    /// Check `tool` against the grant.
    pub fn permit(&self, tool: &str) -> Result<(), GrantError> {
        if self.permits(tool) {
            Ok(())
        } else {
            Err(GrantError::ToolNotGranted {
                grant_id: self.id,
                tool: tool.into(),
                granted: self.tools.iter().cloned().collect::<Vec<_>>().join(", "),
            })
        }
    }
}

/// A mint request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintGrant {
    pub akh: String,
    pub principal: String,
    pub service_principal: String,
    pub tools: GrantTools,
    pub session_id: Option<String>,
    pub ttl: Duration,
}

/// A freshly minted grant: the secret is returned exactly once.
#[derive(Debug, Clone)]
pub struct MintedGrant {
    pub secret: GrantSecret,
    pub grant: DelegationGrant,
}

/// What the presenting call is: the akh it targets, the authenticated core
/// principal presenting it and the session it runs in.
#[derive(Debug, Clone, Copy)]
pub struct GrantPresentation<'a> {
    pub akh: &'a str,
    pub presenter: &'a str,
    pub session_id: &'a str,
}

/// Wall clock for grant issue/expiry, injectable for tests.
pub trait GrantClock: Send + Sync {
    /// Unix milliseconds.
    fn now_ms(&self) -> u64;
}

/// The system clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl GrantClock for SystemClock {
    fn now_ms(&self) -> u64 {
        akh_session_types::now_ms()
    }
}

/// Errors of grant minting and enforcement.
#[derive(Debug, Error, Diagnostic)]
pub enum GrantError {
    #[error("core principal `{principal}` must present a delegation grant")]
    #[diagnostic(
        code(akh::core_api::grant::missing),
        help(
            "Nothing ran. This principal's grant policy is `required`. Create the \
             session with the grant the akh minted for this delegation \
             (`NewSession.grant`), or set `\"grants\": \"optional\"` on the principal \
             in the core config to fall back to its tool scope."
        )
    )]
    Missing { principal: String },

    #[error("delegation grant is unknown or revoked")]
    #[diagnostic(
        code(akh::core_api::grant::unknown),
        help(
            "Nothing ran. The presented grant does not match any live grant of this \
             core. Ask the akh to mint a new one (`_akh/delegation/grant/mint`) and \
             start a new delegated session with it."
        )
    )]
    Unknown,

    #[error("delegation grant {grant_id} expired")]
    #[diagnostic(
        code(akh::core_api::grant::expired),
        help(
            "Nothing ran. Grants are short-lived by design. Mint a new grant for the \
             remaining work, with a TTL that covers it (max 24h)."
        )
    )]
    Expired { grant_id: Uuid, expired_at: u64 },

    #[error("delegation grant {grant_id} is for akh `{granted}`, not `{requested}`")]
    #[diagnostic(
        code(akh::core_api::grant::wrong_akh),
        help("Nothing ran. A grant opens only the akh that minted it; call `{granted}`.")
    )]
    WrongAkh {
        grant_id: Uuid,
        granted: String,
        requested: String,
    },

    #[error("delegation grant {grant_id} may only be presented by `{expected}`, not `{presenter}`")]
    #[diagnostic(
        code(akh::core_api::grant::wrong_presenter),
        help(
            "Nothing ran. The grant is bound to the service principal that brokers \
             the delegated session. Route the call through that service."
        )
    )]
    WrongPresenter {
        grant_id: Uuid,
        expected: String,
        presenter: String,
    },

    #[error("delegation grant {grant_id} is bound to session `{bound}`, not `{presented}`")]
    #[diagnostic(
        code(akh::core_api::grant::session_mismatch),
        help(
            "Nothing ran. A grant belongs to one delegated session. Mint a separate \
             grant for another session."
        )
    )]
    SessionMismatch {
        grant_id: Uuid,
        bound: String,
        presented: String,
    },

    #[error("tool `{tool}` is not granted by delegation grant {grant_id}")]
    #[diagnostic(
        code(akh::core_api::grant::tool_not_granted),
        help("Nothing ran. The grant allows: {granted}. Ask the akh for a grant naming `{tool}`.")
    )]
    ToolNotGranted {
        grant_id: Uuid,
        tool: String,
        granted: String,
    },

    #[error("grant TTL {requested_secs}s exceeds the maximum of {max_secs}s")]
    #[diagnostic(
        code(akh::core_api::grant::ttl_too_long),
        help("Mint with a TTL of at most {max_secs}s and mint again for longer work.")
    )]
    TtlTooLong { requested_secs: u64, max_secs: u64 },

    #[error("invalid grant request: {message}")]
    #[diagnostic(
        code(akh::core_api::grant::invalid_request),
        help(
            "`akh`, `principal` and `service_principal` must be non-empty identifiers \
             (1..256 characters, no control characters) and the TTL must be positive."
        )
    )]
    InvalidRequest { message: String },

    #[error("delegation grants need the /v1 core API, which this akhomed does not serve")]
    #[diagnostic(
        code(akh::core_api::grant::core_api_disabled),
        help(
            "Grants are enforced by the /v1 core API. Start akhomed with \
             AKH_CORE_CONFIG naming a core config, or delegate without a grant."
        )
    )]
    CoreApiDisabled,

    #[error("delegation grant store failed: {message}")]
    #[diagnostic(
        code(akh::core_api::grant::store),
        help("The core operations store rejected the operation; check its disk and retry.")
    )]
    Store { message: String },
}

impl GrantError {
    /// Wire code (`ToolResult.data.code`).
    pub fn wire_code(&self) -> &'static str {
        match self {
            Self::Missing { .. } => "grant_missing",
            Self::Unknown => "grant_unknown",
            Self::Expired { .. } => "grant_expired",
            Self::WrongAkh { .. } => "grant_wrong_akh",
            Self::WrongPresenter { .. } => "grant_wrong_presenter",
            Self::SessionMismatch { .. } => "grant_session_mismatch",
            Self::ToolNotGranted { .. } => "tool_not_granted",
            Self::TtlTooLong { .. } => "grant_ttl_too_long",
            Self::InvalidRequest { .. } => "invalid_request",
            Self::CoreApiDisabled => "core_api_disabled",
            Self::Store { .. } => "grant_store",
        }
    }

    /// A denial the delegate can read: HTTP 200, `success:false`, with the
    /// diagnostic code and help. Nothing ran.
    pub fn into_tool_result(self) -> ToolResult {
        super::denial(&self, self.wire_code())
    }
}

impl From<Fault> for GrantError {
    fn from(fault: Fault) -> Self {
        Self::Store {
            message: fault.to_string(),
        }
    }
}

impl From<GrantError> for Fault {
    fn from(error: GrantError) -> Self {
        let message = match error.help() {
            Some(help) => format!("{error} ({help})"),
            None => error.to_string(),
        };
        Fault::new(error.wire_code(), message)
    }
}

/// Mints, resolves and revokes delegation grants in the core operations store.
#[derive(Clone)]
pub struct GrantAuthority {
    store: Store,
    clock: Arc<dyn GrantClock>,
}

impl std::fmt::Debug for GrantAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrantAuthority").finish_non_exhaustive()
    }
}

impl GrantAuthority {
    /// An authority over `store` using the system clock.
    pub fn new(store: Store) -> Self {
        Self {
            store,
            clock: Arc::new(SystemClock),
        }
    }

    /// Replace the clock (tests inject a manual one).
    pub fn with_clock(mut self, clock: Arc<dyn GrantClock>) -> Self {
        self.clock = clock;
        self
    }

    /// Mint a grant, resolving `request.tools` against the akh's live
    /// `registry`. The secret is returned once and never stored.
    pub fn mint(
        &self,
        request: MintGrant,
        registry: &ToolRegistry,
    ) -> Result<MintedGrant, GrantError> {
        for (field, value) in [
            ("akh", request.akh.as_str()),
            ("principal", request.principal.as_str()),
            ("service_principal", request.service_principal.as_str()),
        ] {
            akh_session_types::validate_key(value).map_err(|fault| GrantError::InvalidRequest {
                message: format!("{field}: {}", fault.message),
            })?;
        }
        if let Some(session) = &request.session_id {
            akh_session_types::validate_key(session).map_err(|fault| {
                GrantError::InvalidRequest {
                    message: format!("session_id: {}", fault.message),
                }
            })?;
        }
        if request.ttl.is_zero() {
            return Err(GrantError::InvalidRequest {
                message: "ttl must be positive".into(),
            });
        }
        if request.ttl > MAX_GRANT_TTL {
            return Err(GrantError::TtlTooLong {
                requested_secs: request.ttl.as_secs(),
                max_secs: MAX_GRANT_TTL.as_secs(),
            });
        }
        let now = self.clock.now_ms();
        let grant = DelegationGrant {
            id: Uuid::now_v7(),
            akh: request.akh,
            principal: request.principal,
            service_principal: request.service_principal,
            tools: request.tools.resolve(registry),
            session_id: request.session_id,
            issued_at: now,
            expires_at: now.saturating_add(request.ttl.as_millis() as u64),
            revoked: false,
        };
        let secret = GrantSecret::generate();
        let storage_key = secret.storage_key();
        self.store.transaction(|tx| {
            // Prune grants that ended (expired or revoked) long ago so the
            // store does not grow without bound.
            for old in tx.scan::<DelegationGrant>(&format!("{GRANT_NAMESPACE}/"))? {
                let ended_at = old.expires_at;
                if ended_at.saturating_add(RETAIN_ENDED_GRANTS_MS) < now {
                    if let Some((old_key, _)) = Self::locate(tx, old.id)? {
                        tx.delete(&old_key)?;
                    }
                    tx.delete(&id_index_key(old.id))?;
                }
            }
            tx.put(&storage_key, &grant)?;
            tx.put(&id_index_key(grant.id), &storage_key)
        })?;
        tracing::info!(
            target: "akh::core_api::grant",
            grant_id = %grant.id, akh = %grant.akh, principal = %grant.principal,
            tools = grant.tools.len(), expires_at = grant.expires_at,
            "delegation grant minted"
        );
        Ok(MintedGrant { secret, grant })
    }

    /// Resolve a presented grant for a call, checking expiry, akh, presenter
    /// and session binding. The tool check is [`DelegationGrant::permit`].
    ///
    /// An unbound grant is bound to the first session that presents it (after
    /// every other check passed), atomically in the store: from then on it
    /// opens that session only. This closes the window between minting and
    /// the akh's own [`Self::bind_session`] after akhsessd answered the
    /// create, which a slow or timed-out create would otherwise leave open.
    pub fn resolve(
        &self,
        presented: &GrantToken,
        at: GrantPresentation<'_>,
    ) -> Result<DelegationGrant, GrantError> {
        let secret = GrantSecret::from_token(presented);
        let grant = self
            .store
            .get::<DelegationGrant>(&secret.storage_key())?
            .filter(|grant| !grant.revoked)
            .ok_or(GrantError::Unknown)?;
        if self.clock.now_ms() >= grant.expires_at {
            return Err(GrantError::Expired {
                grant_id: grant.id,
                expired_at: grant.expires_at,
            });
        }
        if grant.akh != at.akh {
            return Err(GrantError::WrongAkh {
                grant_id: grant.id,
                granted: grant.akh,
                requested: at.akh.into(),
            });
        }
        if grant.service_principal != at.presenter {
            return Err(GrantError::WrongPresenter {
                grant_id: grant.id,
                expected: grant.service_principal,
                presenter: at.presenter.into(),
            });
        }
        match &grant.session_id {
            Some(bound) if bound != at.session_id => Err(GrantError::SessionMismatch {
                grant_id: grant.id,
                bound: bound.clone(),
                presented: at.session_id.into(),
            }),
            Some(_) => Ok(grant),
            // First presentation: bind it. A concurrent first presentation
            // from another session loses inside the store transaction.
            None => self.bind_session(grant.id, at.session_id),
        }
    }

    /// Bind an unbound grant to the session it was handed to. Binding to the
    /// same session again is a no-op; another session is a mismatch.
    pub fn bind_session(&self, id: Uuid, session_id: &str) -> Result<DelegationGrant, GrantError> {
        akh_session_types::validate_key(session_id).map_err(|fault| {
            GrantError::InvalidRequest {
                message: format!("session_id: {}", fault.message),
            }
        })?;
        self.update(id, |grant| match &grant.session_id {
            Some(bound) if bound != session_id => Err(GrantError::SessionMismatch {
                grant_id: grant.id,
                bound: bound.clone(),
                presented: session_id.into(),
            }),
            _ => {
                grant.session_id = Some(session_id.into());
                Ok(())
            }
        })
    }

    /// Revoke a grant by id. Revoking twice is a no-op.
    pub fn revoke(&self, id: Uuid) -> Result<DelegationGrant, GrantError> {
        let grant = self.update(id, |grant| {
            grant.revoked = true;
            Ok(())
        })?;
        tracing::info!(target: "akh::core_api::grant", grant_id = %id, "delegation grant revoked");
        Ok(grant)
    }

    /// Every stored grant (live, expired within the retention window, or
    /// revoked), optionally only those of `akh`, oldest first.
    pub fn list(&self, akh: Option<&str>) -> Result<Vec<DelegationGrant>, GrantError> {
        let mut grants: Vec<DelegationGrant> = self
            .store
            .scan::<DelegationGrant>(&format!("{GRANT_NAMESPACE}/"))?
            .into_iter()
            .filter(|grant| akh.is_none_or(|akh| grant.akh == akh))
            .collect();
        grants.sort_by_key(|grant| grant.id);
        Ok(grants)
    }

    /// Whether `grant` is usable at this authority's current time.
    pub fn is_live(&self, grant: &DelegationGrant) -> bool {
        !grant.revoked && self.clock.now_ms() < grant.expires_at
    }

    fn update(
        &self,
        id: Uuid,
        change: impl FnOnce(&mut DelegationGrant) -> Result<(), GrantError>,
    ) -> Result<DelegationGrant, GrantError> {
        self.store.transaction(|tx| {
            let Some((storage_key, mut grant)) = Self::locate(tx, id)? else {
                return Ok(Err(GrantError::Unknown));
            };
            if let Err(error) = change(&mut grant) {
                return Ok(Err(error));
            }
            tx.put(&storage_key, &grant)?;
            Ok(Ok(grant))
        })?
    }

    /// A grant and its storage key, found through the id index. The index
    /// value is the key (derived from the secret's hash, never the secret).
    fn locate(
        tx: &akh_service_store::Transaction<'_>,
        id: Uuid,
    ) -> akh_session_types::Result<Option<(String, DelegationGrant)>> {
        let Some(storage_key) = tx.get::<String>(&id_index_key(id))? else {
            return Ok(None);
        };
        Ok(tx
            .get::<DelegationGrant>(&storage_key)?
            .map(|grant| (storage_key, grant)))
    }
}

fn id_index_key(id: Uuid) -> String {
    key("delegation_grant_id", &[&id.to_string()])
}

/// Where akhomed publishes the grant authority once `/v1` is up. Minting
/// before that (or with `/v1` disabled) is [`GrantError::CoreApiDisabled`].
pub type GrantAuthoritySlot = Arc<std::sync::OnceLock<GrantAuthority>>;

/// The authority in `slot`, or [`GrantError::CoreApiDisabled`].
pub fn authority_in(slot: &GrantAuthoritySlot) -> Result<&GrantAuthority, GrantError> {
    slot.get().ok_or(GrantError::CoreApiDisabled)
}

/// Mints native-session delegation grants for one akh through the `/v1`
/// grant authority (Phase 53e automatic minting).
#[cfg(feature = "delegation")]
pub struct CoreGrantIssuer {
    slot: GrantAuthoritySlot,
    akh: String,
    registry: Arc<ToolRegistry>,
}

#[cfg(feature = "delegation")]
impl CoreGrantIssuer {
    /// An issuer for `akh`, resolving tool sets against its live `registry`.
    pub fn new(
        slot: GrantAuthoritySlot,
        akh: impl Into<String>,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            slot,
            akh: akh.into(),
            registry,
        }
    }

    fn authority(&self) -> crate::agent::delegation::DelegationResult<&GrantAuthority> {
        authority_in(&self.slot).map_err(unavailable)
    }
}

#[cfg(feature = "delegation")]
fn unavailable(error: GrantError) -> crate::agent::delegation::DelegationError {
    crate::agent::delegation::DelegationError::GrantUnavailable {
        reason: error.to_string(),
    }
}

#[cfg(feature = "delegation")]
fn grant_uuid(id: &str) -> crate::agent::delegation::DelegationResult<Uuid> {
    Uuid::parse_str(id).map_err(|_| unavailable(GrantError::Unknown))
}

#[cfg(feature = "delegation")]
impl crate::agent::delegation::DelegationGrantIssuer for CoreGrantIssuer {
    fn is_available(&self) -> bool {
        authority_in(&self.slot).is_ok()
    }

    fn issue(
        &self,
        request: &crate::agent::delegation::GrantRequest,
    ) -> crate::agent::delegation::DelegationResult<crate::agent::delegation::IssuedGrant> {
        let minted = self
            .authority()?
            .mint(
                MintGrant {
                    akh: self.akh.clone(),
                    principal: request.principal.clone(),
                    service_principal: request.service_principal.clone(),
                    tools: request.tools.clone(),
                    session_id: None,
                    ttl: request.ttl,
                },
                &self.registry,
            )
            .map_err(unavailable)?;
        Ok(crate::agent::delegation::IssuedGrant {
            id: minted.grant.id.to_string(),
            akh: minted.grant.akh,
            token: minted.secret.token(),
        })
    }

    fn bind_session(
        &self,
        grant_id: &str,
        session_id: &str,
    ) -> crate::agent::delegation::DelegationResult<()> {
        self.authority()?
            .bind_session(grant_uuid(grant_id)?, session_id)
            .map(drop)
            .map_err(unavailable)
    }

    fn revoke(&self, grant_id: &str) -> crate::agent::delegation::DelegationResult<()> {
        self.authority()?
            .revoke(grant_uuid(grant_id)?)
            .map(drop)
            .map_err(unavailable)
    }
}
