//! ACP/TCP ingress authentication and per-akh access scoping (Phase 53, ADR 113).
//!
//! akhomed's TCP ACP listener authenticates `initialize` by
//! `_meta.authorization = "Bearer <token>"`. Two credential models exist:
//!
//! - **Scoped principals** (`[[acp.principals]]` in the host config): each
//!   token belongs to one principal, and that principal may only open sessions
//!   and call data-plane methods on the akhs it names. The server-wide control
//!   plane (`_akh/workspace/create`, `_akh/daemon/*`, …) needs `control = true`.
//! - **Legacy shared secret** (`AKHOMED_SHARED_SECRET[_FILE]`): every accepted
//!   token grants every akh and the control plane. It is honoured only when no
//!   principals are configured, and startup logs a warning that it is unscoped.
//!
//! The principals live in the host config rather than in `core.json` because the
//! ACP listener is independent of the `/v1` core API: `/v1` only exists when
//! `AKH_CORE_CONFIG` is set, while ACP ingress is always on when `[acp] tcp_port`
//! is. Coupling them would make a scoped ACP depend on an unrelated surface.
//!
//! # Session ownership
//!
//! Every ACP session records the [`SessionOwner`] that created it at
//! `session/new`: the connection's principal, [`SessionOwner::Legacy`] for the
//! unscoped shared secret, or [`SessionOwner::Local`] for the Unix socket and
//! in-process embeds. `session/load`, `session/prompt` and `session/cancel`
//! check [`AcpAccess::authorize_session`] before touching the session, so two
//! principals granted the same akh cannot use each other's sessions. Session
//! operators (a principal with `control = true`, the legacy shared secret and
//! the local socket, all of which already hold the control plane) may use any
//! session on an akh they are granted; the daemon logs that at `info`.
//!
//! To a non-operator, a session owned by someone else and a session that does
//! not exist produce the *same* refusal (`akh::acp::session_not_owned`, same
//! JSON-RPC code and message), so session ids cannot be probed for existence.
//!
//! Token files are re-read on every `initialize`, so rotating a key (several
//! non-empty lines overlap old and new) takes effect without a restart. Every
//! comparison is constant-time and every candidate is compared against every
//! configured token, so neither timing nor early exit reveals which principal
//! (if any) a token is close to.

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

use super::{SHARED_SECRET_ENV, SHARED_SECRET_FILE_ENV, SharedSecretAuth, constant_time_eq};
use serde::{Deserialize, Serialize};

/// One ACP TCP principal: a token file scoped to a set of akhs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcpPrincipalConfig {
    /// Principal id, used in logs and diagnostics (never the token).
    pub id: String,
    /// Absolute path of the token file. Each non-empty line is one accepted
    /// token (24+ characters, no whitespace); several lines overlap keys during
    /// rotation. Re-read on every `initialize`.
    pub token_file: PathBuf,
    /// Akhs (workspaces) this principal may open sessions on and call the
    /// data plane for. `["*"]` alone grants every akh.
    pub akhs: Vec<String>,
    /// Allow the server-wide control plane (`_akh/workspace/create`,
    /// `_akh/daemon/*`, `_akh/config/*`, relocation, grants, …). Default false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub control: bool,
}

/// Minimum length of a principal token. Matches akhsessd and the `/v1` core
/// API so one key file can serve all three.
pub const MIN_PRINCIPAL_TOKEN_LEN: usize = 24;

/// The `akhs` entry that grants every akh the daemon serves.
pub const ALL_AKHS: &str = "*";

/// Errors of ACP ingress authentication and per-akh scoping.
#[derive(Debug, Error, Diagnostic)]
pub enum AcpAuthError {
    /// A `[[acp.principals]]` entry is malformed.
    #[error("invalid ACP principal `{principal}`: {message}")]
    #[diagnostic(
        code(akh::acp::principal_config),
        help(
            "Each `[[acp.principals]]` entry needs a unique non-empty `id`, an absolute \
             `token_file` and a non-empty `akhs` list (use [\"*\"] alone for every akh). \
             Fix the host config (config.toml), or remove every principal to fall back \
             to the unscoped AKHOMED_SHARED_SECRET."
        )
    )]
    PrincipalConfig { principal: String, message: String },

    /// A principal's token file cannot be read or holds no usable token.
    #[error("cannot load the token file {path} of ACP principal `{principal}`: {message}")]
    #[diagnostic(
        code(akh::acp::principal_token),
        help(
            "The file must be readable by akhomed and hold one or more lines, each a \
             token of at least 24 characters without whitespace (several lines overlap \
             old and new keys during rotation). No two principals may share a token."
        )
    )]
    PrincipalToken {
        principal: String,
        path: PathBuf,
        message: String,
    },

    /// Neither principals nor a shared secret are configured.
    #[error("ACP TCP has no ingress credential: {message}")]
    #[diagnostic(
        code(akh::acp::no_ingress_credential),
        help(
            "Configure `[[acp.principals]]` in the host config (preferred, scoped per akh), \
             or set AKHOMED_SHARED_SECRET_FILE / AKHOMED_SHARED_SECRET (legacy, unscoped). \
             Without either, the TCP listener stays closed; the Unix socket is unaffected."
        )
    )]
    NoIngressCredential { message: String },

    /// A request arrived before a successful authenticated `initialize`.
    #[error("this ACP connection has not authenticated")]
    #[diagnostic(
        code(akh::acp::unauthenticated),
        help(
            "Send `initialize` with `_meta.authorization = \"Bearer <token>\"` first; \
             every other request on a TCP connection is refused until it succeeds."
        )
    )]
    Unauthenticated,

    /// The principal may not use this akh.
    #[error("ACP principal `{principal}` is not granted akh `{workspace}`")]
    #[diagnostic(
        code(akh::acp::workspace_not_granted),
        help(
            "Nothing ran. This principal may use: {granted}. Route the session with \
             `_meta[\"akh/workspace\"]` to one of those, use the token of a principal \
             that holds `{workspace}`, or add `{workspace}` to this principal's `akhs` \
             in the host config `[[acp.principals]]`."
        )
    )]
    WorkspaceNotGranted {
        principal: String,
        workspace: String,
        granted: String,
    },

    /// The principal may not use the server-wide control plane.
    #[error("ACP principal `{principal}` may not call control-plane method `{method}`")]
    #[diagnostic(
        code(akh::acp::control_not_granted),
        help(
            "Control-plane methods span the whole daemon (workspace create/delete, \
             daemon start/stop, config, relocation, grants). Set `control = true` on an \
             operator principal in `[[acp.principals]]`, or use the local Unix socket."
        )
    )]
    ControlNotGranted { principal: String, method: String },

    /// The session was created by another principal, or does not exist. The two
    /// cases are deliberately one error so a non-operator cannot probe ids.
    #[error(
        "ACP session {session_id:?} is unknown, expired, or not owned by principal `{principal}`"
    )]
    #[diagnostic(
        code(akh::acp::session_not_owned),
        help(
            "Nothing ran. An ACP session belongs to the principal that created it with \
             `session/new`; only that principal can load, prompt or cancel it. Sessions \
             also expire (LRU eviction, daemon restart). Open a new session with \
             `session/new`, use the token of the principal that created this one, or ask \
             an operator (a principal with `control = true`) to act on it."
        )
    )]
    SessionNotOwned {
        principal: String,
        session_id: String,
    },
}

impl AcpAuthError {
    /// The diagnostic code as a stable string, for wire payloads.
    #[must_use]
    pub fn code_str(&self) -> String {
        self.code().map(|c| c.to_string()).unwrap_or_default()
    }

    /// A JSON-RPC error carrying this diagnostic (`code`, `message`, `help`).
    ///
    /// [`Self::SessionNotOwned`] uses JSON-RPC invalid-params, the code an
    /// unknown session has always had, so a client that falls back to
    /// `session/new` on a failed `session/load` keeps doing so, and the error
    /// cannot be told apart from an unknown id by its code.
    #[must_use]
    pub fn rpc_error(&self) -> agent_client_protocol::Error {
        let status = match self {
            Self::Unauthenticated => 401,
            Self::WorkspaceNotGranted { .. } | Self::ControlNotGranted { .. } => 403,
            Self::SessionNotOwned { .. } => 404,
            Self::PrincipalConfig { .. }
            | Self::PrincipalToken { .. }
            | Self::NoIngressCredential { .. } => 503,
        };
        let error = match self {
            Self::SessionNotOwned { .. } => {
                let mut error = agent_client_protocol::Error::invalid_params();
                error.message = self.to_string();
                error
            }
            _ => agent_client_protocol::Error::new(status, self.to_string()),
        };
        error.data(serde_json::json!({
            "status": status,
            "code": self.code_str(),
            "message": self.to_string(),
            "help": self.help().map(|h| h.to_string()),
        }))
    }
}

/// Which akhs a principal may use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkhScope {
    /// Every akh the daemon serves (`akhs = ["*"]`, or the legacy secret).
    All,
    /// Exactly these akhs.
    Named(BTreeSet<String>),
}

impl AkhScope {
    /// Parse a config `akhs` list: `["*"]` alone is [`AkhScope::All`].
    fn parse(principal: &str, akhs: &[String]) -> Result<Self, AcpAuthError> {
        let config = |message: &str| AcpAuthError::PrincipalConfig {
            principal: principal.to_string(),
            message: message.to_string(),
        };
        if akhs.is_empty() {
            return Err(config(
                "`akhs` is empty, so the principal could open nothing",
            ));
        }
        if akhs.iter().any(|a| a == ALL_AKHS) {
            return if akhs.len() == 1 {
                Ok(Self::All)
            } else {
                Err(config("`*` must be the only entry of `akhs`"))
            };
        }
        if akhs.iter().any(|a| a.trim().is_empty()) {
            return Err(config("`akhs` contains an empty name"));
        }
        Ok(Self::Named(akhs.iter().cloned().collect()))
    }

    /// Whether `workspace` is within this scope.
    #[must_use]
    pub fn contains(&self, workspace: &str) -> bool {
        match self {
            Self::All => true,
            Self::Named(names) => names.contains(workspace),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::All => "every akh".to_string(),
            Self::Named(names) => names.iter().cloned().collect::<Vec<_>>().join(", "),
        }
    }
}

/// What an authenticated ACP connection may do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcpAccess {
    /// No ingress auth on this listener (Unix socket, in-process embeds): every
    /// akh and the control plane. Sessions it creates are [`SessionOwner::Local`].
    Local,
    /// The legacy unscoped shared secret (TCP without `[[acp.principals]]`):
    /// every akh and the control plane. Sessions it creates are
    /// [`SessionOwner::Legacy`].
    SharedSecret,
    /// A configured principal.
    Scoped(Arc<AcpGrant>),
}

/// Who created an ACP session (recorded at `session/new`).
///
/// Sessions live only in the daemon's memory (they are never persisted), so
/// every tracked session carries an owner from creation. Should a session ever
/// be restored from state that predates ownership, it is [`Self::Legacy`]:
/// usable by the legacy shared secret and by session operators only.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum SessionOwner {
    /// A configured `[[acp.principals]]` principal, by id.
    Principal(String),
    /// The legacy unscoped shared secret, or a session with no recorded owner
    /// (the default).
    #[default]
    Legacy,
    /// The local Unix socket or an in-process embed (no ingress auth).
    Local,
}

impl fmt::Display for SessionOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Principal(id) => write!(f, "principal `{id}`"),
            Self::Legacy => f.write_str("the legacy shared secret"),
            Self::Local => f.write_str("the local socket"),
        }
    }
}

/// How a connection may use an existing session, from
/// [`AcpAccess::authorize_session`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionUse {
    /// The caller created the session.
    Owner,
    /// A session operator uses a session someone else created (log it).
    Operator {
        /// Who created the session.
        owner: SessionOwner,
    },
    /// No such session. Only session operators learn this; the caller reports
    /// the id as unknown (and `session/cancel` is a no-op).
    Unknown,
}

/// The grant of one authenticated principal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcpGrant {
    /// Principal id (for logs and diagnostics; never the token).
    pub principal: String,
    /// Akhs this principal may use.
    pub akhs: AkhScope,
    /// Whether the server-wide control plane is allowed.
    pub control: bool,
}

impl AcpAccess {
    /// Refuse `workspace` unless this access covers it.
    pub fn authorize_workspace(&self, workspace: &str) -> Result<(), AcpAuthError> {
        match self {
            Self::Local | Self::SharedSecret => Ok(()),
            Self::Scoped(grant) if grant.akhs.contains(workspace) => Ok(()),
            Self::Scoped(grant) => Err(AcpAuthError::WorkspaceNotGranted {
                principal: grant.principal.clone(),
                workspace: workspace.to_string(),
                granted: grant.akhs.describe(),
            }),
        }
    }

    /// Refuse a control-plane `method` unless this access allows the control plane.
    pub fn authorize_control(&self, method: &str) -> Result<(), AcpAuthError> {
        match self {
            Self::Local | Self::SharedSecret => Ok(()),
            Self::Scoped(grant) if grant.control => Ok(()),
            Self::Scoped(grant) => Err(AcpAuthError::ControlNotGranted {
                principal: grant.principal.clone(),
                method: method.to_string(),
            }),
        }
    }

    /// The owner recorded on a session this access creates.
    #[must_use]
    pub fn session_owner(&self) -> SessionOwner {
        match self {
            Self::Local => SessionOwner::Local,
            Self::SharedSecret => SessionOwner::Legacy,
            Self::Scoped(grant) => SessionOwner::Principal(grant.principal.clone()),
        }
    }

    /// Whether this access may use sessions other callers created: the holders
    /// of the control plane (`control = true`, the legacy shared secret, the
    /// local socket). Still limited to the akhs the access is granted.
    #[must_use]
    pub fn is_session_operator(&self) -> bool {
        match self {
            Self::Local | Self::SharedSecret => true,
            Self::Scoped(grant) => grant.control,
        }
    }

    /// Decide whether this access may use session `session_id`, whose recorded
    /// owner is `owner` (`None`: no such session).
    ///
    /// A non-operator gets [`AcpAuthError::SessionNotOwned`] both for a session
    /// someone else created and for an unknown id, so the answer reveals
    /// nothing about which ids exist. This check does not cover the akh: the
    /// caller still authorizes the session's workspace afterwards.
    pub fn authorize_session(
        &self,
        session_id: &str,
        owner: Option<&SessionOwner>,
    ) -> Result<SessionUse, AcpAuthError> {
        let caller = self.session_owner();
        match owner {
            Some(owner) if *owner == caller => Ok(SessionUse::Owner),
            Some(owner) if self.is_session_operator() => Ok(SessionUse::Operator {
                owner: owner.clone(),
            }),
            None if self.is_session_operator() => Ok(SessionUse::Unknown),
            _ => Err(AcpAuthError::SessionNotOwned {
                principal: match self {
                    Self::Scoped(grant) => grant.principal.clone(),
                    Self::Local | Self::SharedSecret => caller.to_string(),
                },
                session_id: session_id.to_string(),
            }),
        }
    }
}

/// One configured principal: its grant and its (reloadable) token file.
#[derive(Debug, Clone)]
struct PrincipalEntry {
    grant: Arc<AcpGrant>,
    token_file: PathBuf,
}

impl PrincipalEntry {
    fn tokens(&self) -> Result<Vec<String>, AcpAuthError> {
        let token_error = |message: String| AcpAuthError::PrincipalToken {
            principal: self.grant.principal.clone(),
            path: self.token_file.clone(),
            message,
        };
        let contents = std::fs::read_to_string(&self.token_file)
            .map_err(|e| token_error(format!("read failed: {e}")))?;
        let tokens: Vec<String> = contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect();
        if tokens.is_empty() {
            return Err(token_error("no non-empty token line".into()));
        }
        if tokens
            .iter()
            .any(|t| t.len() < MIN_PRINCIPAL_TOKEN_LEN || t.chars().any(char::is_whitespace))
        {
            return Err(token_error(format!(
                "a token is shorter than {MIN_PRINCIPAL_TOKEN_LEN} characters or contains whitespace"
            )));
        }
        Ok(tokens)
    }
}

/// Token → principal verifier for `[[acp.principals]]`.
#[derive(Debug, Clone)]
pub struct PrincipalAuth {
    principals: Arc<[PrincipalEntry]>,
}

impl PrincipalAuth {
    /// Validate the configured principals and load every token file once, so a
    /// broken config fails at startup rather than at the first client.
    pub fn from_config(configs: &[AcpPrincipalConfig]) -> Result<Self, AcpAuthError> {
        let mut ids = BTreeSet::new();
        let mut principals = Vec::with_capacity(configs.len());
        for config in configs {
            let id = config.id.trim();
            if id.is_empty() {
                return Err(AcpAuthError::PrincipalConfig {
                    principal: config.id.clone(),
                    message: "`id` is empty".into(),
                });
            }
            if !ids.insert(id.to_string()) {
                return Err(AcpAuthError::PrincipalConfig {
                    principal: id.to_string(),
                    message: "`id` is used by more than one principal".into(),
                });
            }
            if !config.token_file.is_absolute() {
                return Err(AcpAuthError::PrincipalConfig {
                    principal: id.to_string(),
                    message: format!(
                        "`token_file` {} is not absolute",
                        config.token_file.display()
                    ),
                });
            }
            principals.push(PrincipalEntry {
                grant: Arc::new(AcpGrant {
                    principal: id.to_string(),
                    akhs: AkhScope::parse(id, &config.akhs)?,
                    control: config.control,
                }),
                token_file: config.token_file.clone(),
            });
        }
        if principals.is_empty() {
            return Err(AcpAuthError::PrincipalConfig {
                principal: String::new(),
                message: "no principals configured".into(),
            });
        }
        let auth = Self {
            principals: principals.into(),
        };
        auth.check_distinct_tokens()?;
        Ok(auth)
    }

    /// Load every token file; refuse a token shared by two principals (it would
    /// make the grant ambiguous).
    fn check_distinct_tokens(&self) -> Result<(), AcpAuthError> {
        let mut seen: Vec<(String, &PrincipalEntry)> = Vec::new();
        for entry in self.principals.iter() {
            for token in entry.tokens()? {
                if let Some((_, other)) = seen.iter().find(|(t, _)| *t == token) {
                    return Err(AcpAuthError::PrincipalToken {
                        principal: entry.grant.principal.clone(),
                        path: entry.token_file.clone(),
                        message: format!(
                            "a token is also accepted for principal `{}`",
                            other.grant.principal
                        ),
                    });
                }
                seen.push((token, entry));
            }
        }
        Ok(())
    }

    /// Resolve `candidate` to its principal's grant.
    ///
    /// Every token of every principal is compared (constant-time each), so the
    /// work done does not depend on which principal matched. A token file that
    /// became unreadable after startup disables only its own principal (logged
    /// at `error`): one broken key must not lock every other client out. When
    /// *no* principal has a readable token file the check fails closed (`Err`,
    /// answered as 503).
    fn resolve(&self, candidate: &str) -> Result<Option<Arc<AcpGrant>>, AcpAuthError> {
        let mut matched: Option<Arc<AcpGrant>> = None;
        let mut ambiguous = false;
        let mut last_error = None;
        let mut readable = 0usize;
        for entry in self.principals.iter() {
            let tokens = match entry.tokens() {
                Ok(tokens) => tokens,
                Err(error) => {
                    tracing::error!(
                        principal = %entry.grant.principal,
                        error = %error,
                        "ACP principal token file unusable; this principal cannot authenticate"
                    );
                    last_error = Some(error);
                    continue;
                }
            };
            readable += 1;
            let mut hit = false;
            for expected in tokens {
                hit |= constant_time_eq(expected.as_bytes(), candidate.as_bytes());
            }
            if hit {
                ambiguous |= matched.is_some();
                matched = Some(Arc::clone(&entry.grant));
            }
        }
        if readable == 0
            && let Some(error) = last_error
        {
            return Err(error);
        }
        if ambiguous {
            tracing::error!(
                "an ACP token matches more than one principal after a token-file change; \
                 denying it until the token files are distinct"
            );
            return Ok(None);
        }
        Ok(matched)
    }

    /// Ids of the configured principals (for startup logs).
    #[must_use]
    pub fn principal_ids(&self) -> Vec<String> {
        self.principals
            .iter()
            .map(|p| p.grant.principal.clone())
            .collect()
    }
}

/// How the TCP listener authenticates `initialize`.
#[derive(Clone)]
pub enum AcpIngressAuth {
    /// Legacy: any accepted token grants every akh and the control plane.
    SharedSecret(SharedSecretAuth),
    /// Per-principal tokens, each scoped to its akhs.
    Principals(PrincipalAuth),
}

impl From<SharedSecretAuth> for AcpIngressAuth {
    fn from(auth: SharedSecretAuth) -> Self {
        Self::SharedSecret(auth)
    }
}

impl From<PrincipalAuth> for AcpIngressAuth {
    fn from(auth: PrincipalAuth) -> Self {
        Self::Principals(auth)
    }
}

/// Why `initialize` was refused.
#[derive(Debug)]
pub(crate) enum IngressDenial {
    /// The token matches nothing (403).
    InvalidCredentials,
    /// The credential source is unreadable (503; fail closed).
    Unavailable(String),
}

impl AcpIngressAuth {
    /// Build the ingress auth for the TCP listener from the host config.
    ///
    /// Principals win: when any are configured the shared secret is not
    /// consulted (a warning says so if it is set). Without principals the
    /// shared secret is used and a warning records that it is unscoped.
    pub fn from_config(principals: &[AcpPrincipalConfig]) -> Result<Self, AcpAuthError> {
        let secret_set = std::env::var_os(SHARED_SECRET_FILE_ENV).is_some()
            || std::env::var_os(SHARED_SECRET_ENV).is_some();
        if !principals.is_empty() {
            let auth = PrincipalAuth::from_config(principals)?;
            if secret_set {
                tracing::warn!(
                    "{SHARED_SECRET_FILE_ENV}/{SHARED_SECRET_ENV} is set but ignored for ACP \
                     TCP ingress because [[acp.principals]] are configured; give operator \
                     access through a principal with akhs = [\"*\"] and control = true"
                );
            }
            tracing::info!(
                principals = ?auth.principal_ids(),
                "ACP TCP ingress: per-principal akh scoping"
            );
            return Ok(Self::Principals(auth));
        }
        let auth = SharedSecretAuth::from_env()
            .map_err(|message| AcpAuthError::NoIngressCredential { message })?;
        tracing::warn!(
            "ACP TCP ingress uses the legacy shared secret, which is NOT scoped per akh: any \
             accepted token can open sessions on every akh and call the control plane. \
             Configure [[acp.principals]] to scope tokens (ADR 113)"
        );
        Ok(Self::SharedSecret(auth))
    }

    /// Authenticate a bearer token (without the `Bearer ` prefix).
    pub(crate) fn authenticate(&self, token: &str) -> Result<AcpAccess, IngressDenial> {
        match self {
            Self::SharedSecret(auth) => match auth.authorizes(token) {
                Ok(true) => Ok(AcpAccess::SharedSecret),
                Ok(false) => Err(IngressDenial::InvalidCredentials),
                Err(error) => Err(IngressDenial::Unavailable(error)),
            },
            Self::Principals(auth) => match auth.resolve(token) {
                Ok(Some(grant)) => Ok(AcpAccess::Scoped(grant)),
                Ok(None) => Err(IngressDenial::InvalidCredentials),
                Err(error) => Err(IngressDenial::Unavailable(error.to_string())),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN_A: &str = "alpha-token-aaaaaaaaaaaaaaaaaaaa";
    const TOKEN_B: &str = "beta-token-bbbbbbbbbbbbbbbbbbbbbb";

    fn principal(
        dir: &std::path::Path,
        id: &str,
        token: &str,
        akhs: &[&str],
    ) -> AcpPrincipalConfig {
        let token_file = dir.join(format!("{id}.key"));
        std::fs::write(&token_file, format!("{token}\n")).unwrap();
        AcpPrincipalConfig {
            id: id.to_string(),
            token_file,
            akhs: akhs.iter().map(|a| (*a).to_string()).collect(),
            control: false,
        }
    }

    #[test]
    fn token_resolves_to_its_own_principal_and_akhs() {
        let dir = tempfile::tempdir().unwrap();
        let auth = PrincipalAuth::from_config(&[
            principal(dir.path(), "janus", TOKEN_A, &["janus"]),
            principal(dir.path(), "vesta", TOKEN_B, &["vesta"]),
        ])
        .unwrap();
        let a = auth.resolve(TOKEN_A).unwrap().expect("token A resolves");
        assert_eq!(a.principal, "janus");
        let access = AcpAccess::Scoped(a);
        access.authorize_workspace("janus").unwrap();
        let err = access.authorize_workspace("vesta").unwrap_err();
        assert_eq!(err.code_str(), "akh::acp::workspace_not_granted");
        assert!(access.authorize_control("_akh/workspace/create").is_err());
        assert!(
            auth.resolve("unknown-token-ccccccccccccccccccc")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn config_rejects_shared_tokens_bad_scopes_and_relative_paths() {
        let dir = tempfile::tempdir().unwrap();
        let shared = PrincipalAuth::from_config(&[
            principal(dir.path(), "a", TOKEN_A, &["a"]),
            principal(dir.path(), "b", TOKEN_A, &["b"]),
        ])
        .unwrap_err();
        assert_eq!(shared.code_str(), "akh::acp::principal_token");

        let empty = PrincipalAuth::from_config(&[principal(dir.path(), "c", TOKEN_A, &[])]);
        assert!(matches!(empty, Err(AcpAuthError::PrincipalConfig { .. })));

        let mixed = PrincipalAuth::from_config(&[principal(dir.path(), "d", TOKEN_A, &["*", "x"])]);
        assert!(matches!(mixed, Err(AcpAuthError::PrincipalConfig { .. })));

        let mut relative = principal(dir.path(), "e", TOKEN_A, &["e"]);
        relative.token_file = PathBuf::from("e.key");
        assert!(PrincipalAuth::from_config(&[relative]).is_err());

        let short = PrincipalAuth::from_config(&[principal(dir.path(), "f", "short", &["f"])]);
        assert!(matches!(short, Err(AcpAuthError::PrincipalToken { .. })));
    }

    #[test]
    fn wildcard_control_principal_is_an_operator() {
        let dir = tempfile::tempdir().unwrap();
        let mut op = principal(dir.path(), "operator", TOKEN_A, &["*"]);
        op.control = true;
        let auth = PrincipalAuth::from_config(&[op]).unwrap();
        let access = AcpAccess::Scoped(auth.resolve(TOKEN_A).unwrap().unwrap());
        access.authorize_workspace("anything").unwrap();
        access.authorize_control("_akh/daemon/stop").unwrap();
    }

    #[test]
    fn sessions_are_usable_by_their_owner_and_session_operators_only() {
        let scoped = |id: &str, control: bool| {
            AcpAccess::Scoped(Arc::new(AcpGrant {
                principal: id.into(),
                akhs: AkhScope::All,
                control,
            }))
        };
        let a = scoped("a", false);
        let b = scoped("b", false);
        let op = scoped("op", true);
        let owned_by_a = SessionOwner::Principal("a".into());

        assert_eq!(a.session_owner(), owned_by_a);
        assert_eq!(
            a.authorize_session("akh-1", Some(&owned_by_a)).unwrap(),
            SessionUse::Owner
        );
        let foreign = b.authorize_session("akh-1", Some(&owned_by_a)).unwrap_err();
        assert_eq!(foreign.code_str(), "akh::acp::session_not_owned");
        let unknown = b.authorize_session("akh-1", None).unwrap_err();
        assert_eq!(
            foreign.to_string(),
            unknown.to_string(),
            "foreign and unknown ids must be indistinguishable to a non-operator"
        );
        let rpc = foreign.rpc_error();
        assert_eq!(
            rpc.code,
            agent_client_protocol::Error::invalid_params().code
        );
        assert_eq!(rpc.data, unknown.rpc_error().data);

        assert_eq!(
            op.authorize_session("akh-1", Some(&owned_by_a)).unwrap(),
            SessionUse::Operator { owner: owned_by_a }
        );
        assert_eq!(
            op.authorize_session("akh-1", None).unwrap(),
            SessionUse::Unknown
        );
        assert!(
            b.authorize_session("akh-2", Some(&SessionOwner::Legacy))
                .is_err()
        );
        assert_eq!(
            AcpAccess::SharedSecret
                .authorize_session("akh-2", Some(&SessionOwner::Legacy))
                .unwrap(),
            SessionUse::Owner
        );
        assert_eq!(
            AcpAccess::Local
                .authorize_session("akh-2", Some(&SessionOwner::Legacy))
                .unwrap(),
            SessionUse::Operator {
                owner: SessionOwner::Legacy
            }
        );
    }

    #[test]
    fn rotated_token_file_is_reloaded() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = principal(dir.path(), "janus", TOKEN_A, &["janus"]);
        let path = cfg.token_file.clone();
        let auth = PrincipalAuth::from_config(&[cfg]).unwrap();
        std::fs::write(&path, format!("{TOKEN_B}\n{TOKEN_A}\n")).unwrap();
        assert!(auth.resolve(TOKEN_B).unwrap().is_some());
        std::fs::write(&path, format!("{TOKEN_B}\n")).unwrap();
        assert!(
            auth.resolve(TOKEN_A).unwrap().is_none(),
            "retired token refused"
        );
    }
}
