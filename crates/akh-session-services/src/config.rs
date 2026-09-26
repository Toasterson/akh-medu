use crate::types::{
    streaming::{StreamingConfig, StreamingConfigError},
    *,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub url: String,
    #[serde(default)]
    pub token_file: PathBuf,
    #[serde(default)]
    pub token_env: Option<String>,
    pub ca_file: Option<PathBuf>,
    #[serde(default)]
    pub allow_insecure_http: bool,
    /// Total request deadline; slow cognitive cold starts may need more than 30s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timeout_secs: Option<u64>,
}

impl Endpoint {
    pub fn validate(&self) -> Result<()> {
        if self
            .request_timeout_secs
            .is_some_and(|value| !(1..=300).contains(&value))
        {
            return Err(Fault::invalid(
                "request_timeout_secs must be between 1 and 300",
            ));
        }
        let expanded = self.url.replace("{sandbox}", "allocated");
        if expanded.contains(['{', '}']) {
            return Err(Fault::invalid("unknown service URL placeholder"));
        }
        let url = url::Url::parse(&expanded).map_err(|_| Fault::invalid("invalid service URL"))?;
        let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
        if !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !matches!(url.scheme(), "http" | "https")
            || (url.scheme() == "http" && !local && !self.allow_insecure_http)
        {
            return Err(Fault::invalid(
                "service URL requires HTTPS or explicitly authorized private HTTP, with no embedded credentials/query",
            ));
        }
        if let Some(name) = &self.token_env {
            if !self.token_file.as_os_str().is_empty()
                || name.is_empty()
                || !name
                    .bytes()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
                || name.as_bytes()[0].is_ascii_digit()
            {
                return Err(Fault::invalid(
                    "endpoint requires one valid credential source",
                ));
            }
        } else if !self.token_file.is_absolute() {
            return Err(Fault::invalid("token_file must be absolute"));
        }
        if self.ca_file.as_ref().is_some_and(|p| !p.is_absolute()) {
            return Err(Fault::invalid("ca_file must be absolute"));
        }
        Ok(())
    }
    pub fn http_client(&self) -> Result<reqwest::Client> {
        self.validate()?;
        let mut builder = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(
                self.request_timeout_secs.unwrap_or(30),
            ))
            .redirect(reqwest::redirect::Policy::none());
        if let Some(path) = &self.ca_file {
            let pem =
                std::fs::read(path).map_err(|_| Fault::invalid("service CA could not be read"))?;
            let certificate = reqwest::Certificate::from_pem(&pem)
                .map_err(|_| Fault::invalid("service CA is invalid PEM"))?;
            builder = builder.add_root_certificate(certificate);
        }
        builder
            .build()
            .map_err(|_| Fault::new("transport", "HTTP client initialization failed"))
    }
    pub fn token(&self) -> Result<String> {
        if let Some(name) = &self.token_env {
            let token = std::env::var(name).map_err(|_| {
                Fault::new("auth_unavailable", "credential environment unavailable")
            })?;
            if token.len() < 24 || token.chars().any(char::is_whitespace) {
                return Err(Fault::new(
                    "auth_unavailable",
                    "invalid credential environment",
                ));
            }
            return Ok(token);
        }
        read_token(&self.token_file)
    }
    pub fn route(&self, path: &str) -> String {
        format!("{}{}", self.url.trim_end_matches('/'), path)
    }
}

pub fn read_token(path: &Path) -> Result<String> {
    let token = std::fs::read_to_string(path)
        .map_err(|_| Fault::new("auth_unavailable", "credential source could not be read"))?;
    let token = token.trim();
    if token.len() < 24 || token.chars().any(|c| c.is_whitespace()) {
        return Err(Fault::new(
            "auth_unavailable",
            "credential must be at least 24 characters without whitespace",
        ));
    }
    Ok(token.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Principal {
    pub id: String,
    pub token_file: PathBuf,
    /// Store only a SHA-256 verifier on less-trusted receiving services.
    #[serde(default)]
    pub token_sha256: bool,
    pub roles: Vec<String>,
    #[serde(default)]
    pub akhs: Vec<String>,
    #[serde(default)]
    pub sources: Vec<String>,
}

impl Principal {
    pub fn credential(&self) -> Result<String> {
        let value = read_token(&self.token_file)?;
        if self.token_sha256
            && (value.len() != 64
                || !value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
        {
            return Err(Fault::invalid(
                "incoming token verifier must be lowercase SHA-256 hex",
            ));
        }
        Ok(value)
    }
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role || r == "admin")
    }
    pub fn can_use_akh(&self, akh: &str) -> bool {
        self.has_role("admin") || self.akhs.iter().any(|a| a == akh)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkhProfile {
    /// Explicit, narrower tool/repository grants for an authorized workspace profile.
    /// This lets coordination sessions keep the Akh identity without Anima assignment credentials.
    #[serde(default)]
    pub session_grants: BTreeMap<String, SessionGrants>,
    /// Nonempty requires immutable repository bootstrap before allocation.
    #[serde(default)]
    pub bootstrap_repositories: Vec<String>,
    pub core_service: Option<String>,
    pub runtime: RuntimeBinding,
    pub instructions: String,
    pub workspace_profiles: Vec<String>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionGrants {
    pub tools: Vec<String>,
    #[serde(default)]
    pub bootstrap_repositories: Vec<String>,
}

pub use akh_session_types::ProviderConfig as RuntimeProfile;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Provisioner {
    /// Existing executor environment, with no implied isolation stronger than its handshake.
    Existing { executor: String, root: String },
    /// Admin-provided deterministic allocation command. JSON stdin/stdout contract, no shell interpolation.
    Command {
        acquire: Vec<String>,
        release: Vec<String>,
    },
    /// Kubernetes SIG Agent Sandbox CRD with a selected, admin-owned manifest template.
    Kubernetes {
        api: String,
        namespace: String,
        resource: String,
        template: serde_json::Value,
        executor: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceProfile {
    #[serde(default)]
    pub repository_bootstrap: Option<GitBootstrapProfile>,
    pub revision: String,
    pub site: String,
    pub provisioner: Provisioner,
    #[serde(default)]
    pub services: Vec<ServiceProfile>,
    #[serde(default)]
    pub retain_on_release: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitBootstrapProfile {
    pub repositories: Vec<String>,
    pub executable: PathBuf,
    /// Private Git/credential-helper configuration installed inside the sandbox.
    pub config_file: Option<PathBuf>,
}

/// Executor-owned Git installation. No executable or configuration is accepted
/// from a model's snapshot arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitSnapshotProfile {
    pub executable: PathBuf,
    pub executable_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitPublicationProfile {
    pub repository: String,
    /// Full prefixes ending in '/', e.g. refs/heads/anima/. No blanket heads/.
    pub branch_prefixes: Vec<String>,
    pub config_file: Option<PathBuf>,
}
impl GitPublicationProfile {
    pub fn validate(&self) -> Result<()> {
        validate_repository(&self.repository)?;
        if self.branch_prefixes.is_empty()
            || self.branch_prefixes.len() > 32
            || self
                .config_file
                .as_ref()
                .is_some_and(|path| !path.is_absolute())
        {
            return Err(Fault::invalid(
                "Git publication needs scoped branch prefixes and absolute private configuration",
            ));
        }
        for prefix in &self.branch_prefixes {
            publication::validate_branch(&format!("{prefix}probe"))?;
            if !prefix.ends_with('/') || prefix == "refs/heads/" {
                return Err(Fault::invalid(
                    "publication prefix must select a work-branch namespace",
                ));
            }
        }
        Ok(())
    }
}

impl GitSnapshotProfile {
    pub fn validate(&self) -> Result<()> {
        if !self.executable.is_absolute()
            || self.executable_sha256.len() != 64
            || !self
                .executable_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Fault::invalid(
                "Git snapshot requires an absolute executable and lowercase SHA-256",
            ));
        }
        Ok(())
    }
}

pub fn validate_repository(repository: &str) -> Result<()> {
    let url = url::Url::parse(repository)
        .map_err(|_| Fault::invalid("repository requires an explicit URL"))?;
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    if repository.len() > 4096
        || repository
            .chars()
            .any(|c| c.is_control() || c.is_whitespace())
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.host_str().is_none()
        || url.path().is_empty()
        || url.path() == "/"
        || (!url.username().is_empty() && url.scheme() != "ssh")
        || !(matches!(url.scheme(), "https" | "ssh")
            || (local && matches!(url.scheme(), "http" | "git")))
    {
        return Err(Fault::invalid(
            "repository requires HTTPS or SSH without embedded secrets; test transports are loopback-only",
        ));
    }
    Ok(())
}

pub fn validate_checkout(request: &bootstrap::BootstrapRepository) -> Result<()> {
    validate_key(&request.request_id)?;
    validate_repository(&request.checkout.repository)?;
    if !matches!(request.checkout.revision.len(), 40 | 64)
        || !request
            .checkout
            .revision
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(Fault::invalid(
            "repository bootstrap requires a complete lowercase commit object ID",
        ));
    }
    Ok(())
}

/// Where a routed event is delivered (Phase 53d, ADR 113 decision 4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RouteTarget {
    /// Invoke an akhsessd automation (`POST /v1/automations/{automation}/invoke`
    /// on the `sessions` endpoint). The pre-53d behaviour.
    Automation { automation: String },
    /// Admit the event into an akh's own reactor:
    /// `POST {endpoints[endpoint].url}/v1/akhs/{akh}/events`. The akh wakes its
    /// event schedules whose source is `<event source>/<event type>`.
    Akh {
        /// Name of a configured `endpoints` entry for akhomed's `/v1` API
        /// (URL, CA and HTTPS policy come from there).
        endpoint: String,
        akh: String,
        /// Bearer credential of the core principal holding the `event` role.
        /// Absent = the endpoint's own credential.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_file: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventRoute {
    /// For replaceable state notifications only, never delegations or commands.
    #[serde(default)]
    pub coalesce_pending: bool,
    pub id: String,
    pub source: String,
    pub event_type: String,
    /// Shorthand for `target = { kind = "automation", automation = ... }`,
    /// kept so existing configs parse unchanged. Exactly one of `automation`
    /// and `target` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub automation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<RouteTarget>,
}

impl EventRoute {
    /// The resolved delivery target. Errors unless exactly one of
    /// `automation` and `target` is set.
    pub fn resolved_target(&self) -> Result<RouteTarget> {
        match (&self.automation, &self.target) {
            (Some(automation), None) => Ok(RouteTarget::Automation {
                automation: automation.clone(),
            }),
            (None, Some(target)) => Ok(target.clone()),
            _ => Err(Fault::invalid(format!(
                "route {} must set exactly one of `automation` and `target`",
                self.id
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerSource {
    pub uri_file: PathBuf,
    pub queue: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoardSource {
    pub endpoint: String,
    pub source: String,
    pub poll_path: String,
    #[serde(default = "board_interval")]
    pub interval_secs: u64,
}
fn board_interval() -> u64 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DjehutySource {
    #[serde(default)]
    pub allowed_authors: Vec<String>,
    #[serde(default)]
    pub message_types: Vec<String>,
    #[serde(default)]
    pub recipient_id: Option<String>,
    pub endpoint: String,
    pub source: String,
    pub threads: Vec<String>,
    #[serde(default)]
    pub ignore_authors: Vec<String>,
    #[serde(default = "board_interval")]
    pub interval_secs: u64,
    #[serde(default = "djehuty_page_size")]
    pub page_size: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DjehutyInbox {
    #[serde(default = "inbox_enabled")]
    pub enabled: bool,
    pub endpoint: String,
    pub source: String,
    pub recipient_id: String,
    #[serde(default)]
    pub allowed_authors: Vec<String>,
    pub message_types: Vec<String>,
    #[serde(default = "board_interval")]
    pub interval_secs: u64,
    #[serde(default = "djehuty_page_size")]
    pub page_size: u16,
}
fn inbox_enabled() -> bool {
    true
}
fn djehuty_page_size() -> u16 {
    50
}

/// Selects the durable store's backend. Absent (the default) keeps every
/// daemon on the original per-kind redb file under `data_dir`; this is what
/// every test and every deployment predating the Postgres backend still
/// gets. `Postgres` moves that daemon's state into the shared `akh-pg`
/// CloudNativePG cluster so it survives node/PVC loss instead of living on
/// a Raspberry Pi SD card.
///
/// The connection string is deliberately never a config field: it is read
/// from the environment variable named by `url_env` at startup, so
/// credentials never need to appear in a (SOPS-decrypted or not) config
/// file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StoreConfig {
    /// Local SQLite WAL database under data_dir, with no network dependency.
    Sqlite,
    Postgres {
        /// Name of the environment variable holding the `postgres://` URL.
        url_env: String,
        /// Postgres schema the store's table lives in. Must already be
        /// creatable by the connecting role (`CREATE SCHEMA IF NOT EXISTS`
        /// runs on open) and must only ever be used for this purpose, since
        /// the database itself is shared with other consumers (e.g. akhomed).
        #[serde(default = "default_store_schema")]
        schema: String,
    },
}
fn default_store_schema() -> String {
    "akh_sessions".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub service_id: String,
    pub public_url: Option<String>,
    pub listen: SocketAddr,
    pub data_dir: PathBuf,
    /// Address for the heap-profiling listener. Deliberately separate from
    /// `listen`: a heap profile contains live session content and tokens, and
    /// `listen` is published through an HTTPRoute. Absent means no profiling
    /// listener at all.
    #[serde(default)]
    pub profiling_listen: Option<SocketAddr>,
    /// Durable store backend. Defaults to the redb file under `data_dir`
    /// when absent; see [`StoreConfig`].
    #[serde(default)]
    pub store: Option<StoreConfig>,
    pub principals: Vec<Principal>,
    #[serde(default)]
    pub endpoints: BTreeMap<String, Endpoint>,
    /// MCP aliases which must never use the endpoint's daemon-wide token file.
    #[serde(default)]
    pub session_mcp_endpoints: Vec<String>,
    #[serde(default)]
    pub akhs: BTreeMap<String, AkhProfile>,
    #[serde(default)]
    pub runtimes: BTreeMap<String, RuntimeProfile>,
    #[serde(default)]
    pub workspaces: BTreeMap<String, WorkspaceProfile>,
    #[serde(default)]
    pub automations: BTreeMap<String, AutomationDefinition>,
    #[serde(default)]
    pub routes: Vec<EventRoute>,
    #[serde(default)]
    pub brokers: Vec<BrokerSource>,
    #[serde(default)]
    pub boards: Vec<BoardSource>,
    #[serde(default)]
    pub djehuty: Vec<DjehutySource>,
    #[serde(default)]
    pub djehuty_inboxes: Vec<DjehutyInbox>,
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    pub workspace_root: Option<PathBuf>,
    #[serde(default)]
    pub git_snapshot: Option<GitSnapshotProfile>,
    #[serde(default)]
    pub git_publication: Vec<GitPublicationProfile>,
    #[serde(default = "unconfined")]
    pub confinement: String,
    /// `deployed` refuses `Colocated` native runtimes (Phase 53 decision 7).
    /// `AKH_DEPLOYMENT=deployed` has the same effect; see [`Config::deployment_mode`].
    #[serde(default)]
    pub deployment: DeploymentMode,
    /// How often akhsessd's dispatcher re-checks each session for stalled
    /// progress or a retirement that can now be finalized. Throttles the
    /// reconciliation *decision*, not the fast per-session dispatch retry.
    #[serde(default = "default_reconciliation_interval_ms")]
    pub reconciliation_interval_ms: u64,
    /// A session (provisioning or mid-turn) with no durable progress for
    /// this long is force-retired so its caller gets a terminal status
    /// instead of an indefinite "provisioning".
    #[serde(default = "default_session_stall_threshold_ms")]
    pub session_stall_threshold_ms: u64,
    /// How long to wait, after retirement is requested, for the owning
    /// runtime to confirm a turn's cancellation before closing it locally as
    /// an unknown outcome. Bounds how long "pending retirement" can persist
    /// when the runtime is unreachable.
    #[serde(default = "default_retirement_cancel_grace_ms")]
    pub retirement_cancel_grace_ms: u64,
    /// How akhrtd forwards native-runtime text and reasoning deltas:
    /// coalesced into paragraphs (default) or token by token. Lives on the
    /// service, not on a runtime profile, so changing it never invalidates
    /// profile conformance evidence.
    #[serde(default)]
    pub streaming: StreamingConfig,
}
fn unconfined() -> String {
    "unconfined".into()
}

/// The fault a deployed service returns for a `Colocated` runtime. The message
/// carries its own remedy because `Fault` has no separate help field.
pub fn colocated_forbidden(subject: &str) -> Fault {
    Fault::new(
        "colocated_forbidden",
        format!(
            "{subject}: execution_mode \"colocated\" is forbidden in deployed mode; \
             native runtime tools would run in the runtime's own address space. \
             Set execution_mode = \"brokered\" with installed conformance \
             evidence so workspace effects go through akhsbxd/akhexecd, or run this \
             service with deployment = \"development\" and without AKH_DEPLOYMENT=deployed \
             for local development only."
        ),
    )
}
/// Diagnostic code for a deployed runtime profile without an explicit
/// `execution_mode`.
pub const EXECUTION_MODE_REQUIRED: &str = "akh::runtime::execution_mode_required";

/// The fault a deployed service returns for runtime profiles that omit
/// `execution_mode`. The omitted value means `colocated` (the development
/// default), so a deployed service will not guess: the operator states it.
pub fn execution_mode_required(subject: &str) -> Fault {
    Fault::new(
        EXECUTION_MODE_REQUIRED,
        format!(
            "{subject}: execution_mode is not set, and deployed mode requires it explicitly \
             (an omitted value means \"colocated\", which deployed services refuse). \
             help: add execution_mode = \"brokered\" to {subject} and install its \
             conformance evidence (akh-runtime-profile), so workspace effects go through \
             akhsbxd/akhexecd; for local development only, run with deployment = \
             \"development\" and without AKH_DEPLOYMENT=deployed."
        ),
    )
}

/// The fault for an invalid `[streaming]` section. `Fault` has no separate
/// help field, so the diagnostic's code becomes the fault code and its help
/// travels in the message.
pub fn streaming_fault(error: StreamingConfigError) -> Fault {
    use miette::Diagnostic;
    let code = error
        .code()
        .map_or_else(|| "invalid_request".to_owned(), |code| code.to_string());
    let help = error
        .help()
        .map(|help| format!(" help: {help}"))
        .unwrap_or_default();
    Fault::new(&code, format!("{error}.{help}"))
}

fn default_reconciliation_interval_ms() -> u64 {
    5 * 60_000
}
fn default_session_stall_threshold_ms() -> u64 {
    15 * 60_000
}
fn default_retirement_cancel_grace_ms() -> u64 {
    2 * 60_000
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| Fault::invalid(e.to_string()))?;
        let config: Self = if path.extension().is_some_and(|e| e == "json") {
            serde_json::from_str(&text).map_err(|e| Fault::invalid(e.to_string()))?
        } else {
            toml::from_str(&text).map_err(|e| Fault::invalid(e.to_string()))?
        };
        config.validate()?;
        Ok(config)
    }
    /// The effective deployment mode: the configured value, made stricter by
    /// `AKH_DEPLOYMENT=deployed` (the environment can never relax it).
    pub fn deployment_mode(&self) -> Result<DeploymentMode> {
        DeploymentMode::resolve_from_env(self.deployment)
            .map_err(|error| Fault::invalid(error.to_string()))
    }

    /// Refuse runtime profiles and akh bindings that would execute workspace
    /// effects in the runtime's own address space while deployed (ADR 113
    /// decision 3). A development configuration is always accepted.
    pub fn validate_deployment(&self, mode: DeploymentMode) -> Result<()> {
        if !mode.is_deployed() {
            return Ok(());
        }
        let omitted = self
            .runtimes
            .iter()
            .filter(|(_, profile)| profile.execution_mode_setting().is_none())
            .map(|(id, _)| format!("runtimes.{id}"))
            .collect::<Vec<_>>();
        if !omitted.is_empty() {
            return Err(execution_mode_required(&omitted.join(", ")));
        }
        let colocated = self
            .runtimes
            .iter()
            .filter(|(_, profile)| profile.execution_mode() == ExecutionMode::Colocated)
            .map(|(id, _)| format!("runtimes.{id}"))
            .chain(
                self.akhs
                    .iter()
                    .filter(|(_, akh)| akh.runtime.execution_mode == ExecutionMode::Colocated)
                    .map(|(id, _)| format!("akhs.{id}.runtime")),
            )
            .collect::<Vec<_>>();
        if colocated.is_empty() {
            return Ok(());
        }
        Err(colocated_forbidden(&colocated.join(", ")))
    }

    pub fn validate(&self) -> Result<()> {
        validate_key(&self.service_id)?;
        self.streaming.validate().map_err(streaming_fault)?;
        self.validate_deployment(self.deployment_mode()?)?;
        if let Some(StoreConfig::Postgres { url_env, schema }) = &self.store
            && (url_env.is_empty() || schema.is_empty())
        {
            return Err(Fault::invalid(
                "store.url_env and store.schema must be non-empty when store.kind is postgres",
            ));
        }
        if let Some(profile) = &self.git_snapshot {
            profile.validate()?;
        }
        let mut repositories = std::collections::HashSet::new();
        for profile in &self.git_publication {
            profile.validate()?;
            if self.git_snapshot.is_none() || !repositories.insert(&profile.repository) {
                return Err(Fault::invalid(
                    "Git publication needs a pinned Git installation and unique repository profiles",
                ));
            }
        }
        if !self.data_dir.is_absolute() || self.principals.is_empty() {
            return Err(Fault::invalid(
                "absolute data_dir and authenticated principals are required",
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for principal in &self.principals {
            validate_key(&principal.id)?;
            if !ids.insert(&principal.id) || !principal.token_file.is_absolute() {
                return Err(Fault::invalid(
                    "principal IDs must be unique and credential paths absolute",
                ));
            }
        }
        for endpoint in self.endpoints.values() {
            endpoint.validate()?;
        }
        let mut scoped = std::collections::HashSet::new();
        for service in &self.session_mcp_endpoints {
            validate_key(service)?;
            if !scoped.insert(service) || self.endpoint(service)?.url.contains('{') {
                return Err(Fault::invalid(
                    "session MCP endpoints must be unique, fixed destinations",
                ));
            }
        }
        let mut sources = std::collections::HashSet::new();
        for source in &self.djehuty {
            validate_key(&source.source)?;
            self.endpoint(&source.endpoint)?;
            for id in source
                .threads
                .iter()
                .chain(&source.allowed_authors)
                .chain(&source.ignore_authors)
                .chain(source.recipient_id.iter())
            {
                uuid::Uuid::parse_str(id)
                    .map_err(|_| Fault::invalid("invalid Djehuty UUID in configuration"))?;
            }
            for kind in &source.message_types {
                validate_key(kind)?;
            }
            if !sources.insert(&source.source)
                || source.threads.is_empty()
                || source.page_size == 0
                || source.page_size > 200
                || source.interval_secs == 0
            {
                return Err(Fault::invalid(
                    "Djehuty sources require unique source IDs, threads, a positive interval and page size 1..200",
                ));
            }
        }
        for source in &self.djehuty_inboxes {
            validate_key(&source.source)?;
            self.endpoint(&source.endpoint)?;
            for id in std::iter::once(&source.recipient_id).chain(&source.allowed_authors) {
                uuid::Uuid::parse_str(id)
                    .map_err(|_| Fault::invalid("invalid Djehuty inbox UUID"))?;
            }
            for kind in &source.message_types {
                validate_key(kind)?;
            }
            if !sources.insert(&source.source)
                || source.message_types.is_empty()
                || source.page_size == 0
                || source.page_size > 200
                || source.interval_secs == 0
            {
                return Err(Fault::invalid(
                    "Djehuty inboxes require unique sources, protocols and positive bounded polling",
                ));
            }
        }
        let mut route_ids = std::collections::HashSet::new();
        for route in &self.routes {
            validate_key(&route.id)?;
            validate_key(&route.source)?;
            validate_key(&route.event_type)?;
            if !route_ids.insert(&route.id) {
                return Err(Fault::invalid("event route IDs must be unique"));
            }
            match route.resolved_target()? {
                RouteTarget::Automation { automation } => validate_key(&automation)?,
                RouteTarget::Akh {
                    endpoint,
                    akh,
                    token_file,
                } => {
                    self.endpoint(&endpoint)?;
                    validate_key(&akh)?;
                    if token_file.as_ref().is_some_and(|path| !path.is_absolute()) {
                        return Err(Fault::invalid("route target token_file must be absolute"));
                    }
                }
            }
        }
        for (id, automation) in &self.automations {
            validate_key(id)?;
            let owner = self
                .principals
                .iter()
                .find(|p| p.id == automation.principal)
                .ok_or_else(|| Fault::invalid("automation owner is not configured"))?;
            let akh = self
                .akhs
                .get(&automation.akh)
                .ok_or_else(|| Fault::invalid("automation Akh is not configured"))?;
            if automation.id != *id
                || automation.max_active == 0
                || automation.max_age_ms == 0
                || automation.sources.is_empty()
                || !owner.can_use_akh(&automation.akh)
                || !akh
                    .workspace_profiles
                    .contains(&automation.workspace_profile)
            {
                return Err(Fault::invalid(
                    "automation requires matching identity, authorized workspace and positive admission limits",
                ));
            }
            for source in &automation.sources {
                validate_key(source)?;
            }
            match &automation.target {
                SessionTarget::Existing { session_id } => validate_key(session_id)?,
                SessionTarget::Conversation { key } => validate_key(key)?,
                _ => {}
            }
        }
        for (name, profile) in &self.akhs {
            validate_key(name)?;
            for repository in &profile.bootstrap_repositories {
                validate_repository(repository)?;
            }
            for workspace in profile.session_grants.keys() {
                self.session_profile(name, workspace)?;
            }
            let mut tools = std::collections::HashSet::new();
            for tool in &profile.tools {
                validate_key(&tool.name)?;
                if !tools.insert(&tool.name)
                    || tool
                        .input_schema
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        != Some("object")
                {
                    return Err(Fault::invalid(
                        "tools require unique names and object input schemas",
                    ));
                }
            }
        }
        for profile in self.workspaces.values() {
            if let Some(git) = &profile.repository_bootstrap {
                if git.repositories.is_empty()
                    || !git.executable.is_absolute()
                    || git.config_file.as_ref().is_some_and(|p| !p.is_absolute())
                {
                    return Err(Fault::invalid(
                        "Git bootstrap requires allowed repositories and absolute sandbox executable/config paths",
                    ));
                }
                for repository in &git.repositories {
                    validate_repository(repository)?;
                }
            }
        }
        Ok(())
    }
    pub fn session_profile(&self, akh: &str, workspace: &str) -> Result<AkhProfile> {
        let mut profile = self
            .akhs
            .get(akh)
            .ok_or_else(|| Fault::missing("Akh profile not found"))?
            .clone();
        if !profile.workspace_profiles.iter().any(|p| p == workspace) {
            return Err(Fault::new(
                "forbidden",
                "workspace profile is not granted to this Akh",
            ));
        }
        if let Some(grants) = profile.session_grants.get(workspace) {
            let names: std::collections::HashSet<_> = grants.tools.iter().collect();
            if names.len() != grants.tools.len()
                || grants
                    .tools
                    .iter()
                    .any(|n| !profile.tools.iter().any(|t| &t.name == n))
                || grants
                    .bootstrap_repositories
                    .iter()
                    .any(|r| !profile.bootstrap_repositories.contains(r))
            {
                return Err(Fault::invalid(
                    "workspace session grants must narrow existing Akh tools and repositories",
                ));
            }
            profile.tools.retain(|t| names.contains(&t.name));
            profile.bootstrap_repositories = grants.bootstrap_repositories.clone();
        }
        Ok(profile)
    }
    pub fn endpoint(&self, id: &str) -> Result<&Endpoint> {
        self.endpoints
            .get(id)
            .ok_or_else(|| Fault::invalid(format!("service endpoint {id} is not configured")))
    }
}

#[cfg(test)]
mod endpoint_tests {
    use super::*;

    #[tokio::test]
    async fn endpoint_deadline_is_configurable_and_bounded() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(|| async {
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                "restored"
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut endpoint: Endpoint = serde_json::from_value(serde_json::json!({
            "url": format!("http://{address}"), "token_file": "/unused"
        }))
        .unwrap();
        assert_eq!(endpoint.request_timeout_secs, None);
        endpoint.validate().unwrap();
        for seconds in [0, 301] {
            endpoint.request_timeout_secs = Some(seconds);
            assert!(endpoint.http_client().is_err());
        }
        endpoint.request_timeout_secs = Some(1);
        let error = endpoint
            .http_client()
            .unwrap()
            .get(&endpoint.url)
            .send()
            .await
            .unwrap_err();
        assert!(error.is_timeout());
        endpoint.request_timeout_secs = Some(3);
        let response = endpoint
            .http_client()
            .unwrap()
            .get(&endpoint.url)
            .send()
            .await
            .unwrap();
        assert_eq!(response.text().await.unwrap(), "restored");
        server.abort();
    }
}

#[cfg(test)]
mod deployment_tests {
    use super::*;
    use serde_json::json;

    fn config(deployment: &str, mode: Option<&str>) -> Config {
        let dir = std::env::temp_dir();
        let mut runtime =
            json!({"revision":"1","kind":"codex","working_dir":dir,"binary":"/usr/bin/true"});
        if let Some(mode) = mode {
            runtime["execution_mode"] = json!(mode);
        }
        serde_json::from_value(json!({
            "service_id":"runtime","listen":"127.0.0.1:0","data_dir":dir,"deployment":deployment,
            "principals":[{"id":"sessions","roles":["session"],"token_file":"/unused"}],
            "runtimes":{"codex-worker":runtime}
        }))
        .unwrap()
    }

    #[test]
    fn deployed_runtime_services_refuse_colocated_profiles() {
        let colocated = config("deployed", Some("colocated"));
        let fault = colocated
            .validate_deployment(DeploymentMode::Deployed)
            .unwrap_err();
        assert_eq!(fault.code, "colocated_forbidden");
        assert!(fault.message.contains("runtimes.codex-worker"), "{fault}");
        // The same profile is accepted for development.
        colocated
            .validate_deployment(DeploymentMode::Development)
            .unwrap();
    }

    #[test]
    fn an_omitted_execution_mode_is_colocated_for_development_and_refused_when_deployed() {
        let omitted = config("deployed", None);
        let profile = &omitted.runtimes["codex-worker"];
        assert_eq!(profile.execution_mode_setting(), None);
        assert_eq!(profile.execution_mode(), ExecutionMode::Colocated);
        let fault = omitted
            .validate_deployment(DeploymentMode::Deployed)
            .unwrap_err();
        assert_eq!(fault.code, EXECUTION_MODE_REQUIRED);
        assert!(fault.message.contains("runtimes.codex-worker"), "{fault}");
        assert!(
            fault.message.contains("execution_mode = \"brokered\""),
            "{fault}"
        );
        // Development keeps the pre-Phase-53 meaning of an omitted mode.
        let development = config("development", None);
        assert_eq!(development.deployment, DeploymentMode::Development);
        development
            .validate_deployment(DeploymentMode::Development)
            .unwrap();
    }

    #[test]
    fn streaming_defaults_to_paragraphs_and_rejects_invalid_settings_with_help() {
        use crate::types::streaming::StreamingMode;
        let dir = std::env::temp_dir();
        let base = json!({
            "service_id":"runtime","listen":"127.0.0.1:0","data_dir":dir,
            "principals":[{"id":"sessions","roles":["session"],"token_file":"/unused"}]
        });
        let config: Config = serde_json::from_value(base.clone()).unwrap();
        assert_eq!(config.streaming, StreamingConfig::default());
        assert_eq!(config.streaming.mode, StreamingMode::Paragraph);
        config.validate().unwrap();

        let mut token = base.clone();
        token["streaming"] = json!({"mode":"token"});
        let config: Config = serde_json::from_value(token).unwrap();
        assert_eq!(config.streaming.mode, StreamingMode::Token);

        let mut invalid = base;
        invalid["streaming"] = json!({"idle_flush_ms":5000,"max_latency_ms":1000});
        let config: Config = serde_json::from_value(invalid).unwrap();
        let fault = config.validate().unwrap_err();
        assert_eq!(
            fault.code,
            "akh::runtime::streaming::idle_exceeds_max_latency"
        );
        assert!(fault.message.contains("help:"), "{fault}");

        let toml_config: Config = toml::from_str(&format!(
            "service_id = \"runtime\"\nlisten = \"127.0.0.1:0\"\ndata_dir = {:?}\n\
             [[principals]]\nid = \"sessions\"\nroles = [\"session\"]\ntoken_file = \"/unused\"\n\
             [streaming]\nmode = \"paragraph\"\nmax_bytes = 2048\n",
            dir.to_string_lossy()
        ))
        .unwrap();
        assert_eq!(toml_config.streaming.max_bytes, 2048);
        assert_eq!(
            toml_config.streaming.idle_flush_ms,
            StreamingConfig::DEFAULT_IDLE_FLUSH_MS
        );
    }

    #[test]
    fn an_explicit_brokered_profile_passes_deployed_validation() {
        let brokered = config("deployed", Some("brokered"));
        assert_eq!(
            brokered.runtimes["codex-worker"].execution_mode(),
            ExecutionMode::Brokered
        );
        brokered
            .validate_deployment(DeploymentMode::Deployed)
            .unwrap();
    }

    #[test]
    fn deployed_services_hold_opencode_profiles_to_the_same_rule() {
        for (mode, refused) in [("colocated", true), ("brokered", false)] {
            let mut config = config("deployed", Some(mode));
            let mut profile = config.runtimes.remove("codex-worker").unwrap();
            profile.kind = RuntimeKind::OpenCode;
            config.runtimes.insert("opencode-worker".into(), profile);
            let result = config.validate_deployment(DeploymentMode::Deployed);
            assert_eq!(result.is_err(), refused, "{mode}");
            if let Err(fault) = result {
                assert_eq!(fault.code, "colocated_forbidden");
                assert!(
                    fault.message.contains("runtimes.opencode-worker"),
                    "{fault}"
                );
            }
        }
    }
}
