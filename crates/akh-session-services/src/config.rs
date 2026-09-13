use crate::types::*;
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
    pub token_file: PathBuf,
    pub ca_file: Option<PathBuf>,
    #[serde(default)]
    pub allow_insecure_http: bool,
}

impl Endpoint {
    pub fn validate(&self) -> Result<()> {
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
        if !self.token_file.is_absolute() {
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
            .timeout(std::time::Duration::from_secs(30))
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventRoute {
    pub id: String,
    pub source: String,
    pub event_type: String,
    pub automation: String,
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
}
fn unconfined() -> String {
    "unconfined".into()
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
    pub fn validate(&self) -> Result<()> {
        validate_key(&self.service_id)?;
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
