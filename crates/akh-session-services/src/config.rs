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
    #[serde(default)]
    pub allow_insecure_http: bool,
}

impl Endpoint {
    pub fn validate(&self) -> Result<()> {
        let url = url::Url::parse(&self.url).map_err(|_| Fault::invalid("invalid service URL"))?;
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
        Ok(())
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
    pub roles: Vec<String>,
    #[serde(default)]
    pub akhs: Vec<String>,
    #[serde(default)]
    pub sources: Vec<String>,
}

impl Principal {
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
    pub runtime: RuntimeBinding,
    pub instructions: String,
    pub workspace_profiles: Vec<String>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
}

pub use akh_runtime::ProviderConfig as RuntimeProfile;

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
pub struct Config {
    pub service_id: String,
    pub listen: SocketAddr,
    pub data_dir: PathBuf,
    pub principals: Vec<Principal>,
    #[serde(default)]
    pub endpoints: BTreeMap<String, Endpoint>,
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
    pub allowed_origins: Vec<String>,
    pub workspace_root: Option<PathBuf>,
    #[serde(default = "unconfined")]
    pub confinement: String,
}
fn unconfined() -> String {
    "unconfined".into()
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
        for (name, profile) in &self.akhs {
            validate_key(name)?;
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
        Ok(())
    }
    pub fn endpoint(&self, id: &str) -> Result<&Endpoint> {
        self.endpoints
            .get(id)
            .ok_or_else(|| Fault::invalid(format!("service endpoint {id} is not configured")))
    }
}
