use crate::{SokarError, SokarResult};
use serde::Deserialize;
use sokar_client::{MediaType, StoreName};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::{Path, PathBuf},
};

/// No literal credentials are accepted. Store option values use the same indirection.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "String")]
pub enum SecretRef {
    Env(String),
    File(PathBuf),
}
impl TryFrom<String> for SecretRef {
    type Error = SokarError;
    fn try_from(value: String) -> SokarResult<Self> {
        if let Some(name) = value
            .strip_prefix("env:")
            .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
        {
            return Ok(Self::Env(name.into()));
        }
        if let Some(path) = value
            .strip_prefix("file:")
            .filter(|v| Path::new(v).is_absolute())
        {
            return Ok(Self::File(path.into()));
        }
        Err(SokarError::Config(
            "expected env:NAME or file:/absolute/path (inline values are forbidden)".into(),
        ))
    }
}
/// A store option value: a plain literal for non-secret settings (bucket,
/// endpoint, region, root) or an `env:`/`file:` indirection for credentials.
/// `literal:` escapes a value that happens to start with a reserved prefix.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "String")]
pub enum OptionValue {
    Literal(String),
    Secret(SecretRef),
}
impl TryFrom<String> for OptionValue {
    type Error = SokarError;
    fn try_from(value: String) -> SokarResult<Self> {
        if value.starts_with("env:") || value.starts_with("file:") {
            return SecretRef::try_from(value).map(Self::Secret);
        }
        if let Some(rest) = value.strip_prefix("literal:") {
            return Ok(Self::Literal(rest.into()));
        }
        if value.is_empty() {
            return Err(SokarError::Config(
                "store option values must not be empty".into(),
            ));
        }
        Ok(Self::Literal(value))
    }
}
impl From<SecretRef> for OptionValue {
    fn from(value: SecretRef) -> Self {
        Self::Secret(value)
    }
}
impl OptionValue {
    pub fn resolve(&self) -> SokarResult<String> {
        match self {
            Self::Literal(value) => Ok(value.clone()),
            Self::Secret(secret) => secret.resolve(),
        }
    }
    /// Renders the value back in config syntax (used by test fixtures).
    pub fn to_config_string(&self) -> String {
        match self {
            Self::Literal(value) => value.clone(),
            Self::Secret(SecretRef::Env(name)) => format!("env:{name}"),
            Self::Secret(SecretRef::File(path)) => format!("file:{}", path.display()),
        }
    }
}
impl SecretRef {
    pub fn resolve(&self) -> SokarResult<String> {
        let value = match self {
            Self::Env(name) => std::env::var(name).map_err(|_| {
                SokarError::Config(format!("cannot resolve environment reference {name}"))
            })?,
            Self::File(path) => std::fs::read_to_string(path).map_err(|_| {
                SokarError::Config(format!("cannot read secret reference {}", path.display()))
            })?,
        };
        let value = value.trim_end_matches(['\r', '\n']);
        if value.is_empty() {
            return Err(SokarError::Config("empty secret reference".into()));
        }
        Ok(value.into())
    }
}
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceKind {
    S3,
    Gcs,
    Azblob,
    Fs,
    Memory,
}
impl ServiceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::S3 => "s3",
            Self::Gcs => "gcs",
            Self::Azblob => "azblob",
            Self::Fs => "fs",
            Self::Memory => "memory",
        }
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreConfig {
    pub name: StoreName,
    pub service: ServiceKind,
    pub prefix: Option<String>,
    #[serde(default)]
    pub options: BTreeMap<String, OptionValue>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrincipalConfig {
    pub id: String,
    pub token_file: Option<PathBuf>,
    pub token_sha256_file: Option<PathBuf>,
}
impl PrincipalConfig {
    pub fn credential(&self) -> SokarResult<(String, bool)> {
        let (path, hashed) = match (&self.token_file, &self.token_sha256_file) {
            (Some(path), None) => (path, false),
            (None, Some(path)) => (path, true),
            _ => {
                return Err(SokarError::Config(
                    "principal needs exactly one token_file or token_sha256_file".into(),
                ));
            }
        };
        let value = SecretRef::File(path.clone()).resolve()?.trim().to_owned();
        if value.len() < 24
            || value.chars().any(char::is_whitespace)
            || value.len() > 4096
            || (hashed && (value.len() != 64 || hex::decode(&value).is_err()))
        {
            return Err(SokarError::Config("invalid credential file format".into()));
        }
        Ok((
            if hashed {
                value.to_ascii_lowercase()
            } else {
                value
            },
            hashed,
        ))
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseConfig {
    pub url: SecretRef,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub api_bind: SocketAddr,
    pub admin_bind: SocketAddr,
    pub public_api_url: Option<url::Url>,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_bind: ([127, 0, 0, 1], 9400).into(),
            admin_bind: ([127, 0, 0, 1], 9401).into(),
            public_api_url: None,
        }
    }
}
/// Literal MCP mount path. Disabled MCP configuration may retain an unchecked
/// path; validate it before mounting (including after toggling enabled).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub struct RoutePath(String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
pub enum RoutePathError {
    #[error("mcp.path is not absolute; start it with /, for example /mcp")]
    #[diagnostic(
        code(sokar::config),
        help("Start mcp.path with '/', for example /mcp.")
    )]
    NotAbsolute,
    #[error("mcp.path has a trailing slash; use a non-root mount without a trailing slash")]
    #[diagnostic(
        code(sokar::config),
        help("Use a non-root mount path without a trailing slash, for example /mcp.")
    )]
    TrailingSlash,
    #[error("mcp.path contains an empty or dot segment; remove empty, . and .. segments")]
    #[diagnostic(code(sokar::config), help("Remove empty, '.' and '..' path segments."))]
    DotSegment,
    #[error("mcp.path contains an illegal byte; use only ASCII letters, digits, /, _, - and .")]
    #[diagnostic(
        code(sokar::config),
        help(
            "Use only ASCII letters, digits, '/', '_', '-' and '.'; route parameters and URL escapes are forbidden."
        )
    )]
    IllegalByte,
    #[error("mcp.path overlaps the reserved /v1 API; choose a mount outside /v1, for example /mcp")]
    #[diagnostic(
        code(sokar::config),
        help("Choose a mount outside /v1, for example /mcp or /agents/sokar.")
    )]
    ReservedV1,
}
impl RoutePath {
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn validate(&self) -> Result<(), RoutePathError> {
        let path = self.as_str();
        if !path.starts_with('/') {
            return Err(RoutePathError::NotAbsolute);
        }
        if path.ends_with('/') {
            return Err(RoutePathError::TrailingSlash);
        }
        if path
            .split('/')
            .skip(1)
            .any(|p| p.is_empty() || p == "." || p == "..")
        {
            return Err(RoutePathError::DotSegment);
        }
        if path
            .bytes()
            .any(|b| !b.is_ascii_alphanumeric() && !b"/_-.".contains(&b))
        {
            return Err(RoutePathError::IllegalByte);
        }
        if path == "/v1" || path.starts_with("/v1/") {
            return Err(RoutePathError::ReservedV1);
        }
        Ok(())
    }
}
impl std::str::FromStr for RoutePath {
    type Err = RoutePathError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::try_from(value.to_owned())
    }
}
impl TryFrom<String> for RoutePath {
    type Error = RoutePathError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        let path = Self(value);
        path.validate()?;
        Ok(path)
    }
}
impl std::fmt::Display for RoutePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
/// Bounds apply to decoded bytes for both upload and download tools.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "RawMcpConfig")]
pub struct McpConfig {
    pub enabled: bool,
    pub path: RoutePath,
    pub max_inline_bytes: usize,
    pub session_idle_secs: u64,
}
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawMcpConfig {
    enabled: bool,
    path: String,
    max_inline_bytes: usize,
    session_idle_secs: u64,
}
impl Default for RawMcpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/mcp".into(),
            max_inline_bytes: 8_388_608,
            session_idle_secs: 900,
        }
    }
}
impl TryFrom<RawMcpConfig> for McpConfig {
    type Error = RoutePathError;
    fn try_from(raw: RawMcpConfig) -> Result<Self, Self::Error> {
        let path = if raw.enabled {
            raw.path.parse()?
        } else {
            RoutePath(raw.path)
        };
        Ok(Self {
            enabled: raw.enabled,
            path,
            max_inline_bytes: raw.max_inline_bytes,
            session_idle_secs: raw.session_idle_secs,
        })
    }
}
impl Default for McpConfig {
    fn default() -> Self {
        Self::try_from(RawMcpConfig::default()).expect("valid default MCP path")
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_body_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_body_bytes: 256 * 1024 * 1024,
        }
    }
}
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeliveryConfig {
    pub inline_media_types: Vec<MediaType>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    /// Omit for an explicitly ephemeral local index; startup reindexes the stores.
    pub database: Option<DatabaseConfig>,
    pub default_store: StoreName,
    #[serde(rename = "store")]
    pub stores: Vec<StoreConfig>,
    #[serde(rename = "principal")]
    pub principals: Vec<PrincipalConfig>,
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub delivery: DeliveryConfig,
}
impl Config {
    pub fn from_file(path: &Path) -> SokarResult<Self> {
        let input = std::fs::read_to_string(path)?;
        // TOML error Display includes the source line, which might contain an accidental secret.
        let config: Self =
            toml::from_str(&input).map_err(|e| SokarError::Config(e.message().to_owned()))?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> SokarResult<()> {
        if self.stores.is_empty() || !self.stores.iter().any(|s| s.name == self.default_store) {
            return Err(SokarError::Config(
                "default_store must name a configured store".into(),
            ));
        }
        let names: std::collections::HashSet<_> = self.stores.iter().map(|s| &s.name).collect();
        if names.len() != self.stores.len() {
            return Err(SokarError::Config("duplicate store name".into()));
        }
        let ids: std::collections::HashSet<_> = self.principals.iter().map(|p| &p.id).collect();
        if self.principals.is_empty()
            || ids.len() != self.principals.len()
            || self.principals.iter().any(|p| p.id.is_empty())
        {
            return Err(SokarError::Config(
                "principals must have unique nonempty ids".into(),
            ));
        }
        for principal in &self.principals {
            principal.credential()?;
        }
        if self.mcp.enabled {
            self.mcp.path.validate()?;
            if self.mcp.max_inline_bytes == 0 || self.mcp.session_idle_secs == 0 {
                return Err(SokarError::Config(
                    "mcp.max_inline_bytes and mcp.session_idle_secs must be positive".into(),
                ));
            }
        }
        if let Some(url) = &self.server.public_api_url
            && (!matches!(url.scheme(), "http" | "https")
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some())
        {
            return Err(SokarError::Config(
                "public_api_url requires HTTP(S) with no credentials, query or fragment".into(),
            ));
        }
        if self.limits.max_body_bytes == 0 {
            return Err(SokarError::Config("max_body_bytes must be positive".into()));
        }
        Ok(())
    }
}
