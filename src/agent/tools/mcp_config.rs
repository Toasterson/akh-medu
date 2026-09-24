//! Static per-akh MCP server configuration (Phase 53i, ADR 113 decision 9).
//!
//! A full akh reaches external MCP tools (Anima, Agent Board, Sokar, Seshat,
//! local services) from its own tool registry. The servers are declared in the
//! akh's workspace TOML as an array of tables under `[mcp]`:
//!
//! ```toml
//! [mcp]
//! require_pushed_mcp = false
//!
//! [[mcp.servers]]
//! name = "anima"
//! url = "https://anima.wegmueller.it/mcp"
//! auth = { token_env = "ROMAN_ANIMA_TOKEN" }
//!
//! [[mcp.servers]]
//! name = "observer"
//! url = "http://127.0.0.1:18436/mcp"
//! tools = ["monitoring_snapshot"]
//! timeout_secs = 60
//! auth = { token_file = "/path/to/observer.key" }
//! ```
//!
//! Every tool a server advertises (or only the `tools` allowlist) is
//! registered as `mcp__<name>__<tool>`. Secrets are never stored in the
//! config: `auth` names an environment variable or a file, resolved at call
//! time so a rotated token file is picked up without a restart.
//!
//! Field-level invariants (tool-name-safe server and tool names) are enforced
//! while deserializing through the [`McpServerName`] / [`McpRemoteToolName`]
//! newtypes; cross-field and cross-entry invariants (duplicates, URL scheme,
//! timeout range, auth fields) by [`validate_mcp_servers`], which the akh
//! config loader runs.

use std::path::PathBuf;
use std::time::Duration;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Longest accepted server name. Keeps `mcp__<server>__<tool>` inside the
/// 64-character tool-name limit of OpenAI-compatible backends for typical
/// remote tool names.
pub const MAX_SERVER_NAME_LEN: usize = 32;

/// Longest accepted fully-qualified tool name (`mcp__<server>__<tool>`), the
/// strictest limit among the chat backends an akh may use.
pub const MAX_QUALIFIED_TOOL_NAME_LEN: usize = 64;

/// Default per-request timeout for a static MCP server.
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Largest accepted `timeout_secs`.
pub const MAX_TIMEOUT_SECS: u64 = 600;

// ── Errors ───────────────────────────────────────────────────────────────────

/// One problem found in the `[[mcp.servers]]` configuration.
#[derive(Debug, Clone, Error, Diagnostic, PartialEq, Eq)]
pub enum McpServerConfigProblem {
    #[error("invalid MCP server name `{name}`: {reason}")]
    #[diagnostic(
        code(akh::mcp::config::server_name),
        help(
            "Server names become the middle segment of `mcp__<server>__<tool>`. Use 1-32 \
             characters from [A-Za-z0-9_-], starting with a letter, without `__` and without \
             a trailing `_` or `-` (for example `anima`, `board`, `vesta-ops`)."
        )
    )]
    InvalidServerName { name: String, reason: &'static str },

    #[error("invalid MCP tool name `{tool}` in an allowlist: {reason}")]
    #[diagnostic(
        code(akh::mcp::config::tool_name),
        help(
            "List the tool name as the remote server advertises it (for example \
             `list_projects`), not the akh-side `mcp__<server>__<tool>` name. Tool names use \
             [A-Za-z0-9_-]."
        )
    )]
    InvalidToolName { tool: String, reason: &'static str },

    #[error("MCP server `{name}` is declared more than once")]
    #[diagnostic(
        code(akh::mcp::config::duplicate_server),
        help(
            "Each `[[mcp.servers]]` entry needs a unique `name`, because the name namespaces \
             its tools. Either remove the duplicate entry or rename one of them (for example \
             `board` and `board-ops`)."
        )
    )]
    DuplicateServer { name: String },

    #[error("MCP server `{server}` has an invalid url `{url}`: {reason}")]
    #[diagnostic(
        code(akh::mcp::config::url),
        help(
            "Use the full streamable-HTTP MCP endpoint including its path, for example \
             `https://anima.wegmueller.it/mcp` or `http://127.0.0.1:18436/mcp`. The url is \
             used verbatim; nothing is appended."
        )
    )]
    InvalidUrl {
        server: String,
        url: String,
        reason: &'static str,
    },

    #[error("MCP server `{server}` lists tool `{tool}` more than once in `tools`")]
    #[diagnostic(
        code(akh::mcp::config::duplicate_tool),
        help("Remove the repeated entry from the `tools` allowlist.")
    )]
    DuplicateAllowlistEntry { server: String, tool: String },

    #[error("MCP server `{server}` has an empty `tools` allowlist")]
    #[diagnostic(
        code(akh::mcp::config::empty_allowlist),
        help(
            "An empty allowlist would register no tools at all. Remove `tools` to expose \
             every tool the server advertises, list the tools to expose, or remove the \
             server entry."
        )
    )]
    EmptyAllowlist { server: String },

    #[error("MCP server `{server}` has timeout_secs = {secs}, outside 1..={max}")]
    #[diagnostic(
        code(akh::mcp::config::timeout),
        help("Set `timeout_secs` between 1 and 600, or omit it to use the 30 second default.")
    )]
    TimeoutOutOfRange { server: String, secs: u64, max: u64 },

    #[error("MCP server `{server}` auth names an invalid environment variable `{var}`")]
    #[diagnostic(
        code(akh::mcp::config::env_var),
        help(
            "Name the variable, not its value: `auth = {{ token_env = \"ROMAN_ANIMA_TOKEN\" }}`. \
             Variable names are non-empty and contain only [A-Za-z0-9_]."
        )
    )]
    InvalidEnvVarName { server: String, var: String },

    #[error("MCP server `{server}` auth names an invalid HTTP header `{header}`")]
    #[diagnostic(
        code(akh::mcp::config::header),
        help(
            "Use a plain header name such as `X-Api-Key` or `Authorization` \
             (letters, digits and `-`)."
        )
    )]
    InvalidHeaderName { server: String, header: String },

    #[error("MCP server `{server}` auth has an empty token_file path")]
    #[diagnostic(
        code(akh::mcp::config::token_file),
        help("Point `token_file` at the file holding the bearer token, or use `token_env`.")
    )]
    EmptyTokenFile { server: String },
}

/// The `[[mcp.servers]]` configuration failed validation.
///
/// Carries every problem found, not just the first, so an operator can fix
/// the whole section in one edit.
#[derive(Debug, Clone, Error, Diagnostic, PartialEq, Eq)]
#[error("invalid [[mcp.servers]] configuration: {} problem(s)", problems.len())]
#[diagnostic(
    code(akh::mcp::config::invalid),
    help(
        "Fix the listed entries in the akh's workspace TOML. Until then this akh's config does \
         not load; to start the akh without its MCP servers, remove the `[[mcp.servers]]` \
         entries."
    )
)]
pub struct McpServersConfigError {
    /// Every problem found.
    #[related]
    pub problems: Vec<McpServerConfigProblem>,
}

/// A static MCP server's credential could not be resolved at call time.
///
/// Messages name the environment variable or file, never the secret value.
#[derive(Debug, Error, Diagnostic)]
pub enum McpAuthError {
    #[error("environment variable `{var}` is not set (or is empty)")]
    #[diagnostic(
        code(akh::mcp::auth::env_unset),
        help(
            "Export `{var}` in the akhomed service environment (for example through the \
             daemon's credential wrapper), then retry; the variable is read on every call. \
             Alternatively switch this server to `token_file`."
        )
    )]
    EnvUnset { var: String },

    #[error("cannot read token file `{path}`")]
    #[diagnostic(
        code(akh::mcp::auth::file_unreadable),
        help(
            "Check that the file exists and is readable by the akhomed user. The file is \
             re-read on every call, so fixing it takes effect without a restart."
        )
    )]
    FileUnreadable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("token file `{path}` is empty")]
    #[diagnostic(
        code(akh::mcp::auth::file_empty),
        help("Write the bearer token into the file (surrounding whitespace is trimmed).")
    )]
    FileEmpty { path: PathBuf },

    #[error("the credential from {source_name} contains control characters")]
    #[diagnostic(
        code(akh::mcp::auth::invalid_value),
        help(
            "HTTP header values cannot contain newlines or other control characters. \
             Re-issue or re-write the credential without them."
        )
    )]
    InvalidValue { source_name: String },
}

// ── Newtypes ─────────────────────────────────────────────────────────────────

fn is_tool_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// A validated static MCP server name: the `<server>` in `mcp__<server>__<tool>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct McpServerName(String);

impl McpServerName {
    /// Validate a server name.
    pub fn parse(name: impl Into<String>) -> Result<Self, McpServerConfigProblem> {
        let name = name.into();
        let reason = if name.is_empty() {
            Some("it is empty")
        } else if name.len() > MAX_SERVER_NAME_LEN {
            Some("it is longer than 32 characters")
        } else if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
            Some("it must start with an ASCII letter")
        } else if !name.chars().all(is_tool_name_char) {
            Some("it contains characters outside [A-Za-z0-9_-]")
        } else if name.contains("__") {
            Some("`__` is the namespace separator in `mcp__<server>__<tool>`")
        } else if name.ends_with(['_', '-']) {
            Some("it ends with `_` or `-`")
        } else {
            None
        };
        match reason {
            Some(reason) => Err(McpServerConfigProblem::InvalidServerName { name, reason }),
            None => Ok(Self(name)),
        }
    }

    /// The name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for McpServerName {
    type Error = McpServerConfigProblem;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<McpServerName> for String {
    fn from(value: McpServerName) -> Self {
        value.0
    }
}

impl std::fmt::Display for McpServerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A validated remote tool name as it appears in a `tools` allowlist.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct McpRemoteToolName(String);

impl McpRemoteToolName {
    /// Validate a remote tool name.
    pub fn parse(tool: impl Into<String>) -> Result<Self, McpServerConfigProblem> {
        let tool = tool.into();
        let reason = if tool.is_empty() {
            Some("it is empty")
        } else if tool.starts_with("mcp__") {
            Some("it is an akh-side `mcp__<server>__<tool>` name, not the remote name")
        } else if !tool.chars().all(is_tool_name_char) {
            Some("it contains characters outside [A-Za-z0-9_-]")
        } else {
            None
        };
        match reason {
            Some(reason) => Err(McpServerConfigProblem::InvalidToolName { tool, reason }),
            None => Ok(Self(tool)),
        }
    }

    /// The name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for McpRemoteToolName {
    type Error = McpServerConfigProblem;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<McpRemoteToolName> for String {
    fn from(value: McpRemoteToolName) -> Self {
        value.0
    }
}

impl std::fmt::Display for McpRemoteToolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Auth ─────────────────────────────────────────────────────────────────────

/// How the akh authenticates to a static MCP server.
///
/// The config names where the secret lives, never the secret itself:
///
/// - omitted or `auth = "none"`: no credential.
/// - `auth = { token_env = "VAR" }`: `Authorization: Bearer $VAR`.
/// - `auth = { token_file = "/path" }`: `Authorization: Bearer <file contents, trimmed>`.
/// - `auth = { header_env = { header = "X-Api-Key", env = "VAR" } }`: `X-Api-Key: $VAR`
///   (the value is sent verbatim, so for `header = "Authorization"` include the scheme).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpServerAuth {
    /// No credential.
    #[default]
    None,
    /// Bearer token read from an environment variable.
    TokenEnv(String),
    /// Bearer token read from a file.
    TokenFile(PathBuf),
    /// Arbitrary header whose value is read from an environment variable.
    HeaderEnv {
        /// Header name, e.g. `X-Api-Key`.
        header: String,
        /// Environment variable holding the header value.
        env: String,
    },
}

impl McpServerAuth {
    /// Whether no credential is configured.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Resolve the credential into the header to send, reading the environment
    /// or file now. `Ok(None)` for [`McpServerAuth::None`].
    pub fn resolve(&self) -> Result<Option<ResolvedAuthHeader>, McpAuthError> {
        let (header, value, source_name) = match self {
            Self::None => return Ok(None),
            Self::TokenEnv(var) => (
                "Authorization".to_string(),
                format!("Bearer {}", read_env(var)?),
                format!("environment variable `{var}`"),
            ),
            Self::TokenFile(path) => {
                let raw = std::fs::read_to_string(path).map_err(|source| {
                    McpAuthError::FileUnreadable {
                        path: path.clone(),
                        source,
                    }
                })?;
                let token = raw.trim();
                if token.is_empty() {
                    return Err(McpAuthError::FileEmpty { path: path.clone() });
                }
                (
                    "Authorization".to_string(),
                    format!("Bearer {token}"),
                    format!("token file `{}`", path.display()),
                )
            }
            Self::HeaderEnv { header, env } => (
                header.clone(),
                read_env(env)?,
                format!("environment variable `{env}`"),
            ),
        };
        if value.chars().any(char::is_control) {
            return Err(McpAuthError::InvalidValue { source_name });
        }
        Ok(Some(ResolvedAuthHeader {
            name: header,
            value,
        }))
    }

    /// A description safe to log: names the variable or file, never a value.
    pub fn describe(&self) -> String {
        match self {
            Self::None => "none".into(),
            Self::TokenEnv(var) => format!("bearer token from env `{var}`"),
            Self::TokenFile(path) => format!("bearer token from file `{}`", path.display()),
            Self::HeaderEnv { header, env } => format!("header `{header}` from env `{env}`"),
        }
    }
}

fn read_env(var: &str) -> Result<String, McpAuthError> {
    match std::env::var(var) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        _ => Err(McpAuthError::EnvUnset {
            var: var.to_string(),
        }),
    }
}

/// A resolved credential header. `Debug` redacts the value.
#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedAuthHeader {
    name: String,
    value: String,
}

impl ResolvedAuthHeader {
    /// Header name (e.g. `Authorization`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Header value. A secret: never log it.
    pub fn expose_value(&self) -> &str {
        &self.value
    }

    /// Whether this is the `Authorization` header.
    pub fn is_authorization(&self) -> bool {
        self.name.eq_ignore_ascii_case("authorization")
    }
}

impl std::fmt::Debug for ResolvedAuthHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedAuthHeader")
            .field("name", &self.name)
            .field("value", &"<redacted>")
            .finish()
    }
}

// ── Server spec ──────────────────────────────────────────────────────────────

/// One `[[mcp.servers]]` entry.
///
/// Unknown keys are rejected (`deny_unknown_fields`): a misspelt `auth` would
/// otherwise silently drop the credential and surface later as an
/// unexplained 401.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerSpec {
    /// Namespace for this server's tools: `mcp__<name>__<tool>`.
    pub name: McpServerName,
    /// Full streamable-HTTP MCP endpoint URL, used verbatim.
    pub url: String,
    /// Optional allowlist of remote tool names to register. `None` registers
    /// every tool the server advertises.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<McpRemoteToolName>>,
    /// Per-request timeout in seconds (default 30, at most 600).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    /// Credential source. Declared last so it serializes after the scalars.
    #[serde(default, skip_serializing_if = "McpServerAuth::is_none")]
    pub auth: McpServerAuth,
}

impl McpServerSpec {
    /// A spec with no allowlist, default timeout and no auth.
    pub fn new(name: McpServerName, url: impl Into<String>) -> Self {
        Self {
            name,
            url: url.into(),
            tools: None,
            timeout_secs: None,
            auth: McpServerAuth::None,
        }
    }

    /// The effective per-request timeout.
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS))
    }

    /// Whether `remote` passes this server's allowlist.
    pub fn allows(&self, remote: &str) -> bool {
        self.tools
            .as_ref()
            .is_none_or(|allow| allow.iter().any(|t| t.as_str() == remote))
    }

    /// The akh-side name for a remote tool: `mcp__<server>__<remote>`.
    pub fn qualified_tool_name(&self, remote: &str) -> String {
        format!("mcp__{}__{remote}", self.name)
    }

    fn problems(&self, out: &mut Vec<McpServerConfigProblem>) {
        let server = self.name.to_string();

        let url_reason = if self.url.chars().any(char::is_whitespace) {
            Some("it contains whitespace")
        } else {
            match self
                .url
                .strip_prefix("https://")
                .or_else(|| self.url.strip_prefix("http://"))
            {
                None => Some("the scheme must be http:// or https://"),
                Some(rest) if rest.is_empty() || rest.starts_with('/') => Some("it has no host"),
                Some(_) => None,
            }
        };
        if let Some(reason) = url_reason {
            out.push(McpServerConfigProblem::InvalidUrl {
                server: server.clone(),
                url: self.url.clone(),
                reason,
            });
        }

        if let Some(tools) = &self.tools {
            if tools.is_empty() {
                out.push(McpServerConfigProblem::EmptyAllowlist {
                    server: server.clone(),
                });
            }
            let mut seen = std::collections::HashSet::new();
            for tool in tools {
                if !seen.insert(tool.as_str()) {
                    out.push(McpServerConfigProblem::DuplicateAllowlistEntry {
                        server: server.clone(),
                        tool: tool.to_string(),
                    });
                }
            }
        }

        if let Some(secs) = self.timeout_secs
            && !(1..=MAX_TIMEOUT_SECS).contains(&secs)
        {
            out.push(McpServerConfigProblem::TimeoutOutOfRange {
                server: server.clone(),
                secs,
                max: MAX_TIMEOUT_SECS,
            });
        }

        let valid_env = |var: &str| {
            !var.is_empty() && var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        };
        match &self.auth {
            McpServerAuth::None => {}
            McpServerAuth::TokenEnv(var) => {
                if !valid_env(var) {
                    out.push(McpServerConfigProblem::InvalidEnvVarName {
                        server,
                        var: var.clone(),
                    });
                }
            }
            McpServerAuth::TokenFile(path) => {
                if path.as_os_str().is_empty() {
                    out.push(McpServerConfigProblem::EmptyTokenFile { server });
                }
            }
            McpServerAuth::HeaderEnv { header, env } => {
                if header.is_empty()
                    || !header
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-')
                {
                    out.push(McpServerConfigProblem::InvalidHeaderName {
                        server: server.clone(),
                        header: header.clone(),
                    });
                }
                if !valid_env(env) {
                    out.push(McpServerConfigProblem::InvalidEnvVarName {
                        server,
                        var: env.clone(),
                    });
                }
            }
        }
    }
}

/// Validate a whole `[[mcp.servers]]` list: per-entry invariants plus unique
/// server names. Returns every problem found.
pub fn validate_mcp_servers(servers: &[McpServerSpec]) -> Result<(), McpServersConfigError> {
    let mut problems = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for spec in servers {
        if !seen.insert(spec.name.as_str()) {
            problems.push(McpServerConfigProblem::DuplicateServer {
                name: spec.name.to_string(),
            });
        }
        spec.problems(&mut problems);
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(McpServersConfigError { problems })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct Doc {
        #[serde(default)]
        servers: Vec<McpServerSpec>,
    }

    fn parse(toml_src: &str) -> Result<Vec<McpServerSpec>, toml::de::Error> {
        toml::from_str::<Doc>(toml_src).map(|d| d.servers)
    }

    #[test]
    fn parses_every_auth_variant() {
        let servers = parse(
            r#"
            [[servers]]
            name = "plain"
            url = "http://127.0.0.1:1/mcp"

            [[servers]]
            name = "explicit-none"
            url = "http://127.0.0.1:1/mcp"
            auth = "none"

            [[servers]]
            name = "anima"
            url = "https://anima.example/mcp"
            auth = { token_env = "ROMAN_ANIMA_TOKEN" }

            [[servers]]
            name = "observer"
            url = "http://127.0.0.1:18436/mcp"
            tools = ["monitoring_snapshot"]
            timeout_secs = 60
            auth = { token_file = "/etc/akh/observer.key" }

            [[servers]]
            name = "grafana"
            url = "https://grafana.example/mcp"
            auth = { header_env = { header = "X-Api-Key", env = "GRAFANA_KEY" } }
            "#,
        )
        .expect("parse");
        assert_eq!(servers.len(), 5);
        assert_eq!(servers[0].auth, McpServerAuth::None);
        assert_eq!(servers[1].auth, McpServerAuth::None);
        assert_eq!(
            servers[2].auth,
            McpServerAuth::TokenEnv("ROMAN_ANIMA_TOKEN".into())
        );
        assert_eq!(
            servers[3].auth,
            McpServerAuth::TokenFile("/etc/akh/observer.key".into())
        );
        assert_eq!(servers[3].timeout(), Duration::from_secs(60));
        assert!(servers[3].allows("monitoring_snapshot"));
        assert!(!servers[3].allows("other"));
        assert!(servers[2].allows("anything"));
        assert_eq!(
            servers[4].auth,
            McpServerAuth::HeaderEnv {
                header: "X-Api-Key".into(),
                env: "GRAFANA_KEY".into()
            }
        );
        validate_mcp_servers(&servers).expect("valid");
    }

    #[test]
    fn unknown_key_in_server_entry_is_rejected() {
        let err = parse(
            r#"
            [[servers]]
            name = "anima"
            url = "https://anima.example/mcp"
            token_env = "ROMAN_ANIMA_TOKEN"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("token_env"), "{err}");
    }

    #[test]
    fn unknown_auth_variant_is_rejected() {
        assert!(
            parse(
                r#"
                [[servers]]
                name = "anima"
                url = "https://anima.example/mcp"
                auth = { token = "literal-secret" }
                "#,
            )
            .is_err()
        );
    }

    #[test]
    fn bad_server_names_are_rejected_at_parse() {
        for bad in [
            "",
            "an ima",
            "a__b",
            "9lives",
            "board_",
            "x-",
            "mcp.x",
            "ab/c",
            "abcdefghijklmnopqrstuvwxyzabcdefg",
        ] {
            assert!(
                McpServerName::parse(bad).is_err(),
                "`{bad}` should be rejected"
            );
            let src = format!("[[servers]]\nname = {bad:?}\nurl = \"http://h/mcp\"\n");
            let err = parse(&src).unwrap_err();
            assert!(
                err.to_string().contains("invalid MCP server name"),
                "{bad}: {err}"
            );
        }
        for good in ["anima", "board", "vesta-ops", "laya_minerva", "a1"] {
            McpServerName::parse(good).expect(good);
        }
    }

    #[test]
    fn bad_allowlist_entries_are_rejected() {
        for bad in ["", "mcp__anima__list", "has space", "dot.ted"] {
            assert!(McpRemoteToolName::parse(bad).is_err(), "`{bad}`");
        }
        McpRemoteToolName::parse("grafana__query_prometheus").expect("double underscore ok");
        let err = parse(
            r#"
            [[servers]]
            name = "anima"
            url = "https://anima.example/mcp"
            tools = ["mcp__anima__list_projects"]
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("invalid MCP tool name"), "{err}");
    }

    #[test]
    fn duplicates_and_cross_field_problems_are_all_reported() {
        let servers = parse(
            r#"
            [[servers]]
            name = "anima"
            url = "https://anima.example/mcp"

            [[servers]]
            name = "anima"
            url = "ftp://anima.example/mcp"
            tools = ["a", "a"]
            timeout_secs = 0
            auth = { header_env = { header = "Bad Header", env = "NOT-VALID" } }

            [[servers]]
            name = "empty"
            url = "http:///mcp"
            tools = []
            auth = { token_file = "" }
            "#,
        )
        .expect("parse");
        let err = validate_mcp_servers(&servers).unwrap_err();
        let p = &err.problems;
        assert!(p.contains(&McpServerConfigProblem::DuplicateServer {
            name: "anima".into()
        }));
        assert!(p.iter().any(|x| matches!(
            x,
            McpServerConfigProblem::InvalidUrl { server, .. } if server == "anima"
        )));
        assert!(
            p.contains(&McpServerConfigProblem::DuplicateAllowlistEntry {
                server: "anima".into(),
                tool: "a".into()
            })
        );
        assert!(
            p.iter()
                .any(|x| matches!(x, McpServerConfigProblem::TimeoutOutOfRange { secs: 0, .. }))
        );
        assert!(
            p.iter()
                .any(|x| matches!(x, McpServerConfigProblem::InvalidHeaderName { .. }))
        );
        assert!(p.iter().any(|x| matches!(
            x,
            McpServerConfigProblem::InvalidEnvVarName { var, .. } if var == "NOT-VALID"
        )));
        assert!(p.contains(&McpServerConfigProblem::EmptyAllowlist {
            server: "empty".into()
        }));
        assert!(p.contains(&McpServerConfigProblem::EmptyTokenFile {
            server: "empty".into()
        }));
        assert!(p.iter().any(|x| matches!(
            x,
            McpServerConfigProblem::InvalidUrl { server, reason, .. }
                if server == "empty" && reason.contains("no host")
        )));
        // Every problem is a diagnostic with a code.
        use miette::Diagnostic;
        assert!(err.code().is_some());
        assert!(p.iter().all(|x| x.code().is_some() && x.help().is_some()));
    }

    #[test]
    fn serialize_round_trips() {
        let servers = parse(
            r#"
            [[servers]]
            name = "grafana"
            url = "https://grafana.example/mcp"
            tools = ["query"]
            timeout_secs = 5
            auth = { header_env = { header = "X-Api-Key", env = "GRAFANA_KEY" } }

            [[servers]]
            name = "plain"
            url = "http://127.0.0.1:1/mcp"
            "#,
        )
        .expect("parse");
        #[derive(Serialize)]
        struct Out<'a> {
            servers: &'a [McpServerSpec],
        }
        let text = toml::to_string_pretty(&Out { servers: &servers }).expect("serialize");
        assert_eq!(parse(&text).expect("reparse"), servers, "{text}");
    }

    #[test]
    fn token_env_resolves_to_bearer_and_debug_redacts() {
        // Unique name: no other test touches it.
        const VAR: &str = "_AKH_TEST_MCP53I_TOKEN_ENV";
        unsafe { std::env::set_var(VAR, "  s3cret-value \n") };
        let header = McpServerAuth::TokenEnv(VAR.into())
            .resolve()
            .expect("resolve")
            .expect("some");
        assert!(header.is_authorization());
        assert_eq!(header.expose_value(), "Bearer s3cret-value");
        assert!(!format!("{header:?}").contains("s3cret"));
        unsafe { std::env::remove_var(VAR) };
    }

    #[test]
    fn missing_env_names_the_variable_only() {
        let err = McpServerAuth::TokenEnv("_AKH_TEST_MCP53I_DEFINITELY_UNSET".into())
            .resolve()
            .unwrap_err();
        assert!(matches!(err, McpAuthError::EnvUnset { .. }));
        assert!(
            err.to_string()
                .contains("_AKH_TEST_MCP53I_DEFINITELY_UNSET")
        );
        let err = McpServerAuth::HeaderEnv {
            header: "X-Api-Key".into(),
            env: "_AKH_TEST_MCP53I_DEFINITELY_UNSET".into(),
        }
        .resolve()
        .unwrap_err();
        assert!(matches!(err, McpAuthError::EnvUnset { .. }));
    }

    #[test]
    fn token_file_is_trimmed_and_reread() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tok");
        std::fs::write(&path, "first\n").unwrap();
        let auth = McpServerAuth::TokenFile(path.clone());
        assert_eq!(
            auth.resolve().unwrap().unwrap().expose_value(),
            "Bearer first"
        );
        std::fs::write(&path, "rotated").unwrap();
        assert_eq!(
            auth.resolve().unwrap().unwrap().expose_value(),
            "Bearer rotated"
        );
        std::fs::write(&path, "  \n").unwrap();
        assert!(matches!(
            auth.resolve().unwrap_err(),
            McpAuthError::FileEmpty { .. }
        ));
        assert!(matches!(
            McpServerAuth::TokenFile(dir.path().join("missing"))
                .resolve()
                .unwrap_err(),
            McpAuthError::FileUnreadable { .. }
        ));
        assert!(McpServerAuth::None.resolve().unwrap().is_none());
    }

    #[test]
    fn describe_never_contains_secret_values() {
        const VAR: &str = "_AKH_TEST_MCP53I_DESCRIBE";
        unsafe { std::env::set_var(VAR, "hidden-value") };
        let auth = McpServerAuth::HeaderEnv {
            header: "X-Api-Key".into(),
            env: VAR.into(),
        };
        assert!(!auth.describe().contains("hidden-value"));
        assert!(auth.describe().contains(VAR));
        unsafe { std::env::remove_var(VAR) };
    }
}
