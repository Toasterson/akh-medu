//! Static per-akh MCP servers in the akh's own tool registry (Phase 53i,
//! ADR 113 decision 9).
//!
//! At agent init, [`attach_static_mcp_servers`] connects every configured
//! `[[mcp.servers]]` entry over streamable-HTTP MCP (the same
//! [`UpstreamMcpClient`] transport the ACP bridge uses), lists its tools and
//! registers each allowed tool as `mcp__<server>__<tool>` in the akh's
//! persistent [`ToolRegistry`]. Those tools are then visible to every consumer
//! of that registry: the agentic turn, the daemon's OODA tool use and the
//! `/v1` core API.
//!
//! Failure policy: a server that is unreachable at init is **never fatal**.
//! It is logged as a warning and retried:
//!
//! - **lazily on first call**: when the server has a `tools` allowlist, each
//!   allowlisted name is registered straight away as a placeholder. Calling a
//!   placeholder re-lists the server (at most once per
//!   [`McpRetryPolicy::min_lazy_interval`]) and, on success, proxies the call
//!   and replaces every placeholder with the full-schema tool;
//! - **in the background**: a retry thread re-lists with exponential backoff
//!   until the server answers (then registers its tools) or the registry is
//!   dropped (the akh slept or was handed off). This is what brings up a
//!   server without an allowlist, whose tool names are unknown until it
//!   answers.
//!
//! Credentials are resolved on every request (so a rotated token file needs no
//! restart) and never logged; logs name the env var or file only.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use miette::Diagnostic;
use serde_json::Value;
use thiserror::Error;

use super::mcp::{McpToolDescriptor, render_call_result};
use super::mcp_config::{
    MAX_QUALIFIED_TOOL_NAME_LEN, McpAuthError, McpServerSpec, McpServersConfigError,
    validate_mcp_servers,
};
use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolRegistry, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolSource,
};
use crate::engine::Engine;
use crate::upstream_client::{UpstreamError, UpstreamMcpClient};

/// Callback run after tools were registered or replaced *after* init (by a
/// lazy call or the retry thread), so the owner can refresh derived views
/// such as the engine's live-tool list.
pub type RegistryChangedHook = Arc<dyn Fn(&ToolRegistry) + Send + Sync>;

/// Retry behaviour for static MCP servers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpRetryPolicy {
    /// First background retry delay after a failed init.
    pub initial_backoff: Duration,
    /// Cap for the exponential background backoff.
    pub max_backoff: Duration,
    /// Minimum gap between two lazy (call-triggered) discovery attempts, so a
    /// turn calling several tools of a dead server pays the timeout once.
    pub min_lazy_interval: Duration,
    /// Upper bound on the per-request timeout used for discovery. Discovery at
    /// init blocks agent construction, so it is kept tighter than calls.
    pub discovery_timeout_cap: Duration,
    /// Whether to run the background retry thread at all.
    pub background_retry: bool,
}

impl Default for McpRetryPolicy {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_secs(30),
            max_backoff: Duration::from_secs(300),
            min_lazy_interval: Duration::from_secs(5),
            discovery_timeout_cap: Duration::from_secs(10),
            background_retry: true,
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// A static MCP server or one of its tools failed.
#[derive(Debug, Error, Diagnostic)]
pub enum StaticMcpError {
    #[error("MCP server `{server}`: credential unavailable")]
    #[diagnostic(
        code(akh::mcp::static_server::auth),
        help(
            "The server's `auth` source could not be read; see the cause below. No request \
             was sent without the credential."
        )
    )]
    Auth {
        server: String,
        #[source]
        #[diagnostic_source]
        source: McpAuthError,
    },

    #[error("MCP server `{server}` ({url}) is unreachable: {message}")]
    #[diagnostic(
        code(akh::mcp::static_server::unreachable),
        help(
            "The akh keeps running without this server's tools and retries: on the next call \
             (at most every few seconds) and in the background. Check that the server is up, \
             that `url` is the full MCP endpoint, and, for HTTP 401/403, that the token \
             named in `auth` is valid for this endpoint."
        )
    )]
    Unreachable {
        server: String,
        url: String,
        message: String,
    },

    #[error(
        "MCP server `{server}` is still unreachable (last error: {last_error}); next retry in {retry_in_secs}s"
    )]
    #[diagnostic(
        code(akh::mcp::static_server::backoff),
        help(
            "A discovery attempt failed moments ago, so this call was not retried yet. \
             Continue without this tool, or call it again later."
        )
    )]
    Backoff {
        server: String,
        last_error: String,
        retry_in_secs: u64,
    },

    #[error("MCP server `{server}` rejected the request (JSON-RPC {code}): {message}")]
    #[diagnostic(
        code(akh::mcp::static_server::rpc),
        help("Check the tool name and arguments against the tool's schema.")
    )]
    Rpc {
        server: String,
        code: i64,
        message: String,
    },

    #[error("MCP server `{server}` sent a response the akh cannot parse: {message}")]
    #[diagnostic(
        code(akh::mcp::static_server::protocol),
        help(
            "The endpoint may not be a streamable-HTTP MCP server. Check that `url` points \
             at the MCP endpoint (usually ending in `/mcp`)."
        )
    )]
    Protocol { server: String, message: String },

    #[error("MCP tool `{tool}` on `{server}` reported an error: {message}")]
    #[diagnostic(
        code(akh::mcp::static_server::tool_error),
        help("The remote tool ran and failed. Read its message, adjust the arguments, and retry.")
    )]
    ToolFailed {
        server: String,
        tool: String,
        message: String,
    },

    #[error("MCP server `{server}` does not advertise tool `{tool}`")]
    #[diagnostic(
        code(akh::mcp::static_server::tool_missing),
        help(
            "The tool is in this server's `tools` allowlist but the server does not offer it. \
             Remove it from the allowlist or check the server's version."
        )
    )]
    ToolNotAdvertised { server: String, tool: String },
}

impl StaticMcpError {
    fn from_upstream(spec: &McpServerSpec, err: UpstreamError) -> Self {
        let server = spec.name.to_string();
        match err {
            UpstreamError::Request { message } => Self::Unreachable {
                server,
                url: spec.url.clone(),
                message,
            },
            UpstreamError::RpcError { code, message } => Self::Rpc {
                server,
                code,
                message,
            },
            UpstreamError::Parse { message } => Self::Protocol { server, message },
            UpstreamError::NotConfigured { service } => Self::Protocol {
                server,
                message: format!("upstream `{service}` not configured"),
            },
        }
    }
}

/// Render a diagnostic for the LLM: message, code, help and cause chain.
fn render_diagnostic(err: &StaticMcpError) -> String {
    let mut out = err.to_string();
    let mut cause = std::error::Error::source(err);
    while let Some(c) = cause {
        out.push_str(&format!("\ncaused by: {c}"));
        cause = c.source();
    }
    if let Some(code) = err.code() {
        out.push_str(&format!("\ncode: {code}"));
    }
    if let Some(help) = err.help() {
        out.push_str(&format!("\nhelp: {help}"));
    }
    if let Some(help) = err.diagnostic_source().and_then(|inner| inner.help()) {
        out.push_str(&format!("\nhelp: {help}"));
    }
    out
}

// ── Attach report ────────────────────────────────────────────────────────────

/// What happened to one server at init.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerAttachOutcome {
    /// Listed and registered.
    Ready {
        server: String,
        /// Registered akh-side tool names, sorted.
        registered: Vec<String>,
        /// Advertised tools left out by the allowlist.
        filtered_out: usize,
        /// Allowlisted names the server does not advertise.
        missing_allowlisted: Vec<String>,
        /// Advertised tools skipped because their name is not tool-name-safe
        /// or the qualified name is too long.
        skipped_invalid: Vec<String>,
    },
    /// Unreachable at init; retried lazily and in the background.
    Pending {
        server: String,
        /// Placeholder names registered for allowlisted tools, sorted.
        placeholders: Vec<String>,
        /// Rendered init error (never contains secrets).
        error: String,
    },
}

/// Result of [`attach_static_mcp_servers`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct McpAttachReport {
    /// One outcome per attached server, in config order.
    pub servers: Vec<McpServerAttachOutcome>,
    /// Entries skipped because the config failed validation (programmatic
    /// callers only; the akh config loader rejects these earlier).
    pub rejected: Vec<String>,
}

impl McpAttachReport {
    /// Total tools registered (ready tools plus placeholders).
    pub fn registered_tools(&self) -> usize {
        self.servers
            .iter()
            .map(|s| match s {
                McpServerAttachOutcome::Ready { registered, .. } => registered.len(),
                McpServerAttachOutcome::Pending { placeholders, .. } => placeholders.len(),
            })
            .sum()
    }
}

// ── Server runtime ───────────────────────────────────────────────────────────

enum ServerState {
    /// Listed; the allowed descriptors keyed by remote name.
    Ready(HashMap<String, McpToolDescriptor>),
    /// Not yet listed successfully.
    Pending {
        last_error: String,
        last_attempt: Instant,
    },
}

/// Runtime state of one static MCP server, shared by all of its tools.
struct StaticMcpServer {
    spec: McpServerSpec,
    registry: Weak<ToolRegistry>,
    hook: Option<RegistryChangedHook>,
    policy: McpRetryPolicy,
    state: Mutex<ServerState>,
    /// Serialises discovery attempts (init, lazy call, retry thread).
    discovery: Mutex<()>,
}

/// Outcome of publishing a server's listed tools into the registry.
struct Published {
    registered: Vec<String>,
    filtered_out: usize,
    missing_allowlisted: Vec<String>,
    skipped_invalid: Vec<String>,
}

fn is_tool_name_safe(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_QUALIFIED_TOOL_NAME_LEN
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl StaticMcpServer {
    fn name(&self) -> &str {
        self.spec.name.as_str()
    }

    fn client(&self, timeout: Duration) -> Result<UpstreamMcpClient, StaticMcpError> {
        let header = self
            .spec
            .auth
            .resolve()
            .map_err(|source| StaticMcpError::Auth {
                server: self.name().to_string(),
                source,
            })?;
        let client = match header {
            None => UpstreamMcpClient::from_endpoint_with_timeout(&self.spec.url, None, timeout),
            Some(h) if h.is_authorization() => UpstreamMcpClient::from_endpoint_with_timeout(
                &self.spec.url,
                Some(h.expose_value().to_string()),
                timeout,
            ),
            Some(h) => UpstreamMcpClient::from_endpoint_with_timeout(&self.spec.url, None, timeout)
                .with_header(h.name(), h.expose_value()),
        };
        Ok(client)
    }

    /// `initialize` + `tools/list`, parsed into descriptors.
    fn list(&self) -> Result<Vec<McpToolDescriptor>, StaticMcpError> {
        let timeout = self.spec.timeout().min(self.policy.discovery_timeout_cap);
        let raw = self
            .client(timeout)?
            .list_tools_raw()
            .map_err(|e| StaticMcpError::from_upstream(&self.spec, e))?;
        Ok(raw.iter().filter_map(McpToolDescriptor::parse).collect())
    }

    fn descriptor(&self, remote: &str) -> Option<McpToolDescriptor> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            ServerState::Ready(tools) => tools.get(remote).cloned(),
            ServerState::Pending { .. } => None,
        }
    }

    fn is_ready(&self) -> bool {
        matches!(
            *self.state.lock().unwrap_or_else(|e| e.into_inner()),
            ServerState::Ready(_)
        )
    }

    /// Discover the server's tools unless it is already listed.
    ///
    /// `respect_backoff` skips the attempt (returning [`StaticMcpError::Backoff`])
    /// when the last failure was less than `min_lazy_interval` ago. On the
    /// Pending → Ready transition the tools are published into the registry.
    fn ensure_ready(
        self: &Arc<Self>,
        respect_backoff: bool,
    ) -> Result<Option<Published>, StaticMcpError> {
        let _serial = self.discovery.lock().unwrap_or_else(|e| e.into_inner());
        {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            match &*state {
                ServerState::Ready(_) => return Ok(None),
                ServerState::Pending {
                    last_error,
                    last_attempt,
                } if respect_backoff => {
                    let elapsed = last_attempt.elapsed();
                    if elapsed < self.policy.min_lazy_interval {
                        return Err(StaticMcpError::Backoff {
                            server: self.name().to_string(),
                            last_error: last_error.clone(),
                            retry_in_secs: (self.policy.min_lazy_interval - elapsed)
                                .as_secs()
                                .max(1),
                        });
                    }
                }
                ServerState::Pending { .. } => {}
            }
        }
        match self.list() {
            Ok(descriptors) => Ok(Some(self.publish(descriptors))),
            Err(err) => {
                *self.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Pending {
                    last_error: err.to_string(),
                    last_attempt: Instant::now(),
                };
                Err(err)
            }
        }
    }

    /// Mark the server Ready with `descriptors` and register its allowed tools,
    /// replacing any placeholders.
    fn publish(self: &Arc<Self>, descriptors: Vec<McpToolDescriptor>) -> Published {
        let mut ready = HashMap::new();
        let mut filtered_out = 0usize;
        let mut skipped_invalid = Vec::new();
        for d in descriptors {
            if !self.spec.allows(&d.remote_name) {
                filtered_out += 1;
                continue;
            }
            let qualified = self.spec.qualified_tool_name(&d.remote_name);
            if !is_tool_name_safe(&qualified) {
                skipped_invalid.push(d.remote_name.clone());
                continue;
            }
            ready.insert(d.remote_name.clone(), d);
        }
        let missing_allowlisted: Vec<String> = self
            .spec
            .tools
            .iter()
            .flatten()
            .map(|t| t.to_string())
            .filter(|t| !ready.contains_key(t))
            .collect();

        let tools: Vec<StaticMcpTool> = ready
            .values()
            .map(|d| StaticMcpTool::ready(Arc::clone(self), d.clone()))
            .collect();
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Ready(ready);

        let mut registered: Vec<String> = tools.iter().map(|t| t.tool_name.clone()).collect();
        registered.sort();
        if let Some(registry) = self.registry.upgrade() {
            for tool in tools {
                registry.register(Box::new(tool));
            }
            // Placeholders for allowlisted tools the server does not offer
            // would only ever fail; drop them.
            for missing in &missing_allowlisted {
                registry.remove(&self.spec.qualified_tool_name(missing));
            }
        }

        let server = self.name();
        tracing::info!(
            target: "akh::mcp",
            server,
            url = %self.spec.url,
            tools = registered.len(),
            filtered_out,
            "static MCP server tools registered"
        );
        if !missing_allowlisted.is_empty() {
            tracing::warn!(
                target: "akh::mcp",
                server,
                missing = ?missing_allowlisted,
                "allowlisted MCP tools are not advertised by the server"
            );
        }
        if !skipped_invalid.is_empty() {
            tracing::warn!(
                target: "akh::mcp",
                server,
                skipped = ?skipped_invalid,
                "MCP tools skipped: name is not tool-name-safe or `mcp__<server>__<tool>` exceeds 64 characters"
            );
        }

        Published {
            registered,
            filtered_out,
            missing_allowlisted,
            skipped_invalid,
        }
    }

    fn notify(&self) {
        if let (Some(hook), Some(registry)) = (&self.hook, self.registry.upgrade()) {
            hook(&registry);
        }
    }

    /// Proxy one call; `Ok` carries the rendered result text.
    fn call(&self, remote: &str, args: Value) -> Result<String, StaticMcpError> {
        let result = self
            .client(self.spec.timeout())?
            .call_tool_result(remote, args)
            .map_err(|e| StaticMcpError::from_upstream(&self.spec, e))?;
        let text = render_call_result(&result);
        if result["isError"].as_bool().unwrap_or(false) {
            return Err(StaticMcpError::ToolFailed {
                server: self.name().to_string(),
                tool: remote.to_string(),
                message: if text.is_empty() {
                    "remote tool returned isError without content".into()
                } else {
                    text
                },
            });
        }
        Ok(text)
    }

    /// Background retry loop; exits once the server is listed or the registry
    /// (and therefore the akh) is gone.
    fn spawn_retry(self: &Arc<Self>) {
        let server = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name(format!("akh-mcp-retry-{}", self.name()))
            .spawn(move || {
                let mut delay = server.policy.initial_backoff;
                loop {
                    std::thread::sleep(delay);
                    if server.registry.strong_count() == 0 || server.is_ready() {
                        return;
                    }
                    match server.ensure_ready(false) {
                        Ok(published) => {
                            if published.is_some() {
                                server.notify();
                            }
                            return;
                        }
                        Err(err) => {
                            tracing::debug!(
                                target: "akh::mcp",
                                server = server.name(),
                                retry_in_secs = delay.as_secs(),
                                "static MCP server still unreachable: {err}"
                            );
                            delay = (delay * 2).min(server.policy.max_backoff);
                        }
                    }
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(
                target: "akh::mcp",
                server = self.name(),
                "could not start the MCP retry thread ({e}); the server is retried only on call"
            );
        }
    }
}

// ── Tool ─────────────────────────────────────────────────────────────────────

/// One remote tool of a static MCP server, registered as `mcp__<server>__<tool>`.
///
/// A tool without a descriptor is a placeholder for an allowlisted tool whose
/// server was unreachable at init; calling it triggers lazy discovery.
struct StaticMcpTool {
    tool_name: String,
    remote_name: String,
    server: Arc<StaticMcpServer>,
    descriptor: Option<McpToolDescriptor>,
}

impl StaticMcpTool {
    fn ready(server: Arc<StaticMcpServer>, descriptor: McpToolDescriptor) -> Self {
        Self {
            tool_name: server.spec.qualified_tool_name(&descriptor.remote_name),
            remote_name: descriptor.remote_name.clone(),
            server,
            descriptor: Some(descriptor),
        }
    }

    fn placeholder(server: Arc<StaticMcpServer>, remote_name: String) -> Self {
        Self {
            tool_name: server.spec.qualified_tool_name(&remote_name),
            remote_name,
            server,
            descriptor: None,
        }
    }

    fn description(&self) -> String {
        match &self.descriptor {
            Some(d) if !d.description.is_empty() => d.description.clone(),
            Some(_) => format!(
                "Remote tool `{}` on MCP server `{}`.",
                self.remote_name,
                self.server.name()
            ),
            None => format!(
                "Remote tool `{}` on MCP server `{}`. The server was unreachable when the akh \
                 started, so its parameter schema is not known yet; calling the tool retries \
                 the connection.",
                self.remote_name,
                self.server.name()
            ),
        }
    }

    fn danger(&self) -> DangerInfo {
        let read_only = self
            .descriptor
            .as_ref()
            .is_some_and(McpToolDescriptor::is_read_only);
        let server = self.server.name();
        DangerInfo {
            effect: ToolEffect::Mutating,
            // Remote tools are external side effects by default. A server's
            // `readOnlyHint` softens that to Cautious, never to Safe: the
            // network hop and the server's own behaviour stay outside the
            // akh's control.
            level: if read_only {
                DangerLevel::Cautious
            } else {
                DangerLevel::Dangerous
            },
            capabilities: HashSet::from([Capability::Network]),
            description: if read_only {
                format!("Calls the external MCP server `{server}` (declared read-only).")
            } else {
                format!("Calls the external MCP server `{server}`, which may change remote state.")
            },
            shadow_triggers: vec![
                "mcp".into(),
                "remote".into(),
                "external".into(),
                server.to_string(),
            ],
        }
    }
}

impl Tool for StaticMcpTool {
    fn signature(&self) -> ToolSignature {
        match &self.descriptor {
            Some(d) => {
                let mut sig = d.signature(&self.tool_name);
                sig.description = self.description();
                sig
            }
            None => ToolSignature {
                name: self.tool_name.clone(),
                description: self.description(),
                parameters: Vec::new(),
            },
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let descriptor = match self
            .descriptor
            .clone()
            .or_else(|| self.server.descriptor(&self.remote_name))
        {
            Some(d) => d,
            None => {
                match self.server.ensure_ready(true) {
                    Ok(Some(_)) => self.server.notify(),
                    Ok(None) => {}
                    Err(err) => return Ok(ToolOutput::err(render_diagnostic(&err))),
                }
                match self.server.descriptor(&self.remote_name) {
                    Some(d) => d,
                    None => {
                        return Ok(ToolOutput::err(render_diagnostic(
                            &StaticMcpError::ToolNotAdvertised {
                                server: self.server.name().to_string(),
                                tool: self.remote_name.clone(),
                            },
                        )));
                    }
                }
            }
        };

        match self
            .server
            .call(&self.remote_name, descriptor.build_args(&input))
        {
            Ok(text) if text.is_empty() => Ok(ToolOutput::ok(format!(
                "{} returned no content.",
                self.tool_name
            ))),
            Ok(text) => Ok(ToolOutput::ok(text)),
            Err(err) => {
                tracing::debug!(
                    target: "akh::mcp",
                    server = self.server.name(),
                    tool = %self.remote_name,
                    "static MCP tool call failed: {err}"
                );
                Ok(ToolOutput::err(render_diagnostic(&err)))
            }
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: self.tool_name.clone(),
            description: self.description(),
            parameters: self
                .descriptor
                .as_ref()
                .map(McpToolDescriptor::param_schemas)
                .unwrap_or_default(),
            danger: self.danger(),
            source: ToolSource::Mcp {
                server: self.server.name().to_string(),
            },
        }
    }
}

// ── Attach ───────────────────────────────────────────────────────────────────

/// Connect every configured static MCP server and register its tools into
/// `registry`.
///
/// Blocking network I/O: call from a blocking context (agent construction
/// inside `spawn_blocking`). Servers are discovered concurrently, so init is
/// bounded by roughly one discovery timeout regardless of server count.
///
/// Never fails: an invalid list (duplicates etc., normally rejected by the
/// config loader) skips the offending entries, and an unreachable server is
/// reported as [`McpServerAttachOutcome::Pending`] and retried as described in
/// the module docs. `hook` runs after any later (post-init) registry change.
pub fn attach_static_mcp_servers(
    registry: &Arc<ToolRegistry>,
    specs: &[McpServerSpec],
    hook: Option<RegistryChangedHook>,
    policy: McpRetryPolicy,
) -> McpAttachReport {
    let mut report = McpAttachReport::default();
    let specs = accepted_specs(specs, &mut report);

    let servers: Vec<Arc<StaticMcpServer>> = specs
        .into_iter()
        .map(|spec| {
            Arc::new(StaticMcpServer {
                spec,
                registry: Arc::downgrade(registry),
                hook: hook.clone(),
                policy,
                state: Mutex::new(ServerState::Pending {
                    last_error: "not yet discovered".into(),
                    // Never blocks the first lazy attempt: backoff is skipped
                    // at init and the retry thread does not respect it.
                    last_attempt: Instant::now(),
                }),
                discovery: Mutex::new(()),
            })
        })
        .collect();

    let results: Vec<Result<Option<Published>, StaticMcpError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = servers
            .iter()
            .map(|server| scope.spawn(move || server.ensure_ready(false)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(StaticMcpError::Protocol {
                        server: "<unknown>".into(),
                        message: "discovery thread panicked".into(),
                    })
                })
            })
            .collect()
    });

    for (server, result) in servers.iter().zip(results) {
        let name = server.name().to_string();
        match result {
            Ok(Some(published)) => report.servers.push(McpServerAttachOutcome::Ready {
                server: name,
                registered: published.registered,
                filtered_out: published.filtered_out,
                missing_allowlisted: published.missing_allowlisted,
                skipped_invalid: published.skipped_invalid,
            }),
            // Unreachable: a fresh server is never Ready before its first attempt.
            Ok(None) => {}
            Err(err) => {
                let mut placeholders: Vec<String> = Vec::new();
                for remote in server.spec.tools.iter().flatten() {
                    let tool = StaticMcpTool::placeholder(Arc::clone(server), remote.to_string());
                    if !is_tool_name_safe(&tool.tool_name) {
                        continue;
                    }
                    placeholders.push(tool.tool_name.clone());
                    registry.register(Box::new(tool));
                }
                placeholders.sort();
                tracing::warn!(
                    target: "akh::mcp",
                    server = %name,
                    url = %server.spec.url,
                    auth = %server.spec.auth.describe(),
                    placeholders = placeholders.len(),
                    "static MCP server unreachable at init; continuing without it and retrying \
                     lazily: {err}"
                );
                if policy.background_retry {
                    server.spawn_retry();
                }
                report.servers.push(McpServerAttachOutcome::Pending {
                    server: name,
                    placeholders,
                    error: render_diagnostic(&err),
                });
            }
        }
    }
    report
}

/// Keep the specs that pass validation; record the rest in `report.rejected`.
fn accepted_specs(specs: &[McpServerSpec], report: &mut McpAttachReport) -> Vec<McpServerSpec> {
    if let Err(McpServersConfigError { problems }) = validate_mcp_servers(specs) {
        for problem in &problems {
            tracing::error!(target: "akh::mcp", "skipping invalid MCP server config: {problem}");
        }
    }
    let mut seen = HashSet::new();
    let mut accepted = Vec::new();
    for spec in specs {
        let valid_alone = validate_mcp_servers(std::slice::from_ref(spec)).is_ok();
        if valid_alone && seen.insert(spec.name.clone()) {
            accepted.push(spec.clone());
        } else {
            report.rejected.push(spec.name.to_string());
        }
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::mcp_config::{McpRemoteToolName, McpServerAuth, McpServerName};
    use serde_json::json;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A minimal streamable-HTTP MCP stub: answers `initialize` (with a session
    /// id), `notifications/initialized` (202), `tools/list` and `tools/call`
    /// (echoing the arguments; tool `fail` returns `isError`). While `up` is
    /// false every request gets HTTP 503. Records the auth headers it saw.
    struct Stub {
        url: String,
        up: Arc<AtomicBool>,
        seen_headers: Arc<Mutex<Vec<String>>>,
        lists: Arc<AtomicUsize>,
    }

    fn stub(tools: Value, up: bool) -> Stub {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let up = Arc::new(AtomicBool::new(up));
        let seen_headers = Arc::new(Mutex::new(Vec::new()));
        let lists = Arc::new(AtomicUsize::new(0));
        let (up2, seen2, lists2) = (up.clone(), seen_headers.clone(), lists.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut writer = stream;
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                let mut len = 0usize;
                loop {
                    line.clear();
                    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                        break;
                    }
                    let lower = line.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                    if lower.starts_with("authorization:") || lower.starts_with("x-api-key:") {
                        seen2.lock().unwrap().push(line.trim().to_string());
                    }
                }
                let mut body = vec![0u8; len];
                let _ = reader.read_exact(&mut body);
                if !up2.load(Ordering::SeqCst) {
                    let _ = writer.write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
                let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                let resp = match req["method"].as_str() {
                    Some("initialize") => json!({"jsonrpc":"2.0","id":req["id"],"result":{
                        "protocolVersion":"2025-06-18","capabilities":{"tools":{}},
                        "serverInfo":{"name":"stub","version":"0"}}}),
                    Some("tools/list") => {
                        lists2.fetch_add(1, Ordering::SeqCst);
                        json!({"jsonrpc":"2.0","id":req["id"],"result":{"tools": tools}})
                    }
                    Some("tools/call") => {
                        let name = req["params"]["name"].as_str().unwrap_or("");
                        let args = req["params"]["arguments"].to_string();
                        json!({"jsonrpc":"2.0","id":req["id"],"result":{
                            "content":[{"type":"text","text": format!("{name}:{args}")}],
                            "isError": name == "fail"}})
                    }
                    _ => {
                        let _ = writer.write_all(
                            b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        );
                        continue;
                    }
                };
                let text = resp.to_string();
                let _ = writer.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: s1\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{text}",
                        text.len()
                    )
                    .as_bytes(),
                );
            }
        });
        Stub {
            url,
            up,
            seen_headers,
            lists,
        }
    }

    fn stub_tools() -> Value {
        json!([
            {"name": "list_projects", "description": "List projects",
             "inputSchema": {"type": "object", "properties": {
                 "page_size": {"type": "integer"}}},
             "annotations": {"readOnlyHint": true}},
            {"name": "create_item", "description": "Create",
             "inputSchema": {"type": "object", "properties": {
                 "title": {"type": "string"}}, "required": ["title"]}},
            {"name": "fail", "description": "Always fails"},
            {"name": "bad name!", "description": "not tool-name-safe"}
        ])
    }

    fn spec(name: &str, url: &str, tools: Option<&[&str]>) -> McpServerSpec {
        let mut s = McpServerSpec::new(McpServerName::parse(name).unwrap(), url);
        s.tools = tools.map(|t| {
            t.iter()
                .map(|n| McpRemoteToolName::parse(*n).unwrap())
                .collect()
        });
        s.timeout_secs = Some(5);
        s
    }

    fn quiet_policy() -> McpRetryPolicy {
        McpRetryPolicy {
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_millis(200),
            min_lazy_interval: Duration::from_millis(0),
            discovery_timeout_cap: Duration::from_secs(5),
            background_retry: false,
        }
    }

    fn engine() -> Engine {
        Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .expect("engine")
    }

    fn names(registry: &ToolRegistry) -> Vec<String> {
        let mut n: Vec<String> = registry.list().into_iter().map(|s| s.name).collect();
        n.sort();
        n
    }

    #[test]
    fn registers_all_safe_tools_under_the_mcp_namespace() {
        let stub = stub(stub_tools(), true);
        let registry = Arc::new(ToolRegistry::new());
        let report = attach_static_mcp_servers(
            &registry,
            &[spec("anima", &stub.url, None)],
            None,
            quiet_policy(),
        );
        assert_eq!(
            names(&registry),
            vec![
                "mcp__anima__create_item",
                "mcp__anima__fail",
                "mcp__anima__list_projects"
            ]
        );
        match &report.servers[0] {
            McpServerAttachOutcome::Ready {
                skipped_invalid, ..
            } => assert_eq!(skipped_invalid, &vec!["bad name!".to_string()]),
            other => panic!("expected ready: {other:?}"),
        }

        // Classification: readOnlyHint → Cautious, default → Dangerous, never Safe.
        let ro = registry.manifest("mcp__anima__list_projects").unwrap();
        assert_eq!(ro.danger.level, DangerLevel::Cautious);
        assert!(ro.danger.capabilities.contains(&Capability::Network));
        assert!(matches!(ro.source, ToolSource::Mcp { ref server } if server == "anima"));
        let rw = registry.manifest("mcp__anima__create_item").unwrap();
        assert_eq!(rw.danger.level, DangerLevel::Dangerous);
        assert!(
            rw.parameters
                .iter()
                .any(|p| p.name == "title" && p.required)
        );
    }

    #[test]
    fn allowlist_filters_and_reports_missing_names() {
        let stub = stub(stub_tools(), true);
        let registry = Arc::new(ToolRegistry::new());
        let report = attach_static_mcp_servers(
            &registry,
            &[spec(
                "board",
                &stub.url,
                Some(&["list_projects", "not_there"]),
            )],
            None,
            quiet_policy(),
        );
        assert_eq!(names(&registry), vec!["mcp__board__list_projects"]);
        assert_eq!(
            report.servers[0],
            McpServerAttachOutcome::Ready {
                server: "board".into(),
                registered: vec!["mcp__board__list_projects".into()],
                filtered_out: 3,
                missing_allowlisted: vec!["not_there".into()],
                skipped_invalid: vec![],
            }
        );
    }

    #[test]
    fn calls_proxy_with_coerced_args_auth_and_is_error_mapping() {
        let stub = stub(stub_tools(), true);
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "tok-53i\n").unwrap();
        let mut s = spec("anima", &stub.url, None);
        s.auth = McpServerAuth::TokenFile(token);
        let registry = Arc::new(ToolRegistry::new());
        attach_static_mcp_servers(&registry, &[s], None, quiet_policy());

        let engine = engine();
        let out = registry
            .execute(
                "mcp__anima__list_projects",
                ToolInput::new().with_param("page_size", "7"),
                &engine,
            )
            .unwrap();
        assert!(out.success, "{}", out.result);
        assert_eq!(out.result, r#"list_projects:{"page_size":7}"#);
        let seen = stub.seen_headers.lock().unwrap().clone();
        assert!(!seen.is_empty());
        assert!(
            seen.iter()
                .all(|h| h.eq_ignore_ascii_case("authorization: Bearer tok-53i")),
            "{seen:?}"
        );

        let failed = registry
            .execute("mcp__anima__fail", ToolInput::new(), &engine)
            .unwrap();
        assert!(!failed.success);
        assert!(
            failed.result.contains("reported an error"),
            "{}",
            failed.result
        );
        assert!(
            failed
                .result
                .contains("akh::mcp::static_server::tool_error"),
            "{}",
            failed.result
        );
    }

    #[test]
    fn header_env_auth_sends_the_custom_header() {
        const VAR: &str = "_AKH_TEST_MCP53I_HEADER_ENV";
        unsafe { std::env::set_var(VAR, "key-53i") };
        let stub = stub(stub_tools(), true);
        let mut s = spec("grafana", &stub.url, Some(&["list_projects"]));
        s.auth = McpServerAuth::HeaderEnv {
            header: "X-Api-Key".into(),
            env: VAR.into(),
        };
        let registry = Arc::new(ToolRegistry::new());
        attach_static_mcp_servers(&registry, &[s], None, quiet_policy());
        assert_eq!(names(&registry), vec!["mcp__grafana__list_projects"]);
        let seen = stub.seen_headers.lock().unwrap().clone();
        assert!(
            seen.iter()
                .all(|h| h.eq_ignore_ascii_case("x-api-key: key-53i")),
            "{seen:?}"
        );
        unsafe { std::env::remove_var(VAR) };
    }

    #[test]
    fn unreachable_at_init_registers_placeholders_and_connects_lazily() {
        let stub = stub(stub_tools(), false);
        let registry = Arc::new(ToolRegistry::new());
        let hook_calls = Arc::new(AtomicUsize::new(0));
        let hook_calls2 = hook_calls.clone();
        let hook: RegistryChangedHook = Arc::new(move |_| {
            hook_calls2.fetch_add(1, Ordering::SeqCst);
        });
        let report = attach_static_mcp_servers(
            &registry,
            &[spec(
                "anima",
                &stub.url,
                Some(&["list_projects", "create_item"]),
            )],
            Some(hook),
            quiet_policy(),
        );
        assert!(matches!(
            &report.servers[0],
            McpServerAttachOutcome::Pending { placeholders, error, .. }
                if placeholders.len() == 2 && error.contains("unreachable")
        ));
        assert_eq!(
            names(&registry),
            vec!["mcp__anima__create_item", "mcp__anima__list_projects"]
        );
        let placeholder = registry.manifest("mcp__anima__create_item").unwrap();
        assert!(placeholder.parameters.is_empty());
        assert_eq!(placeholder.danger.level, DangerLevel::Dangerous);

        // Still down: the call fails with a clear, coded diagnostic.
        let engine = engine();
        let out = registry
            .execute(
                "mcp__anima__create_item",
                ToolInput::new().with_param("title", "x"),
                &engine,
            )
            .unwrap();
        assert!(!out.success);
        assert!(out.result.contains("MCP server `anima`"), "{}", out.result);
        assert!(
            out.result.contains("akh::mcp::static_server::unreachable"),
            "{}",
            out.result
        );

        // Server comes up: the next call discovers, proxies, and upgrades
        // every placeholder to the full-schema tool.
        stub.up.store(true, Ordering::SeqCst);
        let out = registry
            .execute(
                "mcp__anima__create_item",
                ToolInput::new().with_param("title", "x"),
                &engine,
            )
            .unwrap();
        assert!(out.success, "{}", out.result);
        assert_eq!(out.result, r#"create_item:{"title":"x"}"#);
        let upgraded = registry.manifest("mcp__anima__create_item").unwrap();
        assert!(upgraded.parameters.iter().any(|p| p.name == "title"));
        assert_eq!(
            registry
                .manifest("mcp__anima__list_projects")
                .unwrap()
                .danger
                .level,
            DangerLevel::Cautious
        );
        assert_eq!(hook_calls.load(Ordering::SeqCst), 1);

        // Once ready, calls do not re-list.
        let lists = stub.lists.load(Ordering::SeqCst);
        registry
            .execute("mcp__anima__list_projects", ToolInput::new(), &engine)
            .unwrap();
        assert_eq!(stub.lists.load(Ordering::SeqCst), lists);
    }

    #[test]
    fn lazy_retries_respect_the_backoff_window() {
        let stub = stub(stub_tools(), false);
        let registry = Arc::new(ToolRegistry::new());
        let mut policy = quiet_policy();
        policy.min_lazy_interval = Duration::from_secs(60);
        attach_static_mcp_servers(
            &registry,
            &[spec("anima", &stub.url, Some(&["list_projects"]))],
            None,
            policy,
        );
        let out = registry
            .execute("mcp__anima__list_projects", ToolInput::new(), &engine())
            .unwrap();
        assert!(!out.success);
        assert!(
            out.result.contains("akh::mcp::static_server::backoff"),
            "{}",
            out.result
        );
    }

    #[test]
    fn unreachable_without_allowlist_is_brought_up_by_the_retry_thread() {
        let stub = stub(stub_tools(), false);
        let registry = Arc::new(ToolRegistry::new());
        let mut policy = quiet_policy();
        policy.background_retry = true;
        let report =
            attach_static_mcp_servers(&registry, &[spec("seshat", &stub.url, None)], None, policy);
        assert!(matches!(
            &report.servers[0],
            McpServerAttachOutcome::Pending { placeholders, .. } if placeholders.is_empty()
        ));
        assert!(registry.is_empty());

        stub.up.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(10);
        while registry.is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(
            names(&registry),
            vec![
                "mcp__seshat__create_item",
                "mcp__seshat__fail",
                "mcp__seshat__list_projects"
            ]
        );
    }

    #[test]
    fn missing_credential_is_pending_not_fatal_and_sends_nothing() {
        let stub = stub(stub_tools(), true);
        let mut s = spec("anima", &stub.url, Some(&["list_projects"]));
        s.auth = McpServerAuth::TokenEnv("_AKH_TEST_MCP53I_DEFINITELY_UNSET".into());
        let registry = Arc::new(ToolRegistry::new());
        let report = attach_static_mcp_servers(&registry, &[s], None, quiet_policy());
        match &report.servers[0] {
            McpServerAttachOutcome::Pending { error, .. } => {
                assert!(error.contains("credential unavailable"), "{error}");
                assert!(
                    error.contains("_AKH_TEST_MCP53I_DEFINITELY_UNSET"),
                    "{error}"
                );
            }
            other => panic!("expected pending: {other:?}"),
        }
        assert_eq!(stub.lists.load(Ordering::SeqCst), 0);
        assert_eq!(names(&registry), vec!["mcp__anima__list_projects"]);
    }

    #[test]
    fn duplicate_and_invalid_specs_are_skipped_not_fatal() {
        let stub = stub(stub_tools(), true);
        let registry = Arc::new(ToolRegistry::new());
        let mut bad_url = spec("broken", "ftp://nowhere/mcp", None);
        bad_url.timeout_secs = None;
        let report = attach_static_mcp_servers(
            &registry,
            &[
                spec("anima", &stub.url, Some(&["list_projects"])),
                spec("anima", &stub.url, None),
                bad_url,
            ],
            None,
            quiet_policy(),
        );
        assert_eq!(report.rejected, vec!["anima".to_string(), "broken".into()]);
        assert_eq!(report.servers.len(), 1);
        assert_eq!(names(&registry), vec!["mcp__anima__list_projects"]);
    }
}
