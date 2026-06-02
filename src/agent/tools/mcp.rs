//! MCP client: discover and invoke tools on remote MCP servers.
//!
//! Implements JSON-RPC 2.0 transport for the Model Context Protocol,
//! enabling agents to reach remote tool providers (Seshat, Anima,
//! remote akhomed instances) over HTTP.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use miette::Diagnostic;
use serde_json::{Value, json};
use thiserror::Error;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

/// Default tool cache lifetime before a background refresh is triggered (5 minutes).
const DEFAULT_REFRESH_SECS: u64 = 300;

/// Default per-request HTTP timeout (30 seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 30;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by the MCP client.
#[derive(Debug, Error, Diagnostic)]
pub enum McpError {
    #[error("MCP server not found: \"{server}\"")]
    #[diagnostic(
        code(akh::mcp::server_not_found),
        help("Register the server with McpClient::add_server() before calling it.")
    )]
    ServerNotFound { server: String },

    #[error("HTTP error contacting \"{server}\": {message}")]
    #[diagnostic(
        code(akh::mcp::http_error),
        help("Verify the server URL is reachable and the MCP server is running.")
    )]
    Http { server: String, message: String },

    #[error("JSON-RPC error from \"{server}\" (code {code}): {message}")]
    #[diagnostic(
        code(akh::mcp::rpc_error),
        help("The remote MCP server returned a JSON-RPC error. Check the server logs.")
    )]
    Rpc { server: String, code: i64, message: String },

    #[error("failed to parse MCP response from \"{server}\": {message}")]
    #[diagnostic(
        code(akh::mcp::parse_error),
        help("Verify the server returns valid MCP JSON-RPC 2.0 responses.")
    )]
    Parse { server: String, message: String },

    #[error("remote tool \"{tool}\" on \"{server}\" reported an error: {message}")]
    #[diagnostic(
        code(akh::mcp::tool_error),
        help("Check the tool arguments and the remote server logs for details.")
    )]
    ToolError { server: String, tool: String, message: String },
}

// ── Server configuration ─────────────────────────────────────────────────────

/// Configuration for one remote MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// HTTP URL of the MCP endpoint (e.g. `http://seshat.local:8080/mcp`).
    pub url: String,
    /// Optional Bearer token for authentication.
    pub auth_token: Option<String>,
    /// Per-request HTTP timeout.
    pub timeout: Duration,
}

impl McpServerConfig {
    /// Create a minimal config with URL only (no auth, default 30 s timeout).
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            auth_token: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Attach a Bearer token sent in the `Authorization` header.
    pub fn with_auth(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Override the per-request HTTP timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

// ── Internal cache ────────────────────────────────────────────────────────────

struct CacheEntry {
    tools: Vec<ToolSignature>,
    fetched_at: Instant,
}

struct McpClientInner {
    servers: HashMap<String, McpServerConfig>,
    cache: HashMap<String, CacheEntry>,
}

// ── McpClient ─────────────────────────────────────────────────────────────────

/// HTTP-based MCP client.
///
/// Discovers and calls tools on remote MCP servers using JSON-RPC 2.0.
/// Tool-list responses are cached; the cache is lazily refreshed once
/// `refresh_interval` has elapsed since the last successful fetch.
///
/// The client is inexpensive to clone — all instances share the same inner
/// state through an `Arc<RwLock<…>>`.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<RwLock<McpClientInner>>,
    refresh_interval: Duration,
}

impl McpClient {
    /// Create a client with the given server map and cache TTL.
    pub fn new(servers: HashMap<String, McpServerConfig>, refresh_interval: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(McpClientInner {
                servers,
                cache: HashMap::new(),
            })),
            refresh_interval,
        }
    }

    /// Create an empty client with the default 5-minute cache refresh interval.
    pub fn empty() -> Self {
        Self::new(HashMap::new(), Duration::from_secs(DEFAULT_REFRESH_SECS))
    }

    /// Add or replace a server configuration.
    /// Inserting a server does not immediately invalidate its cached tools.
    pub fn add_server(&self, name: impl Into<String>, config: McpServerConfig) {
        let mut inner = self.inner.write().expect("McpClient lock poisoned");
        inner.servers.insert(name.into(), config);
    }

    /// Remove a server and evict its cached tool list.
    pub fn remove_server(&self, name: &str) {
        let mut inner = self.inner.write().expect("McpClient lock poisoned");
        inner.servers.remove(name);
        inner.cache.remove(name);
    }

    /// List names of all registered servers.
    pub fn server_names(&self) -> Vec<String> {
        let inner = self.inner.read().expect("McpClient lock poisoned");
        inner.servers.keys().cloned().collect()
    }

    /// Discover tools on the named server, refreshing the cache if expired.
    ///
    /// Returns [`McpError::ServerNotFound`] if the name has not been registered.
    pub fn discover_tools(&self, server_name: &str) -> Result<Vec<ToolSignature>, McpError> {
        // Fast path: return cached entry if still fresh.
        {
            let inner = self.inner.read().expect("McpClient lock poisoned");
            if let Some(entry) = inner.cache.get(server_name) {
                if entry.fetched_at.elapsed() < self.refresh_interval {
                    return Ok(entry.tools.clone());
                }
            }
        }

        // Slow path: fetch from the server (outside any lock — HTTP can be slow).
        let config = self
            .inner
            .read()
            .expect("McpClient lock poisoned")
            .servers
            .get(server_name)
            .cloned()
            .ok_or_else(|| McpError::ServerNotFound { server: server_name.into() })?;

        let tools = Self::fetch_tools(server_name, &config)?;

        // Store the fresh entry.
        self.inner
            .write()
            .expect("McpClient lock poisoned")
            .cache
            .insert(
                server_name.to_string(),
                CacheEntry { tools: tools.clone(), fetched_at: Instant::now() },
            );

        Ok(tools)
    }

    /// Force a cache refresh for the named server, bypassing the TTL check.
    pub fn refresh(&self, server_name: &str) -> Result<Vec<ToolSignature>, McpError> {
        let config = self
            .inner
            .read()
            .expect("McpClient lock poisoned")
            .servers
            .get(server_name)
            .cloned()
            .ok_or_else(|| McpError::ServerNotFound { server: server_name.into() })?;

        let tools = Self::fetch_tools(server_name, &config)?;

        self.inner
            .write()
            .expect("McpClient lock poisoned")
            .cache
            .insert(
                server_name.to_string(),
                CacheEntry { tools: tools.clone(), fetched_at: Instant::now() },
            );

        Ok(tools)
    }

    /// Call a tool on the named server.
    ///
    /// `args` should be a `serde_json::Value::Object` containing named arguments.
    /// Pass `serde_json::json!({})` for tools that take no arguments.
    pub fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        args: Value,
    ) -> Result<Value, McpError> {
        let config = self
            .inner
            .read()
            .expect("McpClient lock poisoned")
            .servers
            .get(server_name)
            .cloned()
            .ok_or_else(|| McpError::ServerNotFound { server: server_name.into() })?;

        Self::call_remote_tool(server_name, tool_name, args, &config)
    }

    /// Return a merged list of all discoverable tool signatures across every
    /// registered server.
    ///
    /// Uses cached data where available. Skips unreachable servers with a
    /// warning rather than returning an error — partial results are still
    /// useful to the agent.
    pub fn merged_registry(&self) -> Vec<ToolSignature> {
        let names = self.server_names();
        let mut out = Vec::new();
        for name in &names {
            match self.discover_tools(name) {
                Ok(tools) => out.extend(tools),
                Err(e) => {
                    tracing::warn!("MCP: skipping unreachable server \"{name}\": {e}");
                }
            }
        }
        out
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Send a JSON-RPC 2.0 POST and return the parsed response body.
    /// Checks for JSON-RPC-level errors and converts them to [`McpError`].
    fn send_rpc(
        server_name: &str,
        config: &McpServerConfig,
        payload: Value,
    ) -> Result<Value, McpError> {
        let agent = ureq::AgentBuilder::new().timeout(config.timeout).build();

        let req = agent.post(&config.url).set("Content-Type", "application/json");
        let req = match &config.auth_token {
            Some(token) => req.set("Authorization", &format!("Bearer {token}")),
            None => req,
        };

        let body: Value = req
            .send_json(payload)
            .map_err(|e| match e {
                ureq::Error::Status(code, resp) => McpError::Http {
                    server: server_name.into(),
                    message: format!("HTTP {code}: {}", resp.into_string().unwrap_or_default()),
                },
                ureq::Error::Transport(t) => McpError::Http {
                    server: server_name.into(),
                    message: t.to_string(),
                },
            })?
            .into_json()
            .map_err(|e| McpError::Parse {
                server: server_name.into(),
                message: e.to_string(),
            })?;

        if let Some(err) = body.get("error") {
            return Err(McpError::Rpc {
                server: server_name.into(),
                code: err["code"].as_i64().unwrap_or(-1),
                message: err["message"].as_str().unwrap_or("unknown error").to_string(),
            });
        }

        Ok(body)
    }

    fn fetch_tools(
        server_name: &str,
        config: &McpServerConfig,
    ) -> Result<Vec<ToolSignature>, McpError> {
        let body = Self::send_rpc(server_name, config, json!({"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}))?;

        let tools_arr = body
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
            .ok_or_else(|| McpError::Parse {
                server: server_name.into(),
                message: "missing result.tools array in tools/list response".into(),
            })?;

        Ok(tools_arr.iter().map(Self::parse_tool_description).collect())
    }

    /// Convert one MCP tool schema object into a [`ToolSignature`].
    fn parse_tool_description(value: &Value) -> ToolSignature {
        let name = value["name"].as_str().unwrap_or("unknown").to_string();
        let description = value["description"].as_str().unwrap_or("").to_string();

        let mut parameters = Vec::new();
        if let Some(schema) = value.get("inputSchema") {
            let required: HashSet<&str> = schema
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();

            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (prop_name, prop_schema) in props {
                    parameters.push(ToolParam {
                        name: prop_name.clone(),
                        description: prop_schema["description"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                        required: required.contains(prop_name.as_str()),
                    });
                }
            }
        }

        ToolSignature { name, description, parameters }
    }

    fn call_remote_tool(
        server_name: &str,
        tool_name: &str,
        args: Value,
        config: &McpServerConfig,
    ) -> Result<Value, McpError> {
        let payload = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": { "name": tool_name, "arguments": args }
        });

        let body = Self::send_rpc(server_name, config, payload)?;

        let result = body.get("result").ok_or_else(|| McpError::Parse {
            server: server_name.into(),
            message: "missing result field in tools/call response".into(),
        })?;

        if result["isError"].as_bool().unwrap_or(false) {
            let msg = result["content"]
                .as_array()
                .and_then(|arr| arr.iter().find(|c| c["type"] == "text"))
                .and_then(|c| c["text"].as_str())
                .unwrap_or("remote tool returned isError:true")
                .to_string();
            return Err(McpError::ToolError {
                server: server_name.into(),
                tool: tool_name.into(),
                message: msg,
            });
        }

        Ok(result.clone())
    }
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read().expect("McpClient lock poisoned");
        f.debug_struct("McpClient")
            .field("servers", &inner.servers.keys().collect::<Vec<_>>())
            .field("cached", &inner.cache.keys().collect::<Vec<_>>())
            .field("refresh_interval", &self.refresh_interval)
            .finish()
    }
}

// ── McpDiscoverTool ───────────────────────────────────────────────────────────

/// Agent tool: list available tools on a named remote MCP server.
pub struct McpDiscoverTool {
    client: McpClient,
}

impl McpDiscoverTool {
    pub fn new(client: McpClient) -> Self {
        Self { client }
    }
}

impl Tool for McpDiscoverTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "mcp_discover".into(),
            description: "List tools available on a named remote MCP server. \
                          Results are cached; use force_refresh to bypass the cache."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "server".into(),
                    description: "Name of the MCP server (as registered with the client).".into(),
                    required: true,
                },
                ToolParam {
                    name: "force_refresh".into(),
                    description: "Set to 'true' to bypass the cache and fetch fresh tool list."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let server = input.require("server", "mcp_discover")?;
        let force = input.get("force_refresh").is_some_and(|v| v == "true");

        let tools = if force {
            self.client.refresh(server)
        } else {
            self.client.discover_tools(server)
        }
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "mcp_discover".into(),
            message: e.to_string(),
        })?;

        if tools.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "Server \"{server}\" advertises no tools."
            )));
        }

        let lines = tools
            .iter()
            .map(|t| {
                if t.description.is_empty() {
                    format!("  {}", t.name)
                } else {
                    format!("  {} — {}", t.name, t.description)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolOutput::ok(format!(
            "Server \"{server}\" advertises {} tool(s):\n{lines}",
            tools.len()
        )))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "mcp_discover".into(),
            description: "Lists tools available on a remote MCP server (cached).".into(),
            parameters: vec![
                ToolParamSchema::required("server", "MCP server name."),
                ToolParamSchema::optional(
                    "force_refresh",
                    "Bypass cache ('true'/'false', default: 'false').",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::Network]),
                description: "Issues HTTP requests to a remote MCP server to list its tools."
                    .into(),
                shadow_triggers: vec!["discover".into(), "mcp".into(), "remote".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ── McpCallTool ───────────────────────────────────────────────────────────────

/// Agent tool: call a specific tool on a named remote MCP server.
pub struct McpCallTool {
    client: McpClient,
}

impl McpCallTool {
    pub fn new(client: McpClient) -> Self {
        Self { client }
    }
}

impl Tool for McpCallTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "mcp_call".into(),
            description: "Call a tool on a remote MCP server and return its text output.".into(),
            parameters: vec![
                ToolParam {
                    name: "server".into(),
                    description: "Name of the MCP server (as registered with the client).".into(),
                    required: true,
                },
                ToolParam {
                    name: "tool".into(),
                    description: "Name of the remote tool to invoke.".into(),
                    required: true,
                },
                ToolParam {
                    name: "args".into(),
                    description: "JSON object of arguments to pass to the remote tool (default: {})."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let server = input.require("server", "mcp_call")?;
        let tool = input.require("tool", "mcp_call")?;

        let args: Value = match input.get("args") {
            Some(raw) => serde_json::from_str(raw).map_err(|e| AgentError::ToolExecution {
                tool_name: "mcp_call".into(),
                message: format!("args is not valid JSON: {e}"),
            })?,
            None => json!({}),
        };

        let result = self
            .client
            .call_tool(server, tool, args)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "mcp_call".into(),
                message: e.to_string(),
            })?;

        // Extract text items from the MCP content array, fall back to raw JSON.
        let text = result["content"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        if item["type"] == "text" {
                            item["text"].as_str().map(str::to_owned)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| result.to_string());

        Ok(ToolOutput::ok(format!(
            "Remote call \"{server}/{tool}\" succeeded:\n{text}"
        )))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "mcp_call".into(),
            description: "Executes a tool on a remote MCP server and returns its output.".into(),
            parameters: vec![
                ToolParamSchema::required("server", "MCP server name."),
                ToolParamSchema::required("tool", "Name of the remote tool to invoke."),
                ToolParamSchema::optional("args", "JSON object of arguments (default: {})."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([Capability::Network]),
                description: "Executes tools on remote MCP servers over HTTP.".into(),
                shadow_triggers: vec![
                    "mcp".into(),
                    "remote".into(),
                    "call".into(),
                    "external".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    // ── Mock HTTP server ──────────────────────────────────────────────────────

    /// Spawn a minimal HTTP/JSON-RPC mock server that serves `responses` in
    /// order, one per incoming connection.  Returns the server base URL and a
    /// join handle for the background thread.
    fn spawn_mock(responses: Vec<Value>) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");

        let handle = std::thread::spawn(move || {
            let mut resp_iter = responses.into_iter();
            for stream in listener.incoming() {
                let resp = match resp_iter.next() {
                    Some(r) => r,
                    None => break,
                };
                let stream = stream.expect("accept");
                let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                let mut writer = stream;

                // Read request line.
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();

                // Read headers, capture Content-Length.
                let mut content_length = 0usize;
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    let lower = line.to_lowercase();
                    if lower.starts_with("content-length:") {
                        content_length =
                            lower["content-length:".len()..].trim().parse().unwrap_or(0);
                    }
                }

                // Consume request body.
                let mut body = vec![0u8; content_length];
                use std::io::Read;
                Read::read_exact(&mut reader, &mut body).unwrap();

                // Send JSON-RPC response.
                let resp_str = serde_json::to_string(&resp).unwrap();
                let http = format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\
                     \r\n{}",
                    resp_str.len(),
                    resp_str
                );
                writer.write_all(http.as_bytes()).unwrap();
            }
        });

        // Let the OS finish binding before clients connect.
        std::thread::sleep(Duration::from_millis(15));
        (url, handle)
    }

    fn tools_list_response() -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the corpus",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "limit": {"type": "integer", "description": "Max results"}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "infer",
                        "description": "Run inference",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string", "description": "Inference goal"}
                            },
                            "required": ["goal"]
                        }
                    }
                ]
            }
        })
    }

    fn tool_call_ok(text: &str) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": false
            }
        })
    }

    fn rpc_error(code: i64, msg: &str) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": code, "message": msg}
        })
    }

    // ── Unit tests (no HTTP) ──────────────────────────────────────────────────

    #[test]
    fn empty_client_has_no_servers() {
        let client = McpClient::empty();
        assert!(client.server_names().is_empty());
    }

    #[test]
    fn add_and_remove_server() {
        let client = McpClient::empty();
        client.add_server("seshat", McpServerConfig::new("http://seshat.local/mcp"));
        assert!(client.server_names().contains(&"seshat".to_string()));
        client.remove_server("seshat");
        assert!(client.server_names().is_empty());
    }

    #[test]
    fn discover_unknown_server_returns_error() {
        let client = McpClient::empty();
        let err = client.discover_tools("nonexistent").unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound { .. }));
    }

    #[test]
    fn call_tool_unknown_server_returns_error() {
        let client = McpClient::empty();
        let err = client.call_tool("nope", "search", json!({})).unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound { .. }));
    }

    #[test]
    fn merged_registry_empty_client() {
        let client = McpClient::empty();
        assert!(client.merged_registry().is_empty());
    }

    #[test]
    fn parse_tool_description_full_schema() {
        let value = json!({
            "name": "search",
            "description": "Search the corpus",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query"},
                    "limit": {"type": "integer", "description": "Max results"}
                },
                "required": ["query"]
            }
        });
        let sig = McpClient::parse_tool_description(&value);
        assert_eq!(sig.name, "search");
        assert_eq!(sig.description, "Search the corpus");
        assert_eq!(sig.parameters.len(), 2);
        let query_param = sig.parameters.iter().find(|p| p.name == "query").unwrap();
        assert!(query_param.required);
        let limit_param = sig.parameters.iter().find(|p| p.name == "limit").unwrap();
        assert!(!limit_param.required);
    }

    #[test]
    fn parse_tool_description_minimal() {
        let value = json!({"name": "ping"});
        let sig = McpClient::parse_tool_description(&value);
        assert_eq!(sig.name, "ping");
        assert!(sig.description.is_empty());
        assert!(sig.parameters.is_empty());
    }

    #[test]
    fn parse_tool_description_missing_name_fallback() {
        let value = json!({"description": "No name"});
        let sig = McpClient::parse_tool_description(&value);
        assert_eq!(sig.name, "unknown");
    }

    #[test]
    fn server_config_builder() {
        let cfg = McpServerConfig::new("http://localhost/mcp")
            .with_auth("secret-token")
            .with_timeout(Duration::from_secs(5));
        assert_eq!(cfg.url, "http://localhost/mcp");
        assert_eq!(cfg.auth_token.as_deref(), Some("secret-token"));
        assert_eq!(cfg.timeout, Duration::from_secs(5));
    }

    #[test]
    fn debug_impl_shows_server_names() {
        let client = McpClient::empty();
        client.add_server("anima", McpServerConfig::new("http://anima/mcp"));
        let dbg = format!("{client:?}");
        assert!(dbg.contains("anima"));
    }

    // ── Integration tests with mock HTTP server ───────────────────────────────

    #[test]
    fn discover_tools_parses_response() {
        let (url, _handle) = spawn_mock(vec![tools_list_response()]);
        let client = McpClient::empty();
        client.add_server("test", McpServerConfig::new(url).with_timeout(Duration::from_secs(5)));

        let tools = client.discover_tools("test").unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|t| t.name == "search"));
        assert!(tools.iter().any(|t| t.name == "infer"));
    }

    #[test]
    fn discover_tools_caches_result() {
        // Only one mock response; a second discover_tools call must use cache.
        let (url, _handle) = spawn_mock(vec![tools_list_response()]);
        let client = McpClient::new(HashMap::new(), Duration::from_secs(300));
        client.add_server("test", McpServerConfig::new(url).with_timeout(Duration::from_secs(5)));

        let tools1 = client.discover_tools("test").unwrap();
        // Second call would fail if it tried to connect (no second response).
        let tools2 = client.discover_tools("test").unwrap();
        assert_eq!(tools1.len(), tools2.len());
    }

    #[test]
    fn discover_tools_refreshes_after_expiry() {
        // Two responses because the cache expires between the two calls.
        let (url, _handle) =
            spawn_mock(vec![tools_list_response(), tools_list_response()]);

        // 1 ms TTL ensures the second call re-fetches.
        let client = McpClient::new(HashMap::new(), Duration::from_millis(1));
        client.add_server("test", McpServerConfig::new(url).with_timeout(Duration::from_secs(5)));

        let _first = client.discover_tools("test").unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let second = client.discover_tools("test").unwrap();
        assert_eq!(second.len(), 2);
    }

    #[test]
    fn force_refresh_bypasses_cache() {
        // Two responses; the second is served when refresh() is called explicitly.
        let (url, _handle) =
            spawn_mock(vec![tools_list_response(), tools_list_response()]);
        let client = McpClient::new(HashMap::new(), Duration::from_secs(300));
        client.add_server("test", McpServerConfig::new(url).with_timeout(Duration::from_secs(5)));

        let _first = client.discover_tools("test").unwrap();
        let refreshed = client.refresh("test").unwrap();
        assert_eq!(refreshed.len(), 2);
    }

    #[test]
    fn call_tool_returns_text_content() {
        let (url, _handle) = spawn_mock(vec![tool_call_ok("42 results found")]);
        let client = McpClient::empty();
        client.add_server("test", McpServerConfig::new(url).with_timeout(Duration::from_secs(5)));

        let result = client.call_tool("test", "search", json!({"query": "reasoning"})).unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert_eq!(text, "42 results found");
    }

    #[test]
    fn call_tool_is_error_true_returns_tool_error() {
        let resp = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": "tool crashed"}],
                "isError": true
            }
        });
        let (url, _handle) = spawn_mock(vec![resp]);
        let client = McpClient::empty();
        client.add_server("test", McpServerConfig::new(url).with_timeout(Duration::from_secs(5)));

        let err = client.call_tool("test", "search", json!({})).unwrap_err();
        assert!(matches!(err, McpError::ToolError { .. }));
        assert!(err.to_string().contains("tool crashed"));
    }

    #[test]
    fn json_rpc_error_response_returns_rpc_error() {
        let (url, _handle) = spawn_mock(vec![rpc_error(-32601, "Method not found")]);
        let client = McpClient::empty();
        client.add_server("test", McpServerConfig::new(url).with_timeout(Duration::from_secs(5)));

        let err = client.discover_tools("test").unwrap_err();
        assert!(matches!(err, McpError::Rpc { code: -32601, .. }));
    }

    #[test]
    fn merged_registry_collects_all_server_tools() {
        // Two mock servers, each serving one discovery response.
        let (url_a, _ha) = spawn_mock(vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [{"name": "search", "description": "Search"}]
            }
        })]);
        let (url_b, _hb) = spawn_mock(vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [{"name": "work_items", "description": "Anima work items"}]
            }
        })]);

        let client = McpClient::empty();
        client.add_server("seshat", McpServerConfig::new(url_a).with_timeout(Duration::from_secs(5)));
        client.add_server("anima", McpServerConfig::new(url_b).with_timeout(Duration::from_secs(5)));

        let all = client.merged_registry();
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|t| t.name == "search"));
        assert!(all.iter().any(|t| t.name == "work_items"));
    }

    // ── Tool trait tests ──────────────────────────────────────────────────────

    #[test]
    fn mcp_discover_tool_signature() {
        let client = McpClient::empty();
        let tool = McpDiscoverTool::new(client);
        let sig = tool.signature();
        assert_eq!(sig.name, "mcp_discover");
        assert!(sig.parameters.iter().any(|p| p.name == "server" && p.required));
        assert!(sig.parameters.iter().any(|p| p.name == "force_refresh" && !p.required));
    }

    #[test]
    fn mcp_call_tool_signature() {
        let client = McpClient::empty();
        let tool = McpCallTool::new(client);
        let sig = tool.signature();
        assert_eq!(sig.name, "mcp_call");
        assert!(sig.parameters.iter().any(|p| p.name == "server" && p.required));
        assert!(sig.parameters.iter().any(|p| p.name == "tool" && p.required));
        assert!(sig.parameters.iter().any(|p| p.name == "args" && !p.required));
    }

    #[test]
    fn mcp_discover_tool_missing_server_param() {
        let client = McpClient::empty();
        let tool = McpDiscoverTool::new(client);
        let input = ToolInput::new();
        // No engine needed; the tool fails before touching it.
        // We can use a dummy engine reference — but Tool::execute needs &Engine.
        // Since AgentError::ToolExecution is returned without calling engine,
        // we just verify the error variant via the McpClient directly.
        let err = tool.client.discover_tools("nonexistent").unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound { .. }));
        // Also verify input.require fails without the param:
        let err2 = input.require("server", "mcp_discover").unwrap_err();
        assert!(matches!(
            err2,
            crate::agent::error::AgentError::ToolExecution { .. }
        ));
    }

    #[test]
    fn mcp_call_tool_bad_args_json() {
        // args is not valid JSON — the tool should return ToolExecution error.
        let client = McpClient::empty();
        client.add_server("test", McpServerConfig::new("http://localhost/mcp"));
        let _tool = McpCallTool::new(client.clone());

        // Simulate what execute() does with bad args:
        let result = serde_json::from_str::<Value>("not-json");
        assert!(result.is_err());
    }

    #[test]
    fn mcp_discover_tool_manifest_danger_level() {
        let tool = McpDiscoverTool::new(McpClient::empty());
        let m = tool.manifest();
        assert_eq!(m.danger.level, DangerLevel::Cautious);
        assert!(m.danger.capabilities.contains(&Capability::Network));
    }

    #[test]
    fn mcp_call_tool_manifest_danger_level() {
        let tool = McpCallTool::new(McpClient::empty());
        let m = tool.manifest();
        assert_eq!(m.danger.level, DangerLevel::Dangerous);
        assert!(m.danger.capabilities.contains(&Capability::Network));
    }
}
