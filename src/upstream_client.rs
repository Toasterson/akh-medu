//! Lightweight synchronous HTTP-MCP client for calling upstream tool endpoints.
//!
//! Used by the `import_anima_project` and `link_seshat_doc` MCP tools to pull
//! context from Anima (project management) and Seshat (document corpus) into the
//! local knowledge graph.
//!
//! Implements the MCP Streamable HTTP transport: sends an `initialize` request
//! (requesting `rmcp::model::ProtocolVersion::LATEST`) to establish a session,
//! adopts whatever version the upstream negotiates back, then calls the
//! requested tool. All I/O uses `ureq` (blocking) and must be invoked from
//! within `tokio::task::spawn_blocking`.
//!
//! Protocol version: we request the latest version this SDK knows
//! ([`rmcp::model::ProtocolVersion::LATEST`]) and log whatever the server
//! names back in its `InitializeResult` — per the MCP spec the server may
//! answer with a different (older) version than requested, and clients are
//! expected to adopt it. This is a real split in practice: `seshd` (our own
//! `rmcp`-based server) always answers with its fixed, older
//! `V_2025_03_26`, while `t3-code`'s MCP server only accepts the newest
//! `V_2025_06_18` and rejects anything else at the transport level (HTTP
//! 400) — requesting `LATEST` satisfies both, since seshd's `initialize`
//! handler ignores the requested version entirely and t3 wants exactly this
//! one.
//!
//! # Example
//!
//! ```no_run
//! use akh_medu::config::UpstreamEndpointConfig;
//! use akh_medu::upstream_client::UpstreamMcpClient;
//!
//! let cfg = UpstreamEndpointConfig { url: "http://anima.local:4100".into(), auth_header: None };
//! let client = UpstreamMcpClient::new(&cfg);
//! let text = client.call_tool_text("list_work_items", serde_json::json!({"project_id": "abc"})).unwrap();
//! ```

use std::time::Duration;

use miette::Diagnostic;
use rmcp::model::ProtocolVersion;
use serde_json::Value;
use thiserror::Error;

use crate::config::UpstreamEndpointConfig;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from upstream MCP client operations.
#[derive(Debug, Error, Diagnostic)]
pub enum UpstreamError {
    #[error("upstream request failed: {message}")]
    #[diagnostic(
        code(akh::upstream::request),
        help("Check that the upstream service is running and the URL is correct.")
    )]
    Request { message: String },

    #[error("upstream returned RPC error {code}: {message}")]
    #[diagnostic(
        code(akh::upstream::rpc_error),
        help("The upstream MCP tool call was rejected. Check tool name and arguments.")
    )]
    RpcError { code: i64, message: String },

    #[error("failed to parse upstream response: {message}")]
    #[diagnostic(
        code(akh::upstream::parse),
        help("The upstream returned an unexpected response format.")
    )]
    Parse { message: String },

    #[error("upstream not configured for \"{service}\"")]
    #[diagnostic(
        code(akh::upstream::not_configured),
        help(
            "Add [upstreams.{service}] with a url field (and optional auth_header) \
             to akh-medu config.toml."
        )
    )]
    NotConfigured { service: &'static str },
}

pub type UpstreamResult<T> = Result<T, UpstreamError>;

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Synchronous MCP client for a single upstream endpoint.
///
/// Wraps `ureq` and implements the two-step MCP handshake:
/// 1. `initialize` — establishes a session and extracts the `Mcp-Session-Id` header.
/// 2. `tools/call` — invokes the requested tool with that session ID.
///
/// If initialization fails (e.g. the upstream does not require it), the client
/// falls back to a direct tool call without a session ID.
pub struct UpstreamMcpClient {
    /// Fully-resolved MCP endpoint URL to POST to (already includes any
    /// transport path such as `/mcp`). Set verbatim by [`from_endpoint`];
    /// derived as `{base}/mcp` by [`new`].
    ///
    /// [`from_endpoint`]: Self::from_endpoint
    /// [`new`]: Self::new
    endpoint: String,
    auth_header: Option<String>,
    agent: ureq::Agent,
}

impl UpstreamMcpClient {
    /// Create a client from a config-declared upstream, treating `config.url`
    /// as a **base** and appending the `/mcp` transport path.
    ///
    /// This is the convention for statically-configured upstreams (Anima,
    /// Seshat), where the config holds a service root. For a URL that already
    /// names the full MCP endpoint — e.g. an [`McpServer`] pushed over ACP at
    /// `session/new`, per the Claude Code CLI contract — use [`from_endpoint`]
    /// so no path is appended.
    ///
    /// [`McpServer`]: agent_client_protocol::schema::McpServer
    /// [`from_endpoint`]: Self::from_endpoint
    pub fn new(config: &UpstreamEndpointConfig) -> Self {
        let base = config.url.trim_end_matches('/');
        Self::from_endpoint(format!("{base}/mcp"), config.resolved_auth())
    }

    /// Create a client for a **full** MCP endpoint URL, taken verbatim.
    ///
    /// Unlike [`new`], this appends nothing: the caller has already supplied the
    /// complete endpoint to POST to. Use it for MCP servers pushed over ACP so a
    /// server legitimately rooted at `/mcp` (or any other path) is honored
    /// exactly rather than doubled to `/mcp/mcp`.
    ///
    /// [`new`]: Self::new
    pub fn from_endpoint(endpoint: impl Into<String>, auth_header: Option<String>) -> Self {
        Self::from_endpoint_with_timeout(endpoint, auth_header, Duration::from_secs(30))
    }

    /// Create a client for a **full** MCP endpoint URL with an explicit
    /// request timeout, taken verbatim (see [`from_endpoint`]).
    ///
    /// Used for time-sensitive callers such as ACP session-MCP discovery
    /// (`session/prompt`), which runs synchronously inside the prompt turn
    /// and must not let one slow/blackholed pushed server eat the default
    /// 30s budget on every request.
    ///
    /// [`from_endpoint`]: Self::from_endpoint
    pub fn from_endpoint_with_timeout(
        endpoint: impl Into<String>,
        auth_header: Option<String>,
        timeout: Duration,
    ) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(timeout).build();
        Self {
            endpoint: endpoint.into(),
            auth_header,
            agent,
        }
    }

    /// Build a POST request to the resolved MCP endpoint with common MCP headers.
    fn post_mcp(&self) -> ureq::Request {
        let mut req = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json, text/event-stream");
        if let Some(ref auth) = self.auth_header {
            req = req.set("Authorization", auth);
        }
        req
    }

    /// Send a JSON-RPC payload and return `(session_id_header, parsed_body)`.
    fn send_rpc(
        &self,
        req: ureq::Request,
        payload: Value,
    ) -> UpstreamResult<(Option<String>, Value)> {
        let resp = req.send_json(payload).map_err(|e| match e {
            ureq::Error::Status(code, resp) => UpstreamError::Request {
                message: format!(
                    "HTTP {code}: {}",
                    resp.into_string()
                        .unwrap_or_default()
                        .chars()
                        .take(256)
                        .collect::<String>()
                ),
            },
            ureq::Error::Transport(t) => UpstreamError::Request {
                message: format!("transport error: {t}"),
            },
        })?;

        let session_id = resp.header("Mcp-Session-Id").map(str::to_owned);
        let content_type = resp.header("Content-Type").unwrap_or("").to_owned();
        let body = resp.into_string().map_err(|e| UpstreamError::Parse {
            message: format!("reading response body: {e}"),
        })?;

        let json = if content_type.contains("text/event-stream") {
            parse_sse_last_data(&body)?
        } else {
            serde_json::from_str(&body).map_err(|e| UpstreamError::Parse {
                message: format!(
                    "{e}: {}",
                    body.chars().take(256).collect::<String>()
                ),
            })?
        };

        Ok((session_id, json))
    }

    /// Establish an MCP session via the `initialize` handshake.
    ///
    /// Returns the `Mcp-Session-Id` header if the upstream provided one.
    /// Non-fatal RPC errors and network errors during init are logged and
    /// treated as "no session" — callers then attempt a direct call/list
    /// without a session ID, matching upstreams that don't require one.
    fn init_session(&self) -> Option<String> {
        let init_payload = initialize_payload(ProtocolVersion::LATEST);

        match self.send_rpc(self.post_mcp(), init_payload) {
            Ok((sid, json)) => {
                // Non-fatal RPC errors during init are ignored; proceed anyway.
                if !json["error"].is_object() {
                    // Adopt whatever protocol version the upstream negotiated
                    // back in its InitializeResult — the MCP spec lets the
                    // server name a different version than we requested (see
                    // the module doc: seshd always answers with its own
                    // fixed, older version). We don't yet gate any wire-shape
                    // decision on the negotiated version, so an unknown one
                    // is logged loudly (not silently accepted) rather than
                    // failing the call outright.
                    match json["result"]["protocolVersion"].as_str() {
                        Some(v) => {
                            let negotiated: ProtocolVersion = serde_json::from_value(
                                Value::String(v.to_string()),
                            )
                            .unwrap_or(ProtocolVersion::LATEST);
                            if ProtocolVersion::KNOWN_VERSIONS.contains(&negotiated) {
                                tracing::info!(
                                    endpoint = %self.endpoint,
                                    version = %negotiated,
                                    "MCP protocol version negotiated"
                                );
                            } else {
                                tracing::error!(
                                    endpoint = %self.endpoint,
                                    version = %negotiated,
                                    "upstream negotiated an MCP protocol version this SDK \
                                     does not know; proceeding, but the wire format may not \
                                     match what we expect"
                                );
                            }
                        }
                        None => {
                            tracing::debug!(
                                endpoint = %self.endpoint,
                                "upstream initialize response carried no protocolVersion field"
                            );
                        }
                    }
                    // MCP requires the client to follow a successful
                    // `initialize` with a `notifications/initialized`
                    // notification before issuing requests. rmcp-based
                    // servers (seshd's streamable-http) enforce this: a
                    // `tools/call` on a session that never sent it gets an
                    // SSE stream that closes with no data event — which
                    // surfaced as "SSE data JSON parse error: EOF" in the
                    // seshat_curation task. Errors are non-fatal (lenient
                    // servers work either way); some servers answer the
                    // notification with 202/empty, which send_rpc treats as
                    // an error — hence best-effort.
                    let mut req = self.post_mcp();
                    if let Some(ref s) = sid {
                        req = req.set("Mcp-Session-Id", s);
                    }
                    let notif = serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    });
                    if let Err(e) = self.send_rpc(req, notif) {
                        tracing::debug!(
                            error = %e,
                            "notifications/initialized send failed (non-fatal; server may not require it)"
                        );
                    }
                    sid
                } else {
                    tracing::debug!(
                        error = %json["error"],
                        "upstream init returned RPC error; proceeding without session"
                    );
                    None
                }
            }
            Err(e) => {
                // Network errors during init are non-fatal; attempt direct call.
                tracing::debug!(
                    error = %e,
                    "upstream init request failed; attempting direct tool call"
                );
                None
            }
        }
    }

    /// Call a named tool on the upstream MCP endpoint.
    ///
    /// Returns the `content` array from the `CallToolResult`, or an empty vec if
    /// the result contained no content items.
    pub fn call_tool(&self, tool_name: &str, arguments: Value) -> UpstreamResult<Vec<Value>> {
        // Step 1: initialize session.
        let session_id = self.init_session();

        // Step 2: call the tool.
        let tool_payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        });

        let mut req = self.post_mcp();
        if let Some(sid) = session_id.as_deref() {
            req = req.set("Mcp-Session-Id", sid);
        }

        let (_, result_json) = self.send_rpc(req, tool_payload)?;
        check_rpc_error(&result_json)?;

        let content = result_json["result"]["content"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        Ok(content)
    }

    /// List tools advertised by the upstream MCP endpoint (name + description).
    pub fn list_tools(&self) -> UpstreamResult<Vec<(String, String)>> {
        // Step 1: initialize session (same handshake as `call_tool`).
        let session_id = self.init_session();

        // Step 2: request the tool list.
        let list_payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        });

        let mut req = self.post_mcp();
        if let Some(sid) = session_id.as_deref() {
            req = req.set("Mcp-Session-Id", sid);
        }

        let (_, result_json) = self.send_rpc(req, list_payload)?;
        check_rpc_error(&result_json)?;

        Ok(parse_tools_list(&result_json))
    }

    /// List the upstream's tools as their raw MCP descriptor objects, each
    /// carrying `name`, `description`, and `inputSchema`. Callers that need the
    /// parameter schema (e.g. to build proxy tools) use this instead of
    /// [`list_tools`], which returns only name + description.
    pub fn list_tools_raw(&self) -> UpstreamResult<Vec<Value>> {
        let session_id = self.init_session();

        let list_payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        });

        let mut req = self.post_mcp();
        if let Some(sid) = session_id.as_deref() {
            req = req.set("Mcp-Session-Id", sid);
        }

        let (_, result_json) = self.send_rpc(req, list_payload)?;
        check_rpc_error(&result_json)?;

        Ok(result_json["result"]["tools"]
            .as_array()
            .cloned()
            .unwrap_or_default())
    }

    /// Call a tool and return the concatenated text from all `text`-type content items.
    pub fn call_tool_text(&self, tool_name: &str, arguments: Value) -> UpstreamResult<String> {
        let content = self.call_tool(tool_name, arguments)?;
        Ok(content
            .iter()
            .filter_map(|c| c["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the `initialize` JSON-RPC request body, requesting `version` as the
/// MCP protocol version. Extracted so the requested version is unit-testable
/// without a network call.
fn initialize_payload(version: ProtocolVersion) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": version,
            "capabilities": {},
            "clientInfo": {
                "name": "akhomed-bridge",
                "version": env!("CARGO_PKG_VERSION")
            }
        }
    })
}

fn check_rpc_error(json: &Value) -> UpstreamResult<()> {
    if json["error"].is_object() {
        Err(UpstreamError::RpcError {
            code: json["error"]["code"].as_i64().unwrap_or(-1),
            message: json["error"]["message"]
                .as_str()
                .unwrap_or("unknown RPC error")
                .to_string(),
        })
    } else {
        Ok(())
    }
}

/// Parse a `tools/list` JSON-RPC response body into `(name, description)` pairs.
fn parse_tools_list(result_json: &Value) -> Vec<(String, String)> {
    result_json["result"]["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|t| {
            let name = t["name"].as_str().unwrap_or_default().to_string();
            let description = t["description"].as_str().unwrap_or_default().to_string();
            (name, description)
        })
        .collect()
}

/// Extract the last `data:` line from an SSE stream body and parse it as JSON.
fn parse_sse_last_data(body: &str) -> UpstreamResult<Value> {
    let last_data = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .last()
        .ok_or_else(|| UpstreamError::Parse {
            message: "no data events found in SSE stream".to_string(),
        })?;

    serde_json::from_str(last_data).map_err(|e| UpstreamError::Parse {
        message: format!("SSE data JSON parse error: {e}"),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_payload_requests_the_latest_known_protocol_version() {
        // Regression for the t3-code MCP discovery HTTP 400: t3's server only
        // accepts `2025-06-18` and hard-rejects anything else at the
        // transport level. `ProtocolVersion::LATEST` must resolve to exactly
        // that string, or the handshake breaks again.
        let payload = initialize_payload(ProtocolVersion::LATEST);
        assert_eq!(payload["params"]["protocolVersion"], "2025-06-18");
        assert_eq!(ProtocolVersion::LATEST, ProtocolVersion::V_2025_06_18);
    }

    #[test]
    fn new_appends_mcp_path_to_base_url() {
        let cfg = UpstreamEndpointConfig {
            url: "http://anima.local:4100/".into(),
            auth_header: None,
        };
        // Config convention: url is a base, `/mcp` is the transport path.
        assert_eq!(UpstreamMcpClient::new(&cfg).endpoint, "http://anima.local:4100/mcp");
    }

    #[test]
    fn from_endpoint_with_timeout_is_honored_by_default_ctor() {
        // `from_endpoint` should just be the 30s-default sugar over
        // `from_endpoint_with_timeout` — same endpoint handling either way.
        let a = UpstreamMcpClient::from_endpoint("https://x.local/mcp", None);
        let b = UpstreamMcpClient::from_endpoint_with_timeout(
            "https://x.local/mcp",
            None,
            Duration::from_secs(30),
        );
        assert_eq!(a.endpoint, b.endpoint);
    }

    #[test]
    fn from_endpoint_takes_url_verbatim() {
        // ACP-pushed servers carry the full endpoint; never double `/mcp/mcp`,
        // and honor a server legitimately rooted at `/mcp`.
        let c = UpstreamMcpClient::from_endpoint("https://seshat.local/mcp", None);
        assert_eq!(c.endpoint, "https://seshat.local/mcp");
        let rooted = UpstreamMcpClient::from_endpoint("https://x.local/custom/path", None);
        assert_eq!(rooted.endpoint, "https://x.local/custom/path");
    }

    #[test]
    fn parse_sse_last_data_extracts_last_event() {
        let sse = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}}\n";
        let val = parse_sse_last_data(sse).unwrap();
        assert_eq!(val["result"]["content"][0]["text"], "ok");
    }

    #[test]
    fn parse_sse_no_data_returns_error() {
        let sse = "event: done\n\n";
        assert!(parse_sse_last_data(sse).is_err());
    }

    #[test]
    fn parse_sse_skips_done_sentinel() {
        let sse = "data: {\"ok\":true}\ndata: [DONE]\n";
        let val = parse_sse_last_data(sse).unwrap();
        assert_eq!(val["ok"], true);
    }

    #[test]
    fn check_rpc_error_passes_success() {
        let ok = serde_json::json!({"jsonrpc":"2.0","id":1,"result":{}});
        assert!(check_rpc_error(&ok).is_ok());
    }

    #[test]
    fn check_rpc_error_returns_err() {
        let err = serde_json::json!({"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}});
        let e = check_rpc_error(&err).unwrap_err();
        match e {
            UpstreamError::RpcError { code, .. } => assert_eq!(code, -32601),
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn parse_tools_list_extracts_name_and_description() {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {"name": "search", "description": "Search the corpus"},
                    {"name": "list_work_items", "description": "List project work items"}
                ]
            }
        });
        let tools = parse_tools_list(&body);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0], ("search".to_string(), "Search the corpus".to_string()));
        assert_eq!(
            tools[1],
            ("list_work_items".to_string(), "List project work items".to_string())
        );
    }

    #[test]
    fn parse_tools_list_missing_tools_array_is_empty() {
        let body = serde_json::json!({"jsonrpc": "2.0", "id": 2, "result": {}});
        assert!(parse_tools_list(&body).is_empty());
    }
}
