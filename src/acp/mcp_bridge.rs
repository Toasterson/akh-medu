//! MCP-over-ACP bridge: honor the MCP servers a client pushes at `session/new`.
//!
//! ACP clients advertise their MCP servers in `NewSessionRequest.mcp_servers` —
//! the same mechanism a spawned CLI receives via `--mcp-config`. We distill each
//! pushed server into a [`SessionMcpServerCfg`], record it per session, and (in
//! the daemon build) connect to the HTTP ones at prompt time, discover their
//! tools, and register namespaced proxy tools (`mcp__{server}__{tool}`) into the
//! akh's registry for the duration of that turn.
//!
//! Only the streamable-HTTP transport is bridged today; stdio / SSE / ACP-over-
//! channel servers are counted and logged but not connected.

use agent_client_protocol::schema::McpServer;

/// A client-pushed HTTP MCP server, distilled to what the bridge acts on.
///
/// Non-HTTP transports never produce one of these — they are reported via the
/// `unsupported` count from [`parse_mcp_servers`] instead.
#[derive(Debug, Clone)]
pub struct SessionMcpServerCfg {
    /// Human-readable server name the client supplied (used for namespacing).
    pub name: String,
    /// MCP endpoint base URL.
    pub url: String,
    /// Bearer/authorization header value, lifted from an `Authorization` header
    /// if the client supplied one.
    pub auth_header: Option<String>,
}

/// Parse ACP `mcp_servers` into the bridge's own config.
///
/// Returns the HTTP servers we can bridge plus the number of pushed servers on
/// transports we don't honor yet (stdio / SSE / ACP). Case-insensitively lifts
/// an `Authorization` header into the single auth slot the HTTP client supports;
/// any other headers are dropped.
pub fn parse_mcp_servers(servers: &[McpServer]) -> (Vec<SessionMcpServerCfg>, usize) {
    let mut out = Vec::new();
    let mut unsupported = 0usize;
    for server in servers {
        match server {
            McpServer::Http(http) => {
                let auth_header = http
                    .headers
                    .iter()
                    .find(|h| h.name.eq_ignore_ascii_case("authorization"))
                    .map(|h| h.value.clone());
                out.push(SessionMcpServerCfg {
                    name: http.name.clone(),
                    url: http.url.clone(),
                    auth_header,
                });
            }
            // Stdio / SSE / ACP-over-channel / future transports (the enum is
            // `#[non_exhaustive]`): not bridged yet.
            _ => unsupported += 1,
        }
    }
    (out, unsupported)
}

// ── Proxy tools (daemon build only — needs the HTTP MCP client) ──────────────

#[cfg(feature = "daemon")]
pub use proxy::build_session_tools;

#[cfg(feature = "daemon")]
mod proxy {
    use std::collections::HashSet;

    use serde_json::{Map, Value};

    use super::SessionMcpServerCfg;
    use crate::agent::error::AgentResult;
    use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
    use crate::agent::tool_manifest::{
        Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
    };
    use crate::config::UpstreamEndpointConfig;
    use crate::engine::Engine;
    use crate::upstream_client::UpstreamMcpClient;

    /// One parameter of a proxied remote tool, carrying enough type info to
    /// coerce the LLM's string argument back to the JSON type the server wants.
    struct ProxyParam {
        name: String,
        /// JSON Schema `type` string (`"string"`, `"integer"`, `"number"`,
        /// `"boolean"`, …); anything else is passed through as a string.
        ty: String,
        required: bool,
        description: String,
    }

    /// A single remote MCP tool exposed to the akh as a first-class tool.
    ///
    /// Named `mcp__{server}__{remote}` (the Claude-Code convention), it forwards
    /// each call over streamable-HTTP MCP to the client-pushed server.
    pub struct McpProxyTool {
        tool_name: String,
        remote_name: String,
        description: String,
        params: Vec<ProxyParam>,
        url: String,
        auth_header: Option<String>,
    }

    /// Sanitize a server name into the `mcp__server__tool` namespace segment.
    fn sanitize(name: &str) -> String {
        name.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }

    impl McpProxyTool {
        /// Build a proxy from one raw MCP tool descriptor (`{name, description,
        /// inputSchema}`). Returns `None` if the descriptor has no usable name.
        fn from_descriptor(
            server: &str,
            url: &str,
            auth_header: Option<&str>,
            descriptor: &Value,
        ) -> Option<Self> {
            let remote_name = descriptor["name"].as_str()?.to_string();
            if remote_name.is_empty() {
                return None;
            }
            let description = descriptor["description"].as_str().unwrap_or("").to_string();

            let required: HashSet<String> = descriptor["inputSchema"]["required"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let mut params = Vec::new();
            if let Some(props) = descriptor["inputSchema"]["properties"].as_object() {
                for (pname, pschema) in props {
                    params.push(ProxyParam {
                        name: pname.clone(),
                        ty: pschema["type"].as_str().unwrap_or("string").to_string(),
                        required: required.contains(pname),
                        description: pschema["description"].as_str().unwrap_or("").to_string(),
                    });
                }
            }

            Some(Self {
                tool_name: format!("mcp__{}__{}", sanitize(server), remote_name),
                remote_name,
                description,
                params,
                url: url.to_string(),
                auth_header: auth_header.map(String::from),
            })
        }

        /// Coerce the flat string `ToolInput` into a JSON argument object typed
        /// per the remote tool's schema (so `usize`/`f32`/`bool` params
        /// round-trip instead of arriving as strings the server rejects).
        fn build_args(&self, input: &ToolInput) -> Value {
            let mut args = Map::new();
            for p in &self.params {
                let Some(raw) = input.get(&p.name) else { continue };
                let value = match p.ty.as_str() {
                    "integer" => raw
                        .parse::<i64>()
                        .map(Value::from)
                        .unwrap_or_else(|_| Value::from(raw)),
                    "number" => raw
                        .parse::<f64>()
                        .map(Value::from)
                        .unwrap_or_else(|_| Value::from(raw)),
                    "boolean" => raw
                        .parse::<bool>()
                        .map(Value::from)
                        .unwrap_or_else(|_| Value::from(raw)),
                    _ => Value::from(raw),
                };
                args.insert(p.name.clone(), value);
            }
            Value::Object(args)
        }
    }

    impl Tool for McpProxyTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: self.tool_name.clone(),
                description: self.description.clone(),
                parameters: self
                    .params
                    .iter()
                    .map(|p| ToolParam {
                        name: p.name.clone(),
                        description: p.description.clone(),
                        required: p.required,
                    })
                    .collect(),
            }
        }

        fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
            let args = self.build_args(&input);
            let client = UpstreamMcpClient::new(&UpstreamEndpointConfig {
                url: self.url.clone(),
                auth_header: self.auth_header.clone(),
            });
            match client.call_tool_text(&self.remote_name, args) {
                Ok(text) if text.is_empty() => {
                    Ok(ToolOutput::ok(format!("{} returned no content.", self.remote_name)))
                }
                Ok(text) => Ok(ToolOutput::ok(text)),
                Err(e) => Ok(ToolOutput::err(format!(
                    "upstream MCP tool \"{}\" failed: {e}",
                    self.remote_name
                ))),
            }
        }

        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: self.tool_name.clone(),
                description: self.description.clone(),
                parameters: self
                    .params
                    .iter()
                    .map(|p| {
                        if p.required {
                            ToolParamSchema::required(&p.name, &p.description)
                        } else {
                            ToolParamSchema::optional(&p.name, &p.description)
                        }
                    })
                    .collect(),
                danger: DangerInfo {
                    level: DangerLevel::Dangerous,
                    capabilities: HashSet::from([Capability::Network]),
                    description: "Calls a client-provided MCP server over the network.".into(),
                    shadow_triggers: vec!["mcp".into(), "remote".into(), "external".into()],
                },
                source: ToolSource::Native,
            }
        }
    }

    /// Connect to each pushed HTTP MCP server, discover its tools, and build a
    /// proxy `Tool` for each. Discovery failures are logged and skipped so one
    /// unreachable server never breaks a turn. Runs blocking network I/O — call
    /// only from a blocking context (the ADR-027 turn closure).
    pub fn build_session_tools(cfgs: &[SessionMcpServerCfg]) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> = Vec::new();
        for cfg in cfgs {
            let client = UpstreamMcpClient::new(&UpstreamEndpointConfig {
                url: cfg.url.clone(),
                auth_header: cfg.auth_header.clone(),
            });
            match client.list_tools_raw() {
                Ok(descriptors) => {
                    let before = tools.len();
                    for descriptor in &descriptors {
                        if let Some(tool) = McpProxyTool::from_descriptor(
                            &cfg.name,
                            &cfg.url,
                            cfg.auth_header.as_deref(),
                            descriptor,
                        ) {
                            tools.push(Box::new(tool));
                        }
                    }
                    tracing::info!(
                        server = %cfg.name,
                        url = %cfg.url,
                        tools = tools.len() - before,
                        "session MCP server tools bridged"
                    );
                }
                Err(e) => tracing::warn!(
                    server = %cfg.name,
                    url = %cfg.url,
                    "session MCP discovery failed, skipping: {e}"
                ),
            }
        }
        tools
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::agent::tool::Tool;

        fn descriptor() -> Value {
            serde_json::json!({
                "name": "seshat_search",
                "description": "Search the corpus",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the query"},
                        "top_k": {"type": "integer", "description": "max results"},
                        "min_similarity": {"type": "number"}
                    },
                    "required": ["query"]
                }
            })
        }

        #[test]
        fn from_descriptor_namespaces_and_maps_params() {
            let t = McpProxyTool::from_descriptor("seshat", "http://x/mcp", None, &descriptor())
                .expect("proxy");
            let sig = t.signature();
            assert_eq!(sig.name, "mcp__seshat__seshat_search");
            assert_eq!(t.remote_name, "seshat_search");
            let query = sig.parameters.iter().find(|p| p.name == "query").unwrap();
            assert!(query.required);
            let top_k = sig.parameters.iter().find(|p| p.name == "top_k").unwrap();
            assert!(!top_k.required);
        }

        #[test]
        fn from_descriptor_sanitizes_server_name() {
            let t = McpProxyTool::from_descriptor("an ima.1", "http://x", None, &descriptor())
                .expect("proxy");
            assert_eq!(t.signature().name, "mcp__an_ima_1__seshat_search");
        }

        #[test]
        fn from_descriptor_rejects_nameless() {
            assert!(
                McpProxyTool::from_descriptor("s", "http://x", None, &serde_json::json!({})).is_none()
            );
        }

        #[test]
        fn build_args_coerces_types_by_schema() {
            let t = McpProxyTool::from_descriptor("seshat", "http://x", None, &descriptor())
                .expect("proxy");
            let input = ToolInput::new()
                .with_param("query", "reasoning")
                .with_param("top_k", "5")
                .with_param("min_similarity", "0.7");
            let args = t.build_args(&input);
            assert_eq!(args["query"], Value::from("reasoning"));
            assert_eq!(args["top_k"], Value::from(5i64));
            assert_eq!(args["min_similarity"], Value::from(0.7f64));
        }

        #[test]
        fn build_args_omits_unset_params() {
            let t = McpProxyTool::from_descriptor("seshat", "http://x", None, &descriptor())
                .expect("proxy");
            let args = t.build_args(&ToolInput::new().with_param("query", "q"));
            assert!(args.get("top_k").is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::{HttpHeader, McpServer, McpServerHttp};

    #[test]
    fn parse_http_server_lifts_authorization_header() {
        let mut http = McpServerHttp::new("seshat", "http://seshat.local:4200");
        http.headers.push(HttpHeader::new("Authorization", "Bearer tok"));
        http.headers.push(HttpHeader::new("X-Other", "ignored"));
        let (cfgs, unsupported) = parse_mcp_servers(&[McpServer::Http(http)]);
        assert_eq!(unsupported, 0);
        assert_eq!(cfgs.len(), 1);
        assert_eq!(cfgs[0].name, "seshat");
        assert_eq!(cfgs[0].url, "http://seshat.local:4200");
        assert_eq!(cfgs[0].auth_header.as_deref(), Some("Bearer tok"));
    }

    #[test]
    fn parse_http_server_without_auth() {
        let http = McpServerHttp::new("anima", "http://anima.local/mcp");
        let (cfgs, unsupported) = parse_mcp_servers(&[McpServer::Http(http)]);
        assert_eq!(unsupported, 0);
        assert_eq!(cfgs.len(), 1);
        assert!(cfgs[0].auth_header.is_none());
    }

    #[test]
    fn parse_empty_is_empty() {
        let (cfgs, unsupported) = parse_mcp_servers(&[]);
        assert!(cfgs.is_empty());
        assert_eq!(unsupported, 0);
    }
}
