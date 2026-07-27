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

/// Tool descriptors discovered from one client-pushed MCP server, cached for
/// the session's lifetime after the first successful discovery pass.
///
/// Carries everything [`build_tools_from_discovered`] needs to rebuild fresh
/// [`McpProxyTool`](proxy::McpProxyTool)s each turn with **zero network
/// I/O** — discovery (`initialize` + `tools/list`, blocking HTTP) runs at
/// most once per ACP session; every later `session/prompt` in that session
/// reuses this.
#[derive(Debug, Clone)]
pub struct DiscoveredServerTools {
    /// The client-supplied server name (used for tool namespacing).
    pub name: String,
    /// MCP endpoint base URL.
    pub url: String,
    /// Bearer/authorization header value, if the client supplied one.
    pub auth_header: Option<String>,
    /// Raw `{name, description, inputSchema}` tool descriptors as returned by
    /// `tools/list`.
    pub descriptors: Vec<serde_json::Value>,
}

/// Aggregate failure reported by [`discover_session_tools_strict`] when one or
/// more client-pushed MCP servers did not come back after the bounded retry.
///
/// The session **must not** continue tool-less — the work item "Turns die
/// quietly" requires the turn to terminate with the upstream status visible
/// in the thread. The caller streams an `error[acp]` message carrying the
/// `summaries` and ends the turn with a stop reason Anima can distinguish
/// from a clean completion.
#[derive(Debug, Clone)]
pub struct DiscoveryFailure {
    /// How many servers the caller asked the bridge to reach.
    pub attempted: usize,
    /// How many servers actually came up before the failure was returned.
    pub succeeded: usize,
    /// One human-readable line per failing server: `name (url): error_text`.
    pub summaries: Vec<String>,
}

impl DiscoveryFailure {
    /// Render the failure as a single line for an `error[acp]` message.
    pub fn summary(&self) -> String {
        if self.summaries.len() == 1 {
            self.summaries[0].clone()
        } else {
            format!(
                "{} of {} MCP servers failed discovery: {}",
                self.summaries.len(),
                self.attempted,
                self.summaries.join("; ")
            )
        }
    }
}

// ── Proxy tools (daemon build only — needs the HTTP MCP client) ──────────────

#[cfg(feature = "daemon")]
pub use proxy::{build_tools_from_discovered, discover_session_tools, discover_session_tools_strict};

#[cfg(feature = "daemon")]
mod proxy {
    use std::collections::HashSet;
    use std::time::Duration;

    use serde_json::{Map, Value};

    use super::{DiscoveredServerTools, DiscoveryFailure, SessionMcpServerCfg};
    use crate::agent::error::AgentResult;
    use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
    use crate::agent::tool_manifest::{
        Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
    };
    use crate::engine::Engine;
    use crate::message::{AkhMessage, MessageSink};
    use crate::upstream_client::UpstreamMcpClient;

    /// Per-server discovery budget. Discovery runs synchronously inside the
    /// prompt turn (see `AkhAcpAgent::run_turn`), so a slow/blackholed server
    /// must not eat the full `session/prompt` round trip the way the client's
    /// [`UpstreamMcpClient::new`]/[`from_endpoint`] 30s default would; a
    /// handful of seconds is plenty for a healthy `initialize` + `tools/list`.
    /// Discovery only ever runs once per session (see the cache in
    /// `run_turn`), so this budget is paid at most once per pushed server.
    ///
    /// [`from_endpoint`]: UpstreamMcpClient::from_endpoint
    const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);

    /// Backoff between discovery retries in [`discover_with_retries`]. Short
    /// enough that a single transient HTTP 401 (the work item's failure mode)
    /// rides through without blowing the prompt turn's idle budget; long
    /// enough that a flap-the-door upstream is not hammered.
    const DISCOVERY_RETRY_BACKOFF: Duration = Duration::from_millis(250);

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
        server: String,
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
                server: server.to_string(),
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
            let client = UpstreamMcpClient::from_endpoint(self.url.clone(), self.auth_header.clone());
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
                source: ToolSource::Mcp { server: self.server.clone() },
            }
        }
    }

    /// Connect to each pushed HTTP MCP server and discover its tools, once.
    ///
    /// Runs blocking network I/O under a tight [`DISCOVERY_TIMEOUT`] per
    /// server (not the client's normal 30s call budget) — call only from a
    /// blocking context (the ADR-027 turn closure), and only when the
    /// session's discovery hasn't already been cached (see `run_turn`).
    ///
    /// Servers are discovered **concurrently**, one OS thread per pushed
    /// server (`std::thread::scope`, joined before returning) rather than
    /// serially: a session with N pushed servers previously paid up to
    /// `N * DISCOVERY_TIMEOUT` wall-clock on the one-time cache-miss path if
    /// several were slow/unreachable, since each `list_tools_raw()` blocked
    /// the next server's attempt. Concurrent discovery bounds the whole pass
    /// to roughly one `DISCOVERY_TIMEOUT` regardless of server count. This
    /// function already only runs from a blocking context (never inside the
    /// async runtime), so spawning plain OS threads for the duration of one
    /// discovery pass is safe and needs no extra runtime.
    ///
    /// A server that fails discovery is skipped, logged via `tracing::warn`,
    /// **and** reported through `sink` as a visible [`AkhMessage::system`] so
    /// the client learns its server was dropped instead of the turn simply
    /// reading as if the server never existed. Output order matches `cfgs`
    /// (independent of which thread finishes first).
    ///
    /// **Legacy behaviour**: callers that want a tool-less session to *fail
    /// the turn* rather than silently continue should use
    /// [`discover_session_tools_strict`] instead — see the work item
    /// "Turns die quietly: … MCP discovery that 401s and 'skips'".
    pub fn discover_session_tools(
        cfgs: &[SessionMcpServerCfg],
        sink: &dyn MessageSink,
    ) -> Vec<DiscoveredServerTools> {
        let (ok, failed, _) = discover_with_retries(cfgs, 0);
        for server in &ok {
            tracing::info!(
                server = %server.name,
                url = %server.url,
                tools = server.descriptors.len(),
                "session MCP server tools discovered (cached for the session)"
            );
        }
        for (name, url, e) in &failed {
            tracing::warn!(
                server = %name,
                url = %url,
                "session MCP discovery failed, skipping: {e}"
            );
            sink.emit(&AkhMessage::system(format!(
                "MCP server \"{name}\" was skipped for this session: tool discovery failed ({e})"
            )));
        }
        ok
    }

    /// Strict variant of [`discover_session_tools`]: a session that loses any
    /// pushed MCP server **fails the turn** rather than continuing tool-less.
    ///
    /// Each failing server gets one bounded retry (after a short backoff) to
    /// ride through the intermittent HTTP 401s the work item documents — a
    /// second failure terminates the session with the upstream status in the
    /// reason. The success path matches [`discover_session_tools`] (caching
    /// the descriptors for the rest of the session); the failure path returns
    /// a [`DiscoveryFailure`] carrying every server that did not come back, so
    /// the caller can stream a single `error[acp]` message and end the turn
    /// with a stop reason Anima can distinguish from a clean completion.
    ///
    /// Call only from a blocking context (see [`discover_session_tools`]).
    pub fn discover_session_tools_strict(
        cfgs: &[SessionMcpServerCfg],
        sink: &dyn MessageSink,
    ) -> Result<Vec<DiscoveredServerTools>, DiscoveryFailure> {
        const RETRIES: usize = 1;

        let (ok, failed, attempted) = discover_with_retries(cfgs, RETRIES);

        // Log every server that came up so the daemon record shows the
        // healthy half of the pass too.
        for server in &ok {
            tracing::info!(
                server = %server.name,
                url = %server.url,
                tools = server.descriptors.len(),
                "session MCP server tools discovered (cached for the session)"
            );
        }

        if failed.is_empty() {
            return Ok(ok);
        }

        // One or more servers did not come back after `RETRIES` retries: fail
        // the session. Emit a per-server `error[acp]` so the upstream status
        // is visible in the thread (the work item requires "the upstream
        // status in the reason"), then return the aggregate failure.
        let mut summaries = Vec::with_capacity(failed.len());
        for (name, url, err) in &failed {
            tracing::warn!(
                server = %name,
                url = %url,
                retries = RETRIES,
                "session MCP discovery failed after retries — failing the session (tool-less continuation is unsafe)"
            );
            sink.emit(&AkhMessage::error(
                "acp",
                format!(
                    "MCP server \"{name}\" ({url}) discovery failed after {RETRIES} retr\
                     {plural}: {err}",
                    plural = if RETRIES == 1 { "y" } else { "ies" }
                ),
            ));
            summaries.push(format!("{name} ({url}): {err}"));
        }

        Err(DiscoveryFailure {
            attempted,
            succeeded: ok.len(),
            summaries,
        })
    }

    /// One pass of concurrent discovery, optionally retrying the failing
    /// servers `retries` more times with a short backoff between passes.
    ///
    /// Returns `(ok, failed, attempted)` where `failed` lists every server
    /// that did not come back after the final retry, and `attempted` is the
    /// total number of servers the caller asked us to reach. The retry loop
    /// only re-attempts the servers that failed the previous pass, so a
    /// session with one flaky server and three healthy ones pays at most
    /// `1 * DISCOVERY_TIMEOUT` per retry, not `4 * DISCOVERY_TIMEOUT`.
    fn discover_with_retries(
        cfgs: &[SessionMcpServerCfg],
        retries: usize,
    ) -> (Vec<DiscoveredServerTools>, Vec<(String, String, String)>, usize) {
        let attempted = cfgs.len();
        if cfgs.is_empty() {
            return (Vec::new(), Vec::new(), 0);
        }

        // First pass: try every server concurrently.
        let mut ok: Vec<DiscoveredServerTools> = Vec::new();
        let mut failed: Vec<(String, String, String)> = Vec::new();
        let (pass_ok, pass_err) = discover_pass(cfgs);
        ok.extend(pass_ok);
        failed.extend(pass_err);

        // Bounded retries: re-attempt only the servers that failed, once per
        // retry, with a short backoff between passes. A pass that fully
        // resolves clears `failed`; a pass that produces new failures
        // replaces `failed` with the still-failing set.
        for _ in 0..retries {
            if failed.is_empty() {
                break;
            }
            std::thread::sleep(DISCOVERY_RETRY_BACKOFF);
            let retry_cfgs: Vec<SessionMcpServerCfg> = failed
                .iter()
                .filter_map(|(name, url, _)| {
                    cfgs.iter()
                        .find(|c| c.name == *name && c.url == *url)
                        .cloned()
                })
                .collect();
            let (retry_ok, retry_err) = discover_pass(&retry_cfgs);
            // Promote any retries that succeeded.
            ok.extend(retry_ok);
            // `failed` becomes whatever is still failing after this retry.
            failed = retry_err;
        }

        (ok, failed, attempted)
    }

    /// One concurrent discovery pass over `cfgs`. Returns the servers that
    /// came up plus the `(name, url, error_text)` triple for each that did
    /// not. Output order matches `cfgs` (independent of which thread finishes
    /// first) so retries re-issue in a stable order.
    fn discover_pass(
        cfgs: &[SessionMcpServerCfg],
    ) -> (Vec<DiscoveredServerTools>, Vec<(String, String, String)>) {
        if cfgs.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let results: Vec<Result<DiscoveredServerTools, (String, String, String)>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = cfgs
                    .iter()
                    .map(|cfg| {
                        scope.spawn(move || {
                            let client = UpstreamMcpClient::from_endpoint_with_timeout(
                                cfg.url.clone(),
                                cfg.auth_header.clone(),
                                DISCOVERY_TIMEOUT,
                            );
                            match client.list_tools_raw() {
                                Ok(descriptors) => Ok(DiscoveredServerTools {
                                    name: cfg.name.clone(),
                                    url: cfg.url.clone(),
                                    auth_header: cfg.auth_header.clone(),
                                    descriptors,
                                }),
                                Err(e) => Err((cfg.name.clone(), cfg.url.clone(), e.to_string())),
                            }
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| {
                        Err((
                            "<unknown>".to_string(),
                            "<unknown>".to_string(),
                            "discovery thread panicked".to_string(),
                        ))
                    }))
                    .collect()
            });

        let mut ok = Vec::new();
        let mut failed = Vec::new();
        for result in results {
            match result {
                Ok(server) => ok.push(server),
                Err(triple) => failed.push(triple),
            }
        }
        (ok, failed)
    }

    /// Build proxy `Tool`s from previously-discovered descriptors — no
    /// network I/O. Called once per prompt turn (cheap JSON→struct work) so
    /// each turn gets fresh `McpProxyTool` instances even though discovery
    /// itself ran once for the whole session.
    pub fn build_tools_from_discovered(discovered: &[DiscoveredServerTools]) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> = Vec::new();
        for server in discovered {
            for descriptor in &server.descriptors {
                if let Some(tool) = McpProxyTool::from_descriptor(
                    &server.name,
                    &server.url,
                    server.auth_header.as_deref(),
                    descriptor,
                ) {
                    tools.push(Box::new(tool));
                }
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

        /// `discover_session_tools_strict` must fail the session when a pushed
        /// MCP server is unreachable, instead of silently skipping it and
        /// continuing tool-less (work item "Turns die quietly" defect 2).
        ///
        /// Uses a port that is practically guaranteed to refuse the TCP
        /// connection (port 1 is unprivileged but no daemon binds it on a
        /// test runner), so `list_tools_raw` returns a transport error and
        /// the strict variant returns `Err(DiscoveryFailure)` after its one
        /// bounded retry. The legacy `discover_session_tools` would have
        /// returned an empty `Vec` here, leaving the session tool-less.
        #[test]
        fn strict_discovery_fails_the_session_when_a_server_is_unreachable() {
            let cfg = SessionMcpServerCfg {
                name: "anima".into(),
                url: "http://127.0.0.1:1/mcp".into(),
                auth_header: None,
            };
            let sink = crate::message::VecSink::new();
            let result = discover_session_tools_strict(&[cfg], &sink);
            assert!(
                result.is_err(),
                "strict discovery must fail when a pushed server is unreachable"
            );
            let failure = result.unwrap_err();
            assert_eq!(failure.attempted, 1);
            assert_eq!(failure.succeeded, 0);
            assert_eq!(failure.summaries.len(), 1);
            assert!(
                failure.summaries[0].contains("anima"),
                "failure summary should name the server: {}",
                failure.summaries[0]
            );
            // The strict variant streams an `error[acp]` per failing server
            // so the upstream status is visible in the thread — verify at
            // least one error message landed in the sink.
            let has_error = sink.messages().iter().any(|m| matches!(
                m,
                crate::message::AkhMessage::Error { code, .. } if code == "acp"
            ));
            assert!(
                has_error,
                "strict discovery should stream an error[acp] message for the failing server"
            );
        }

        /// `discover_session_tools_strict` succeeds when every pushed server
        /// comes up. We can't run a real MCP server in a unit test, but we
        /// can verify the empty-cfgs path: no servers pushed ⇒ no failure,
        /// no tools, no error messages. (The healthy-nonempty path is
        /// exercised by the ACP integration tests that push a real server.)
        #[test]
        fn strict_discovery_with_no_servers_is_ok_with_no_tools() {
            let sink = crate::message::VecSink::new();
            let result = discover_session_tools_strict(&[], &sink);
            assert!(result.is_ok(), "empty cfgs should not fail");
            assert!(result.unwrap().is_empty());
            assert!(
                sink.is_empty(),
                "no messages should be emitted when there are no servers to discover"
            );
        }

        /// `discover_session_tools` (the legacy non-strict variant) preserves
        /// its pre-fix behaviour for back-compat: an unreachable server is
        /// skipped with a `system` message, not a session-failing error.
        #[test]
        fn legacy_discovery_skips_unreachable_server_without_failing() {
            let cfg = SessionMcpServerCfg {
                name: "anima".into(),
                url: "http://127.0.0.1:1/mcp".into(),
                auth_header: None,
            };
            let sink = crate::message::VecSink::new();
            let result = discover_session_tools(&[cfg], &sink);
            assert!(
                result.is_empty(),
                "legacy discovery should return an empty Vec when the server is unreachable"
            );
            // The legacy path emits a `system` (skip) message, not an error.
            let has_system = sink.messages().iter().any(|m| matches!(
                m,
                crate::message::AkhMessage::System { text } if text.contains("skipped")
            ));
            assert!(
                has_system,
                "legacy discovery should emit a 'skipped' system message for the failing server"
            );
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
