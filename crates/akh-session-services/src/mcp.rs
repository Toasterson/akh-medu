//! Bounded Streamable HTTP tool calls, with SDK-owned protocol negotiation.
//! A transport reconnect may GET a response stream; it never repeats tools/call.
use crate::{config::Endpoint, types::*};
use rmcp::model::ProtocolVersion;
use serde_json::{Value, json};
use std::time::Duration;

const LIMIT: usize = 8 * 1024 * 1024;
struct Peer {
    endpoint: Endpoint,
    token: String,
    client: reqwest::Client,
    session: Option<String>,
    version: Option<String>,
}
impl Peer {
    fn request(&self, method: reqwest::Method) -> Result<reqwest::RequestBuilder> {
        let mut request = self
            .client
            .request(method, &self.endpoint.url)
            .bearer_auth(&self.token)
            .header("Accept", "application/json, text/event-stream");
        if let Some(session) = &self.session {
            request = request.header("MCP-Session-Id", session);
        }
        if let Some(version) = &self.version {
            request = request.header("MCP-Protocol-Version", version);
        }
        Ok(request)
    }
    async fn notify(&self, value: Value) -> Result<()> {
        let response = self
            .request(reqwest::Method::POST)?
            .json(&value)
            .send()
            .await
            .map_err(transport)?;
        if response.status() != reqwest::StatusCode::ACCEPTED {
            return Err(Fault::new(
                "mcp_protocol",
                "MCP notification/response was not accepted",
            ));
        }
        Ok(())
    }
    async fn message(&self, value: Value, expected: &Value) -> Result<Option<Value>> {
        if value["jsonrpc"] != "2.0" {
            return Err(Fault::new("mcp_protocol", "invalid MCP JSON-RPC version"));
        }
        if value.get("method").is_some() {
            if value.get("id").is_some() {
                let response = if value["method"] == "ping" {
                    json!({"jsonrpc":"2.0","id":value["id"],"result":{}})
                } else {
                    // Sampling, elicitation and roots are not advertised by this bounded tool client.
                    json!({"jsonrpc":"2.0","id":value["id"],"error":{"code":-32601,"message":"Client capability not advertised"}})
                };
                self.notify(response).await?;
            }
            return Ok(None);
        }
        if &value["id"] != expected
            || (value.get("result").is_some() == value.get("error").is_some())
        {
            return Err(Fault::new(
                "mcp_protocol",
                "MCP response does not match the outstanding request",
            ));
        }
        Ok(Some(value))
    }
    async fn rpc(&mut self, value: Value) -> Result<Value> {
        let expected = value["id"].clone();
        let mut response = self
            .request(reqwest::Method::POST)?
            .json(&value)
            .send()
            .await
            .map_err(transport)?;
        let mut total = 0;
        let mut last_event = None::<String>;
        let mut delay = 1000;
        for attempt in 0..=3 {
            if !response.status().is_success() {
                return Err(Fault::new(
                    "mcp_protocol",
                    format!(
                        "MCP returned HTTP {}; request will not be reposted",
                        response.status().as_u16()
                    ),
                ));
            }
            if value["method"] == "initialize" && attempt == 0 {
                if let Some(header) = response.headers().get("MCP-Session-Id") {
                    let session = header
                        .to_str()
                        .map_err(|_| Fault::invalid("invalid MCP session header"))?;
                    if session.is_empty()
                        || session.len() > 4096
                        || !session.bytes().all(|b| (0x21..=0x7e).contains(&b))
                    {
                        return Err(Fault::invalid("invalid MCP session identifier"));
                    }
                    self.session = Some(session.into());
                }
            }
            let content = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_owned();
            if content.starts_with("application/json") {
                let mut bytes = vec![];
                while let Some(chunk) = response.chunk().await.map_err(transport)? {
                    total += chunk.len();
                    if total > LIMIT {
                        return Err(Fault::new("capacity", "MCP response exceeds byte limit"));
                    }
                    bytes.extend_from_slice(&chunk);
                }
                let value = serde_json::from_slice(&bytes)
                    .map_err(|_| Fault::invalid("invalid MCP JSON"))?;
                return self.message(value, &expected).await?.ok_or_else(|| {
                    Fault::new("mcp_protocol", "MCP JSON response has no request result")
                });
            }
            if !content.starts_with("text/event-stream") {
                return Err(Fault::new(
                    "mcp_protocol",
                    "MCP response must be JSON or SSE",
                ));
            }
            let mut buffer = vec![];
            let mut data = String::new();
            let mut pending_event = None::<Option<String>>;
            let mut skip_lf = false;
            let mut first_line = true;
            loop {
                let chunk = match response.chunk().await {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => break,
                    Err(_) if last_event.is_some() => break,
                    Err(error) => return Err(transport(error)),
                };
                total += chunk.len();
                if total > LIMIT {
                    return Err(Fault::new("capacity", "MCP stream exceeds byte limit"));
                }
                buffer.extend_from_slice(&chunk);
                if skip_lf {
                    if buffer.first() == Some(&b'\n') {
                        buffer.remove(0);
                    }
                    skip_lf = false;
                }
                while let Some(end) = buffer.iter().position(|b| matches!(b, b'\n' | b'\r')) {
                    let cr = buffer[end] == b'\r';
                    let consumed = if cr && buffer.get(end + 1) == Some(&b'\n') {
                        end + 2
                    } else {
                        end + 1
                    };
                    skip_lf = cr && consumed == buffer.len() && consumed == end + 1;
                    let line = std::str::from_utf8(&buffer[..end])
                        .map_err(|_| Fault::invalid("MCP stream is not UTF-8"))?
                        .trim_end_matches('\r');
                    let line = if first_line {
                        first_line = false;
                        line.strip_prefix('\u{feff}').unwrap_or(line)
                    } else {
                        line
                    };
                    if line.is_empty() {
                        if let Some(id) = pending_event.take() {
                            last_event = id;
                        }
                        if !data.trim().is_empty() {
                            let value = serde_json::from_str(&data)
                                .map_err(|_| Fault::invalid("MCP SSE event is invalid JSON"))?;
                            if let Some(reply) = self.message(value, &expected).await? {
                                return Ok(reply);
                            }
                        }
                        data.clear();
                    } else if let Some(value) = line.strip_prefix("data:") {
                        data.push_str(value.strip_prefix(' ').unwrap_or(value));
                        data.push('\n');
                    } else if let Some(value) = line.strip_prefix("id:") {
                        let value = value.strip_prefix(' ').unwrap_or(value);
                        if value.len() > 4096 || value.contains(['\0', '\r']) {
                            return Err(Fault::invalid("invalid MCP event ID"));
                        }
                        pending_event = Some(if value.is_empty() {
                            None
                        } else {
                            Some(value.into())
                        });
                    } else if let Some(value) = line.strip_prefix("retry:") {
                        if let Ok(value) = value.trim().parse::<u64>() {
                            delay = value;
                        }
                    }
                    buffer.drain(..consumed);
                }
            }
            if attempt == 3 || last_event.is_none() || delay > 30_000 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(delay)).await;
            // Resumption retrieves the response to the original call, without repeating its effects.
            response = self
                .request(reqwest::Method::GET)?
                .header("Last-Event-ID", last_event.as_ref().unwrap())
                .send()
                .await
                .map_err(transport)?;
        }
        Err(Fault::new(
            "transport",
            "MCP stream ended without a result; tool call will not be repeated",
        ))
    }
}
fn transport(_: reqwest::Error) -> Fault {
    Fault::new(
        "transport",
        "MCP transport failed; effects may have occurred",
    )
}

pub async fn call(
    endpoint: &Endpoint,
    id: &str,
    tool: &str,
    arguments: Value,
) -> Result<ToolResult> {
    call_with_token(endpoint, endpoint.token()?, id, tool, arguments).await
}

pub async fn call_with_token(
    endpoint: &Endpoint,
    token: String,
    id: &str,
    tool: &str,
    arguments: Value,
) -> Result<ToolResult> {
    let mut peer = Peer {
        endpoint: endpoint.clone(),
        token,
        client: endpoint.http_client()?,
        session: None,
        version: None,
    };
    let work = async {
        let init=peer.rpc(json!({"jsonrpc":"2.0","id":format!("init-{id}"),"method":"initialize","params":{"protocolVersion":ProtocolVersion::LATEST,"capabilities":{},"clientInfo":{"name":"akh-session-broker","version":env!("CARGO_PKG_VERSION")}}})).await?;
        let selected: ProtocolVersion =
            serde_json::from_value(init["result"]["protocolVersion"].clone()).map_err(|_| {
                Fault::new(
                    "mcp_protocol",
                    "MCP initialization returned no protocol version",
                )
            })?;
        if !ProtocolVersion::KNOWN_VERSIONS.contains(&selected)
            || selected == ProtocolVersion::V_2024_11_05
        {
            return Err(Fault::new(
                "unsupported",
                "server selected an unsupported Streamable HTTP version",
            ));
        }
        peer.version = Some(selected.to_string());
        if !init["result"]["capabilities"]["tools"].is_object() {
            return Err(Fault::new(
                "unsupported",
                "MCP server did not advertise tools",
            ));
        }
        peer.notify(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
            .await?;
        // Metadata binds opt-in server receipts to this durable broker effect.
        // It is separate from model arguments and does not authorize POST replay.
        let reply=peer.rpc(json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":tool,"arguments":arguments,"_meta":{"it.wegmueller.akh/operation-id":id}}})).await?;
        let result = if reply.get("error").is_some() {
            ToolResult {
                success: false,
                // Tool errors are actionable model input, just like successful
                // tool content. Keep the server's explanation but never echo
                // this peer's bearer credential into a durable transcript.
                text: reply["error"]["message"]
                    .as_str()
                    .unwrap_or("MCP server rejected the tool request")
                    .replace(&peer.token, "[redacted]"),
                data: Some(json!({"code":"mcp_error", "rpc_code":reply["error"]["code"].as_i64()})),
                receipt: None,
            }
        } else {
            let result: rmcp::model::CallToolResult =
                serde_json::from_value(reply["result"].clone()).map_err(|_| {
                    Fault::new(
                        "mcp_protocol",
                        "MCP tool result is invalid for the negotiated protocol",
                    )
                })?;
            ToolResult {
                success: !result.is_error.unwrap_or(false),
                text: reply["result"]["content"].to_string(),
                data: Some(reply["result"].clone()),
                receipt: None,
            }
        };
        if peer.session.is_some() {
            // Only close after a known result. Lost results keep their server-side evidence.
            let _ = tokio::time::timeout(
                Duration::from_secs(2),
                peer.request(reqwest::Method::DELETE)?.send(),
            )
            .await;
        }
        Ok(result)
    };
    tokio::time::timeout(Duration::from_secs(90), work)
        .await
        .unwrap_or_else(|_| {
            Err(Fault::new(
                "transport",
                "MCP exchange timed out; request will not be repeated",
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        extract::State,
        http::{HeaderMap, StatusCode},
        response::{IntoResponse, Response},
        routing::post,
    };
    use std::sync::{Arc, Mutex};
    #[derive(Default)]
    struct Trace {
        posts: Vec<Value>,
        gets: Vec<String>,
        deletes: usize,
        mode: u8,
    }
    async fn post_message(
        State(state): State<Arc<Mutex<Trace>>>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response {
        assert!(
            headers["authorization"]
                .to_str()
                .unwrap()
                .starts_with("Bearer fixture-mcp-")
        );
        assert!(
            headers["accept"]
                .to_str()
                .unwrap()
                .contains("text/event-stream")
        );
        let mut trace = state.lock().unwrap();
        trace.posts.push(body.clone());
        if body["method"] == "initialize" {
            return ([("MCP-Session-Id","owned-session")],Json(json!({"jsonrpc":"2.0","id":body["id"],"result":{"protocolVersion":ProtocolVersion::LATEST,"capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1"}}}))).into_response();
        }
        assert_eq!(headers["mcp-session-id"], "owned-session");
        assert_eq!(
            headers["mcp-protocol-version"],
            ProtocolVersion::LATEST.as_str()
        );
        if body["method"] == "notifications/initialized" || body.get("method").is_none() {
            return StatusCode::ACCEPTED.into_response();
        }
        assert_eq!(body["method"], "tools/call");
        assert_eq!(
            body["params"]["_meta"]["it.wegmueller.akh/operation-id"],
            "operation"
        );
        assert_eq!(body["params"]["arguments"], json!({"n":7}));
        assert!(
            trace
                .posts
                .iter()
                .any(|p| p["method"] == "notifications/initialized")
        );
        if trace.mode == 0 {
            return Json(json!({"jsonrpc":"2.0","id":body["id"],"result":{"content":[{"type":"text","text":"done"}],"isError":false}})).into_response();
        }
        if trace.mode == 2 {
            return ([("content-type", "text/event-stream")], "data: \n\n").into_response();
        }
        if trace.mode == 3 {
            return Json(json!({"jsonrpc":"2.0","id":body["id"],"error":{"code":-32602,"message":"Missing grep evidence; credential fixture-mcp-token-0123456789012345"}})).into_response();
        }
        (
            [("content-type", "text/event-stream")],
            "id: completed-event\r\ndata: \r\n\r\nid: incomplete-event\r\ndata: {",
        )
            .into_response()
    }
    async fn get_stream(State(state): State<Arc<Mutex<Trace>>>, headers: HeaderMap) -> Response {
        let mut trace = state.lock().unwrap();
        trace
            .gets
            .push(headers["last-event-id"].to_str().unwrap().into());
        let call = trace
            .posts
            .iter()
            .find(|p| p["method"] == "tools/call")
            .unwrap();
        let result = json!({"jsonrpc":"2.0","id":call["id"],"result":{"content":[{"type":"text","text":"resumed"}],"isError":false}});
        ([("content-type","text/event-stream")],format!("data: {{\"jsonrpc\":\"2.0\",\"id\":\"server-ping\",\"method\":\"ping\"}}\n\ndata: {result}\n\n")).into_response()
    }
    async fn delete_session(
        State(state): State<Arc<Mutex<Trace>>>,
        headers: HeaderMap,
    ) -> StatusCode {
        assert_eq!(headers["mcp-session-id"], "owned-session");
        state.lock().unwrap().deletes += 1;
        StatusCode::NO_CONTENT
    }
    #[tokio::test]
    async fn negotiates_json_and_sse_and_resumes_only_with_get_without_reposting_tools() {
        for mode in 0..=3 {
            let trace = Arc::new(Mutex::new(Trace {
                mode,
                ..Default::default()
            }));
            let app = Router::new()
                .route(
                    "/mcp",
                    post(post_message).get(get_stream).delete(delete_session),
                )
                .with_state(trace.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/mcp", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let dir = tempfile::tempdir().unwrap();
            let token_file = dir.path().join("token");
            std::fs::write(&token_file, "fixture-mcp-token-0123456789012345").unwrap();
            let result = call(
                &Endpoint {
                    url,
                    token_file,
                    ca_file: None,
                    allow_insecure_http: false,
                    request_timeout_secs: None,
                },
                "operation",
                "tool",
                json!({"n":7}),
            )
            .await;
            let trace = trace.lock().unwrap();
            assert_eq!(
                trace
                    .posts
                    .iter()
                    .filter(|p| p["method"] == "tools/call")
                    .count(),
                1
            );
            if mode == 2 {
                assert!(result.is_err());
                assert_eq!(trace.deletes, 0);
            } else if mode == 3 {
                let result = result.unwrap();
                assert!(!result.success);
                assert_eq!(result.text, "Missing grep evidence; credential [redacted]");
                assert_eq!(result.data.unwrap()["rpc_code"], -32602);
                assert_eq!(trace.deletes, 1);
            } else {
                assert!(result.unwrap().success);
                assert_eq!(trace.deletes, 1);
            }
            if mode == 1 {
                assert_eq!(trace.gets, vec!["completed-event"]);
                assert!(
                    trace
                        .posts
                        .iter()
                        .any(|p| p["id"] == "server-ping" && p.get("result").is_some())
                );
            }
            server.abort();
        }
    }
}
