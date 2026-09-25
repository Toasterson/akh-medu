//! Per-turn loopback MCP endpoint that carries Akh tools into OpenCode.
//!
//! OpenCode's ACP adapter accepts client-supplied MCP servers but has no
//! client-side tool callback, so the bridge serves the turn's Akh tool catalog
//! as a Streamable-HTTP MCP server on `127.0.0.1:<ephemeral>` guarded by a
//! per-turn bearer token. Only `POST` with JSON responses is implemented (the
//! spec lets a server decline the optional SSE stream with 405). Tool calls
//! are handed to the turn's thread, which executes them through the host
//! broker exactly like Codex dynamic tools; this endpoint never runs anything.
use miette::Diagnostic;
use serde::Serialize;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use thiserror::Error;

const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 8 * 1024 * 1024;
/// Newest first; an unknown requested version is answered with the newest.
const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];
const IO_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Error, Diagnostic)]
pub enum McpError {
    #[error("could not open the loopback MCP endpoint for Akh tools")]
    #[diagnostic(
        code(akh::runtime::opencode::mcp_listen),
        help(
            "the runtime binds 127.0.0.1 on an ephemeral port for each OpenCode turn; allow loopback listening for the runtime process"
        )
    )]
    Listen(#[source] std::io::Error),

    #[error("could not read randomness for a per-turn credential")]
    #[diagnostic(
        code(akh::runtime::opencode::entropy),
        help("/dev/urandom must be readable by the runtime process")
    )]
    Entropy(#[source] std::io::Error),
}

/// One tool in the catalog OpenCode lists.
#[derive(Debug, Clone, Serialize)]
pub(super) struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

impl McpTool {
    /// Tools arrive from the runtime in chat-completions shape.
    pub(super) fn from_function(tool: &Value) -> Self {
        let function = &tool["function"];
        Self {
            name: function["name"].as_str().unwrap_or_default().to_owned(),
            description: function["description"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            input_schema: match &function["parameters"] {
                Value::Object(schema) => Value::Object(schema.clone()),
                _ => json!({"type": "object"}),
            },
        }
    }
}

/// A tool call waiting for the turn's thread.
pub(super) struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub reply: mpsc::SyncSender<ToolReply>,
}

pub(super) enum ToolReply {
    /// A serialized `ToolResult` (success or denial).
    Result(Value),
    /// The turn failed while executing the call; nothing more runs.
    Failed(String),
}

struct Shared {
    tools: Vec<McpTool>,
    authorization: String,
    calls: mpsc::SyncSender<ToolCall>,
    deadline: Instant,
    sequence: AtomicU64,
    /// Set once an authenticated client listed the catalog.
    listed: AtomicBool,
}

pub(super) struct LoopbackMcp {
    url: String,
    token: String,
    calls: mpsc::Receiver<ToolCall>,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    acceptor: Option<JoinHandle<()>>,
}

impl LoopbackMcp {
    pub(super) fn start(tools: Vec<McpTool>, deadline: Instant) -> Result<Self, McpError> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(McpError::Listen)?;
        listener.set_nonblocking(true).map_err(McpError::Listen)?;
        let address = listener.local_addr().map_err(McpError::Listen)?;
        let token = random_token()?;
        let (sender, calls) = mpsc::sync_channel(16);
        let shared = Arc::new(Shared {
            tools,
            authorization: format!("Bearer {token}"),
            calls: sender,
            deadline,
            sequence: AtomicU64::new(0),
            listed: AtomicBool::new(false),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let accepting = shared.clone();
        let acceptor = std::thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let shared = accepting.clone();
                        std::thread::spawn(move || serve(stream, &shared));
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            }
        });
        Ok(Self {
            url: format!("http://{address}/mcp"),
            token,
            calls,
            shared,
            stop,
            acceptor: Some(acceptor),
        })
    }

    /// ACP `McpServer` entry for `session/new` and `session/load`.
    pub(super) fn acp_server(&self, name: &str) -> Value {
        json!({"type": "http", "name": name, "url": self.url,
            "headers": [{"name": "Authorization", "value": format!("Bearer {}", self.token)}]})
    }

    /// Whether OpenCode connected and listed the Akh tools. OpenCode ignores
    /// MCP registration failures, so the bridge checks this itself.
    pub(super) fn catalog_served(&self) -> bool {
        self.shared.listed.load(Ordering::Relaxed)
    }

    pub(super) fn next_call(&self) -> Option<ToolCall> {
        self.calls.try_recv().ok()
    }
}

impl Drop for LoopbackMcp {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(acceptor) = self.acceptor.take() {
            let _ = acceptor.join();
        }
    }
}

/// 256 bits from the OS, hex encoded.
pub(super) fn random_token() -> Result<String, McpError> {
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .map_err(McpError::Entropy)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn same(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

/// A request that is answered without reaching JSON-RPC dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reject {
    BadRequest,
    Unauthorized,
    NotFound,
    MethodNotAllowed,
    LengthRequired,
    TooLarge,
}

impl Reject {
    fn status(self) -> &'static str {
        match self {
            Self::BadRequest => "400 Bad Request",
            Self::Unauthorized => "401 Unauthorized",
            Self::NotFound => "404 Not Found",
            Self::MethodNotAllowed => "405 Method Not Allowed",
            Self::LengthRequired => "411 Length Required",
            Self::TooLarge => "413 Payload Too Large",
        }
    }
}

struct Head {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
}

impl Head {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

fn read_head(reader: &mut impl BufRead) -> Result<Head, Reject> {
    let mut lines = Vec::new();
    let mut total = 0;
    loop {
        let mut line = String::new();
        // Bound each line too, so one endless header cannot grow the buffer.
        let read = Read::take(&mut *reader, (MAX_HEAD - total + 1) as u64)
            .read_line(&mut line)
            .map_err(|_| Reject::BadRequest)?;
        total += read;
        if read == 0 || total > MAX_HEAD {
            return Err(Reject::BadRequest);
        }
        let line = line.trim_end_matches(['\r', '\n']).to_owned();
        if line.is_empty() {
            break;
        }
        lines.push(line);
    }
    let mut request_line = lines.first().ok_or(Reject::BadRequest)?.split(' ');
    let method = request_line.next().ok_or(Reject::BadRequest)?.to_owned();
    let target = request_line.next().ok_or(Reject::BadRequest)?;
    let path = target.split('?').next().unwrap_or_default().to_owned();
    let headers = lines[1..]
        .iter()
        .map(|line| {
            line.split_once(':')
                .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
                .ok_or(Reject::BadRequest)
        })
        .collect::<Result<_, _>>()?;
    Ok(Head {
        method,
        path,
        headers,
    })
}

enum Answer {
    Json(Value),
    Accepted,
}

fn read_message(reader: &mut impl BufRead, shared: &Shared) -> Result<Answer, Reject> {
    let head = read_head(reader)?;
    if head.path != "/mcp" {
        return Err(Reject::NotFound);
    }
    let authorized = head
        .header("authorization")
        .is_some_and(|value| same(value.as_bytes(), shared.authorization.as_bytes()));
    if !authorized {
        return Err(Reject::Unauthorized);
    }
    if head.method != "POST" {
        return Err(Reject::MethodNotAllowed);
    }
    if head.header("transfer-encoding").is_some() {
        return Err(Reject::LengthRequired);
    }
    let length: usize = head
        .header("content-length")
        .ok_or(Reject::LengthRequired)?
        .parse()
        .map_err(|_| Reject::BadRequest)?;
    if length > MAX_BODY {
        return Err(Reject::TooLarge);
    }
    let mut body = vec![0; length];
    reader
        .read_exact(&mut body)
        .map_err(|_| Reject::BadRequest)?;
    let message: Value = match serde_json::from_slice(&body) {
        Ok(message) => message,
        Err(_) => {
            return Ok(Answer::Json(
                json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}}),
            ));
        }
    };
    Ok(dispatch(shared, &message).map_or(Answer::Accepted, Answer::Json))
}

fn serve(stream: TcpStream, shared: &Shared) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(read_half);
    let (status, body, extra) = match read_message(&mut reader, shared) {
        Ok(Answer::Json(value)) => ("200 OK", value.to_string(), ""),
        Ok(Answer::Accepted) => ("202 Accepted", String::new(), ""),
        Err(reject @ Reject::MethodNotAllowed) => {
            (reject.status(), String::new(), "Allow: POST\r\n")
        }
        Err(reject) => (reject.status(), String::new(), ""),
    };
    let content_type = if body.is_empty() {
        ""
    } else {
        "Content-Type: application/json\r\n"
    };
    let mut writer = stream;
    let _ = write!(
        writer,
        "HTTP/1.1 {status}\r\n{content_type}{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = writer.flush();
}

/// JSON-RPC dispatch. `None` for notifications and client responses.
fn dispatch(shared: &Shared, message: &Value) -> Option<Value> {
    let id = message.get("id").filter(|id| !id.is_null())?.clone();
    let Some(method) = message["method"].as_str() else {
        // A response to a request this server never sends.
        return None;
    };
    let params = &message["params"];
    let outcome = match method {
        "initialize" => {
            let requested = params["protocolVersion"].as_str().unwrap_or_default();
            let version = PROTOCOL_VERSIONS
                .iter()
                .find(|v| **v == requested)
                .unwrap_or(&PROTOCOL_VERSIONS[0]);
            Ok(json!({"protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "akh", "title": "Akh-Medu", "version": env!("CARGO_PKG_VERSION")}}))
        }
        "ping" => Ok(json!({})),
        "tools/list" => {
            shared.listed.store(true, Ordering::Relaxed);
            Ok(json!({"tools": shared.tools}))
        }
        "tools/call" => call(shared, params),
        _ => Err((-32601, "method not found".to_owned())),
    };
    Some(match outcome {
        Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
        Err((code, message)) => {
            json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
        }
    })
}

fn call(shared: &Shared, params: &Value) -> Result<Value, (i64, String)> {
    let name = params["name"]
        .as_str()
        .ok_or((-32602, "tools/call requires a name".to_owned()))?;
    let arguments = match &params["arguments"] {
        Value::Null => json!({}),
        other => other.clone(),
    };
    let (reply, answer) = mpsc::sync_channel(1);
    let sequence = shared.sequence.fetch_add(1, Ordering::Relaxed);
    shared
        .calls
        .send(ToolCall {
            id: format!("mcp-{sequence}"),
            name: name.to_owned(),
            arguments,
            reply,
        })
        .map_err(|_| (-32603, "the Akh turn has ended".to_owned()))?;
    let wait = shared.deadline.saturating_duration_since(Instant::now());
    match answer.recv_timeout(wait) {
        Ok(ToolReply::Result(result)) => {
            // Structured results (process handles, receipts) must survive
            // MCP's text content, as with the Codex and Claude bridges.
            let text = if result["data"].is_null() {
                result["text"].as_str().unwrap_or_default().to_owned()
            } else {
                result.to_string()
            };
            Ok(json!({"content": [{"type": "text", "text": text}],
                "isError": result["success"] != true}))
        }
        Ok(ToolReply::Failed(message)) => Err((-32603, message)),
        Err(_) => Err((
            -32603,
            "the Akh turn ended before the tool completed".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(url: &str, request: &str) -> (u16, String) {
        let address = url
            .strip_prefix("http://")
            .unwrap()
            .split('/')
            .next()
            .unwrap();
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let status = response[9..12].parse().unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(head.contains("Connection: close"));
        (status, format!("{head}\r\n\r\n{body}"))
    }

    fn post(url: &str, authorization: Option<&str>, body: &Value) -> (u16, Value) {
        let body = body.to_string();
        let auth = authorization
            .map(|a| format!("Authorization: {a}\r\n"))
            .unwrap_or_default();
        let (status, response) = raw(
            url,
            &format!(
                "POST /mcp HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        );
        let body = response.split_once("\r\n\r\n").unwrap().1;
        (status, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    fn server() -> LoopbackMcp {
        LoopbackMcp::start(
            vec![McpTool::from_function(&json!({"function":{"name":"echo",
                "description":"echo","parameters":{"type":"object"}}}))],
            Instant::now() + Duration::from_secs(5),
        )
        .unwrap()
    }

    #[test]
    fn serves_the_catalog_and_hands_calls_to_the_turn_thread() {
        let server = server();
        let bearer = format!("Bearer {}", server.token);
        let (status, init) = post(
            &server.url,
            Some(&bearer),
            &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}),
        );
        assert_eq!(status, 200);
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(
            init["result"]["capabilities"]["tools"]["listChanged"],
            false
        );
        let notification = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
        assert_eq!(post(&server.url, Some(&bearer), &notification).0, 202);
        let (_, list) = post(
            &server.url,
            Some(&bearer),
            &json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        );
        assert_eq!(list["result"]["tools"][0]["name"], "echo");
        assert_eq!(list["result"]["tools"][0]["inputSchema"]["type"], "object");
        assert!(server.catalog_served());

        let caller = {
            let (url, bearer) = (server.url.clone(), bearer.clone());
            std::thread::spawn(move || {
                post(
                    &url,
                    Some(&bearer),
                    &json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
                    "params":{"name":"echo","arguments":{"n":7}}}),
                )
            })
        };
        let call = loop {
            if let Some(call) = server.next_call() {
                break call;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(call.name, "echo");
        assert_eq!(call.arguments["n"], 7);
        call.reply
            .send(ToolReply::Result(
                json!({"success":true,"text":"7","data":{"process_id":"p"}}),
            ))
            .unwrap();
        let (_, result) = caller.join().unwrap();
        let text: Value =
            serde_json::from_str(result["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text["data"]["process_id"], "p");
        assert_eq!(result["result"]["isError"], false);
    }

    #[test]
    fn refuses_other_callers_methods_and_ended_turns() {
        let server = server();
        let request = json!({"jsonrpc":"2.0","id":1,"method":"tools/list"});
        assert_eq!(post(&server.url, None, &request).0, 401);
        assert_eq!(post(&server.url, Some("Bearer wrong"), &request).0, 401);
        assert!(
            !server.catalog_served(),
            "unauthenticated listing does not count"
        );
        let bearer = format!("Bearer {}", server.token);
        let (_, error) = post(
            &server.url,
            Some(&bearer),
            &json!({"jsonrpc":"2.0","id":9,"method":"resources/list"}),
        );
        assert_eq!(error["error"]["code"], -32601);
        let (status, response) = raw(
            &server.url,
            &format!("GET /mcp HTTP/1.1\r\nAuthorization: {bearer}\r\n\r\n"),
        );
        assert_eq!(status, 405);
        assert!(response.contains("Allow: POST"));
        let (status, _) = raw(
            &server.url,
            &format!(
                "POST /other HTTP/1.1\r\nAuthorization: {bearer}\r\nContent-Length: 0\r\n\r\n"
            ),
        );
        assert_eq!(status, 404);

        // Once the turn is gone, a late call fails instead of hanging.
        let shared = Shared {
            tools: vec![],
            authorization: String::new(),
            calls: mpsc::sync_channel(1).0,
            deadline: Instant::now() + Duration::from_secs(1),
            sequence: AtomicU64::new(0),
            listed: AtomicBool::new(false),
        };
        let error = call(&shared, &json!({"name":"echo"})).unwrap_err();
        assert_eq!(error.0, -32603);
    }

    #[test]
    fn tokens_are_fresh_and_compared_in_full() {
        let (a, b) = (random_token().unwrap(), random_token().unwrap());
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"ab"));
    }
}
