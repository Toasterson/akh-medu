//! ACP stdio bridge for upstream editors that launch agent commands.
//!
//! Two targets:
//!
//! - **akhsessd** (default): stdio to the session service's `/acp` WebSocket.
//!   The bearer token authenticates the WebSocket handshake; `--akh` and
//!   `--workspace-profile` become `_meta.akh` / `_meta.workspaceProfile` on
//!   `session/new`.
//! - **akhomed** (`--akhomed`, Phase 53d): stdio to akhomed's own ACP TCP
//!   listener (`tcp://host:port`, newline-delimited JSON-RPC). akhomed does not
//!   read `_meta.akh`: it authenticates `initialize` by
//!   `_meta.authorization = "Bearer <token>"` and routes a session by
//!   `_meta["akh/workspace"]`. The bridge injects both, so a T3 entrypoint only
//!   swaps `--endpoint` and adds `--akhomed`. The token is the one of an
//!   `[[acp.principals]]` entry whose `akhs` include `--akh` (a scoped
//!   principal is refused with `akh::acp::workspace_not_granted` on any other
//!   akh), or the legacy unscoped `AKHOMED_SHARED_SECRET` when akhomed has no
//!   principals configured.
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::{collections::HashSet, path::PathBuf};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt};
use tokio_tungstenite::{
    Connector, connect_async_tls_with_config,
    tungstenite::{Message, client::IntoClientRequest},
};

#[derive(Parser)]
struct Args {
    /// `wss://…/acp` (akhsessd) or, with `--akhomed`, `tcp://host:port`.
    #[arg(long)]
    endpoint: String,
    #[arg(long)]
    token_file: PathBuf,
    /// Optional private CA bundle for a work/homelab WSS ingress.
    #[arg(long)]
    ca_file: Option<PathBuf>,
    #[arg(long)]
    akh: Option<String>,
    /// akhsessd only; akhomed has no workspace profiles and ignores it.
    #[arg(long)]
    workspace_profile: Option<String>,
    #[arg(long)]
    session: Option<String>,
    #[arg(long)]
    allow_insecure_ws: bool,
    /// Target akhomed's ACP TCP listener instead of akhsessd's WebSocket.
    #[arg(long)]
    akhomed: bool,
    /// With `--akhomed`: permit a non-loopback plaintext TCP endpoint. The
    /// shared secret then crosses the network unencrypted; use only on a
    /// trusted private link.
    #[arg(long)]
    allow_remote_tcp: bool,
    /// T3's Grok ACP driver authenticates again after transport authentication.
    /// Acknowledge its cached-token method only after the bearer-authenticated
    /// WebSocket handshake has succeeded. No client filesystem RPCs are enabled.
    #[arg(long)]
    t3_compat: bool,
}

type BoxError = Box<dyn std::error::Error>;

/// Which ACP server the bridge talks to; decides the `_meta` keys.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    SessionService,
    Akhomed,
}

/// What to do with one message read from the editor.
enum Outgoing {
    /// Send it to the server.
    Forward(Value),
    /// Answer the editor locally; nothing goes to the server.
    Reply(Value),
}

/// Editor-side protocol adaptation shared by both transports.
struct Rewriter<'a> {
    args: &'a Args,
    target: Target,
    /// `Bearer <token>`, injected into akhomed's `initialize`.
    bearer: String,
    /// Request ids of `session/new` calls rewritten to `session/load`.
    rewritten: HashSet<String>,
}

impl Rewriter<'_> {
    fn outgoing(&mut self, mut message: Value) -> Outgoing {
        let method = message["method"].as_str().map(str::to_owned);
        if self.args.t3_compat && matches!(method.as_deref(), Some("session/new" | "session/load"))
        {
            // T3 injects its own MCP server. Roman sessions use only the
            // operator's daemon grants; never forward editor credentials
            // or capabilities that can expose its project filesystem.
            message["params"]["mcpServers"] = json!([]);
            if let Some(params) = message["params"].as_object_mut() {
                params.remove("additionalDirectories");
            }
        }
        if self.args.t3_compat
            && method.as_deref() == Some("authenticate")
            && message["params"]["methodId"] == "cached_token"
        {
            return Outgoing::Reply(json!({"jsonrpc":"2.0","id":message["id"],"result":{}}));
        }
        if self.target == Target::Akhomed && method.as_deref() == Some("initialize") {
            message["params"]["_meta"]["authorization"] = json!(self.bearer);
        }
        if method.as_deref() == Some("session/new") {
            if let Some(session) = &self.args.session {
                self.rewritten.insert(message["id"].to_string());
                message["method"] = json!("session/load");
                message["params"]["sessionId"] = json!(session);
            } else {
                match self.target {
                    Target::SessionService => {
                        if let Some(akh) = &self.args.akh {
                            message["params"]["_meta"]["akh"] = json!(akh);
                        }
                        if let Some(profile) = &self.args.workspace_profile {
                            message["params"]["_meta"]["workspaceProfile"] = json!(profile);
                        }
                    }
                    Target::Akhomed => {
                        if let Some(akh) = &self.args.akh {
                            message["params"]["_meta"]["akh/workspace"] = json!(akh);
                        }
                    }
                }
            }
        }
        Outgoing::Forward(message)
    }

    fn incoming(&mut self, mut message: Value) -> Value {
        if self.rewritten.remove(&message["id"].to_string()) && message.get("result").is_some() {
            message["result"]["sessionId"] = json!(self.args.session);
        }
        message
    }
}

fn is_loopback(url: &url::Url) -> bool {
    matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let url = url::Url::parse(&args.endpoint)?;
    let token = akh_session_services::config::read_token(&args.token_file)?;
    if url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err("the endpoint may not embed credentials, a query or a fragment".into());
    }
    if args.akhomed {
        if url.scheme() != "tcp" || url.port().is_none() {
            return Err(
                "--akhomed needs a tcp://host:port endpoint (akhomed's ACP TCP listener)".into(),
            );
        }
        if !is_loopback(&url) && !args.allow_remote_tcp {
            return Err(
                "akhomed ACP TCP is plaintext; use a loopback endpoint or pass --allow-remote-tcp for a trusted private link"
                    .into(),
            );
        }
        if args.ca_file.is_some() {
            return Err("--ca-file applies to WSS only".into());
        }
        return run_akhomed(&args, &url, token).await;
    }
    if !matches!(url.scheme(), "ws" | "wss")
        || (url.scheme() == "ws" && !is_loopback(&url) && !args.allow_insecure_ws)
    {
        return Err(
            "use WSS or explicitly authorized private WS, without embedded credentials/query"
                .into(),
        );
    }
    run_session_service(&args, &url, token).await
}

async fn run_akhomed(args: &Args, url: &url::Url, token: String) -> Result<(), BoxError> {
    let host = url.host_str().unwrap_or_default().trim_matches(['[', ']']);
    let port = url.port().unwrap_or_default();
    let stream = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::net::TcpStream::connect((host, port)),
    )
    .await??;
    let _ = stream.set_nodelay(true);
    let (read_half, mut write_half) = stream.into_split();
    let mut server = tokio::io::BufReader::new(read_half);
    let mut input = tokio::io::BufReader::new(tokio::io::stdin());
    let mut output = tokio::io::stdout();
    let mut rewriter = Rewriter {
        args,
        target: Target::Akhomed,
        bearer: format!("Bearer {token}"),
        rewritten: HashSet::new(),
    };
    let (mut line, mut reply) = (Vec::new(), Vec::new());
    loop {
        tokio::select! {
            read = read_frame(&mut input, &mut line, 2 * 1024 * 1024) => {
                if read? == 0 { break; }
                let message: Value = serde_json::from_slice(&line)?;
                line.clear();
                match rewriter.outgoing(message) {
                    Outgoing::Reply(answer) => write_line(&mut output, &answer).await?,
                    Outgoing::Forward(message) => write_line(&mut write_half, &message).await?,
                }
            }
            read = read_frame(&mut server, &mut reply, 8 * 1024 * 1024) => {
                if read? == 0 { break; }
                let message: Value = serde_json::from_slice(&reply)?;
                reply.clear();
                write_line(&mut output, &rewriter.incoming(message)).await?;
            }
        }
    }
    write_half.shutdown().await?;
    Ok(())
}

async fn run_session_service(args: &Args, url: &url::Url, token: String) -> Result<(), BoxError> {
    let mut request = args.endpoint.as_str().into_client_request()?;
    request
        .headers_mut()
        .insert("authorization", format!("Bearer {token}").parse()?);
    let connector = if let Some(path) = &args.ca_file {
        if url.scheme() != "wss" {
            return Err("--ca-file requires WSS".into());
        }
        let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut roots = rustls::RootCertStore::empty();
        for certificate in rustls_pemfile::certs(&mut reader) {
            roots.add(certificate?)?;
        }
        if roots.is_empty() {
            return Err("private CA bundle contains no certificates".into());
        }
        let tls = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_no_client_auth();
        Some(Connector::Rustls(std::sync::Arc::new(tls)))
    } else {
        None
    };
    let ws_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(8 * 1024 * 1024))
        .max_frame_size(Some(8 * 1024 * 1024));
    let (socket, _) = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        connect_async_tls_with_config(request, Some(ws_config), false, connector),
    )
    .await??;
    let (mut outgoing, mut incoming) = socket.split();
    let mut input = tokio::io::BufReader::new(tokio::io::stdin());
    let mut output = tokio::io::stdout();
    let mut rewriter = Rewriter {
        args,
        target: Target::SessionService,
        bearer: String::new(),
        rewritten: HashSet::new(),
    };
    let mut line = Vec::new();
    loop {
        tokio::select! {
            read = read_frame(&mut input, &mut line, 2 * 1024 * 1024) => {
                if read? == 0 { break; }
                let message: Value = serde_json::from_slice(&line)?;
                line.clear();
                match rewriter.outgoing(message) {
                    Outgoing::Reply(answer) => write_line(&mut output, &answer).await?,
                    Outgoing::Forward(message) => {
                        outgoing.send(Message::Text(message.to_string().into())).await?
                    }
                }
            }
            frame = incoming.next() => match frame {
                Some(Ok(Message::Text(text))) => {
                    let message: Value = serde_json::from_str(&text)?;
                    write_line(&mut output, &rewriter.incoming(message)).await?;
                }
                Some(Ok(Message::Ping(bytes))) => outgoing.send(Message::Pong(bytes)).await?,
                Some(Ok(Message::Pong(_))) => {}
                Some(Err(error)) => return Err(error.into()),
                _ => break,
            },
        }
    }
    outgoing.close().await?;
    Ok(())
}

async fn write_line(
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    message: &Value,
) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(message)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await
}

async fn read_frame(
    reader: &mut (impl AsyncBufRead + Unpin),
    frame: &mut Vec<u8>,
    limit: usize,
) -> std::io::Result<usize> {
    loop {
        let bytes = reader.fill_buf().await?;
        if bytes.is_empty() {
            return Ok(frame.len());
        }
        let newline = bytes.iter().position(|b| *b == b'\n');
        let count = newline.map(|i| i + 1).unwrap_or(bytes.len());
        if frame.len() + count > limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ACP message exceeds the frame limit",
            ));
        }
        frame.extend_from_slice(&bytes[..count]);
        reader.consume(count);
        if newline.is_some() {
            return Ok(frame.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec![
            "akh-acp",
            "--endpoint",
            "tcp://127.0.0.1:18432",
            "--token-file",
            "/unused",
            "--akh",
            "janus",
            "--workspace-profile",
            "janus",
        ];
        argv.extend_from_slice(extra);
        Args::parse_from(argv)
    }

    fn forwarded(outgoing: Outgoing) -> Value {
        match outgoing {
            Outgoing::Forward(message) => message,
            Outgoing::Reply(reply) => panic!("expected a forward, got reply {reply}"),
        }
    }

    #[test]
    fn akhomed_mode_injects_the_bearer_and_the_workspace_route() {
        let args = args(&["--akhomed", "--t3-compat"]);
        let mut rewriter = Rewriter {
            args: &args,
            target: Target::Akhomed,
            bearer: "Bearer secret-token".into(),
            rewritten: HashSet::new(),
        };
        let init = forwarded(rewriter.outgoing(
            json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":1}}),
        ));
        assert_eq!(
            init["params"]["_meta"]["authorization"],
            "Bearer secret-token"
        );
        let new = forwarded(
            rewriter.outgoing(json!({"jsonrpc":"2.0","id":1,"method":"session/new",
            "params":{"cwd":"/tmp","mcpServers":[{"name":"t3"}]}})),
        );
        assert_eq!(new["params"]["_meta"]["akh/workspace"], "janus");
        assert!(new["params"]["_meta"].get("akh").is_none());
        assert_eq!(new["params"]["mcpServers"], json!([]));
        // T3's cached-token re-authentication is answered locally.
        assert!(matches!(
            rewriter.outgoing(json!({"jsonrpc":"2.0","id":2,"method":"authenticate",
                "params":{"methodId":"cached_token"}})),
            Outgoing::Reply(_)
        ));
    }

    #[test]
    fn session_service_mode_is_unchanged() {
        let args = args(&[]);
        let mut rewriter = Rewriter {
            args: &args,
            target: Target::SessionService,
            bearer: String::new(),
            rewritten: HashSet::new(),
        };
        let init = forwarded(
            rewriter.outgoing(json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}})),
        );
        assert!(
            init["params"].get("_meta").is_none(),
            "token stays in the WS handshake"
        );
        let new = forwarded(rewriter.outgoing(
            json!({"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/tmp"}}),
        ));
        assert_eq!(new["params"]["_meta"]["akh"], "janus");
        assert_eq!(new["params"]["_meta"]["workspaceProfile"], "janus");
    }
}
