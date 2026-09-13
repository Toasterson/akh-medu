//! ACP stdio-to-WebSocket bridge for upstream editors that launch agent commands.
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::{collections::HashSet, path::PathBuf};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio_tungstenite::{
    Connector, connect_async_tls_with_config,
    tungstenite::{Message, client::IntoClientRequest},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    endpoint: String,
    #[arg(long)]
    token_file: PathBuf,
    /// Optional private CA bundle for a work/homelab WSS ingress.
    #[arg(long)]
    ca_file: Option<PathBuf>,
    #[arg(long)]
    akh: Option<String>,
    #[arg(long)]
    workspace_profile: Option<String>,
    #[arg(long)]
    session: Option<String>,
    #[arg(long)]
    allow_insecure_ws: bool,
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let url = url::Url::parse(&args.endpoint)?;
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    if !matches!(url.scheme(), "ws" | "wss")
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || (url.scheme() == "ws" && !local && !args.allow_insecure_ws)
    {
        return Err(
            "use WSS or explicitly authorized private WS, without embedded credentials/query"
                .into(),
        );
    }
    let mut request = args.endpoint.into_client_request()?;
    let token = akh_session_services::config::read_token(&args.token_file)?;
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
    let mut rewritten = HashSet::new();
    let mut line = Vec::new();
    loop {
        tokio::select! {
            read=read_frame(&mut input,&mut line)=>{
                if read?==0{break;}
                let mut message:Value=serde_json::from_slice(&line)?;line.clear();
                if message["method"]=="session/new"{
                    if let Some(session)=&args.session{
                        rewritten.insert(message["id"].to_string());message["method"]=json!("session/load");message["params"]["sessionId"]=json!(session);
                    }else{
                        if let Some(akh)=&args.akh{message["params"]["_meta"]["akh"]=json!(akh);}
                        if let Some(profile)=&args.workspace_profile{message["params"]["_meta"]["workspaceProfile"]=json!(profile);}
                    }
                }
                outgoing.send(Message::Text(message.to_string().into())).await?;
            }
            frame=incoming.next()=>match frame{
                Some(Ok(Message::Text(text)))=>{
                    let mut message:Value=serde_json::from_str(&text)?;
                    if rewritten.remove(&message["id"].to_string())&&message.get("result").is_some(){message["result"]["sessionId"]=json!(args.session);}
                    let mut bytes=serde_json::to_vec(&message)?;bytes.push(b'\n');output.write_all(&bytes).await?;output.flush().await?;
                }
                Some(Ok(Message::Ping(bytes)))=>outgoing.send(Message::Pong(bytes)).await?,
                Some(Ok(Message::Pong(_)))=>{},
                Some(Err(error))=>return Err(error.into()),
                _=>break,
            },
        }
    }
    outgoing.close().await?;
    Ok(())
}
async fn read_frame(
    reader: &mut tokio::io::BufReader<tokio::io::Stdin>,
    frame: &mut Vec<u8>,
) -> std::io::Result<usize> {
    loop {
        let bytes = reader.fill_buf().await?;
        if bytes.is_empty() {
            return Ok(frame.len());
        }
        let newline = bytes.iter().position(|b| *b == b'\n');
        let count = newline.map(|i| i + 1).unwrap_or(bytes.len());
        if frame.len() + count > 2 * 1024 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ACP message exceeds 2 MiB",
            ));
        }
        frame.extend_from_slice(&bytes[..count]);
        reader.consume(count);
        if newline.is_some() {
            return Ok(frame.len());
        }
    }
}
