//! Measurement harness, never a daemon admission path. The caller supplies a
//! private profile/job and answers supplied-tool requests on stdin. No conformance
//! certificate is created; the real runtime daemon keeps its admission gate.
use akh_runtime::Host;
use akh_session_services::{
    runtime_profile,
    types::{ExecutionMode, ProviderConfig, RuntimeEvent, RuntimeJobRequest, ToolResult},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{BufRead, Read, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Deserialize)]
struct Probe {
    profile: ProviderConfig,
    request: RuntimeJobRequest,
}
fn send(value: Value) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{value}")
        .and_then(|_| out.flush())
        .map_err(|_| "probe output closed".into())
}
struct Witness;
impl Host for Witness {
    fn event(&mut self, event: RuntimeEvent) -> Result<(), String> {
        send(json!({"type":"event","event":event}))
    }
    fn tool(&mut self, id: &str, name: &str, arguments: Value) -> Result<ToolResult, String> {
        send(json!({"type":"tool","id":id,"name":name,"arguments":arguments}))?;
        let mut line = String::new();
        std::io::stdin()
            .lock()
            .take(1024 * 1024 + 1)
            .read_line(&mut line)
            .map_err(|_| "probe tool response unavailable")?;
        if line.len() > 1024 * 1024 {
            return Err("probe tool response exceeds limit".into());
        }
        serde_json::from_str(&line).map_err(|_| "invalid probe tool response".into())
    }
    fn approval(&mut self, _id: &str, _name: &str, _args: &Value) -> Result<bool, String> {
        Err("brokered probe unexpectedly requested native approval".into())
    }
    fn question(&mut self, _id: &str, _questions: &Value) -> Result<Value, String> {
        Err("probe unexpectedly requested human input".into())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("probe JSON path required")?;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("probe input exceeds limit".into());
    }
    let probe: Probe = serde_json::from_slice(&bytes)?;
    if probe.profile.execution_mode() != ExecutionMode::Brokered || probe.profile.isolation.is_none()
    {
        return Err("probe requires a private brokered profile".into());
    }
    let _lease = runtime_profile::acquire(&probe.profile).map_err(|e| e.message)?;
    let cancel = Arc::new(AtomicBool::new(false));
    let signal_cancel = cancel.clone();
    // Let the provider transport unwind and reap its own process group when
    // the harness terminates early. Closing stdin also unblocks a tool reply.
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("signal runtime");
        runtime.block_on(async {
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("SIGTERM");
            tokio::select! { _=term.recv()=>{}, _=tokio::signal::ctrl_c()=>{} }
            signal_cancel.store(true, Ordering::Relaxed);
        });
    });
    match akh_runtime::run(&probe.profile, &probe.request, &mut Witness, &cancel) {
        Ok(output) => send(json!({"type":"completed","output":output}))?,
        Err(error) => {
            send(json!({"type":"failed","error":error}))?;
            std::process::exit(1);
        }
    }
    Ok(())
}
