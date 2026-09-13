use crate::{config::Config, http, store::Store, types::SCHEMA_VERSION};
#[cfg(any(
    feature = "session",
    feature = "runtime",
    feature = "executor",
    feature = "sandbox",
    feature = "events"
))]
use axum::routing::post;
use axum::{Json, Router, extract::DefaultBodyLimit, routing::get};
use serde_json::json;
#[cfg(any(feature = "session", feature = "runtime", feature = "events"))]
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};
#[cfg(feature = "session")]
mod sessions;
#[cfg(feature = "session")]
pub use sessions::{SessionApp, controller, session_router};
#[cfg(feature = "runtime")]
mod runtimes;
#[cfg(feature = "runtime")]
pub use runtimes::runtime_router;
#[cfg(feature = "executor")]
mod executors;
#[cfg(feature = "executor")]
pub use executors::executor_router;
#[cfg(feature = "sandbox")]
mod sandboxes;
#[cfg(feature = "sandbox")]
pub use sandboxes::sandbox_router;
#[cfg(feature = "events")]
mod events;
#[cfg(feature = "executor")]
use crate::executor::Executor;
#[cfg(feature = "events")]
use crate::ingress::Events;
#[cfg(feature = "runtime")]
use crate::runtime::Runtime;
#[cfg(feature = "sandbox")]
use crate::sandbox::Sandboxes;
#[cfg(feature = "events")]
pub use events::event_router;
pub fn protect(router: Router, config: Arc<Config>) -> Router {
    router.layer(DefaultBodyLimit::max(2 * 1024 * 1024)).layer(
        axum::middleware::from_fn_with_state(config, http::authorization),
    )
}
#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    check: bool,
}
pub async fn run(kind: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    let args = Args::parse();
    let config = Arc::new(Config::load(&args.config)?);
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    for principal in &config.principals {
        principal.credential()?;
    }
    if args.check {
        println!("{kind}: configuration valid");
        return Ok(());
    }
    let store = Store::open(&config.data_dir.join(format!("{kind}.redb")))?;
    #[allow(unused_mut)]
    let mut tasks: Vec<tokio::task::JoinHandle<()>> = vec![];
    #[cfg(feature = "runtime")]
    #[allow(unused_assignments)]
    let mut runtime_cleanup = None;
    #[cfg(feature = "executor")]
    #[allow(unused_assignments)]
    let mut executor_cleanup = None;
    let router = match kind {
        #[cfg(feature = "session")]
        "akhsessd" => {
            let app = SessionApp::new(config.clone(), store)?;
            let worker = app.clone();
            tasks.push(tokio::spawn(worker.run_dispatcher()));
            session_router(app)
        }
        #[cfg(feature = "runtime")]
        "akhrtd" => {
            let runtime = Runtime::new(config.clone(), store)?;
            runtime_cleanup = Some(runtime.clone());
            let worker = runtime.clone();
            let interval = Duration::from_millis(config.reconciliation_interval_ms);
            tasks.push(tokio::spawn(async move {
                loop {
                    match worker.resolve_stale_unknown_jobs() {
                        Ok(resolved) => {
                            for id in resolved {
                                tracing::warn!(
                                    job = %id,
                                    "runtime automatically closed a job whose outcome stayed unknown past the reconciliation stall threshold"
                                );
                            }
                        }
                        Err(error) => {
                            tracing::error!(code=%error.code, "stale runtime job reconciliation failed")
                        }
                    }
                    tokio::time::sleep(interval).await;
                }
            }));
            runtime_router(runtime)
        }
        #[cfg(feature = "executor")]
        "akhexecd" => {
            let executor = Executor::new(config.clone(), store)?;
            executor_cleanup = Some(executor.clone());
            let worker = executor.clone();
            tasks.push(tokio::spawn(async move {
                loop {
                    if let Err(error) = worker.expire_leases() {
                        tracing::error!(code=%error.code,"workspace lease reconciliation failed");
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }));
            executor_router(executor)
        }
        #[cfg(feature = "sandbox")]
        "akhsbxd" => sandbox_router(Sandboxes::new(config.clone(), store)?),
        #[cfg(feature = "events")]
        "akheventd" => {
            let service = Events::new(config.clone(), store)?;
            service.sources();
            let worker = service.clone();
            tasks.push(tokio::spawn(async move {
                loop {
                    if let Err(error) = worker.drain().await {
                        tracing::error!(code=%error.code,"event outbox drain failed");
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }));
            event_router(service)
        }
        _ => return Err("unknown service".into()),
    };
    let identity = config.service_id.clone();
    let router=router.route("/healthz",get(move||{let identity=identity.clone();async move{Json(json!({"service_id":identity,"schema_version":SCHEMA_VERSION,"status":"listening"}))}}));
    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    tracing::info!(service=%config.service_id,address=%listener.local_addr()?,"service listening");
    use std::future::IntoFuture;
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = stopped.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! { result=&mut server=>{return Ok(result?);}, _=shutdown()=>{} }
    let _ = stop.send(());
    for task in tasks {
        task.abort();
    }
    #[cfg(feature = "runtime")]
    if let Some(runtime) = runtime_cleanup {
        runtime.cancel_all();
    }
    #[cfg(feature = "executor")]
    if let Some(executor) = executor_cleanup {
        executor.cancel_all();
    }
    match tokio::time::timeout(std::time::Duration::from_secs(10), server).await {
        Ok(result) => result?,
        Err(_) => tracing::warn!(
            "HTTP drain deadline reached; unresolved durable work remains fenced on restart"
        ),
    }
    Ok(())
}
async fn shutdown() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
