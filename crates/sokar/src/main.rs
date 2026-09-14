#![forbid(unsafe_code)]
use clap::{Parser, Subcommand};
use sokar::{
    Sokar, SokarError, SokarResult,
    config::Config,
    index::{Index, MemoryIndex, PgIndex},
    store::StoreRegistry,
};
use std::{path::PathBuf, process::ExitCode, sync::Arc};

#[derive(Parser)]
#[command(
    name = "sokard",
    version,
    about = "Sokar content-addressed artifact daemon"
)]
struct Args {
    #[arg(long, default_value = "sokar.toml", global = true)]
    config: PathBuf,
    #[arg(long,conflicts_with_all=["check_stores","migrate"])]
    check: bool,
    #[arg(long,conflicts_with_all=["migrate"])]
    check_stores: bool,
    #[arg(long)]
    migrate: bool,
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    Reindex {
        #[arg(long)]
        store: sokar_client::StoreName,
    },
}
#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sokar=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{:?}", miette::Report::new(e));
            ExitCode::from(2)
        }
    }
}
async fn run(args: Args) -> SokarResult<()> {
    if args.command.is_some() && (args.check || args.check_stores || args.migrate) {
        return Err(SokarError::Config(
            "reindex cannot be combined with --check, --check-stores or --migrate".into(),
        ));
    }
    let config = Arc::new(Config::from_file(&args.config)?);
    if args.migrate {
        let db = config
            .database
            .as_ref()
            .ok_or_else(|| SokarError::Config("--migrate requires [database]".into()))?;
        PgIndex::connect(&db.url.resolve()?)
            .await?
            .migrate()
            .await?;
        println!("sokar migrations complete");
        return Ok(());
    }
    let stores = Arc::new(StoreRegistry::build(&config.stores).await?);
    if args.check_stores {
        stores.check(true).await?;
        println!("all Sokar stores checked");
        return Ok(());
    }
    let index: Arc<dyn Index> = if let Some(db) = &config.database {
        let index = PgIndex::connect(&db.url.resolve()?).await?;
        index.migrate().await?;
        Arc::new(index)
    } else {
        tracing::warn!("using ephemeral local index; idempotency records do not survive restart");
        Arc::new(MemoryIndex::default())
    };
    let state = Arc::new(Sokar::new(config.clone(), stores.clone(), index));
    if args.check {
        state.index.ping().await?;
        stores.check(true).await?;
        println!("Sokar config, index and all stores checked");
        return Ok(());
    }
    if let Some(Command::Reindex { store }) = args.command {
        let count = state.reindex(&store).await?;
        println!(
            "STORE\tCOLLECTIONS\tARTIFACTS\tBLOBS\n{store}\t{}\t{}\t{}",
            count.collections, count.artifacts, count.blobs
        );
        return Ok(());
    }
    if config.database.is_none() {
        for store in stores.stores.keys() {
            state.reindex(store).await?;
        }
    }
    let api = tokio::net::TcpListener::bind(config.server.api_bind).await?;
    let admin = tokio::net::TcpListener::bind(config.server.admin_bind).await?;
    tracing::info!(api=%api.local_addr()?,admin=%admin.local_addr()?,"Sokar listening");
    tokio::try_join!(
        axum::serve(api, sokar::http::api(state.clone())).with_graceful_shutdown(shutdown()),
        axum::serve(admin, sokar::http::admin(state)).with_graceful_shutdown(shutdown()),
    )?;
    Ok(())
}
async fn shutdown() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=terminate.recv()=>{} }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}
