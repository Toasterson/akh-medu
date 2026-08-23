//! `hekad` — Heka daemon CLI.
//!
//! Phase 42a: `migrate` creates the heka_* schema.
//! Phase 42b: `eval` runs the LongMemEval / LoCoMo retrieval harness.
//! Phase 42c: `embed` subcommand — loads the ONNX MiniLM-L6-v2 model and
//! runs the embed worker loop, polling `heka_drawer_chunks` for unembedded
//! rows and writing vectors to `heka_drawer_embeddings`.
//!
//! Usage:
//!   hekad migrate [config_path]
//!   hekad eval longmemeval --mode raw
//!   hekad eval locomo --top-k auto
//!   hekad embed [config_path]      Run the embed worker (foreground, Ctrl-C to stop)

use std::path::PathBuf;
#[cfg(feature = "embedding")]
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, WrapErr};
use sea_orm_migration::MigratorTrait;

use heka::config::{HekaConfig, connect_database};
#[cfg(feature = "embedding")]
use heka::embedder::{SentenceEmbedder, resolve_model_dir};
use heka::eval::EvalCommand;
use heka::migration::Migrator;

/// Heka daemon CLI.
#[derive(Debug, Parser)]
#[command(name = "hekad", about = "Heka — Postgres+pgvector evaluation store")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Create the heka_* schema on the target PostgreSQL database.
    Migrate {
        /// Path to heka.toml (defaults to ~/.config/heka/heka.toml).
        config_path: Option<PathBuf>,
    },

    /// Run the long-term memory evaluation harness.
    Eval {
        #[command(subcommand)]
        command: EvalCommand,
    },

    /// Run the ONNX embedding worker in the foreground.
    Embed {
        /// Path to heka.toml (defaults to ~/.config/heka/heka.toml).
        config_path: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "heka=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Migrate { config_path } => run_migrate(config_path).await,
        Commands::Eval { command } => heka::eval::run(&command).await,
        Commands::Embed { config_path } => run_embed(config_path).await,
    }
}

async fn run_migrate(config_path: Option<PathBuf>) -> miette::Result<()> {
    let config_path = config_path.unwrap_or_else(default_config_path);

    let config = HekaConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;

    let db = connect_database(&config.database.url, config.database.max_connections)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL")?;

    tracing::info!("running Heka migrations");
    Migrator::up(&db, None)
        .await
        .into_diagnostic()
        .wrap_err("running migrations")?;

    tracing::info!("Heka migrations complete — 5 tables + HNSW index created");
    Ok(())
}

/// Default config file path: `~/.config/heka/heka.toml`.
fn default_config_path() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("heka")
        .join("heka.toml")
}

/// Run the embed worker: load the ONNX model, connect to the DB, and
/// poll for unembedded chunks forever (or until Ctrl-C).
#[cfg(feature = "embedding")]
async fn run_embed(config_path: Option<PathBuf>) -> miette::Result<()> {
    let config_path = config_path.unwrap_or_else(default_config_path);

    let config = HekaConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;

    let db = connect_database(&config.database.url, config.database.max_connections)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL")?;

    // Load the ONNX embedding model.
    let model_dir = config.model_dir();
    let model_name = &config.embedding.model;
    let resolved = resolve_model_dir(&model_dir, model_name)
        .wrap_err_with(|| format!("resolving embedding model '{model_name}'"))?;

    tracing::info!(model = model_name, dir = %resolved.display(), "loading ONNX model");
    let embedder = SentenceEmbedder::load(&resolved).wrap_err("loading ONNX embedding model")?;
    let shared: heka::embedder::SharedEmbedder = Arc::new(Mutex::new(embedder));

    tracing::info!(
        batch_size = config.embedding.batch_size,
        poll_secs = config.embedding.poll_interval_secs,
        "starting heka embed worker — Ctrl-C to stop"
    );

    // Run the embed worker loop. This runs forever.
    heka::embed_worker::run_embed_worker(Arc::new(db), shared, config.embedding.clone()).await;

    Ok(())
}

#[cfg(not(feature = "embedding"))]
async fn run_embed(_config_path: Option<PathBuf>) -> miette::Result<()> {
    Err(miette::miette!(
        "the embed worker requires the `embedding` feature; rebuild hekad with default features"
    ))
}
