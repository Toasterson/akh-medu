//! `hekad` — Heka daemon CLI.
//!
//! Phase 42a: `migrate` creates the heka_* schema.
//! Phase 42b: `eval` runs the LongMemEval / LoCoMo retrieval harness.
//!
//! Usage:
//!   hekad migrate [config_path]
//!   hekad eval longmemeval --mode raw
//!   hekad eval locomo --top-k auto

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, WrapErr};
use sea_orm::Database;
use sea_orm_migration::MigratorTrait;

use heka::config::HekaConfig;
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
    }
}

async fn run_migrate(config_path: Option<PathBuf>) -> miette::Result<()> {
    let config_path = config_path.unwrap_or_else(default_config_path);

    let config = HekaConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;

    tracing::info!(url = %config.database.url, "connecting to PostgreSQL");
    let db = Database::connect(&config.database.url)
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
