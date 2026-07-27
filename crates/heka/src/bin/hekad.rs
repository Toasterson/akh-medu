//! `hekad` — Heka daemon CLI.
//!
//! Phase 42a: only the `migrate` subcommand is implemented. It loads
//! `heka.toml`, connects to PostgreSQL, and runs the SeaORM migrator to
//! create the 5 `heka_*` tables + HNSW index.
//!
//! Usage:
//!   hekad migrate [config_path]
//!   hekad migrate           # uses default config path ~/.config/heka/heka.toml

use std::path::PathBuf;

use miette::{IntoDiagnostic, WrapErr};
use sea_orm::Database;
use sea_orm_migration::MigratorTrait;

use heka::config::HekaConfig;
use heka::migration::Migrator;

#[tokio::main]
async fn main() -> miette::Result<()> {
    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "heka=info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    // Parse subcommand.
    if args.len() < 2 || args[1] == "help" || args[1] == "--help" {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "migrate" => run_migrate(&args[2..]).await,
        cmd => {
            eprintln!("unknown subcommand: {cmd}");
            print_usage();
            std::process::exit(2);
        }
    }
}

async fn run_migrate(args: &[String]) -> miette::Result<()> {
    let config_path = if let Some(p) = args.first() {
        PathBuf::from(p)
    } else {
        default_config_path()
    };

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

fn print_usage() {
    eprintln!("hekad — Heka daemon\n");
    eprintln!("USAGE:");
    eprintln!("  hekad migrate [config_path]   Create the heka_* schema on the target DB");
    eprintln!("  hekad help                     Show this message");
    eprintln!();
    eprintln!("Config defaults to ~/.config/heka/heka.toml");
    eprintln!("DATABASE_URL env var overrides [database] url in the config");
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