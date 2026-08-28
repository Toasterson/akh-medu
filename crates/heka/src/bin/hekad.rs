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
//!   hekad register-akh <name> [--description <desc>]   Register an akh in heka_akhs
//!   hekad eval longmemeval --mode raw
//!   hekad eval locomo --top-k auto
//!   hekad embed [config_path]      Run the embed worker (foreground, Ctrl-C to stop)
//!   hekad repair-legacy-title-hashes --drawer-id <UUID>

use std::path::PathBuf;
#[cfg(feature = "embedding")]
use std::sync::Arc;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, WrapErr};
use sea_orm_migration::MigratorTrait;
use uuid::Uuid;

use heka::config::{HekaConfig, connect_database};
#[cfg(feature = "embedding")]
use heka::embedder::{SentenceEmbedder, resolve_model_dir};
use heka::eval::EvalCommand;
use heka::migration::{Migrator, repair_rejected_legacy_title_hashes};

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

    /// Register an akh in the heka_akhs table.
    RegisterAkh {
        /// Unique akh name (e.g. "tecton", "thoth").
        name: String,
        /// Optional human-readable description.
        #[arg(long)]
        description: Option<String>,
        /// Path to heka.toml (defaults to ~/.config/heka/heka.toml).
        #[arg(long)]
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

    /// Explicitly repair verified drawers written by the rejected 000002
    /// sha256(title) backfill. No automatic timestamp inference is performed.
    RepairLegacyTitleHashes {
        /// Exact drawer UUID to approve for repair. Repeat for an atomic batch
        /// of at most 256 unique drawers.
        #[arg(long = "drawer-id", required = true, value_name = "UUID", num_args = 1)]
        drawer_ids: Vec<Uuid>,

        /// Path to heka.toml (defaults to ~/.config/heka/heka.toml).
        #[arg(long)]
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
        Commands::RegisterAkh {
            name,
            description,
            config_path,
        } => run_register_akh(&name, description, config_path).await,
        Commands::Eval { command } => heka::eval::run(&command).await,
        Commands::Embed { config_path } => run_embed(config_path).await,
        Commands::RepairLegacyTitleHashes {
            drawer_ids,
            config_path,
        } => run_legacy_title_hash_repair(config_path, &drawer_ids).await,
    }
}

async fn run_legacy_title_hash_repair(
    config_path: Option<PathBuf>,
    drawer_ids: &[Uuid],
) -> miette::Result<()> {
    let config_path = config_path.unwrap_or_else(default_config_path);
    let config = HekaConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;
    let db = connect_database(&config.database.url, config.database.max_connections)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL")?;
    let stats = repair_rejected_legacy_title_hashes(&db, drawer_ids)
        .await
        .into_diagnostic()
        .wrap_err("repairing explicitly approved rejected-000002 drawer hashes")?;
    tracing::info!(
        repaired = stats.repaired,
        already_repaired = stats.already_repaired,
        "explicit legacy title-hash repair complete"
    );
    Ok(())
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

/// Register an akh in `heka_akhs`.
///
/// Creates a new row with the given name and optional description. If an akh
/// with the same name already exists, updates its description (upsert).
async fn run_register_akh(
    name: &str,
    description: Option<String>,
    config_path: Option<PathBuf>,
) -> miette::Result<()> {
    use sea_orm::{ActiveModelTrait, ColumnTrait, EntityTrait, QueryFilter, Set};

    let config_path = config_path.unwrap_or_else(default_config_path);

    let config = HekaConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;

    let db = connect_database(&config.database.url, config.database.max_connections)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL")?;

    // Check if akh already exists — if so, update description and return.
    if let Some(existing) = heka::entity::heka_akhs::Entity::find()
        .filter(heka::entity::heka_akhs::Column::Name.eq(name))
        .one(&db)
        .await
        .into_diagnostic()
        .wrap_err_with(|| format!("looking up akh '{name}'"))?
    {
        if let Some(desc) = &description {
            let mut active: heka::entity::heka_akhs::ActiveModel = existing.into();
            active.description = Set(Some(desc.clone()));
            active.updated_at = Set(chrono::Utc::now());
            active.update(&db).await.into_diagnostic()?;
            tracing::info!(akh = name, "updated existing akh registration");
        } else {
            tracing::info!(akh = name, "akh already registered — no changes");
        }
        return Ok(());
    }

    // Insert new akh row.
    let now = chrono::Utc::now();
    let new_akh = heka::entity::heka_akhs::ActiveModel {
        id: Set(Uuid::new_v4()),
        name: Set(name.to_string()),
        description: Set(description),
        personality: Set(None),
        avatar_url: Set(None),
        token_hash: Set(None),
        created_at: Set(now),
        updated_at: Set(now),
    };

    let saved = new_akh.insert(&db).await.into_diagnostic()?;

    tracing::info!(
        akh = name,
        id = %saved.id,
        "registered new akh in heka_akhs"
    );

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
    let shared = heka::embedder::SharedEmbedder::new(embedder);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_legacy_repair_accepts_repeated_uuid_flags() {
        let first = "018f55b2-1df7-7b0a-b9ca-2a880942782b";
        let second = "018f55b2-1df7-7b0a-b9ca-2a880942782c";
        let cli = Cli::try_parse_from([
            "hekad",
            "repair-legacy-title-hashes",
            "--drawer-id",
            first,
            "--drawer-id",
            second,
            "--config-path",
            "/tmp/explicit-heka.toml",
        ])
        .unwrap();
        match cli.command {
            Commands::RepairLegacyTitleHashes {
                drawer_ids,
                config_path,
            } => {
                assert_eq!(drawer_ids.len(), 2);
                assert_eq!(drawer_ids[0].to_string(), first);
                assert_eq!(drawer_ids[1].to_string(), second);
                assert_eq!(config_path, Some(PathBuf::from("/tmp/explicit-heka.toml")));
            }
            _ => panic!("parsed the wrong command"),
        }
        assert!(
            Cli::try_parse_from(["hekad", "repair-legacy-title-hashes", "--config-path", "x"])
                .is_err()
        );
    }
}
