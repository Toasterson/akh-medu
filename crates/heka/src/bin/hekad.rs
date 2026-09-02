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
//!   hekad register-akh <name> [--description <desc>] [--token]   Register an akh
//!   hekad set-token <name>                                        Set/rotate bearer token
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
        /// Mint a bearer token and store its SHA-256 hash. The plaintext
        /// token is printed once and never stored. Required for MCP access.
        #[arg(long)]
        token: bool,
        /// Path to heka.toml (defaults to ~/.config/heka/heka.toml).
        #[arg(long)]
        config_path: Option<PathBuf>,
    },

    /// Set or rotate the bearer token for an existing akh.
    SetToken {
        /// Name of the akh to update.
        name: String,
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

    /// Run the MCP server (/health + /akh/{akh_id}/mcp) alongside the embed
    /// worker. Ctrl-C drains the listener and stops the worker loop.
    Serve {
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
            token,
            config_path,
        } => run_register_akh(&name, description, token, config_path).await,
        Commands::SetToken { name, config_path } => run_set_token(&name, config_path).await,
        Commands::Eval { command } => heka::eval::run(&command).await,
        Commands::Embed { config_path } => run_embed(config_path).await,
        Commands::Serve { config_path } => run_serve(config_path).await,
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
/// When `mint_token` is true, mints a bearer token and stores its SHA-256
/// hash; the plaintext token is printed once to stdout.
async fn run_register_akh(
    name: &str,
    description: Option<String>,
    mint_token: bool,
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
        let mut active: heka::entity::heka_akhs::ActiveModel = existing.into();
        if let Some(desc) = &description {
            active.description = Set(Some(desc.clone()));
        }
        if mint_token {
            let (plaintext, hash) = heka::auth::mint_token();
            active.token_hash = Set(Some(hash));
            println!("Bearer token for {name}: {plaintext}");
            tracing::info!(akh = name, "minted new bearer token");
        }
        if description.is_some() || mint_token {
            active.updated_at = Set(chrono::Utc::now());
            active.update(&db).await.into_diagnostic()?;
            tracing::info!(akh = name, "updated existing akh registration");
        } else {
            tracing::info!(akh = name, "akh already registered — no changes");
        }
        return Ok(());
    }

    // Insert new akh row.
    let token_hash = if mint_token {
        let (plaintext, hash) = heka::auth::mint_token();
        println!("Bearer token for {name}: {plaintext}");
        Some(hash)
    } else {
        None
    };

    let now = chrono::Utc::now();
    let new_akh = heka::entity::heka_akhs::ActiveModel {
        id: Set(Uuid::new_v4()),
        name: Set(name.to_string()),
        description: Set(description),
        personality: Set(None),
        avatar_url: Set(None),
        token_hash: Set(token_hash),
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

/// Set or rotate the bearer token for an existing akh.
///
/// Mints a fresh 32-byte token, stores its SHA-256 hash in `heka_akhs.token_hash`,
/// and prints the plaintext once to stdout. The old token (if any) is
/// immediately invalidated.
async fn run_set_token(name: &str, config_path: Option<PathBuf>) -> miette::Result<()> {
    use sea_orm::{ActiveModelTrait, ColumnTrait, EntityTrait, QueryFilter, Set};

    let config_path = config_path.unwrap_or_else(default_config_path);
    let config = HekaConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;
    let db = connect_database(&config.database.url, config.database.max_connections)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL")?;

    let existing = heka::entity::heka_akhs::Entity::find()
        .filter(heka::entity::heka_akhs::Column::Name.eq(name))
        .one(&db)
        .await
        .into_diagnostic()
        .wrap_err_with(|| format!("looking up akh '{name}'"))?
        .ok_or_else(|| miette::miette!("akh '{name}' not found — register it first with `hekad register-akh {name} --token`"))?;

    let (plaintext, hash) = heka::auth::mint_token();
    let mut active: heka::entity::heka_akhs::ActiveModel = existing.into();
    active.token_hash = Set(Some(hash));
    active.updated_at = Set(chrono::Utc::now());
    active.update(&db).await.into_diagnostic()?;

    println!("Bearer token for {name}: {plaintext}");
    tracing::info!(akh = name, "bearer token set (plaintext shown once)");

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

/// Load ONNX model + embedder for `serve`. Without the `embedding` feature
/// the server still runs — MCP tools and /health do not embed; the backlog
/// then waits for a feature-enabled embed worker process.
#[cfg(feature = "embedding")]
fn load_shared_embedder(config: &HekaConfig) -> miette::Result<heka::embedder::SharedEmbedder> {
    let model_dir = config.model_dir();
    let model_name = &config.embedding.model;
    let resolved = resolve_model_dir(&model_dir, model_name)
        .wrap_err_with(|| format!("resolving embedding model '{model_name}'"))?;

    tracing::info!(model = model_name, dir = %resolved.display(), "loading ONNX model");
    let embedder = SentenceEmbedder::load(&resolved).wrap_err("loading ONNX embedding model")?;
    Ok(heka::embedder::SharedEmbedder::new(embedder))
}

/// Run the HTTP MCP surface and the embed worker in the same process.
///
/// Ctrl-C (SIGINT) triggers one shutdown signal consumed by both halves:
/// `heka::serve::serve` stops accepting connections and drains in-flight MCP
/// sessions, and the embed worker loop exits after its current batch — no
/// half-finished batch transaction is orphaned.
async fn run_serve(config_path: Option<PathBuf>) -> miette::Result<()> {
    let config_path = config_path.unwrap_or_else(default_config_path);
    let config = HekaConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;

    let db = Arc::new(
        connect_database(&config.database.url, config.database.max_connections)
            .await
            .into_diagnostic()
            .wrap_err("connecting to PostgreSQL")?,
    );

    let bind_addr: std::net::SocketAddr = config
        .server
        .bind
        .parse()
        .into_diagnostic()
        .wrap_err_with(|| format!("invalid server.bind address '{}'", config.server.bind))?;
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .into_diagnostic()
        .wrap_err_with(|| format!("binding TCP listener on {bind_addr}"))?;

    // One shutdown trigger shared by HTTP drain and the worker loop.
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received — draining HTTP and stopping the embed worker");
    };
    let (worker_stop_tx, worker_stop_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = shutdown.then(|()| async move {
        let _ = worker_stop_tx.send(());
    });
    tokio::pin!(shutdown);

    // Embed worker alongside the server: drains when the stop signal fires.
    // The worker polls every ~1s minimum between batches, so worst-case
    // shutdown latency is one batch + one poll interval.
    let (worker_done_tx, worker_done_rx) = tokio::sync::oneshot::channel::<()>();
    #[cfg(feature = "embedding")]
    {
        let embedder = load_shared_embedder(&config)?;
        let worker_db = Arc::clone(&db);
        let poll_secs = config.embedding.poll_interval_secs.max(1);
        let batch_size = config.embedding.batch_size;
        let model_name = config.embedding.model.clone();
        tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(poll_secs);
            let mut stop = std::pin::pin!(worker_stop_rx);
            tracing::info!(
                batch_size,
                poll_secs,
                "heka embed worker started (serve mode)"
            );
            loop {
                match heka::embed_worker::embed_pending_batch(
                    &worker_db,
                    &embedder,
                    batch_size,
                    &model_name,
                )
                .await
                {
                    Ok(stats) => {
                        if stats.chunks_processed > 0 {
                            tracing::info!(
                                chunks = stats.chunks_processed,
                                embeddings = stats.embeddings_created,
                                "heka embed batch complete"
                            );
                        }
                    }
                    Err(error) => tracing::error!(%error, "heka embed worker error"),
                }
                if tokio::time::timeout(interval, &mut stop).await.is_ok() {
                    break;
                }
            }
            tracing::info!("heka embed worker drained");
            let _ = worker_done_tx.send(());
        });
    }
    #[cfg(not(feature = "embedding"))]
    {
        tracing::warn!(
            "hekad built without the `embedding` feature — serve runs MCP + health only; \
             start a feature-enabled embed worker separately to drain the backlog"
        );
        drop(worker_stop_rx);
        tokio::spawn(async move {
            let _ = worker_done_tx.send(());
        });
    }

    tracing::info!(bind = %bind_addr, "hekad serve listening (/health + /akh/{{akh_id}}/mcp)");
    let serve_result = heka::serve::serve(listener, Arc::clone(&db), shutdown).await;

    // Listener is drained and closed; wait for the worker drain so process
    // exit never abandons an open batch transaction.
    let _ = worker_done_rx.await;
    tracing::info!("hekad serve shutdown complete");

    serve_result
        .into_diagnostic()
        .wrap_err("serving hekad MCP surface")
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
