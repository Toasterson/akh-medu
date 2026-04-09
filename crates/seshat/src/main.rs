//! Seshat's Archive — `seshd` service binary.
//!
//! Runs the Seshat MCP server that mediates pgvector access for akh-medu instances.

use std::path::PathBuf;
use std::sync::Arc;

use miette::{IntoDiagnostic, WrapErr};
use sea_orm::Database;
use sea_orm_migration::MigratorTrait;

use seshat::config::SeshatConfig;
use seshat::mcp::{SeshatMcpServer, SeshatMcpState};
use seshat::migration::Migrator;

#[tokio::main]
async fn main() -> miette::Result<()> {
    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "seshat=info".parse().unwrap()),
        )
        .init();

    // Parse CLI arguments.
    let args: Vec<String> = std::env::args().collect();

    // Handle `migrate` subcommand.
    if args.len() > 1 && args[1] == "migrate" {
        let config_path = if args.len() > 2 {
            PathBuf::from(&args[2])
        } else {
            default_config_path()
        };
        let config = SeshatConfig::from_file(&config_path)
            .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;
        let db = Database::connect(&config.database.url)
            .await
            .into_diagnostic()
            .wrap_err("connecting to PostgreSQL")?;
        Migrator::up(&db, None)
            .await
            .into_diagnostic()
            .wrap_err("running migrations")?;
        tracing::info!("migrations complete");
        return Ok(());
    }

    let config_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        default_config_path()
    };

    // Load configuration.
    let config = SeshatConfig::from_file(&config_path)
        .wrap_err_with(|| format!("loading config from {}", config_path.display()))?;

    tracing::info!(bind = %config.server.bind, "starting Seshat's Archive");

    // Connect to PostgreSQL.
    let db = Database::connect(&config.database.url)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL")?;

    tracing::info!("connected to PostgreSQL");

    // Run migrations.
    Migrator::up(&db, None)
        .await
        .into_diagnostic()
        .wrap_err("running migrations")?;

    tracing::info!("migrations complete");

    let db = Arc::new(db);

    // Load embedding model (if feature enabled).
    #[cfg(feature = "embedding")]
    let embedder = {
        let model_dir =
            seshat::embedder::resolve_model_dir(&config.model_dir(), &config.embedding.model);
        match model_dir {
            Ok(dir) => match seshat::embedder::SentenceEmbedder::load(&dir) {
                Ok(e) => {
                    tracing::info!(model = %config.embedding.model, "embedding model loaded");
                    Some(Arc::new(std::sync::Mutex::new(e)))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "embedding model unavailable — search disabled");
                    None
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "embedding model unavailable — search disabled");
                None
            }
        }
    };
    #[cfg(not(feature = "embedding"))]
    let embedder: Option<seshat::embedder::SharedEmbedder> = None;

    // Start embedding worker (background task).
    #[cfg(feature = "embedding")]
    if let Some(ref emb) = embedder {
        let db_clone = Arc::clone(&db);
        let emb_clone = Arc::clone(emb);
        let embed_config = config.embedding.clone();
        tokio::spawn(async move {
            seshat::embed_worker::run_embed_worker(db_clone, emb_clone, embed_config).await;
        });
        tracing::info!("embedding worker started");
    }

    // Build MCP server.
    let state = Arc::new(SeshatMcpState {
        db: Arc::clone(&db),
        embedder,
    });

    let mcp_server = SeshatMcpServer::new(state);

    // Mount MCP server on axum with streamable HTTP transport.
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };

    let session_manager = Arc::new(LocalSessionManager::default());
    let mcp_config = StreamableHttpServerConfig::default();
    let mcp_service = StreamableHttpService::new(
        move || Ok(mcp_server.clone()),
        session_manager,
        mcp_config,
    );

    let app = axum::Router::new().nest_service("/mcp", mcp_service);

    let bind_addr: std::net::SocketAddr = config
        .server
        .bind
        .parse()
        .map_err(|e| seshat::SeshatError::Config(format!("invalid bind address: {e}")))
        .into_diagnostic()?;

    tracing::info!(bind = %bind_addr, "Seshat's Archive MCP server listening");

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .into_diagnostic()
        .wrap_err("binding TCP listener")?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .into_diagnostic()
        .wrap_err("serving")?;

    Ok(())
}

/// Default config file path: `~/.config/seshat/seshat.toml`.
fn default_config_path() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("seshat")
        .join("seshat.toml")
}
