//! Heka service configuration, parsed from `heka.toml`.
//!
//! Mirrors `crates/seshat/src/config.rs` — TOML config with environment-variable
//! overrides for K8s secrets.

use sea_orm::{ConnectOptions, Database, DatabaseConnection};
use serde::{Deserialize, Serialize};

/// Top-level Heka service configuration, parsed from `heka.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HekaConfig {
    pub database: DatabaseConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub eval: EvalConfig,
}

/// PostgreSQL connection settings.
#[derive(Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL.
    /// Example: `postgres://user:pass@localhost:5432/akhmedu`
    pub url: String,

    /// Maximum number of connections in the pool.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

impl std::fmt::Debug for DatabaseConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DatabaseConfig")
            .field("url", &redact_database_url(&self.url))
            .field("max_connections", &self.max_connections)
            .finish()
    }
}

/// Service (hekad) settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address the `hekad serve` listener binds (config: `server.bind`).
    #[serde(default = "default_bind")]
    pub bind: String,
}

/// Embedding model and worker settings.
///
/// Mirrors `seshat::config::EmbeddingConfig`. The embed worker polls
/// `heka_drawer_chunks` for rows with `embedded_at IS NULL`, embeds
/// them in batches, and writes the vectors to `heka_drawer_embeddings`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Sentence embedding model name (for model directory lookup).
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// Number of chunks to embed per batch.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Seconds between embed worker polls for unembedded chunks.
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,

    /// Directory for downloaded models.
    /// Defaults to `~/.local/share/heka/models/`.
    pub model_dir: Option<String>,
}

/// Evaluation data settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Path to the evaluation dataset (JSONL or CSV) for recall/precision scoring.
    #[serde(default)]
    pub data_path: String,
}

// --- Defaults ---

fn default_max_connections() -> u32 {
    10
}

fn default_bind() -> String {
    "127.0.0.1:9091".into()
}

fn default_embedding_model() -> String {
    "all-MiniLM-L6-v2".into()
}

fn default_batch_size() -> usize {
    32
}

fn default_poll_interval() -> u64 {
    30
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            batch_size: default_batch_size(),
            poll_interval_secs: default_poll_interval(),
            model_dir: None,
        }
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            data_path: String::new(),
        }
    }
}

impl HekaConfig {
    /// Load config from a TOML file path.
    ///
    /// After parsing, environment variable overrides are applied:
    /// - `DATABASE_URL` overrides `database.url`
    /// - `HEKA_BIND` overrides `server.bind`
    pub fn from_file(path: &std::path::Path) -> crate::HekaResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::HekaError::Config(format!("cannot read {}: {e}", path.display()))
        })?;
        // `toml::de::Error` can render the source line that failed. Discard its
        // display text because an adjacent database URL may contain secrets.
        let mut config: Self = toml::from_str(&content).map_err(|_| {
            crate::HekaError::Config("invalid TOML configuration (contents redacted)".to_string())
        })?;

        // Allow DATABASE_URL env var to override the TOML value (K8s secrets).
        if let Ok(url) = std::env::var("DATABASE_URL") {
            config.database.url = url;
        }
        // Allow HEKA_BIND env var to override the bind address.
        if let Ok(bind) = std::env::var("HEKA_BIND") {
            config.server.bind = bind;
        }

        Ok(config)
    }

    /// Resolve the model directory, defaulting to `~/.local/share/heka/models/`.
    pub fn model_dir(&self) -> std::path::PathBuf {
        if let Some(ref dir) = self.embedding.model_dir {
            std::path::PathBuf::from(dir)
        } else if let Some(data) = dirs_fallback() {
            data.join("heka").join("models")
        } else {
            std::path::PathBuf::from("models")
        }
    }
}

/// Best-effort XDG data home without pulling in the `dirs` crate.
fn dirs_fallback() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/share"))
        })
}

/// Return a log-safe database target without userinfo, database names, query
/// parameters, or fragments. Invalid URLs are fully redacted rather than
/// echoed back into logs.
pub fn redact_database_url(url: &str) -> String {
    let Some((scheme, _)) = url.split_once("://") else {
        return "<redacted-database-url>".to_string();
    };
    if !matches!(scheme, "postgres" | "postgresql") {
        return "<redacted-database-url>".to_string();
    }
    let Ok(parsed) = url.parse::<sea_orm::sqlx::postgres::PgConnectOptions>() else {
        return "<redacted-database-url>".to_string();
    };
    let host = parsed.get_host();
    if host.is_empty() || host.bytes().any(|byte| byte.is_ascii_control()) {
        return "<redacted-database-url>".to_string();
    }
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    };

    format!("{scheme}://{host}:{}/<database>", parsed.get_port())
}

/// Emit the standard connection event with a credential-safe target.
pub fn log_database_connection(url: &str) {
    tracing::info!(database = %redact_database_url(url), "connecting to PostgreSQL");
}

/// Connect without allowing SeaORM's URL-bearing parse/driver errors to reach
/// user-facing diagnostics. The redacted target remains useful for operations;
/// credentials, database names, and query secrets never enter the error.
pub async fn connect_database(
    url: &str,
    max_connections: u32,
) -> crate::HekaResult<DatabaseConnection> {
    if max_connections == 0 {
        return Err(crate::HekaError::Config(
            "database max_connections must be greater than zero".to_string(),
        ));
    }
    let target = redact_database_url(url);
    log_database_connection(url);
    let mut options = private_postgres_connect_options(url)?;
    options.max_connections(max_connections).sqlx_logging(false);
    Database::connect(options).await.map_err(|_| {
        crate::HekaError::Config(format!(
            "database connection to {target} failed; inspect server availability and credentials"
        ))
    })
}

/// Keep the credential-bearing URL out of SeaORM's instrumented
/// `ConnectOptions`. The connector parses this harmless routing URL, then the
/// skipped callback replaces it with the already-parsed private options.
fn private_postgres_connect_options(url: &str) -> crate::HekaResult<ConnectOptions> {
    let parsed: sea_orm::sqlx::postgres::PgConnectOptions = url.parse().map_err(|_| {
        crate::HekaError::Config(format!(
            "database URL {} is not a valid PostgreSQL connection string",
            redact_database_url(url)
        ))
    })?;
    let mut options = ConnectOptions::new("postgres://redacted.invalid/postgres");
    options.map_sqlx_postgres_opts(move |_| parsed.clone());
    Ok(options)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use super::{
        DatabaseConfig, HekaConfig, connect_database, log_database_connection,
        private_postgres_connect_options, redact_database_url,
    };

    #[derive(Clone)]
    struct CapturedLog(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedLog {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().write(bytes)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.0.lock().unwrap().flush()
        }
    }

    #[test]
    fn database_log_target_never_contains_credentials_or_query_secrets() {
        let raw = "postgres://alice:p%40ssword@db.internal:5432/heka?password=second-secret&sslmode=require";
        let redacted = redact_database_url(raw);
        assert_eq!(redacted, "postgres://db.internal:5432/<database>");
        for secret in ["alice", "p%40ssword", "heka", "second-secret", "sslmode"] {
            assert!(!redacted.contains(secret));
        }

        assert_eq!(
            redact_database_url("not-a-url-containing-super-secret"),
            "<redacted-database-url>"
        );
    }

    #[test]
    fn emitted_connection_event_contains_no_url_secrets() {
        let raw =
            "postgres://alice:password@db.internal:5432/private_db?application_name=query-secret";
        let output = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer({
                let output = Arc::clone(&output);
                move || CapturedLog(Arc::clone(&output))
            })
            .finish();

        tracing::subscriber::with_default(subscriber, || log_database_connection(raw));

        let logged = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert!(logged.contains("postgres://db.internal:5432/<database>"));
        for secret in [
            "alice",
            "password",
            "private_db",
            "query-secret",
            "application_name",
        ] {
            assert!(!logged.contains(secret), "secret appeared in log: {logged}");
        }
    }

    #[tokio::test]
    async fn malformed_connection_error_never_echoes_the_raw_url() {
        let raw = "postgres://alice:password@/private_db?sslpassword=query-secret";
        let error = connect_database(raw, 1).await.unwrap_err().to_string();
        for secret in [
            "alice",
            "password",
            "private_db",
            "query-secret",
            "sslpassword",
        ] {
            assert!(!error.contains(secret), "secret appeared in error: {error}");
        }
        assert!(error.contains("<redacted-database-url>"));
    }

    #[test]
    fn instrumented_connect_options_never_hold_the_raw_url() {
        let raw =
            "postgres://alice:password@db.internal:5432/private_db?application_name=query-secret";
        let options = private_postgres_connect_options(raw).unwrap();
        let debug = format!("{options:?}");

        assert_eq!(options.get_url(), "postgres://redacted.invalid/postgres");
        for secret in [
            "alice",
            "password",
            "db.internal",
            "private_db",
            "query-secret",
            "application_name",
        ] {
            assert!(
                !debug.contains(secret),
                "secret appeared in ConnectOptions debug output: {debug}"
            );
        }
    }

    #[test]
    fn config_debug_and_parse_errors_never_expose_database_credentials() {
        let raw =
            "postgres://alice:password@db.internal:5432/private_db?application_name=query-secret";
        let database = DatabaseConfig {
            url: raw.to_string(),
            max_connections: 2,
        };
        let debug = format!("{database:?}");

        let path = std::env::temp_dir().join(format!(
            "heka-private-config-test-{}.toml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            format!("[database]\nurl = {raw:?}\nmax_connections = \"invalid\"\n"),
        )
        .unwrap();
        let error = HekaConfig::from_file(&path).unwrap_err().to_string();
        std::fs::remove_file(path).unwrap();

        for rendered in [&debug, &error] {
            for secret in ["alice", "password", "private_db", "query-secret"] {
                assert!(
                    !rendered.contains(secret),
                    "secret appeared in config diagnostic: {rendered}"
                );
            }
        }
    }
}
