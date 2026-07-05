//! `akhomed-migrate` — run akhomed-store schema migrations against PostgreSQL.
//!
//! Connection URL resolution, highest precedence first:
//!   1. `--url <URL>`
//!   2. `DATABASE_URL` environment variable
//!   3. `[postgres].url` in the TOML file given by `--config <PATH>`
//!
//! Usage:
//!   akhomed-migrate --url postgres://user:pass@host:5432/akhmedu
//!   akhomed-migrate --config /etc/akh-medu/config.toml
//!   DATABASE_URL=postgres://... akhomed-migrate

use std::path::PathBuf;

use miette::{IntoDiagnostic, WrapErr, miette};
use serde::Deserialize;

/// Minimal view of akhomed's config.toml — just enough to read `[postgres]`.
/// We parse our own slice rather than depending on the main crate (which would
/// create a dependency cycle).
#[derive(Debug, Deserialize)]
struct ConfigFile {
    postgres: Option<PostgresSection>,
}

#[derive(Debug, Deserialize)]
struct PostgresSection {
    url: String,
    #[serde(default)]
    max_connections: Option<u32>,
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "akhomed_store=info,akhomed_migrate=info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mut url_flag: Option<String> = None;
    let mut config_flag: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => {
                url_flag = Some(next_value(&args, &mut i, "--url")?);
            }
            "--config" => {
                config_flag = Some(PathBuf::from(next_value(&args, &mut i, "--config")?));
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => {
                return Err(miette!(
                    "unexpected argument: {other}\n\nRun `akhomed-migrate --help` for usage."
                ));
            }
        }
        i += 1;
    }

    let (url, max_connections) = resolve_connection(url_flag, config_flag)?;

    let mut opts = akhomed_store::PoolOptions::new(url);
    if let Some(max) = max_connections {
        opts.max_connections = max;
    }

    // Advisory-locked migration: opens its own dedicated single connection,
    // serializing against any other host migrating the same cluster at once.
    tracing::info!("connecting to PostgreSQL and applying migrations (advisory-locked)");
    akhomed_store::run_migrations_locked(&opts)
        .await
        .wrap_err("applying akhomed-store migrations")?;

    tracing::info!("akhomed-store migrations complete");
    Ok(())
}

/// Resolve the connection URL (and optional pool size) from flags, env, or file.
fn resolve_connection(
    url_flag: Option<String>,
    config_flag: Option<PathBuf>,
) -> miette::Result<(String, Option<u32>)> {
    if let Some(url) = url_flag {
        return Ok((url, None));
    }
    if let Ok(url) = std::env::var("DATABASE_URL") {
        return Ok((url, None));
    }
    if let Some(path) = config_flag {
        let content = std::fs::read_to_string(&path)
            .into_diagnostic()
            .wrap_err_with(|| format!("reading config {}", path.display()))?;
        let parsed: ConfigFile = toml::from_str(&content)
            .into_diagnostic()
            .wrap_err_with(|| format!("parsing TOML config {}", path.display()))?;
        let pg = parsed.postgres.ok_or_else(|| {
            miette!(
                "config {} has no [postgres] section; nothing to migrate against",
                path.display()
            )
        })?;
        return Ok((pg.url, pg.max_connections));
    }

    Err(miette!(
        "no connection URL provided.\n\nProvide one of:\n  \
         --url <URL>\n  DATABASE_URL=<URL>\n  --config <config.toml> (with a [postgres] section)"
    ))
}

/// Pull the value following a flag, advancing the index.
fn next_value(args: &[String], i: &mut usize, flag: &str) -> miette::Result<String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| miette!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "akhomed-migrate — run akhomed-store schema migrations\n\n\
         USAGE:\n  \
         akhomed-migrate [--url <URL> | --config <PATH>]\n\n\
         The connection URL is resolved in order: --url, then DATABASE_URL,\n\
         then the [postgres].url field of the --config TOML file."
    );
}
