//! `akh-derive` — singleton derivation worker for the agent-session
//! transcript harvest (ADR 108).
//!
//! `--migrate` runs pending migrations (advisory-locked, via
//! [`akh_transcripts::pool::run_migrations_locked`]) and exits 0 — the Helm
//! `migrate` initContainer's path, run with the `harvest` owner role's
//! certificate rather than `harvest_derive`'s: default privileges for
//! `harvest_derive` only apply to objects `harvest` created
//! (`docs/ops/transcript-harvest.md`).
//!
//! Otherwise this binary takes the singleton advisory lock
//! ([`akh_transcripts::derive::acquire_singleton`]) — a second instance
//! against the same database refuses to start rather than queuing up behind
//! the first, per the derivation-worker ticket's runtime decision — and
//! either runs one pass (`--once`) or loops forever at
//! `AKH_TRANSCRIPTS_DERIVE_INTERVAL_SECS`.
//!
//! Env:
//! - `AKH_TRANSCRIPTS_DATABASE_URL` (required)
//! - `AKH_TRANSCRIPTS_MODEL_DIR` (required unless `--no-embed`)
//! - `AKH_TRANSCRIPTS_DERIVE_INTERVAL_SECS` (default 30)
//! - `AKH_TRANSCRIPTS_DERIVE_BATCH` (default 200 — pending *streams* per pass)
//! - `RUST_LOG`

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clap::Parser;

use akh_embed::SentenceEmbedder;
use akh_transcripts::TranscriptError;
use akh_transcripts::derive::{self, DeriveConfig};
use akh_transcripts::pool;

/// Derivation worker for the agent-session transcript harvest: normalizes
/// `th_raw_records` into `th_events`, rolls up `th_sessions`, chunks
/// `prompt`/`reply` runs into `th_chunks`, and embeds them.
#[derive(Parser, Debug)]
#[command(name = "akh-derive", version, about)]
struct Cli {
    /// Run pending migrations against AKH_TRANSCRIPTS_DATABASE_URL, then
    /// exit 0, instead of deriving anything.
    #[arg(long)]
    migrate: bool,

    /// Run exactly one derivation pass and exit, instead of looping forever.
    #[arg(long)]
    once: bool,

    /// Skip the embedding phase. AKH_TRANSCRIPTS_MODEL_DIR becomes optional
    /// in this mode — for environments without the MiniLM model on disk.
    #[arg(long)]
    no_embed: bool,
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn env_u64(name: &str, default: u64) -> u64 {
    env_var(name)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let Some(database_url) = env_var("AKH_TRANSCRIPTS_DATABASE_URL") else {
        eprintln!(
            "akh-derive: AKH_TRANSCRIPTS_DATABASE_URL is required — see \
             docs/ops/transcript-harvest.md for the mTLS connection-string shape."
        );
        return ExitCode::FAILURE;
    };

    if cli.migrate {
        return match pool::run_migrations_locked(&database_url).await {
            Ok(()) => {
                tracing::info!(target: "akh_transcripts::derive", "akh-derive --migrate: migrations applied");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err:?}");
                ExitCode::FAILURE
            }
        };
    }

    let embedder = if cli.no_embed {
        None
    } else {
        let Some(model_dir) = env_var("AKH_TRANSCRIPTS_MODEL_DIR") else {
            eprintln!(
                "akh-derive: AKH_TRANSCRIPTS_MODEL_DIR is required unless --no-embed is passed."
            );
            return ExitCode::FAILURE;
        };
        match SentenceEmbedder::load(&PathBuf::from(&model_dir)) {
            Ok(embedder) => Some(Arc::new(Mutex::new(embedder))),
            Err(err) => {
                eprintln!("akh-derive: failed to load embedding model from {model_dir}: {err}");
                return ExitCode::FAILURE;
            }
        }
    };

    // Held for the rest of the process's life — see `derive::acquire_singleton`.
    let _guard = match derive::acquire_singleton(&database_url).await {
        Ok(guard) => guard,
        Err(TranscriptError::DeriveAlreadyRunning) => {
            eprintln!(
                "akh-derive: another akh-derive process already holds the singleton advisory \
                 lock for this database; refusing to start a second one \
                 (derivation-worker.md, decision 4)."
            );
            return ExitCode::FAILURE;
        }
        Err(err) => {
            eprintln!("{err:?}");
            return ExitCode::FAILURE;
        }
    };

    let db = match pool::connect(&database_url).await {
        Ok(db) => db,
        Err(err) => {
            eprintln!("{err:?}");
            return ExitCode::FAILURE;
        }
    };

    let config = DeriveConfig {
        batch_streams: env_u64(
            "AKH_TRANSCRIPTS_DERIVE_BATCH",
            derive::DEFAULT_BATCH_STREAMS,
        ),
        embed_batch: if cli.no_embed {
            None
        } else {
            Some(derive::DEFAULT_EMBED_BATCH)
        },
    };

    if cli.once {
        return match derive::run_pass(&db, &config, embedder.as_ref()).await {
            Ok(summary) => {
                tracing::info!(target: "akh_transcripts::derive", "{}", summary);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err:?}");
                ExitCode::FAILURE
            }
        };
    }

    let interval = Duration::from_secs(env_u64("AKH_TRANSCRIPTS_DERIVE_INTERVAL_SECS", 30));
    tracing::info!(target: "akh_transcripts::derive", 
        interval_secs = interval.as_secs(),
        batch_streams = config.batch_streams,
        embed = embedder.is_some(),
        "akh-derive: singleton acquired, entering the derivation loop"
    );
    loop {
        match _guard.reassert().await {
            Ok(true) => {}
            Ok(false) => {
                tracing::error!(target: "akh_transcripts::derive", "akh-derive: singleton advisory lock is now held by another process; exiting so the supervisor restarts us cleanly");
                std::process::exit(3);
            }
            Err(err) => tracing::error!(target: "akh_transcripts::derive", "akh-derive: could not re-assert the singleton lock: {err:?}"),
        }
        match derive::run_pass(&db, &config, embedder.as_ref()).await {
            Ok(summary) => tracing::info!(target: "akh_transcripts::derive", "{}", summary),
            Err(err) => tracing::error!(target: "akh_transcripts::derive", "{:?}", err),
        }
        tokio::time::sleep(interval).await;
    }
}
