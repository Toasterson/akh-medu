//! `heka-capture` — Claude Code stop-hook capture binary.
//!
//! Reads a Claude Code stop-hook JSON payload from stdin, parses session
//! turns, and calls `Ingest::write_drawer` for each turn. Designed as the
//! target of the `~/.claude/settings.json` Stop hook:
//!
//! ```json
//! {
//!   "hooks": {
//!     "Stop": [{
//!       "hooks": [{
//!         "type": "command",
//!         "command": "heka-capture"
//!       }]
//!     }]
//!   }
//! }
//! ```
//!
//! ## Input format
//!
//! Claude Code provides `session_id`, `transcript_path`, and
//! `last_assistant_message`. `heka-capture` parses user and assistant text from
//! the transcript JSONL and uses `last_assistant_message` when the asynchronously
//! written transcript does not yet contain the final response.
//!
//! The pre-Phase-42c inline `transcript`/`turns` format remains accepted. Each
//! inline turn contains:
//! - `speaker`: "user" | "assistant" | "system"
//! - `body`: the turn text content
//! - optional `contact`: freeform contact label
//! - optional `topics`: array of topic strings
//!
//! If the payload has a flat `turns` array (older format), that is accepted too.
//! Non-JSON stdin text is ingested as a single "system" turn, which keeps
//! `heka-capture` usable as a pipe target:
//!
//! ```sh
//! echo "some text to remember" | heka-capture
//! ```
//!
//! ## Configuration
//!
//! Database URL is read from `DATABASE_URL` env var or `heka.toml`.
//! The akh is identified by `HEKA_AKH_NAME` env var (default: "tecton").

use std::path::PathBuf;

use miette::{IntoDiagnostic, WrapErr};
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};

use heka::HekaConfig;
use heka::capture::{parse_capture_input, read_hook_input};
use heka::config::connect_database;
use heka::entity::heka_akhs;
use heka::ingest::write_drawer;

#[tokio::main]
async fn main() -> miette::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "heka=info".parse().unwrap()),
        )
        .init();

    // Read stdin.
    let input = read_hook_input(std::io::stdin().lock())
        .into_diagnostic()
        .wrap_err("reading stdin")?;

    if input.trim().is_empty() {
        tracing::warn!("heka-capture: empty stdin, nothing to ingest");
        return Ok(());
    }

    let batch = parse_capture_input(&input)
        .into_diagnostic()
        .wrap_err("parsing capture input")?;
    let turns = &batch.turns;

    if turns.is_empty() {
        tracing::warn!("heka-capture: no turns in payload, nothing to ingest");
        return Ok(());
    }

    // Connect to the database.
    let (db_url, max_connections) = match std::env::var("DATABASE_URL") {
        Ok(url) => (url, 10),
        Err(_) => {
            let config_path = default_config_path();
            let database = HekaConfig::from_file(&config_path)
                .into_diagnostic()
                .wrap_err_with(|| format!("loading config from {}", config_path.display()))?
                .database;
            (database.url, database.max_connections)
        }
    };
    let db = connect_database(&db_url, max_connections)
        .await
        .into_diagnostic()
        .wrap_err("connecting to PostgreSQL")?;

    // Resolve the akh ID from HEKA_AKH_NAME (default "tecton").
    let akh_name = std::env::var("HEKA_AKH_NAME").unwrap_or_else(|_| "tecton".to_string());
    let akh = heka_akhs::Entity::find()
        .filter(heka_akhs::Column::Name.eq(&akh_name))
        .one(&db)
        .await
        .into_diagnostic()
        .wrap_err_with(|| format!("looking up akh '{akh_name}'"))?
        .ok_or_else(|| {
            miette::miette!(
                "akh '{akh_name}' not found in heka_akhs — run `hekad register-akh {akh_name}` to register it before enabling the capture hook"
            )
        })?;

    let session_id = batch.session_id.as_deref();
    let default_contact = batch.contact.as_deref();

    let mut ingested = 0usize;
    let mut duplicated = 0usize;

    for turn in turns {
        let body = match &turn.body {
            Some(b) if !b.trim().is_empty() => b,
            _ => continue,
        };

        let speaker = turn.speaker.as_deref();
        let contact = turn.contact.as_deref().or(default_contact);
        let topics = turn.topics.clone().unwrap_or_default();

        let summary = write_drawer(&db, akh.id, session_id, speaker, body, contact, &topics)
            .await
            .into_diagnostic()
            .wrap_err("writing drawer")?;

        if summary.duplicated {
            duplicated += 1;
        } else {
            ingested += 1;
        }
    }

    tracing::info!(
        total_turns = turns.len(),
        ingested,
        duplicated,
        akh = %akh_name,
        session = ?session_id,
        "heka-capture complete"
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
