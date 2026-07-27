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
//!       "type": "command",
//!       "command": "heka-capture"
//!     }]
//!   }
//! }
//! ```
//!
//! ## Input format
//!
//! The stop-hook payload is a JSON object with a `session_id` field and a
//! `transcript` array of turn objects, each with:
//! - `speaker`: "user" | "assistant" | "system"
//! - `body`: the turn text content
//! - optional `contact`: freeform contact label
//! - optional `topics`: array of topic strings
//!
//! If the payload has a flat `turns` array (older format), that is accepted too.
//! If no transcript is present, the raw stdin text is ingested as a single
//! "system" turn — this makes `heka-capture` usable as a pipe target:
//!
//! ```sh
//! echo "some text to remember" | heka-capture
//! ```
//!
//! ## Configuration
//!
//! Database URL is read from `DATABASE_URL` env var or `heka.toml`.
//! The akh is identified by `HEKA_AKH_NAME` env var (default: "tecton").

use std::io::Read;

use miette::{IntoDiagnostic, WrapErr};
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter};
use serde::Deserialize;
use uuid::Uuid;

use heka::entity::heka_akhs;
use heka::ingest::write_drawer;

/// Claude Code stop-hook payload.
#[derive(Debug, Deserialize, Default)]
struct StopHookPayload {
    session_id: Option<String>,
    /// Full transcript format (preferred): array of turns.
    transcript: Option<Vec<TurnPayload>>,
    /// Legacy format: flat turns array.
    turns: Option<Vec<TurnPayload>>,
    /// Contact label (applied to all turns if present).
    contact: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct TurnPayload {
    speaker: Option<String>,
    body: Option<String>,
    /// Per-turn contact override.
    contact: Option<String>,
    /// Per-turn topics.
    topics: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "heka=info".parse().unwrap()),
        )
        .init();

    // Read stdin.
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .into_diagnostic()
        .wrap_err("reading stdin")?;

    if input.trim().is_empty() {
        tracing::warn!("heka-capture: empty stdin, nothing to ingest");
        return Ok(());
    }

    // Parse the payload. If it's not valid JSON, treat the raw text as a
    // single system turn (pipe-friendly mode).
    let payload: StopHookPayload = if input.trim_start().starts_with('{') {
        serde_json::from_str(&input)
            .into_diagnostic()
            .wrap_err("parsing stop-hook JSON payload")?
    } else {
        StopHookPayload {
            transcript: Some(vec![TurnPayload {
                speaker: Some("system".to_string()),
                body: Some(input.clone()),
                contact: None,
                topics: None,
            }]),
            ..Default::default()
        }
    };

    // Collect turns from transcript or legacy turns field.
    let turns = payload.transcript.or(payload.turns).unwrap_or_default();

    if turns.is_empty() {
        tracing::warn!("heka-capture: no turns in payload, nothing to ingest");
        return Ok(());
    }

    // Connect to the database.
    let db_url = std::env::var("DATABASE_URL").map_err(|_| {
        miette::miette!("DATABASE_URL env var is required (or set [database] url in heka.toml)")
    })?;
    let db = sea_orm::Database::connect(&db_url)
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
                "akh '{akh_name}' not found in heka_akhs — register it first via `hekad register-akh`"
            )
        })?;

    let session_id = payload.session_id.as_deref();
    let default_contact = payload.contact.as_deref();

    let mut ingested = 0usize;
    let mut duplicated = 0usize;

    for turn in &turns {
        let body = match &turn.body {
            Some(b) if !b.trim().is_empty() => b,
            _ => continue,
        };

        let speaker = turn.speaker.as_deref();
        let contact = turn.contact.as_deref().or(default_contact);
        let topics = turn.topics.clone().unwrap_or_default();

        let summary = write_drawer(
            &db,
            akh.id,
            session_id,
            speaker,
            body,
            contact,
            &topics,
        )
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