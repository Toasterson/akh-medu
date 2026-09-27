//! `akh-harvest` — the transcript harvester binary (increment 3 of
//! `docs/ai/plans/2026-09-19-transcript-harvest.md`, ADR 119).
//!
//! An idempotent one-shot CLI, driven by a platform timer
//! (`docs/wayfinder/transcript-harvest/tickets/process-model.md`): discover
//! Claude Code, Codex and OpenCode sessions on this machine, resume from a
//! per-stream watermark, push new records to `harvest-pg`, advance the
//! watermark, exit. All orchestration logic lives in
//! [`akh_transcripts::harvest`] — this file is CLI parsing, environment
//! resolution, logging setup, and the exit-code contract.
//!
//! ## Environment
//!
//! | variable | required | default |
//! |---|---|---|
//! | `AKH_TRANSCRIPTS_DATABASE_URL` | yes | — |
//! | `AKH_TRANSCRIPTS_MACHINE` | no | this host's hostname |
//! | `AKH_TRANSCRIPTS_STATE_DIR` | no | `$XDG_STATE_HOME/akh-transcripts`, else `~/.local/state/akh-transcripts` |
//! | `AKH_TRANSCRIPTS_CLAUDE_ROOT` | no | `~/.claude/projects` |
//! | `AKH_TRANSCRIPTS_CODEX_ROOT` | no | `~/.codex/sessions` |
//! | `AKH_TRANSCRIPTS_OPENCODE_DB` | no | `~/.local/share/opencode/opencode.db` |
//! | `RUST_LOG` | no | `akh_harvest=info,akh_transcripts=info` |
//!
//! See `docs/ops/transcript-harvest.md` for the live deployment (auth, env
//! file location, systemd units).
//!
//! ## Exit code
//!
//! `0` even when individual streams failed to read or insert — those are
//! logged and folded into the `failed_streams` count in the final summary
//! line; the timer retries on its own schedule. Non-zero only when the
//! database is unreachable or the state directory is unusable, both
//! reported as a miette diagnostic naming the fix.

use clap::{Parser, ValueEnum};

use akh_transcripts::harvest::{HarvestConfig, RunOptions, run};
use akh_transcripts::model::Source;

/// Agent-session transcript harvester (ADR 119).
#[derive(Debug, Parser)]
#[command(name = "akh-harvest", about = "Agent-session transcript harvester (ADR 119)")]
struct Cli {
    /// Run once and exit. This is the only mode this binary has ever had —
    /// the flag exists because the systemd unit already passes it and a
    /// `--watch` mode remains a possible future addition
    /// (`process-model.md`).
    #[arg(long)]
    once: bool,

    /// Discover, read and forward-fill only. Writes nothing to the database
    /// and touches no watermark file.
    #[arg(long)]
    dry_run: bool,

    /// Restrict this run to one or more sources. Repeatable
    /// (`--source claude_code --source codex`). Omit for every source.
    #[arg(long = "source", value_enum)]
    source: Vec<CliSource>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "snake_case")]
enum CliSource {
    ClaudeCode,
    Codex,
    Opencode,
}

impl From<CliSource> for Source {
    fn from(value: CliSource) -> Self {
        match value {
            CliSource::ClaudeCode => Source::ClaudeCode,
            CliSource::Codex => Source::Codex,
            CliSource::Opencode => Source::Opencode,
        }
    }
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "akh_harvest=info,akh_transcripts=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();
    // `--once` changes nothing about how this binary behaves (it has never
    // supported anything else); logged so the field is not silently dead —
    // its only purpose is accepting the flag the systemd unit already
    // passes, per `process-model.md`'s "a `--watch` mode remains available
    // later" note.
    tracing::debug!(once = cli.once, dry_run = cli.dry_run, "akh-harvest starting");

    let cfg = HarvestConfig::from_env()?;
    let opts = RunOptions {
        dry_run: cli.dry_run,
        sources: cli.source.into_iter().map(Source::from).collect(),
    };

    let summary = run(&cfg, &opts).await?;
    // Printed unconditionally (not gated on RUST_LOG) so the operator- and
    // acceptance-facing summary line is always visible on stdout, exactly as
    // the contract's final line specifies. Counts only — never raw record
    // content, which may hold credentials (contract point: "never print raw
    // record content").
    println!("{}", summary.log_line());

    Ok(())
}
