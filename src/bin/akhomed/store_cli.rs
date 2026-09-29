//! `akhomed store …` — offline store maintenance (ADR 120).
//!
//! Runs before any daemon state is built: no config is loaded, no socket is
//! bound, no akh is woken. The only verb today is `split-per-akh`, which
//! copies a shared-layout SQLite store into the per-akh layout
//! ([`akhomed_store::split`]).

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};

/// `akhomed store` command line.
#[derive(Debug, Parser)]
#[command(
    name = "akhomed store",
    about = "Offline maintenance of akhomed's backing store"
)]
struct StoreCli {
    #[command(subcommand)]
    command: StoreCommand,
}

#[derive(Debug, Subcommand)]
enum StoreCommand {
    /// Copy a shared-layout SQLite store into per-akh files (ADR 120).
    ///
    /// Creates `--host` plus `akh.<name>.sqlite3` beside it for every akh in
    /// the source, copies each akh's rows in one transaction per file,
    /// verifies row counts and SHA-256 digests per table, and writes an
    /// evidence JSON. The source is opened read-only and never modified; the
    /// run refuses while an akhomed holds the source's lock files.
    SplitPerAkh(SplitArgs),
}

#[derive(Debug, clap::Args)]
struct SplitArgs {
    /// The shared-layout database: a path or the `sqlite://…` URL from
    /// `[database] url`.
    #[arg(long)]
    source: String,
    /// The new host database to create (must not exist). Akh files are
    /// created beside it; use this path as the new `[database] url`.
    #[arg(long)]
    host: PathBuf,
    /// Where to write the evidence JSON (must not exist). Default:
    /// `<host>.split-evidence.json`.
    #[arg(long)]
    evidence: Option<PathBuf>,
    /// Print what would be copied (per akh, per table) and exit. Read-only;
    /// works while the daemon runs.
    #[arg(long)]
    dry_run: bool,
    /// Leave rows that belong to no akh in the source instead of refusing.
    #[arg(long)]
    allow_orphans: bool,
    /// Print the plan / evidence as JSON instead of text.
    #[arg(long)]
    json: bool,
    /// SQLite busy timeout for the connections the split opens.
    #[arg(long, default_value_t = 30)]
    busy_timeout_secs: u64,
}

/// Run `akhomed store <args>` and return the process exit code.
pub async fn run(args: impl IntoIterator<Item = String>) -> i32 {
    let cli = match StoreCli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return if error.use_stderr() { 2 } else { 0 };
        }
    };
    match cli.command {
        StoreCommand::SplitPerAkh(args) => match split(args).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("{:?}", miette::Report::new(error));
                1
            }
        },
    }
}

async fn split(args: SplitArgs) -> Result<(), akhomed_store::split::SplitError> {
    use akhomed_store::split;
    let source =
        split::sqlite_url_path(&args.source).unwrap_or_else(|| PathBuf::from(&args.source));
    let request = split::SplitRequest {
        source,
        host: args.host.clone(),
        allow_orphans: args.allow_orphans,
        busy_timeout: Duration::from_secs(args.busy_timeout_secs),
    };
    if args.dry_run {
        let plan = split::plan(&request).await?;
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan).unwrap_or_default()
            );
        } else {
            print_plan(&plan);
        }
        return Ok(());
    }
    let evidence_path = args
        .evidence
        .unwrap_or_else(|| PathBuf::from(format!("{}.split-evidence.json", args.host.display())));
    let evidence = split::execute(&request, &evidence_path).await?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&evidence).unwrap_or_default()
        );
        return Ok(());
    }
    println!("split verified: {}", evidence.verified);
    println!(
        "source unchanged: {} (sha256 {})",
        evidence.source.sha256_before == evidence.source.sha256_after,
        evidence.source.sha256_after
    );
    println!("host  {}", evidence.host.path.display());
    print_rows(
        evidence
            .host
            .tables
            .iter()
            .map(|t| (t.table.as_str(), t.target_rows)),
    );
    for akh in &evidence.akhs {
        println!(
            "akh {} ({}) -> {}",
            akh.name,
            akh.akh_id,
            akh.target.path.display()
        );
        print_rows(
            akh.target
                .tables
                .iter()
                .map(|t| (t.table.as_str(), t.target_rows)),
        );
    }
    if !evidence.orphans_left_in_source.is_empty() {
        println!("orphans left in the source:");
        print_rows(
            evidence
                .orphans_left_in_source
                .iter()
                .map(|t| (t.table.as_str(), t.rows)),
        );
    }
    println!("evidence {}", evidence_path.display());
    println!(
        "next: set [database] url = \"sqlite://{}?mode=rwc\" and layout = \"per_akh\", then start akhomed",
        evidence.host.path.display()
    );
    Ok(())
}

fn print_rows<'a>(rows: impl Iterator<Item = (&'a str, u64)>) {
    let mut total = 0u64;
    for (table, count) in rows {
        total += count;
        if count > 0 {
            println!("  {table:<34} {count:>9}");
        }
    }
    println!("  {:<34} {total:>9}", "(total rows)");
}

fn print_plan(plan: &akhomed_store::split::SplitPlan) {
    println!("dry run — nothing is created or modified");
    println!("source {} (read-only)", plan.source.display());
    println!("host   {} (would be created)", plan.host.display());
    if plan.in_use.is_empty() {
        println!("in use: no lock file beside the source is held");
    } else {
        println!("in use: the real run refuses until these are released (stop akhomed):");
        for lock in &plan.in_use {
            println!("  {}", lock.display());
        }
    }
    println!("migrations: {}", plan.source_migrations.len());
    println!("host tables:");
    print_rows(plan.host_tables.iter().map(|t| (t.table.as_str(), t.rows)));
    for akh in &plan.akhs {
        println!(
            "akh {} ({}) -> {}",
            akh.name,
            akh.akh_id,
            akh.path.display()
        );
        print_rows(akh.tables.iter().map(|t| (t.table.as_str(), t.rows)));
    }
    if plan.orphans.is_empty() {
        println!("orphans: none");
    } else {
        println!(
            "orphans (rows attributed to no akh; the real run refuses without --allow-orphans):"
        );
        print_rows(plan.orphans.iter().map(|t| (t.table.as_str(), t.rows)));
    }
}
