//! Phase 40 wake-path measurement harness.
//!
//! Measures the cost of rebuilding derived structures at wake:
//! e-graph re-saturation (`--what egraph`), HNSW(VSA) index rebuild
//! (`--what hnsw`), or both (`--what both`). See
//! `docs/ai/plans/2026-04-25-phase40-egraph-persistence.md` (Anima ticket
//! `019dc59a-513a-74a1-a088-d1524d891095`) for methodology and the decision
//! criteria the output feeds.
//!
//! Each grid cell runs in its own subprocess so peak-RSS (`VmHWM`) is
//! attributable per cell. The parent writes raw JSON results plus a
//! markdown table snippet for the report.

mod arms;
mod spec;
mod theory;

use std::io::Write as _;
use std::path::PathBuf;
use std::process::Command;

use clap::{Parser, Subcommand, ValueEnum};
use miette::{Context, IntoDiagnostic, Result, miette};

use spec::{CellResult, CellSpec, Limits, RuleSet, median};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum What {
    Egraph,
    Hnsw,
    Both,
}

#[derive(Parser)]
#[command(
    name = "akh-egraph-bench",
    about = "Phase 40 wake-path measurement harness (e-graph re-saturation + HNSW rebuild)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Which arm(s) to measure.
    #[arg(long, value_enum, default_value = "egraph")]
    what: What,

    /// Write raw per-cell JSON results here.
    #[arg(long, default_value = "phase40-results.json")]
    out_json: PathBuf,

    /// Write markdown result tables here (report building blocks).
    #[arg(long, default_value = "phase40-tables.md")]
    report: PathBuf,

    /// Reduced smoke-test grid (sizes capped at 1k, 2 reps).
    #[arg(long)]
    quick: bool,

    /// RNG seed for the synthetic theory generator.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Subcommand)]
enum Cmd {
    /// Internal: run a single cell (JSON spec via --spec) and print a JSON
    /// result line on stdout. Invoked by the parent per grid cell.
    Cell {
        #[arg(long)]
        spec: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(Cmd::Cell { spec }) = &cli.cmd {
        let spec: CellSpec = serde_json::from_str(spec)
            .into_diagnostic()
            .wrap_err("cell spec is not valid JSON — the parent and cell binaries disagree")?;
        let result = arms::run_cell(&spec);
        println!(
            "{}",
            serde_json::to_string(&result)
                .into_diagnostic()
                .wrap_err("serializing cell result")?
        );
        return Ok(());
    }

    let grid = build_grid(cli.what, cli.quick, cli.seed);
    eprintln!("phase40 harness: {} cells", grid.len());

    let exe = std::env::current_exe()
        .into_diagnostic()
        .wrap_err("cannot locate own executable for cell subprocesses")?;

    let mut results: Vec<CellResult> = Vec::with_capacity(grid.len());
    for (i, cell) in grid.iter().enumerate() {
        let spec_json = serde_json::to_string(cell).into_diagnostic()?;
        eprintln!("[{}/{}] {}", i + 1, grid.len(), spec_json);
        let t = std::time::Instant::now();
        let out = Command::new(&exe)
            .arg("cell")
            .arg("--spec")
            .arg(&spec_json)
            .output()
            .into_diagnostic()
            .wrap_err("spawning cell subprocess")?;
        if !out.status.success() {
            return Err(miette!(
                code = "phase40::cell_failed",
                help = format!("re-run the failing cell directly: akh-egraph-bench cell --spec '{spec_json}'"),
                "cell {} failed with {}: {}",
                spec_json,
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        // Libraries under measurement may print noise to stdout (hnsw_rs
        // logs "setting number of points ..."), so the cell contract is:
        // the *last* non-empty stdout line is the JSON payload.
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        let result: CellResult = serde_json::from_str(line.trim())
            .into_diagnostic()
            .wrap_err_with(|| format!("parsing cell result line: {line}"))?;
        eprintln!("    done in {:.1}s", t.elapsed().as_secs_f64());

        // Persist incrementally (JSONL sidecar) so a failing late cell
        // cannot lose the results of every cell before it.
        let mut partial = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(cli.out_json.with_extension("partial.jsonl"))
            .into_diagnostic()
            .wrap_err("opening incremental JSONL sidecar")?;
        writeln!(partial, "{}", line.trim()).into_diagnostic()?;

        results.push(result);
    }

    let json = serde_json::to_string_pretty(&results).into_diagnostic()?;
    std::fs::write(&cli.out_json, json)
        .into_diagnostic()
        .wrap_err_with(|| format!("writing {}", cli.out_json.display()))?;

    let tables = render_tables(&results);
    std::fs::write(&cli.report, &tables)
        .into_diagnostic()
        .wrap_err_with(|| format!("writing {}", cli.report.display()))?;

    // Also echo the tables so an interactive run sees them immediately.
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(tables.as_bytes()).into_diagnostic()?;
    eprintln!(
        "\nraw JSON: {} — tables: {}",
        cli.out_json.display(),
        cli.report.display()
    );
    Ok(())
}

/// The measurement grid from the plan (§40b), plus the Q2 fork arm.
fn build_grid(what: What, quick: bool, seed: u64) -> Vec<CellSpec> {
    let mut grid = Vec::new();

    if matches!(what, What::Egraph | What::Both) {
        let (sizes, reps): (&[usize], usize) = if quick {
            (&[100, 1_000], 2)
        } else {
            (&[100, 1_000, 10_000, 100_000], 5)
        };
        // Production-limit sweep: the actual wake-path exposure.
        for &n_facts in sizes {
            for rule_set in [RuleSet::Small, RuleSet::Medium, RuleSet::Large] {
                for branching in [1usize, 3, 10] {
                    grid.push(CellSpec::Egraph {
                        n_facts,
                        rule_set,
                        branching,
                        limits: Limits::Prod,
                        seed,
                        reps,
                    });
                }
            }
        }
        // Generous-limit sweep: what full saturation would cost without the
        // engine's guard rails (the number the ADR thresholds compare).
        let generous_sizes: &[usize] = if quick { &[1_000] } else { &[1_000, 10_000, 100_000] };
        for &n_facts in generous_sizes {
            grid.push(CellSpec::Egraph {
                n_facts,
                rule_set: RuleSet::Medium,
                branching: 3,
                limits: Limits::Generous,
                seed,
                reps: 3,
            });
        }
        if !quick {
            // The deliberate worst case: the non-terminating epistemic pack
            // under generous limits.
            grid.push(CellSpec::Egraph {
                n_facts: 10_000,
                rule_set: RuleSet::Large,
                branching: 3,
                limits: Limits::Generous,
                seed,
                reps: 3,
            });
        }
        // Q2 fork workload.
        grid.push(CellSpec::Fork {
            n_facts: if quick { 200 } else { 1_000 },
            forks: if quick { 10 } else { 100 },
            seed,
        });
    }

    if matches!(what, What::Hnsw | What::Both) {
        let sizes: &[usize] = if quick { &[1_000] } else { &[1_000, 10_000, 100_000] };
        for &n_items in sizes {
            grid.push(CellSpec::Hnsw { n_items, reps: 3 });
        }
    }

    grid
}

fn render_tables(results: &[CellResult]) -> String {
    let mut s = String::new();
    let mut egraph_rows = String::new();
    let mut hnsw_rows = String::new();
    let mut fork_rows = String::new();

    for r in results {
        match r {
            CellResult::Egraph {
                spec,
                build_expr_ms,
                initial_nodes,
                reps,
                peak_rss_mb,
            } => {
                let CellSpec::Egraph {
                    n_facts,
                    rule_set,
                    branching,
                    limits,
                    ..
                } = spec.as_ref()
                else {
                    continue;
                };
                let sat: Vec<f64> = reps.iter().map(|x| x.saturate_ms).collect();
                let ext: Vec<f64> = reps.iter().map(|x| x.extract_ms).collect();
                let last = reps.last().expect("at least one rep");
                egraph_rows.push_str(&format!(
                    "| {} | {} | {} | {} | {:.1} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {} | {} | {:.0} |\n",
                    n_facts,
                    rule_set.label(),
                    branching,
                    limits.label(),
                    build_expr_ms,
                    initial_nodes,
                    median(&sat),
                    sat.iter().cloned().fold(f64::INFINITY, f64::min),
                    sat.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    median(&ext),
                    last.stop_reason,
                    last.e_classes,
                    last.e_nodes,
                    peak_rss_mb,
                ));
            }
            CellResult::Hnsw {
                spec,
                cold_build_ms,
                rebuild_ms,
                peak_rss_mb,
            } => {
                let CellSpec::Hnsw { n_items, .. } = spec.as_ref() else {
                    continue;
                };
                hnsw_rows.push_str(&format!(
                    "| {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} |\n",
                    n_items,
                    cold_build_ms,
                    median(rebuild_ms),
                    rebuild_ms.iter().cloned().fold(f64::INFINITY, f64::min),
                    rebuild_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    peak_rss_mb,
                ));
            }
            CellResult::Fork {
                spec,
                base_saturate_ms,
                fork_ms,
                peak_rss_mb,
            } => {
                let CellSpec::Fork { n_facts, forks, .. } = spec.as_ref() else {
                    continue;
                };
                let total: f64 = fork_ms.iter().sum();
                fork_rows.push_str(&format!(
                    "| {} | {} | {:.1} | {:.2} | {:.2} | {:.1} | {:.0} |\n",
                    n_facts,
                    forks,
                    base_saturate_ms,
                    median(fork_ms),
                    fork_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    total,
                    peak_rss_mb,
                ));
            }
        }
    }

    if !egraph_rows.is_empty() {
        s.push_str("### E-graph re-saturation\n\n");
        s.push_str(
            "| n_facts | rules | branch | limits | build ms | init nodes | sat p50 ms | sat min | sat max | extract p50 ms | stop | e-classes | e-nodes | peak RSS MiB |\n",
        );
        s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        s.push_str(&egraph_rows);
        s.push('\n');
    }
    if !fork_rows.is_empty() {
        s.push_str("### Q2 fork workload (clone + perturb + re-run, medium rules, prod limits)\n\n");
        s.push_str(
            "| n_facts | forks | base sat ms | fork p50 ms | fork max ms | total ms | peak RSS MiB |\n",
        );
        s.push_str("|---|---|---|---|---|---|---|\n");
        s.push_str(&fork_rows);
        s.push('\n');
    }
    if !hnsw_rows.is_empty() {
        s.push_str("### HNSW(VSA) rebuild (10 000-bit vectors)\n\n");
        s.push_str(
            "| n_items | cold build ms | rebuild p50 ms | rebuild min | rebuild max | peak RSS MiB |\n",
        );
        s.push_str("|---|---|---|---|---|---|\n");
        s.push_str(&hnsw_rows);
        s.push('\n');
    }
    s
}
