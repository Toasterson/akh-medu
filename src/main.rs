//! akh-medu CLI: neuro-symbolic AI engine.

use std::collections::HashSet;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::error::EngineError;
use akh_medu::graph::traverse::TraversalConfig;
use akh_medu::graph::Triple;
use akh_medu::infer::InferenceQuery;
use akh_medu::provenance::DerivationKind;
use akh_medu::symbol::SymbolId;
use akh_medu::vsa::Dimension;

#[derive(Parser)]
#[command(name = "akh-medu", version, about = "Neuro-symbolic AI engine")]
struct Cli {
    /// Data directory for persistent storage.
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,

    /// Hypervector dimension.
    #[arg(long, global = true, default_value = "10000")]
    dimension: usize,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new akh-medu data directory.
    Init,

    /// Ingest triples from a JSON file (label-based or numeric format).
    Ingest {
        /// Path to JSON file with triples.
        #[arg(long)]
        file: PathBuf,
    },

    /// Query the knowledge base using spreading-activation inference.
    Query {
        /// Seed symbols (comma-separated names or IDs, e.g. "Sun,Moon" or "1,2").
        #[arg(long)]
        seeds: String,

        /// Number of results to return.
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Maximum inference depth.
        #[arg(long, default_value = "1")]
        max_depth: usize,
    },

    /// Traverse the knowledge graph from seed nodes using BFS.
    Traverse {
        /// Seed symbols (comma-separated names or IDs).
        #[arg(long)]
        seeds: String,

        /// Maximum traversal depth.
        #[arg(long, default_value = "3")]
        max_depth: usize,

        /// Only follow these predicates (comma-separated, optional).
        #[arg(long)]
        predicates: Option<String>,

        /// Minimum confidence threshold.
        #[arg(long, default_value = "0.0")]
        min_confidence: f32,

        /// Maximum number of triples to collect.
        #[arg(long, default_value = "1000")]
        max_results: usize,

        /// Output format: "summary" or "json".
        #[arg(long, default_value = "summary")]
        format: String,
    },

    /// Run a SPARQL query against the knowledge graph.
    Sparql {
        /// Inline SPARQL query string.
        #[arg(long)]
        query: Option<String>,

        /// Path to a SPARQL query file (.rq).
        #[arg(long)]
        file: Option<PathBuf>,
    },

    /// Simplify a symbolic expression using e-graph rewriting.
    Reason {
        /// Expression to simplify (e.g. "(not (not x))").
        #[arg(long)]
        expr: String,

        /// Show rule count and extra details.
        #[arg(long)]
        verbose: bool,
    },

    /// Search for symbols similar to a given symbol via VSA.
    Search {
        /// Symbol name or numeric ID to search around.
        #[arg(long)]
        symbol: String,

        /// Number of results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Compute an analogy: A:B :: C:? via VSA bind/unbind.
    Analogy {
        /// First symbol (A).
        #[arg(long)]
        a: String,

        /// Second symbol (B).
        #[arg(long)]
        b: String,

        /// Third symbol (C).
        #[arg(long)]
        c: String,

        /// Number of results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Recover the filler for a (subject, predicate) pair via VSA unbind.
    Filler {
        /// Subject symbol name or ID.
        #[arg(long)]
        subject: String,

        /// Predicate symbol name or ID.
        #[arg(long)]
        predicate: String,

        /// Number of results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Show engine info and statistics.
    Info,

    /// List and inspect symbols.
    Symbols {
        #[command(subcommand)]
        action: SymbolAction,
    },

    /// Export engine data as JSON.
    Export {
        #[command(subcommand)]
        action: ExportAction,
    },

    /// Manage skillpacks.
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
}

#[derive(Subcommand)]
enum SymbolAction {
    /// List all registered symbols.
    List,
    /// Show details of a specific symbol (by name or ID).
    Show {
        /// Symbol name or numeric ID.
        name_or_id: String,
    },
}

#[derive(Subcommand)]
enum ExportAction {
    /// Export the symbol table as JSON.
    Symbols,
    /// Export all triples as JSON.
    Triples,
    /// Export provenance chain for a symbol as JSON.
    Provenance {
        /// Symbol name or numeric ID.
        name_or_id: String,
    },
}

#[derive(Subcommand)]
enum SkillAction {
    /// List all discovered skillpacks.
    List,
    /// Load (discover + warm + activate) a skillpack.
    Load {
        /// Skill name (directory name under skills/).
        name: String,
    },
    /// Unload (deactivate) a skillpack.
    Unload {
        /// Skill name.
        name: String,
    },
    /// Show details about a skillpack.
    Info {
        /// Skill name.
        name: String,
    },
    /// Scaffold a new skillpack with template files.
    Scaffold {
        /// Name for the new skillpack.
        name: String,
    },
}

fn main() -> Result<()> {
    miette::set_hook(Box::new(|_| {
        Box::new(
            miette::MietteHandlerOpts::new()
                .terminal_links(true)
                .unicode(true)
                .context_lines(3)
                .build(),
        )
    }))
    .ok(); // Ignore error if hook already set (e.g., in tests)

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let config = EngineConfig {
        dimension: Dimension(cli.dimension),
        data_dir: cli.data_dir.clone(),
        ..Default::default()
    };

    match cli.command {
        Commands::Init => {
            let data_dir = cli.data_dir.unwrap_or_else(|| PathBuf::from(".akh-medu"));
            let config = EngineConfig {
                data_dir: Some(data_dir.clone()),
                dimension: Dimension(cli.dimension),
                ..Default::default()
            };
            let engine = Engine::new(config).into_diagnostic()?;
            println!("Initialized akh-medu at {}", data_dir.display());
            println!("{}", engine.info());
        }

        Commands::Ingest { file } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let content = std::fs::read_to_string(&file).into_diagnostic()?;

            let triples: Vec<serde_json::Value> =
                serde_json::from_str(&content).into_diagnostic()?;

            if triples.is_empty() {
                println!("No triples found in {}", file.display());
                return Ok(());
            }

            // Auto-detect format from first element.
            let first = &triples[0];
            let is_label_format = first.get("subject").is_some();
            let is_numeric_format = first.get("s").is_some();

            if is_label_format {
                // Label-based format: {"subject": "Sun", "predicate": "is-a", "object": "Star"}
                let mut label_triples = Vec::new();
                for (i, val) in triples.iter().enumerate() {
                    let subject = val["subject"].as_str().ok_or_else(|| {
                        EngineError::IngestFormat {
                            message: format!("triple {i}: missing or non-string 'subject' field"),
                        }
                    }).into_diagnostic()?;
                    let predicate = val["predicate"].as_str().ok_or_else(|| {
                        EngineError::IngestFormat {
                            message: format!("triple {i}: missing or non-string 'predicate' field"),
                        }
                    }).into_diagnostic()?;
                    let object = val["object"].as_str().ok_or_else(|| {
                        EngineError::IngestFormat {
                            message: format!("triple {i}: missing or non-string 'object' field"),
                        }
                    }).into_diagnostic()?;
                    let confidence = val["confidence"].as_f64().unwrap_or(1.0) as f32;

                    label_triples.push((
                        subject.to_string(),
                        predicate.to_string(),
                        object.to_string(),
                        confidence,
                    ));
                }

                let (created, ingested) = engine
                    .ingest_label_triples(&label_triples)
                    .into_diagnostic()?;

                // Persist after ingest.
                let _ = engine.persist();

                println!(
                    "Ingested {ingested} triples ({created} new symbols) from {}",
                    file.display()
                );
            } else if is_numeric_format {
                // Numeric format: {"s": 1, "p": 2, "o": 3}
                let mut count = 0;
                for val in &triples {
                    let s = val["s"].as_u64().unwrap_or(0);
                    let p = val["p"].as_u64().unwrap_or(0);
                    let o = val["o"].as_u64().unwrap_or(0);
                    let confidence = val["confidence"].as_f64().unwrap_or(1.0) as f32;

                    if let (Some(s), Some(p), Some(o)) =
                        (SymbolId::new(s), SymbolId::new(p), SymbolId::new(o))
                    {
                        engine
                            .add_triple(&Triple::new(s, p, o).with_confidence(confidence))
                            .into_diagnostic()?;
                        count += 1;
                    }
                }
                println!("Ingested {count} triples from {}", file.display());
            } else {
                return Err(EngineError::IngestFormat {
                    message: "unrecognized triple format in first element".into(),
                })
                .into_diagnostic();
            }

            println!("{}", engine.info());
        }

        Commands::Query {
            seeds,
            top_k,
            max_depth,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let seed_ids: std::result::Result<Vec<SymbolId>, _> = seeds
                .split(',')
                .map(|s| engine.resolve_symbol(s.trim()))
                .collect();
            let seed_ids = seed_ids.into_diagnostic()?;

            if seed_ids.is_empty() {
                miette::bail!("no valid seed symbols provided");
            }

            let query = InferenceQuery {
                seeds: seed_ids,
                top_k,
                max_depth,
                ..Default::default()
            };

            let result = engine.infer(&query).into_diagnostic()?;

            println!("Inference results (top {top_k}, depth {max_depth}):");
            for (i, (sym_id, confidence)) in result.activations.iter().enumerate() {
                let label = engine.resolve_label(*sym_id);
                println!(
                    "  {}. \"{}\" / {} (confidence: {:.4})",
                    i + 1,
                    label,
                    sym_id,
                    confidence
                );
            }

            if !result.provenance.is_empty() {
                println!("\nProvenance:");
                for record in &result.provenance {
                    let derived_label = engine.resolve_label(record.derived_id);
                    let kind_desc = format_derivation_kind(&record.kind, &engine);
                    println!(
                        "  \"{}\" / {} depth={} confidence={:.4} [{}]",
                        derived_label,
                        record.derived_id,
                        record.depth,
                        record.confidence,
                        kind_desc
                    );
                }
            }
        }

        Commands::Traverse {
            seeds,
            max_depth,
            predicates,
            min_confidence,
            max_results,
            format,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let seed_ids: std::result::Result<Vec<SymbolId>, _> = seeds
                .split(',')
                .map(|s| engine.resolve_symbol(s.trim()))
                .collect();
            let seed_ids = seed_ids.into_diagnostic()?;

            let predicate_filter: HashSet<SymbolId> = if let Some(ref preds) = predicates {
                let ids: std::result::Result<Vec<SymbolId>, _> = preds
                    .split(',')
                    .map(|s| engine.resolve_symbol(s.trim()))
                    .collect();
                ids.into_diagnostic()?.into_iter().collect()
            } else {
                HashSet::new()
            };

            let traverse_config = TraversalConfig {
                max_depth,
                predicate_filter,
                min_confidence,
                max_results,
            };

            let result = engine.traverse(&seed_ids, traverse_config).into_diagnostic()?;

            if format == "json" {
                let json_triples: Vec<serde_json::Value> = result
                    .triples
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "subject": engine.resolve_label(t.subject),
                            "predicate": engine.resolve_label(t.predicate),
                            "object": engine.resolve_label(t.object),
                            "confidence": t.confidence,
                        })
                    })
                    .collect();
                let json = serde_json::to_string_pretty(&json_triples).into_diagnostic()?;
                println!("{json}");
            } else {
                println!(
                    "Traversal: {} triples, {} nodes, depth {}",
                    result.triples.len(),
                    result.visited.len(),
                    result.depth_reached
                );
                for t in &result.triples {
                    println!(
                        "  \"{}\" -> {} -> \"{}\"  [{:.2}]",
                        engine.resolve_label(t.subject),
                        engine.resolve_label(t.predicate),
                        engine.resolve_label(t.object),
                        t.confidence,
                    );
                }
            }
        }

        Commands::Sparql { query, file } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let sparql_str = if let Some(q) = query {
                q
            } else if let Some(path) = file {
                std::fs::read_to_string(&path).into_diagnostic()?
            } else {
                miette::bail!("provide either --query or --file for SPARQL");
            };

            let results = engine.sparql_query(&sparql_str).into_diagnostic()?;

            if results.is_empty() {
                println!("No results.");
            } else {
                // Print header from first row's variable names.
                if let Some(first_row) = results.first() {
                    let header: Vec<&str> = first_row.iter().map(|(k, _)| k.as_str()).collect();
                    println!("{}", header.join("\t"));
                }
                for row in &results {
                    let vals: Vec<&str> = row.iter().map(|(_, v)| v.as_str()).collect();
                    println!("{}", vals.join("\t"));
                }
            }
        }

        Commands::Reason { expr, verbose } => {
            let engine = Engine::new(config).into_diagnostic()?;

            if verbose {
                let rules = engine.all_rules();
                println!("Active rules: {}", rules.len());
            }

            println!("Input:      {expr}");
            let simplified = engine.simplify_expression(&expr).into_diagnostic()?;
            println!("Simplified: {simplified}");
        }

        Commands::Search { symbol, top_k } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let sym_id = engine.resolve_symbol(&symbol).into_diagnostic()?;
            let label = engine.resolve_label(sym_id);

            let results = engine.search_similar_to(sym_id, top_k).into_diagnostic()?;

            println!("Similar to \"{label}\" (top {top_k}):");
            for (i, sr) in results.iter().enumerate() {
                let sr_label = engine.resolve_label(sr.symbol_id);
                println!(
                    "  {}. \"{}\" / {} (similarity: {:.4})",
                    i + 1,
                    sr_label,
                    sr.symbol_id,
                    sr.similarity
                );
            }
        }

        Commands::Analogy { a, b, c, top_k } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let a_id = engine.resolve_symbol(&a).into_diagnostic()?;
            let b_id = engine.resolve_symbol(&b).into_diagnostic()?;
            let c_id = engine.resolve_symbol(&c).into_diagnostic()?;

            let a_label = engine.resolve_label(a_id);
            let b_label = engine.resolve_label(b_id);
            let c_label = engine.resolve_label(c_id);

            let results = engine.infer_analogy(a_id, b_id, c_id, top_k).into_diagnostic()?;

            println!("Analogy: \"{a_label}\" : \"{b_label}\" :: \"{c_label}\" : ?");
            for (i, (sym_id, confidence)) in results.iter().enumerate() {
                let label = engine.resolve_label(*sym_id);
                println!(
                    "  {}. \"{}\" / {} (confidence: {:.4})",
                    i + 1,
                    label,
                    sym_id,
                    confidence
                );
            }
        }

        Commands::Filler {
            subject,
            predicate,
            top_k,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let subj_id = engine.resolve_symbol(&subject).into_diagnostic()?;
            let pred_id = engine.resolve_symbol(&predicate).into_diagnostic()?;

            let subj_label = engine.resolve_label(subj_id);
            let pred_label = engine.resolve_label(pred_id);

            let results = engine
                .recover_filler(subj_id, pred_id, top_k)
                .into_diagnostic()?;

            println!("Filler for (\"{subj_label}\", \"{pred_label}\"):");
            for (i, (sym_id, similarity)) in results.iter().enumerate() {
                let label = engine.resolve_label(*sym_id);
                println!(
                    "  {}. \"{}\" / {} (similarity: {:.4})",
                    i + 1,
                    label,
                    sym_id,
                    similarity
                );
            }
        }

        Commands::Info => {
            let engine = Engine::new(config).into_diagnostic()?;
            println!("{}", engine.info());
        }

        Commands::Symbols { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                SymbolAction::List => {
                    let symbols = engine.all_symbols();
                    if symbols.is_empty() {
                        println!("No symbols registered.");
                    } else {
                        println!("Symbols ({}):", symbols.len());
                        for meta in &symbols {
                            println!(
                                "  {} / {} [{}]",
                                meta.label, meta.id, meta.kind
                            );
                        }
                    }
                }
                SymbolAction::Show { name_or_id } => {
                    let id = engine.resolve_symbol(&name_or_id).into_diagnostic()?;
                    let meta = engine.get_symbol_meta(id).into_diagnostic()?;
                    println!("Symbol: \"{}\"", meta.label);
                    println!("  id:         {}", meta.id);
                    println!("  kind:       {}", meta.kind);
                    println!("  created_at: {}", meta.created_at);

                    let from = engine.triples_from(id);
                    if !from.is_empty() {
                        println!("  outgoing triples ({}):", from.len());
                        for t in &from {
                            println!(
                                "    -> {} -> \"{}\"",
                                engine.resolve_label(t.predicate),
                                engine.resolve_label(t.object)
                            );
                        }
                    }

                    let to = engine.triples_to(id);
                    if !to.is_empty() {
                        println!("  incoming triples ({}):", to.len());
                        for t in &to {
                            println!(
                                "    \"{}\" -> {} ->",
                                engine.resolve_label(t.subject),
                                engine.resolve_label(t.predicate)
                            );
                        }
                    }
                }
            }
        }

        Commands::Export { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                ExportAction::Symbols => {
                    let exports = engine.export_symbol_table();
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
                ExportAction::Triples => {
                    let exports = engine.export_triples();
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
                ExportAction::Provenance { name_or_id } => {
                    let id = engine.resolve_symbol(&name_or_id).into_diagnostic()?;
                    let exports = engine.export_provenance_chain(id).into_diagnostic()?;
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
            }
        }

        Commands::Skill { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                SkillAction::List => {
                    let skills = engine.list_skills();
                    if skills.is_empty() {
                        println!("No skillpacks discovered.");
                    } else {
                        println!("Skillpacks ({}):", skills.len());
                        for skill in &skills {
                            println!(
                                "  {} ({}) [{}] - {}",
                                skill.id, skill.version, skill.state, skill.description
                            );
                        }
                    }
                }
                SkillAction::Load { name } => {
                    let activation = engine.load_skill(&name).into_diagnostic()?;
                    println!("Loaded skill: {}", activation.skill_id);
                    println!("  triples: {}", activation.triples_loaded);
                    println!("  rules:   {}", activation.rules_loaded);
                    println!("  memory:  {} bytes", activation.memory_bytes);
                }
                SkillAction::Unload { name } => {
                    engine.unload_skill(&name).into_diagnostic()?;
                    println!("Unloaded skill: {name}");
                }
                SkillAction::Info { name } => {
                    let info = engine.skill_info(&name).into_diagnostic()?;
                    println!("Skill: {}", info.id);
                    println!("  name:        {}", info.name);
                    println!("  version:     {}", info.version);
                    println!("  description: {}", info.description);
                    println!("  state:       {}", info.state);
                    println!("  domains:     {}", info.domains.join(", "));
                    println!("  triples:     {}", info.triple_count);
                    println!("  rules:       {}", info.rule_count);
                }
                SkillAction::Scaffold { name } => {
                    let data_dir = cli
                        .data_dir
                        .as_deref()
                        .unwrap_or_else(|| std::path::Path::new(".akh-medu"));
                    let skill_dir = data_dir.join("skills").join(&name);
                    std::fs::create_dir_all(&skill_dir).into_diagnostic()?;

                    // Write skill.json template.
                    let manifest = serde_json::json!({
                        "id": name,
                        "name": name,
                        "version": "0.1.0",
                        "description": format!("{name} knowledge domain"),
                        "domains": [&name],
                        "weight_size_bytes": 0,
                        "triples_file": "triples.json",
                        "rules_file": "rules.txt"
                    });
                    std::fs::write(
                        skill_dir.join("skill.json"),
                        serde_json::to_string_pretty(&manifest).into_diagnostic()?,
                    )
                    .into_diagnostic()?;

                    // Write example triples.json in label-based format.
                    let triples = serde_json::json!([
                        {
                            "subject": "ExampleEntity",
                            "predicate": "is-a",
                            "object": "Category",
                            "confidence": 1.0
                        }
                    ]);
                    std::fs::write(
                        skill_dir.join("triples.json"),
                        serde_json::to_string_pretty(&triples).into_diagnostic()?,
                    )
                    .into_diagnostic()?;

                    // Write rules.txt template.
                    std::fs::write(
                        skill_dir.join("rules.txt"),
                        "# Rewrite rules for this skillpack.\n\
                         # Format: <lhs-pattern> => <rhs-pattern>\n\
                         # Example:\n\
                         # (similar ?x ?y) => (similar ?y ?x)\n",
                    )
                    .into_diagnostic()?;

                    println!("Scaffolded skill '{}' at {}", name, skill_dir.display());
                    println!("  skill.json   - manifest (edit name, domains, description)");
                    println!("  triples.json - knowledge triples (label-based format)");
                    println!("  rules.txt    - rewrite rules");
                }
            }
        }
    }

    Ok(())
}

/// Format a DerivationKind with human-readable label resolution.
fn format_derivation_kind(kind: &DerivationKind, engine: &Engine) -> String {
    match kind {
        DerivationKind::Extracted => "extracted".to_string(),
        DerivationKind::Seed => "seed".to_string(),
        DerivationKind::GraphEdge { from, predicate } => {
            format!(
                "graph edge from \"{}\" via \"{}\"",
                engine.resolve_label(*from),
                engine.resolve_label(*predicate)
            )
        }
        DerivationKind::VsaRecovery {
            from,
            predicate,
            similarity,
        } => {
            format!(
                "VSA recovery from \"{}\" via \"{}\" (sim: {:.4})",
                engine.resolve_label(*from),
                engine.resolve_label(*predicate),
                similarity
            )
        }
        DerivationKind::Analogy { a, b, c } => {
            format!(
                "analogy \"{}\":\"{}\" :: \"{}\":?",
                engine.resolve_label(*a),
                engine.resolve_label(*b),
                engine.resolve_label(*c)
            )
        }
        DerivationKind::FillerRecovery {
            subject,
            predicate,
        } => {
            format!(
                "filler recovery (\"{}\", \"{}\")",
                engine.resolve_label(*subject),
                engine.resolve_label(*predicate)
            )
        }
        DerivationKind::Reasoned => "reasoned".to_string(),
        DerivationKind::Aggregated => "aggregated".to_string(),
    }
}
