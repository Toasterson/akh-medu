//! Isolated R1 bridge exposing bounded native forward-chaining with provenance.
//! Never connects to a daemon; the caller must supply a fresh exam data directory.
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use akh_medu::agent::tools::{
    kg_delete::KgDeleteTool, kg_query::KgQueryTool, memory_recall::MemoryRecallTool,
    memory_write::MemoryWriteTool,
};
use akh_medu::agent::{AgentPredicates, Tool, ToolInput};
use akh_medu::autonomous::rule_engine::{RuleEngine, RuleEngineConfig};
use akh_medu::autonomous::rules::{InferenceRule, RuleKind, RuleSet, RuleTerm, TriplePattern};
use akh_medu::engine::{Engine, EngineConfig};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("exam data directory required")?,
    );
    std::fs::create_dir_all(&data_dir)?;
    let engine = Engine::new(EngineConfig {
        data_dir: Some(data_dir),
        ..EngineConfig::default()
    })?;
    let writer = MemoryWriteTool::new(AgentPredicates::init(&engine)?);
    let recall = MemoryRecallTool::new(AgentPredicates::init(&engine)?);
    let query = KgQueryTool;
    let delete = KgDeleteTool;
    let tools: [&dyn Tool; 4] = [&writer, &recall, &query, &delete];
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let response = (|| -> Result<Value, Box<dyn std::error::Error>> {
            let request: Value = serde_json::from_str(&line?)?;
            let name = request["tool"].as_str().ok_or("tool must be a string")?;
            // Host-only endpoints: never registered in candidate PublicEpisode.
            if name == "exam_owned_snapshot" {
                engine.persist()?;
                let rows: Vec<_> = engine.all_triples().iter().map(|t| json!({
                    "edge": t, "labels": {"subject": engine.resolve_label(t.subject),
                    "predicate": engine.resolve_label(t.predicate), "object": engine.resolve_label(t.object)}
                })).collect();
                return Ok(json!({"success": true, "rows": rows}));
            }
            if name == "exam_retract_owned" {
                let sources: Vec<akh_medu::graph::Triple> =
                    serde_json::from_value(request["arguments"]["sources"].clone())?;
                let derived: Vec<akh_medu::graph::Triple> =
                    serde_json::from_value(request["arguments"]["derived"].clone())?;
                let removed = engine.retract_owned_rule_snapshot(&sources, &derived)?;
                engine.persist()?;
                return Ok(json!({"success": true, "removed": removed}));
            }
            if name == "exam_export" {
                engine.persist()?;
                return Ok(json!({"success": true, "triples": engine.export_triples()}));
            }
            if name == "exam_run_rules" {
                let specs = request["arguments"]["rules"]
                    .as_array()
                    .ok_or("rules array required")?;
                if specs.len() > 16 {
                    return Err("at most 16 rules".into());
                }
                let mut rules = Vec::new();
                for spec in specs {
                    let parse =
                        |key: &str| -> Result<Vec<TriplePattern>, Box<dyn std::error::Error>> {
                            let values = spec[key].as_array().ok_or("patterns array required")?;
                            if values.is_empty() || values.len() > 8 {
                                return Err("one to eight patterns required".into());
                            }
                            let mut patterns = Vec::new();
                            for value in values {
                                let pattern = TriplePattern::parse(
                                    value.as_str().ok_or("pattern string required")?,
                                )?;
                                for term in [&pattern.subject, &pattern.predicate, &pattern.object]
                                {
                                    if let RuleTerm::Label(label) = term {
                                        engine.resolve_or_create_entity(label)?;
                                    }
                                }
                                patterns.push(pattern);
                            }
                            Ok(patterns)
                        };
                    let mut rule = InferenceRule::new(
                        spec["name"].as_str().ok_or("rule name required")?,
                        RuleKind::Custom {
                            name: "r1-declared-policy".into(),
                        },
                    )
                    .with_antecedents(parse("antecedents")?)
                    .with_consequents(parse("consequents")?);
                    rule.max_derivations_per_iteration = 64;
                    rules.push(rule);
                }
                let result = RuleEngine::new(RuleEngineConfig {
                    max_iterations: 4,
                    max_new_triples: 64,
                    min_confidence: 0.99,
                    auto_commit: true,
                })
                .with_rules(RuleSet {
                    name: "r1-declared-policy".into(),
                    rules,
                    source: "versioned experiment rule artifact".into(),
                })
                .run(&engine)?;
                let label = |t: &akh_medu::graph::Triple| {
                    json!({
                    "subject": engine.resolve_label(t.subject), "predicate": engine.resolve_label(t.predicate),
                    "object": engine.resolve_label(t.object), "confidence": t.confidence })
                };
                let derived: Vec<_> = result
                    .derived
                    .iter()
                    .map(|d| {
                        json!({
                    "triple": label(&d.triple), "rule": d.rule_name,
                    "antecedents": d.antecedent_triples.iter().map(&label).collect::<Vec<_>>(),
                    "iteration": d.iteration, "confidence": d.confidence })
                    })
                    .collect();
                engine.persist()?;
                return Ok(json!({"success": true, "derived": derived,
                    "iterations": result.iterations, "reached_fixpoint": result.reached_fixpoint,
                    "rule_stats": result.rule_stats}));
            }
            let tool = tools
                .iter()
                .find(|t| t.signature().name == name)
                .ok_or("unknown exam tool")?;
            let params = request["arguments"]
                .as_object()
                .ok_or("arguments must be an object")?
                .clone();
            let output = tool.execute(&engine, ToolInput { params })?;
            engine.persist()?;
            Ok(json!({"success": output.success, "result": output.result}))
        })();
        let value = response.unwrap_or_else(|e| json!({"success": false, "error": e.to_string()}));
        serde_json::to_writer(&mut stdout, &value)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    engine.persist()?;
    Ok(())
}
