//! Isolated R1 bridge exposing bounded native forward-chaining with provenance.
//! Never connects to a daemon; the caller must supply a fresh exam data directory.
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::collections::HashMap;
use akh_medu::agent::causal::{ActionSchema, CausalManager, CausalPattern, CausalEffect, EffectKind, PatternElement, StateTransition};

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
    let mut causal = CausalManager::restore(&engine)?;
    let mut predictions: HashMap<String, StateTransition> = HashMap::new();
    let mut next_prediction = 0u64;
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
            if name == "exam_causal_predict" {
                let args = &request["arguments"];
                let parse = |key: &str| -> Result<Vec<CausalPattern>, Box<dyn std::error::Error>> {
                    let rows = args[key].as_array().ok_or("literal triple array required")?;
                    if rows.is_empty() || rows.len() > 8 { return Err("one to eight literal triples required".into()); }
                    let mut patterns = Vec::new();
                    for row in rows {
                        let values = row.as_array().ok_or("triple must be an array")?;
                        if values.len() != 3 { return Err("three labels required".into()); }
                        let mut ids = Vec::new();
                        for (i, v) in values.iter().enumerate() {
                            let label = v.as_str().ok_or("label must be string")?;
                            if label.is_empty() || label.len() > 128 { return Err("label length invalid".into()); }
                            ids.push(if i == 1 { engine.resolve_or_create_relation(label)? } else { engine.resolve_or_create_entity(label)? });
                        }
                        patterns.push(CausalPattern { subject: PatternElement::Concrete(ids[0]), predicate: PatternElement::Concrete(ids[1]), object: PatternElement::Concrete(ids[2]), negated: false });
                    }
                    Ok(patterns)
                };
                let preconditions = parse("preconditions")?;
                let effects = parse("effects")?.into_iter().map(|p| CausalEffect { kind: EffectKind::Assert, subject: p.subject, predicate: p.predicate, object: p.object, confidence: 1.0 }).collect();
                let schema = ActionSchema { action_id: engine.resolve_or_create_entity("r0-causal-repair")?, name: "r0-causal-repair".into(), description: "Sourced experimental schema".into(), parameters: vec![], preconditions, effects, success_rate: 0.0, execution_count: 0 };
                let applicable = schema.is_applicable(&engine.all_triples());
                causal.register_action_schema(schema);
                causal.persist(&engine)?;
                if !applicable { return Ok(json!({"success": true, "applicable": false})); }
                let transition = causal.predict_effects("r0-causal-repair", &engine)?;
                if transition.assertions.is_empty() { return Err("empty prediction refused".into()); }
                next_prediction += 1;
                let id = format!("prediction-{}-{}", std::process::id(), next_prediction);
                let result = json!({"success": true, "applicable": true, "prediction": id, "assertions": transition.assertions});
                predictions.insert(id, transition);
                return Ok(result);
            }
            if name == "exam_causal_verify" {
                let id = request["arguments"]["prediction"].as_str().ok_or("prediction id required")?;
                let transition = predictions.get(id).ok_or("unknown prediction in this session")?;
                return Ok(json!({"success": true, "verified": causal.verify_prediction(transition, &engine)}));
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
