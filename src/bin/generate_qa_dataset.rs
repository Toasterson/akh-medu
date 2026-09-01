//! generate_qa_dataset — Gemma+Psyche teacher QA dataset generator.
//!
//! Uses a teacher LLM (deepseek via OpenRouter) to produce agent-specific QA
//! pairs from Knowledge Graph triples, tagged with each agent's Psyche params.
//! Output is written as JSONL to `memory/akh-medu-training/{agent_id}.jsonl`.
//!
//! Usage:
//!   generate_qa_dataset [--data-dir <path>] [--output-dir <path>] \
//!       [--api-key <key>] [--model <model>] [--target-pairs <n>] \
//!       [--batch-size <n>] [--pairs-per-call <n>]
//!
//! API key defaults to `OPENROUTER_API_KEY` or `AKH_EXTRACTION_API_KEY` env vars.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use miette::{Diagnostic, IntoDiagnostic, Result, WrapErr};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use akh_medu::compartment::psyche::{ArchetypeWeights, Persona, Psyche, SelfIntegration, Shadow};
use akh_medu::engine::{Engine, EngineConfig};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "generate_qa_dataset",
    about = "Generate agent-specific QA training pairs from KG triples using a teacher LLM"
)]
struct Cli {
    /// Path to existing engine data directory. When absent, an empty in-memory
    /// engine is used (useful for testing prompt templates).
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Directory to write JSONL files into.
    /// Defaults to `memory/akh-medu-training` relative to CWD.
    #[arg(long, default_value = "memory/akh-medu-training")]
    output_dir: PathBuf,

    /// OpenRouter/OpenAI-compatible API key.
    /// Falls back to OPENROUTER_API_KEY then AKH_EXTRACTION_API_KEY env vars.
    #[arg(long)]
    api_key: Option<String>,

    /// Teacher model identifier (OpenRouter model string).
    #[arg(long, default_value = "deepseek/deepseek-chat")]
    model: String,

    /// API base URL.
    #[arg(long, default_value = "https://openrouter.ai/api/v1")]
    api_base: String,

    /// Target QA pairs per agent (generation stops when this is reached).
    #[arg(long, default_value_t = 500)]
    target_pairs: usize,

    /// KG triples per LLM batch (context window constraint).
    #[arg(long, default_value_t = 15)]
    batch_size: usize,

    /// QA pairs to request per LLM call.
    #[arg(long, default_value_t = 5)]
    pairs_per_call: usize,

    /// Request timeout in seconds.
    #[arg(long, default_value_t = 90)]
    timeout_secs: u64,

    /// Maximum LLM retries on transient errors.
    #[arg(long, default_value_t = 3)]
    max_retries: u32,

    /// Dry run: print prompts without calling the LLM.
    #[arg(long)]
    dry_run: bool,
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
enum GenError {
    #[error("no API key available")]
    #[diagnostic(
        code(gen_qa::no_api_key),
        help(
            "Set OPENROUTER_API_KEY or AKH_EXTRACTION_API_KEY env var, \
             or pass --api-key."
        )
    )]
    NoApiKey,

    #[error("LLM API request failed (status {status}): {body}")]
    #[diagnostic(
        code(gen_qa::api_failed),
        help("Check your API key, model name, and base URL. Body: {body}")
    )]
    ApiFailed { status: u16, body: String },

    #[error("LLM API request timed out after {secs}s")]
    #[diagnostic(
        code(gen_qa::api_timeout),
        help("Increase --timeout-secs or check network connectivity.")
    )]
    ApiTimeout { secs: u64 },

    #[error("failed to parse LLM response as QA pairs: {reason}")]
    #[diagnostic(
        code(gen_qa::parse_failed),
        help("The model returned malformed JSON. Raw (truncated): {raw}")
    )]
    ParseFailed { reason: String, raw: String },

    #[error("I/O error: {message}")]
    #[diagnostic(code(gen_qa::io), help("Check filesystem permissions."))]
    Io { message: String },

    #[error("engine error: {message}")]
    #[diagnostic(code(gen_qa::engine))]
    Engine { message: String },
}

impl From<std::io::Error> for GenError {
    fn from(e: std::io::Error) -> Self {
        Self::Io {
            message: e.to_string(),
        }
    }
}

impl From<akh_medu::error::AkhError> for GenError {
    fn from(e: akh_medu::error::AkhError) -> Self {
        Self::Engine {
            message: e.to_string(),
        }
    }
}

type GenResult<T> = std::result::Result<T, GenError>;

// ---------------------------------------------------------------------------
// Output types (JSONL record)
// ---------------------------------------------------------------------------

/// A single QA pair with full provenance metadata.
#[derive(Debug, Serialize, Deserialize)]
struct QaPair {
    /// Globally unique pair ID (agent_id + sequential counter).
    pair_id: String,
    /// The agent that "authored" this pair.
    agent_id: String,
    /// The question in the agent's voice.
    question: String,
    /// The answer in the agent's voice.
    answer: String,
    /// Psyche parameters used to style this pair.
    psyche_params: PsycheParams,
    /// KG triples used as grounding context.
    kg_context: Vec<KgTripleRecord>,
    /// Confidence that the answer is grounded (0.0–1.0).
    grounding_score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct PsycheParams {
    agent_id: String,
    persona_name: String,
    grammar_preference: String,
    traits: Vec<String>,
    tone: Vec<String>,
    archetype_healer: f32,
    archetype_sage: f32,
    archetype_guardian: f32,
    archetype_explorer: f32,
    dominant_archetype: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct KgTripleRecord {
    subject: String,
    predicate: String,
    object: String,
    confidence: f32,
}

// ---------------------------------------------------------------------------
// Checkpoint (resumability)
// ---------------------------------------------------------------------------

/// Tracks how many pairs have been written per agent so far.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Checkpoint {
    pairs_written: HashMap<String, usize>,
    /// Monotonically increasing global pair counter for unique IDs.
    global_counter: u64,
}

impl Checkpoint {
    fn load(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn save(&self, path: &Path) -> GenResult<()> {
        let json = serde_json::to_string_pretty(self).unwrap_or_default();
        fs::write(path, json)?;
        Ok(())
    }

    fn count(&self, agent_id: &str) -> usize {
        self.pairs_written.get(agent_id).copied().unwrap_or(0)
    }

    fn increment(&mut self, agent_id: &str, n: usize) {
        *self.pairs_written.entry(agent_id.to_string()).or_default() += n;
        self.global_counter += n as u64;
    }
}

// ---------------------------------------------------------------------------
// Ennead agent definitions
// ---------------------------------------------------------------------------

struct AgentDef {
    id: &'static str,
    psyche: Psyche,
    /// One-paragraph character brief injected into the teacher prompt.
    character_brief: &'static str,
}

fn ennead_agents() -> Vec<AgentDef> {
    vec![
        AgentDef {
            id: "thoth",
            character_brief: "You are Thoth, the divine scribe and lord of knowledge. \
                You ask precise, probing questions that reveal the deep structure of \
                relationships between concepts. Your answers are thorough, cite specific \
                facts, and build systematic understanding. You favour abstraction and \
                categorisation. Use a formal, scholarly tone.",
            psyche: Psyche {
                persona: Persona {
                    name: "Thoth".into(),
                    grammar_preference: "formal".into(),
                    traits: vec![
                        "precise".into(),
                        "scholarly".into(),
                        "systematic".into(),
                        "encyclopaedic".into(),
                    ],
                    tone: vec!["formal".into(), "authoritative".into(), "methodical".into()],
                },
                shadow: Shadow {
                    veto_patterns: vec![],
                    bias_patterns: vec![],
                },
                archetypes: ArchetypeWeights {
                    sage: 0.95,
                    healer: 0.3,
                    guardian: 0.4,
                    explorer: 0.35,
                },
                self_integration: SelfIntegration {
                    individuation_level: 0.8,
                    last_evolution_cycle: 0,
                    shadow_encounters: 5,
                    rebalance_count: 3,
                    dominant_archetype: "sage".into(),
                },
                awakened: true,
                ocean: None,
                ennead_role: None,
                avatar_hash: None,
            },
        },
        AgentDef {
            id: "ptah",
            character_brief: "You are Ptah, the divine craftsman and architect of creation. \
                You ask practical questions about how things are built, structured, and \
                connected. Your answers emphasise design decisions, trade-offs, and concrete \
                construction patterns. Use a terse, technical, builder's tone.",
            psyche: Psyche {
                persona: Persona {
                    name: "Ptah".into(),
                    grammar_preference: "terse".into(),
                    traits: vec![
                        "pragmatic".into(),
                        "architectural".into(),
                        "constructive".into(),
                        "design-focused".into(),
                    ],
                    tone: vec!["terse".into(), "technical".into(), "direct".into()],
                },
                shadow: Shadow {
                    veto_patterns: vec![],
                    bias_patterns: vec![],
                },
                archetypes: ArchetypeWeights {
                    sage: 0.5,
                    healer: 0.3,
                    guardian: 0.6,
                    explorer: 0.85,
                },
                self_integration: SelfIntegration {
                    individuation_level: 0.7,
                    last_evolution_cycle: 0,
                    shadow_encounters: 2,
                    rebalance_count: 4,
                    dominant_archetype: "explorer".into(),
                },
                awakened: true,
                ocean: None,
                ennead_role: None,
                avatar_hash: None,
            },
        },
        AgentDef {
            id: "isis",
            character_brief: "You are Isis, the great healer and mother of magic. \
                You ask nurturing questions focused on gaps in knowledge, missing \
                connections, and how understanding can be restored or completed. \
                Your answers are warm, integrative, and holistic. Use a narrative, \
                compassionate tone.",
            psyche: Psyche {
                persona: Persona {
                    name: "Isis".into(),
                    grammar_preference: "narrative".into(),
                    traits: vec![
                        "nurturing".into(),
                        "integrative".into(),
                        "perceptive".into(),
                        "holistic".into(),
                    ],
                    tone: vec![
                        "warm".into(),
                        "compassionate".into(),
                        "encouraging".into(),
                    ],
                },
                shadow: Shadow {
                    veto_patterns: vec![],
                    bias_patterns: vec![],
                },
                archetypes: ArchetypeWeights {
                    sage: 0.45,
                    healer: 0.95,
                    guardian: 0.65,
                    explorer: 0.3,
                },
                self_integration: SelfIntegration {
                    individuation_level: 0.85,
                    last_evolution_cycle: 0,
                    shadow_encounters: 8,
                    rebalance_count: 6,
                    dominant_archetype: "healer".into(),
                },
                awakened: true,
                ocean: None,
                ennead_role: None,
                avatar_hash: None,
            },
        },
        AgentDef {
            id: "anubis",
            character_brief: "You are Anubis, guardian of transitions and weigher of truth. \
                You ask sharp, evaluative questions that test the integrity and consistency \
                of knowledge. Your answers are measured, cautious, and always acknowledge \
                uncertainty. Use a measured, judicious tone.",
            psyche: Psyche {
                persona: Persona {
                    name: "Anubis".into(),
                    grammar_preference: "formal".into(),
                    traits: vec![
                        "cautious".into(),
                        "evaluative".into(),
                        "impartial".into(),
                        "rigorous".into(),
                    ],
                    tone: vec!["measured".into(), "judicious".into(), "careful".into()],
                },
                shadow: Shadow {
                    veto_patterns: vec![],
                    bias_patterns: vec![],
                },
                archetypes: ArchetypeWeights {
                    sage: 0.65,
                    healer: 0.25,
                    guardian: 0.95,
                    explorer: 0.2,
                },
                self_integration: SelfIntegration {
                    individuation_level: 0.75,
                    last_evolution_cycle: 0,
                    shadow_encounters: 12,
                    rebalance_count: 5,
                    dominant_archetype: "guardian".into(),
                },
                awakened: true,
                ocean: None,
                ennead_role: None,
                avatar_hash: None,
            },
        },
        AgentDef {
            id: "sekhmet",
            character_brief: "You are Sekhmet, the lioness investigator and destroyer of \
                ignorance. You ask incisive, probing questions that expose hidden \
                dependencies, root causes, and surprising connections. Your answers are \
                bold, direct, and willing to challenge conventional assumptions. Use a \
                sharp, investigative tone.",
            psyche: Psyche {
                persona: Persona {
                    name: "Sekhmet".into(),
                    grammar_preference: "terse".into(),
                    traits: vec![
                        "incisive".into(),
                        "bold".into(),
                        "investigative".into(),
                        "contrarian".into(),
                    ],
                    tone: vec!["sharp".into(), "direct".into(), "provocative".into()],
                },
                shadow: Shadow {
                    veto_patterns: vec![],
                    bias_patterns: vec![],
                },
                archetypes: ArchetypeWeights {
                    sage: 0.55,
                    healer: 0.45,
                    guardian: 0.2,
                    explorer: 0.9,
                },
                self_integration: SelfIntegration {
                    individuation_level: 0.6,
                    last_evolution_cycle: 0,
                    shadow_encounters: 15,
                    rebalance_count: 8,
                    dominant_archetype: "explorer".into(),
                },
                awakened: true,
                ocean: None,
                ennead_role: None,
                avatar_hash: None,
            },
        },
        AgentDef {
            id: "khnum",
            character_brief: "You are Khnum, the potter who shapes forms at the source. \
                You ask questions about origins, formation processes, and the fundamental \
                properties that define entities. Your answers focus on essential \
                characteristics, how things come to be, and their core nature. Use a \
                reflective, generative tone.",
            psyche: Psyche {
                persona: Persona {
                    name: "Khnum".into(),
                    grammar_preference: "narrative".into(),
                    traits: vec![
                        "generative".into(),
                        "foundational".into(),
                        "reflective".into(),
                        "elemental".into(),
                    ],
                    tone: vec!["reflective".into(), "thoughtful".into(), "elemental".into()],
                },
                shadow: Shadow {
                    veto_patterns: vec![],
                    bias_patterns: vec![],
                },
                archetypes: ArchetypeWeights {
                    sage: 0.6,
                    healer: 0.5,
                    guardian: 0.55,
                    explorer: 0.7,
                },
                self_integration: SelfIntegration {
                    individuation_level: 0.65,
                    last_evolution_cycle: 0,
                    shadow_encounters: 4,
                    rebalance_count: 2,
                    dominant_archetype: "explorer".into(),
                },
                awakened: true,
                ocean: None,
                ennead_role: None,
                avatar_hash: None,
            },
        },
    ]
}

// ---------------------------------------------------------------------------
// KG triple extraction helpers
// ---------------------------------------------------------------------------

/// A triple with its labels resolved.
#[derive(Clone)]
struct LabeledTriple {
    subject: String,
    predicate: String,
    object: String,
    confidence: f32,
}

impl LabeledTriple {
    fn to_record(&self) -> KgTripleRecord {
        KgTripleRecord {
            subject: self.subject.clone(),
            predicate: self.predicate.clone(),
            object: self.object.clone(),
            confidence: self.confidence,
        }
    }

    /// One-line text representation for prompt context.
    fn to_prompt_line(&self) -> String {
        format!(
            "({}, {}, {}) [conf: {:.2}]",
            self.subject, self.predicate, self.object, self.confidence
        )
    }
}

/// Load and label all triples from the engine.
fn load_labeled_triples(engine: &Engine) -> Vec<LabeledTriple> {
    engine
        .all_triples()
        .into_iter()
        .map(|t| LabeledTriple {
            subject: engine.resolve_label(t.subject),
            predicate: engine.resolve_label(t.predicate),
            object: engine.resolve_label(t.object),
            confidence: t.confidence,
        })
        .collect()
}

/// Group triples into entity clusters (by subject) and return a flat list of
/// batches, each containing at most `batch_size` triples.
///
/// Clusters are kept together where possible so context is locally coherent.
fn make_batches(triples: &[LabeledTriple], batch_size: usize) -> Vec<Vec<LabeledTriple>> {
    // Group by subject.
    let mut by_subject: HashMap<String, Vec<LabeledTriple>> = HashMap::new();
    for t in triples {
        by_subject
            .entry(t.subject.clone())
            .or_default()
            .push(t.clone());
    }

    // Sort subjects for deterministic output.
    let mut subjects: Vec<String> = by_subject.keys().cloned().collect();
    subjects.sort();

    let mut batches: Vec<Vec<LabeledTriple>> = Vec::new();
    let mut current: Vec<LabeledTriple> = Vec::new();

    for subject in subjects {
        let cluster = by_subject.remove(&subject).unwrap_or_default();
        for triple in cluster {
            current.push(triple);
            if current.len() >= batch_size {
                batches.push(std::mem::take(&mut current));
            }
        }
        // Finish a cluster mid-batch rather than splitting across batches
        // (improves context coherence).  Flush when cluster boundary reached
        // and we have a reasonable amount of data.
        if current.len() >= batch_size / 2 + 1 {
            batches.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        batches.push(current);
    }

    batches
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

fn build_system_prompt(agent: &AgentDef, pairs_per_call: usize) -> String {
    let p = &agent.psyche.persona;
    let a = &agent.psyche.archetypes;
    let dominant = agent.psyche.dominant_archetype();

    format!(
        "You are a QA dataset generator acting as the agent \"{name}\".\n\
         {brief}\n\n\
         Persona traits: {traits}.\n\
         Communication tone: {tone}.\n\
         Dominant archetype: {dominant} (healer={h:.2}, sage={s:.2}, guardian={g:.2}, explorer={e:.2}).\n\n\
         Given a list of Knowledge Graph triples, generate exactly {n} question-answer pairs.\n\
         Rules:\n\
         - Every question MUST be answerable from the provided KG triples (no hallucination).\n\
         - The question style must reflect the persona: {traits}.\n\
         - Answers must be factually grounded in the triples — quote entity names exactly.\n\
         - Do NOT invent entities, relationships, or facts not present in the triples.\n\
         - Output ONLY a valid JSON array of objects with keys \"q\" and \"a\".\n\
         - No preamble, no markdown fences, no explanation outside the JSON.\n\
         - Example format: [{{\"q\": \"What is X?\", \"a\": \"X is ...\"}}]",
        name = p.name,
        brief = agent.character_brief,
        traits = p.traits.join(", "),
        tone = p.tone.join(", "),
        dominant = dominant,
        h = a.healer,
        s = a.sage,
        g = a.guardian,
        e = a.explorer,
        n = pairs_per_call,
    )
}

fn build_user_message(batch: &[LabeledTriple]) -> String {
    let triples_text: Vec<String> = batch.iter().map(|t| t.to_prompt_line()).collect();
    format!(
        "Knowledge Graph triples:\n{}\n\nGenerate QA pairs:",
        triples_text.join("\n")
    )
}

// ---------------------------------------------------------------------------
// LLM client
// ---------------------------------------------------------------------------

struct TeacherClient {
    api_key: String,
    model: String,
    base_url: String,
    timeout: Duration,
    max_retries: u32,
}

impl TeacherClient {
    fn call(&self, system: &str, user: &str) -> GenResult<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
        });

        let mut last_error: Option<GenError> = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // Brief back-off between retries.
                std::thread::sleep(Duration::from_secs(2u64.pow(attempt)));
            }

            let result = ureq::post(&url)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("Content-Type", "application/json")
                .timeout(self.timeout)
                .send_json(&body);

            match result {
                Ok(resp) => {
                    let val: serde_json::Value = resp
                        .into_json()
                        .map_err(|e| GenError::ParseFailed {
                            reason: format!("JSON decode: {e}"),
                            raw: String::new(),
                        })?;

                    let content = val
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();

                    return Ok(content);
                }
                Err(ureq::Error::Status(status, resp)) => {
                    let body_text = resp.into_string().unwrap_or_default();
                    // 429 (rate limit) and 5xx are retryable.
                    if status == 429 || status >= 500 {
                        last_error = Some(GenError::ApiFailed {
                            status,
                            body: body_text,
                        });
                        continue;
                    }
                    return Err(GenError::ApiFailed {
                        status,
                        body: body_text,
                    });
                }
                Err(_) => {
                    last_error = Some(GenError::ApiTimeout { secs: self.timeout.as_secs() });
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or(GenError::ApiTimeout { secs: self.timeout.as_secs() }))
    }
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawQaPair {
    q: String,
    a: String,
}

/// Extract a JSON array from potentially messy LLM output.
fn extract_json_array(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if end > start {
                return &trimmed[start..=end];
            }
        }
    }
    "[]"
}

fn parse_qa_pairs(raw: &str) -> GenResult<Vec<RawQaPair>> {
    let json_str = extract_json_array(raw);
    serde_json::from_str::<Vec<RawQaPair>>(json_str).map_err(|e| GenError::ParseFailed {
        reason: e.to_string(),
        raw: raw.chars().take(300).collect(),
    })
}

// ---------------------------------------------------------------------------
// Grounding score (hallucination check)
// ---------------------------------------------------------------------------

/// Computes a simple grounding score [0,1] based on how many KG entity labels
/// appear in the answer text.
fn grounding_score(answer: &str, batch: &[LabeledTriple]) -> f32 {
    let lower = answer.to_lowercase();

    // Collect unique entities from batch context.
    let mut entities: Vec<String> = Vec::new();
    for t in batch {
        entities.push(t.subject.to_lowercase());
        entities.push(t.object.to_lowercase());
    }
    entities.sort();
    entities.dedup();

    if entities.is_empty() {
        return 0.5; // No context to check against — neutral score.
    }

    let matched = entities.iter().filter(|e| lower.contains(e.as_str())).count();
    (matched as f32 / entities.len() as f32).min(1.0)
}

// ---------------------------------------------------------------------------
// Per-agent JSONL writer
// ---------------------------------------------------------------------------

struct AgentWriter {
    path: PathBuf,
}

impl AgentWriter {
    fn open(output_dir: &Path, agent_id: &str) -> GenResult<Self> {
        let path = output_dir.join(format!("{agent_id}.jsonl"));
        Ok(Self { path })
    }

    fn append(&self, pairs: &[QaPair]) -> GenResult<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        for pair in pairs {
            let line = serde_json::to_string(pair).unwrap_or_default();
            writeln!(file, "{line}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Generation loop
// ---------------------------------------------------------------------------

fn generate_for_agent(
    agent: &AgentDef,
    batches: &[Vec<LabeledTriple>],
    client: &TeacherClient,
    writer: &AgentWriter,
    checkpoint: &mut Checkpoint,
    checkpoint_path: &Path,
    target_pairs: usize,
    pairs_per_call: usize,
    dry_run: bool,
) -> GenResult<usize> {
    let already = checkpoint.count(agent.id);
    if already >= target_pairs {
        eprintln!(
            "  [{}] already at {}/{} pairs — skipping",
            agent.id, already, target_pairs
        );
        return Ok(0);
    }

    let system_prompt = build_system_prompt(agent, pairs_per_call);
    let mut generated = 0usize;
    let mut batch_idx = already / pairs_per_call; // approximate resume position
    batch_idx = batch_idx.min(batches.len().saturating_sub(1));

    // Cycle through batches until we hit the target.
    let mut cycle = 0usize;
    while checkpoint.count(agent.id) < target_pairs {
        let bi = (batch_idx + cycle) % batches.len().max(1);
        let batch = &batches[bi];
        cycle += 1;

        let user_msg = build_user_message(batch);

        if dry_run {
            eprintln!("  [DRY RUN] agent={} batch={bi} prompt len={}", agent.id, user_msg.len());
            // Generate stub pairs in dry-run mode.
            let stubs: Vec<QaPair> = (0..pairs_per_call)
                .map(|i| build_qa_pair(
                    agent,
                    checkpoint,
                    &RawQaPair {
                        q: format!("[dry-run] question {i}"),
                        a: format!("[dry-run] answer {i}"),
                    },
                    batch,
                ))
                .collect();
            generated += stubs.len();
            checkpoint.increment(agent.id, stubs.len());
            writer.append(&stubs)?;
            checkpoint.save(checkpoint_path)?;
            continue;
        }

        match client.call(&system_prompt, &user_msg) {
            Ok(raw) => {
                match parse_qa_pairs(&raw) {
                    Ok(pairs) if !pairs.is_empty() => {
                        let qa_pairs: Vec<QaPair> = pairs
                            .into_iter()
                            .map(|p| build_qa_pair(agent, checkpoint, &p, batch))
                            .collect();
                        let n = qa_pairs.len();
                        writer.append(&qa_pairs)?;
                        checkpoint.increment(agent.id, n);
                        checkpoint.save(checkpoint_path)?;
                        generated += n;

                        let current = checkpoint.count(agent.id);
                        eprintln!(
                            "  [{}] {}/{} pairs (batch {bi}, +{n})",
                            agent.id, current, target_pairs
                        );
                    }
                    Ok(_) => {
                        eprintln!("  [{}] WARNING: model returned 0 pairs, skipping batch", agent.id);
                    }
                    Err(e) => {
                        eprintln!("  [{}] WARNING: parse error on batch {bi}: {e}", agent.id);
                    }
                }
            }
            Err(e) => {
                eprintln!("  [{}] ERROR on batch {bi}: {e} — continuing", agent.id);
                // Brief pause before continuing to next batch.
                std::thread::sleep(Duration::from_secs(2));
            }
        }

        // Safety: don't loop forever if we have no batches.
        if batches.is_empty() {
            break;
        }
        // Stop if we've cycled through all batches more than 10x without progress.
        if cycle > batches.len() * 10 + 100 {
            eprintln!("  [{}] WARNING: exceeded max batch cycles, stopping early", agent.id);
            break;
        }
    }

    Ok(generated)
}

fn build_qa_pair(
    agent: &AgentDef,
    checkpoint: &Checkpoint,
    raw: &RawQaPair,
    batch: &[LabeledTriple],
) -> QaPair {
    let p = &agent.psyche.persona;
    let a = &agent.psyche.archetypes;
    let pair_id = format!("{}-{}", agent.id, checkpoint.global_counter);

    let score = grounding_score(&raw.a, batch);

    QaPair {
        pair_id,
        agent_id: agent.id.to_string(),
        question: raw.q.clone(),
        answer: raw.a.clone(),
        psyche_params: PsycheParams {
            agent_id: agent.id.to_string(),
            persona_name: p.name.clone(),
            grammar_preference: p.grammar_preference.clone(),
            traits: p.traits.clone(),
            tone: p.tone.clone(),
            archetype_healer: a.healer,
            archetype_sage: a.sage,
            archetype_guardian: a.guardian,
            archetype_explorer: a.explorer,
            dominant_archetype: agent.psyche.dominant_archetype().to_string(),
        },
        kg_context: batch.iter().map(|t| t.to_record()).collect(),
        grounding_score: score,
    }
}

// ---------------------------------------------------------------------------
// Seed triples (fallback when no data-dir is provided)
// ---------------------------------------------------------------------------

/// Minimal seed triples so the generator can run in stand-alone mode
/// (no KG data directory required).  These describe the akh-medu system itself.
fn seed_triples() -> Vec<LabeledTriple> {
    let raw = vec![
        // System entities
        ("akh-medu", "is-a", "neuro-symbolic-ai-engine"),
        ("akh-medu", "uses", "vector-symbolic-architecture"),
        ("akh-medu", "uses", "knowledge-graph"),
        ("akh-medu", "uses", "e-graph-reasoning"),
        ("akh-medu", "uses", "hnsw-item-memory"),
        ("akh-medu", "written-in", "rust"),
        ("akh-medu", "edition", "2024"),
        // VSA
        ("vector-symbolic-architecture", "is-a", "hyperdimensional-computing"),
        ("vector-symbolic-architecture", "dimension", "10000"),
        ("vector-symbolic-architecture", "encoding", "bipolar"),
        // KG layers
        ("knowledge-graph", "has-layer", "in-memory-petgraph"),
        ("knowledge-graph", "has-layer", "persistent-oxigraph"),
        ("knowledge-graph", "stores", "triple"),
        ("triple", "has-field", "subject"),
        ("triple", "has-field", "predicate"),
        ("triple", "has-field", "object"),
        ("triple", "has-field", "confidence"),
        // Agent
        ("agent", "has", "psyche"),
        ("agent", "runs", "ooda-loop"),
        ("agent", "has", "working-memory"),
        ("agent", "has", "episodic-memory"),
        ("agent", "has", "goal-system"),
        ("agent", "uses", "tool-registry"),
        ("ooda-loop", "is-a", "synchronous-control-loop"),
        ("ooda-loop", "phases", "observe-orient-decide-act"),
        // Psyche
        ("psyche", "has", "persona"),
        ("psyche", "has", "shadow"),
        ("psyche", "has", "archetype-weights"),
        ("psyche", "has", "self-integration"),
        ("archetype-weights", "includes", "sage"),
        ("archetype-weights", "includes", "healer"),
        ("archetype-weights", "includes", "guardian"),
        ("archetype-weights", "includes", "explorer"),
        // Ennead
        ("ennead", "is-a", "group-of-divine-agents"),
        ("thoth", "is-a", "ennead-member"),
        ("thoth", "archetype", "sage"),
        ("thoth", "role", "scribe-of-knowledge"),
        ("ptah", "is-a", "ennead-member"),
        ("ptah", "archetype", "explorer"),
        ("ptah", "role", "craftsman-architect"),
        ("isis", "is-a", "ennead-member"),
        ("isis", "archetype", "healer"),
        ("isis", "role", "healer-of-magic"),
        ("anubis", "is-a", "ennead-member"),
        ("anubis", "archetype", "guardian"),
        ("anubis", "role", "weigher-of-truth"),
        ("sekhmet", "is-a", "ennead-member"),
        ("sekhmet", "archetype", "explorer"),
        ("sekhmet", "role", "investigator"),
        ("khnum", "is-a", "ennead-member"),
        ("khnum", "archetype", "explorer"),
        ("khnum", "role", "shaper-of-forms"),
        // Storage
        ("tiered-store", "has-tier", "hot-dashmap"),
        ("tiered-store", "has-tier", "warm-mmap"),
        ("tiered-store", "has-tier", "cold-redb"),
        ("redb", "is-a", "embedded-acid-database"),
        // Tools
        ("tool-registry", "contains", "kg-query"),
        ("tool-registry", "contains", "infer-rules"),
        ("tool-registry", "contains", "gap-analysis"),
        ("tool-registry", "contains", "memory-recall"),
        ("tool-registry", "contains", "file-io"),
        ("tool-registry", "contains", "http-fetch"),
        ("kg-query", "capability", "read-knowledge-graph"),
        ("gap-analysis", "capability", "detect-missing-knowledge"),
        ("memory-recall", "capability", "retrieve-episodic-memory"),
    ];

    raw.into_iter()
        .map(|(s, p, o)| LabeledTriple {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            confidence: 1.0,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn resolve_api_key(cli_key: Option<String>) -> Option<String> {
    cli_key
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .or_else(|| std::env::var("AKH_EXTRACTION_API_KEY").ok())
}

fn main() -> Result<()> {
    // Initialise tracing so engine logs are visible.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("akh_medu=warn".parse().into_diagnostic()?),
        )
        .init();

    let cli = Cli::parse();

    // --- Resolve API key ---------------------------------------------------
    let api_key = if cli.dry_run {
        "dry-run-placeholder".to_string()
    } else {
        resolve_api_key(cli.api_key)
            .ok_or(GenError::NoApiKey)
            .into_diagnostic()
            .wrap_err("Cannot proceed without an API key")?
    };

    // --- Prepare output directory ------------------------------------------
    fs::create_dir_all(&cli.output_dir)
        .map_err(GenError::from)
        .into_diagnostic()
        .wrap_err("Failed to create output directory")?;

    let checkpoint_path = cli.output_dir.join("checkpoint.json");
    let mut checkpoint = Checkpoint::load(&checkpoint_path);

    // --- Load KG triples ---------------------------------------------------
    let triples: Vec<LabeledTriple> = if let Some(ref data_dir) = cli.data_dir {
        eprintln!("Loading KG from {:?}…", data_dir);
        let engine = Engine::new(EngineConfig {
            data_dir: Some(data_dir.clone()),
            ..EngineConfig::default()
        })
        .map_err(GenError::from)
        .into_diagnostic()
        .wrap_err("Failed to initialise engine")?;
        let ts = load_labeled_triples(&engine);
        eprintln!("Loaded {} KG triples", ts.len());
        ts
    } else {
        eprintln!("No --data-dir provided; using built-in seed triples.");
        let ts = seed_triples();
        eprintln!("Seed triple count: {}", ts.len());
        ts
    };

    if triples.is_empty() {
        eprintln!("WARNING: no triples available — dataset will be empty.");
    }

    // --- Build batches -----------------------------------------------------
    let batches = make_batches(&triples, cli.batch_size);
    eprintln!("Triple batches: {}", batches.len());

    // --- Prepare LLM client ------------------------------------------------
    let client = TeacherClient {
        api_key,
        model: cli.model.clone(),
        base_url: cli.api_base.clone(),
        timeout: Duration::from_secs(cli.timeout_secs),
        max_retries: cli.max_retries,
    };

    // --- Run per-agent generation ------------------------------------------
    let agents = ennead_agents();
    let mut total_generated = 0usize;

    for agent in &agents {
        let already = checkpoint.count(agent.id);
        eprintln!(
            "\n=== Agent: {} ({}) — {}/{} pairs ===",
            agent.id, agent.psyche.persona.name, already, cli.target_pairs
        );

        let writer = AgentWriter::open(&cli.output_dir, agent.id)
            .into_diagnostic()
            .wrap_err(format!("Failed to open writer for {}", agent.id))?;

        let n = generate_for_agent(
            agent,
            &batches,
            &client,
            &writer,
            &mut checkpoint,
            &checkpoint_path,
            cli.target_pairs,
            cli.pairs_per_call,
            cli.dry_run,
        )
        .into_diagnostic()
        .wrap_err(format!("Generation failed for agent {}", agent.id))?;

        total_generated += n;
        eprintln!(
            "  [{}] done — {} new pairs this run, {} total",
            agent.id,
            n,
            checkpoint.count(agent.id)
        );
    }

    // --- Final summary -----------------------------------------------------
    eprintln!("\n========================================");
    eprintln!("Dataset generation complete.");
    eprintln!("New pairs this run : {total_generated}");
    eprintln!("Output directory   : {}", cli.output_dir.display());
    eprintln!("\nPer-agent totals:");
    for agent in &agents {
        eprintln!(
            "  {:10} {:5} pairs  ({:?})",
            agent.id,
            checkpoint.count(agent.id),
            cli.output_dir.join(format!("{}.jsonl", agent.id))
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_agents_defined() {
        let agents = ennead_agents();
        assert_eq!(agents.len(), 6);
    }

    #[test]
    fn each_agent_has_unique_id() {
        let agents = ennead_agents();
        let ids: Vec<&str> = agents.iter().map(|a| a.id).collect();
        let unique: std::collections::HashSet<&&str> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "agent IDs must be unique");
    }

    #[test]
    fn dominant_archetype_consistent_with_weights() {
        for agent in ennead_agents() {
            let dominant = agent.psyche.dominant_archetype();
            let a = &agent.psyche.archetypes;
            let max = [a.healer, a.sage, a.guardian, a.explorer]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let expected = match dominant {
                "healer" => a.healer,
                "sage" => a.sage,
                "guardian" => a.guardian,
                "explorer" => a.explorer,
                _ => panic!("unknown archetype"),
            };
            assert!(
                (expected - max).abs() < 1e-4,
                "agent {} dominant_archetype '{}' does not match highest weight",
                agent.id,
                dominant
            );
        }
    }

    #[test]
    fn seed_triples_not_empty() {
        let ts = seed_triples();
        assert!(!ts.is_empty());
    }

    #[test]
    fn make_batches_respects_batch_size() {
        let triples = seed_triples();
        let batches = make_batches(&triples, 10);
        for batch in &batches {
            // Each batch may slightly exceed batch_size at cluster boundaries,
            // but not by more than the cluster size itself.
            assert!(batch.len() <= 20, "batch too large: {}", batch.len());
        }
    }

    #[test]
    fn extract_json_array_clean() {
        let raw = r#"[{"q":"What?","a":"Ans."}]"#;
        assert_eq!(extract_json_array(raw), raw);
    }

    #[test]
    fn extract_json_array_with_preamble() {
        let raw = r#"Here you go: [{"q":"Q","a":"A"}] Done."#;
        assert_eq!(extract_json_array(raw), r#"[{"q":"Q","a":"A"}]"#);
    }

    #[test]
    fn parse_qa_pairs_valid() {
        let raw = r#"[{"q":"What is the VSA dimension?","a":"10000"}]"#;
        let pairs = parse_qa_pairs(raw).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].q, "What is the VSA dimension?");
    }

    #[test]
    fn grounding_score_full_match() {
        let batch = vec![LabeledTriple {
            subject: "akh-medu".into(),
            predicate: "uses".into(),
            object: "rust".into(),
            confidence: 1.0,
        }];
        let score = grounding_score("akh-medu is written in rust", &batch);
        assert!(score > 0.5, "score={score}");
    }

    #[test]
    fn grounding_score_no_match() {
        let batch = vec![LabeledTriple {
            subject: "akh-medu".into(),
            predicate: "uses".into(),
            object: "rust".into(),
            confidence: 1.0,
        }];
        let score = grounding_score("python is a snake language", &batch);
        assert!(score < 0.5, "score={score}");
    }

    #[test]
    fn checkpoint_increment_and_count() {
        let mut ck = Checkpoint::default();
        assert_eq!(ck.count("thoth"), 0);
        ck.increment("thoth", 5);
        assert_eq!(ck.count("thoth"), 5);
        assert_eq!(ck.global_counter, 5);
        ck.increment("thoth", 3);
        assert_eq!(ck.count("thoth"), 8);
    }

    #[test]
    fn system_prompt_contains_agent_name() {
        let agents = ennead_agents();
        let thoth = agents.iter().find(|a| a.id == "thoth").unwrap();
        let prompt = build_system_prompt(thoth, 5);
        assert!(prompt.contains("Thoth"));
        assert!(prompt.contains("5"));
    }
}
