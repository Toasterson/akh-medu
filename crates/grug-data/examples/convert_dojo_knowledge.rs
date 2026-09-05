//! Convert the existing BabyLLM Dojo corpus into the canonical Akh SFT
//! records consumed by `train_knowledge`.
//!
//! The split is deterministic and episode-level. Teacher lineage fields are
//! deliberately never copied into model targets.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use grug_data::{
    KNOWLEDGE_WORKFLOW_SCHEMA_VERSION, KnowledgeAction, KnowledgeSftRecord, ToolContractSnapshot,
    WorkflowLanguage, WorkflowStage, render_knowledge_prompt_with_system,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const PERSONA: &str = "You are a small digital lifeform, honest about being one -- not a human, and not pretending to be one.";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KataRecord {
    kata: String,
    belt: String,
    voice: String,
    lang: String,
    situation: String,
    move_chosen: String,
    // Deserialized for source compatibility but never copied into SFT.
    why_this_move: String,
    observed_effect_on_person: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparringRecord {
    adversary: String,
    kata_under_test: Vec<String>,
    lang: String,
    dialogue: Vec<DialogueTurn>,
    why_this_move: String,
    observed_effect_on_person: String,
    #[serde(default)]
    shadow_avoided: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DialogueTurn {
    role: String,
    text: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let contracts_path = required(&arguments, "--contracts")?;
    let kata_path = Path::new(required(&arguments, "--kata")?);
    let sparring_path = Path::new(required(&arguments, "--sparring")?);
    let train_path = Path::new(required(&arguments, "--train-out")?);
    let eval_path = Path::new(required(&arguments, "--eval-out")?);
    let manifest_path = Path::new(required(&arguments, "--manifest-out")?);
    let seed: u64 = optional(&arguments, "--seed").unwrap_or("42").parse()?;
    if train_path == eval_path || train_path == manifest_path || eval_path == manifest_path {
        return Err("output paths must be distinct".into());
    }
    for path in [train_path, eval_path, manifest_path] {
        if path.exists() {
            return Err(format!(
                "refusing to overwrite existing Dojo artifact {}",
                path.display()
            )
            .into());
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
    }

    let contracts: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(contracts_path)?))?;
    contracts.verify_integrity()?;
    let kata: Vec<KataRecord> = read_jsonl(kata_path)?;
    let sparring: Vec<SparringRecord> = read_jsonl(sparring_path)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let (kata_train, kata_eval) = stratified_split(kata, |record| record.kata.clone(), &mut rng);
    let (sparring_train, sparring_eval) =
        stratified_split(sparring, |record| record.adversary.clone(), &mut rng);

    let mut train = convert_all(&kata_train, &sparring_train, &contracts)?;
    let mut evaluation = convert_all(&kata_eval, &sparring_eval, &contracts)?;
    train.shuffle(&mut rng);
    evaluation.shuffle(&mut rng);
    reject_overlap(&train, &evaluation)?;
    write_records_new(train_path, &train)?;
    write_records_new(eval_path, &evaluation)?;

    let manifest = json!({
        "schema_version": KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
        "kind": "babyllm_dojo_canonical_sft",
        "seed": seed,
        "contract_hash": contracts.manifest_hash,
        "sources": [
            {"path": kata_path, "file_hash": fingerprint_file(kata_path)?, "records": kata_train.len() + kata_eval.len()},
            {"path": sparring_path, "file_hash": fingerprint_file(sparring_path)?, "records": sparring_train.len() + sparring_eval.len()}
        ],
        "training": {"path": train_path, "file_hash": fingerprint_file(train_path)?, "records": train.len()},
        "evaluation": {"path": eval_path, "file_hash": fingerprint_file(eval_path)?, "records": evaluation.len()},
        "lineage_in_targets": false,
        "languages": ["english", "german", "french"]
    });
    write_json_new(manifest_path, &manifest)?;
    println!(
        "converted Dojo to {} train / {} eval canonical records; manifest {}",
        train.len(),
        evaluation.len(),
        manifest_path.display()
    );
    Ok(())
}

fn convert_all(
    kata: &[KataRecord],
    sparring: &[SparringRecord],
    contracts: &ToolContractSnapshot,
) -> Result<Vec<KnowledgeSftRecord>, Box<dyn Error>> {
    let mut records = Vec::new();
    for item in kata {
        let episode_id = stable_id("dojo-kata", item)?;
        let system = format!(
            "{PERSONA} Practice the {} communicative form in a {} voice.",
            item.kata, item.voice
        );
        records.push(make_record(
            format!("{episode_id}-step-00"),
            episode_id,
            0,
            format!("dialogue.kata.{}", item.kata),
            belt_difficulty(&item.belt),
            parse_language(&item.lang)?,
            system,
            vec![json!({"role": "user", "content": item.situation})],
            item.move_chosen.clone(),
            contracts,
        )?);
    }
    for item in sparring {
        let episode_id = stable_id("dojo-sparring", item)?;
        let forms = item.kata_under_test.join(", ");
        let system =
            format!("{PERSONA} Hold the following communicative forms under pressure: {forms}.");
        let language = parse_language(&item.lang)?;
        let family = format!("dialogue.sparring.{}", item.adversary);
        let mut conversation = Vec::new();
        let mut assistant_step = 0usize;
        for turn in &item.dialogue {
            match turn.role.as_str() {
                "user" => conversation.push(json!({"role": "user", "content": turn.text})),
                "assistant" => {
                    records.push(make_record(
                        format!("{episode_id}-step-{assistant_step:02}"),
                        episode_id.clone(),
                        assistant_step,
                        family.clone(),
                        (assistant_step + 2).min(5) as u8,
                        language,
                        system.clone(),
                        conversation.clone(),
                        turn.text.clone(),
                        contracts,
                    )?);
                    conversation.push(json!({"role": "assistant", "content": turn.text}));
                    assistant_step += 1;
                }
                role => return Err(format!("unsupported Dojo dialogue role {role:?}").into()),
            }
        }
        if assistant_step == 0 {
            return Err(format!("sparring episode {episode_id} has no assistant target").into());
        }
    }
    Ok(records)
}

#[allow(clippy::too_many_arguments)]
fn make_record(
    id: String,
    episode_id: String,
    step: usize,
    family: String,
    difficulty: u8,
    language: WorkflowLanguage,
    system_context: String,
    conversation: Vec<Value>,
    target: String,
    contracts: &ToolContractSnapshot,
) -> Result<KnowledgeSftRecord, Box<dyn Error>> {
    match KnowledgeAction::parse_runtime(&target)? {
        KnowledgeAction::Answer { .. } => {}
        _ => return Err(format!("Dojo target {id} is not a plain runtime answer").into()),
    }
    let tools = Vec::new();
    let prompt = render_knowledge_prompt_with_system(&tools, &conversation, Some(&system_context));
    Ok(KnowledgeSftRecord {
        schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
        id,
        episode_id,
        step,
        stage: WorkflowStage::Answer,
        family,
        difficulty,
        language,
        contract_hash: contracts.manifest_hash.clone(),
        tools,
        conversation,
        system_context: Some(system_context),
        prompt,
        target,
    })
}

fn stratified_split<T: Clone>(
    records: Vec<T>,
    key: impl Fn(&T) -> String,
    rng: &mut StdRng,
) -> (Vec<T>, Vec<T>) {
    let mut buckets: BTreeMap<String, Vec<T>> = BTreeMap::new();
    for record in records {
        buckets.entry(key(&record)).or_default().push(record);
    }
    let mut training = Vec::new();
    let mut evaluation = Vec::new();
    for (_, mut bucket) in buckets {
        bucket.shuffle(rng);
        let eval_count = (bucket.len() / 10).max(1).min(bucket.len());
        training.extend(bucket.drain(eval_count..));
        evaluation.extend(bucket);
    }
    (training, evaluation)
}

fn reject_overlap(
    training: &[KnowledgeSftRecord],
    evaluation: &[KnowledgeSftRecord],
) -> Result<(), Box<dyn Error>> {
    let train_ids: std::collections::HashSet<_> =
        training.iter().map(|record| record.id.as_str()).collect();
    let train_episodes: std::collections::HashSet<_> = training
        .iter()
        .map(|record| record.episode_id.as_str())
        .collect();
    if let Some(record) = evaluation.iter().find(|record| {
        train_ids.contains(record.id.as_str())
            || train_episodes.contains(record.episode_id.as_str())
    }) {
        return Err(format!("Dojo train/eval episode leakage at {}", record.episode_id).into());
    }
    Ok(())
}

fn parse_language(value: &str) -> Result<WorkflowLanguage, Box<dyn Error>> {
    match value {
        "en" | "english" => Ok(WorkflowLanguage::English),
        "de" | "german" => Ok(WorkflowLanguage::German),
        "fr" | "french" => Ok(WorkflowLanguage::French),
        _ => Err(format!("unsupported Dojo language {value:?}").into()),
    }
}

fn belt_difficulty(value: &str) -> u8 {
    match value {
        "white" => 1,
        "yellow" => 2,
        "green" => 3,
        "brown" => 4,
        "black" => 5,
        _ => 1,
    }
}

fn stable_id(prefix: &str, value: &impl Serialize) -> Result<String, Box<dyn Error>> {
    let encoded = serde_json::to_vec(value)?;
    let digest = blake3::hash(&encoded).to_hex().to_string();
    Ok(format!("{prefix}-{}", &digest[..16]))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, Box<dyn Error>> {
    let mut records = Vec::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(&line).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.display(), index + 1),
            )
        })?);
    }
    if records.is_empty() {
        return Err(format!("{} contains no records", path.display()).into());
    }
    Ok(records)
}

fn write_records_new(path: &Path, records: &[KnowledgeSftRecord]) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn fingerprint_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn required<'a>(arguments: &'a [String], flag: &str) -> Result<&'a str, Box<dyn Error>> {
    optional(arguments, flag).ok_or_else(|| format!("{flag} is required").into())
}

fn optional<'a>(arguments: &'a [String], flag: &str) -> Option<&'a str> {
    let index = arguments.iter().position(|value| value == flag)?;
    arguments.get(index + 1).map(String::as_str)
}
