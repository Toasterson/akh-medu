//! Prepare immutable, decontaminated natural-language shards for BabyLLM.
//!
//! Inputs are canonical `FoundationRecord` JSONL files. Keeping acquisition
//! separate from normalization means Hugging Face, an offline mirror, or a
//! future provider can all feed the same audited boundary without teaching
//! the trainer provider-specific schemas.
//!
//! ```text
//! cargo run -p grug-data --release --example prepare_foundation -- \
//!   --train-jsonl staging/fineweb-edu-en.jsonl \
//!   --train-jsonl staging/fineweb2-hq-de.jsonl \
//!   --eval-jsonl staging/foundation-eval.jsonl \
//!   --out runs/foundation/tool-native-v1/corpus --run-id tool-native-v1
//! ```

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use grug_data::{
    FOUNDATION_CORPUS_GENERATOR_VERSION, FOUNDATION_CORPUS_MANIFEST_SCHEMA_VERSION,
    FoundationCorpusManifest, FoundationDecontaminationSummary, FoundationRecord, FoundationShard,
    FoundationSourceSummary, FoundationSplit, FoundationSplitManifest,
    update_foundation_fingerprint,
};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug)]
struct Args {
    train_jsonl: Vec<PathBuf>,
    eval_jsonl: Vec<PathBuf>,
    out: PathBuf,
    run_id: String,
    records_per_shard: u64,
    ngram_size: usize,
    allowed_languages: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SourceKey {
    split: FoundationSplit,
    source: String,
    revision: String,
    license: String,
    language: String,
}

#[derive(Default)]
struct CorpusCounts {
    languages: BTreeMap<String, u64>,
    sources: BTreeMap<SourceKey, u64>,
}

impl CorpusCounts {
    fn observe(&mut self, split: FoundationSplit, record: &FoundationRecord) {
        *self.languages.entry(record.language.clone()).or_default() += 1;
        *self
            .sources
            .entry(SourceKey {
                split,
                source: record.source.clone(),
                revision: record.source_revision.clone(),
                license: record.license.clone(),
                language: record.language.clone(),
            })
            .or_default() += 1;
    }

    fn source_summaries(self) -> Vec<FoundationSourceSummary> {
        self.sources
            .into_iter()
            .map(|(key, records)| FoundationSourceSummary {
                split: key.split,
                source: key.source,
                source_revision: key.revision,
                license: key.license,
                language: key.language,
                records,
            })
            .collect()
    }
}

struct ShardWriter {
    root: PathBuf,
    split: FoundationSplit,
    records_per_shard: u64,
    next_index: usize,
    current: Option<OpenShard>,
    shards: Vec<FoundationShard>,
    records: u64,
    jsonl_bytes: u64,
    fingerprint: blake3::Hasher,
}

struct OpenShard {
    relative: String,
    writer: BufWriter<File>,
    hasher: blake3::Hasher,
    records: u64,
    jsonl_bytes: u64,
}

impl ShardWriter {
    fn new(root: &Path, split: FoundationSplit, records_per_shard: u64) -> Self {
        Self {
            root: root.to_path_buf(),
            split,
            records_per_shard,
            next_index: 0,
            current: None,
            shards: Vec::new(),
            records: 0,
            jsonl_bytes: 0,
            fingerprint: blake3::Hasher::new(),
        }
    }

    fn write_record(&mut self, record: &FoundationRecord) -> Result<(), Box<dyn Error>> {
        if self
            .current
            .as_ref()
            .is_some_and(|shard| shard.records >= self.records_per_shard)
        {
            self.finish_current()?;
        }
        if self.current.is_none() {
            let relative = format!("{}/part-{:05}.jsonl", self.split.as_str(), self.next_index);
            self.next_index += 1;
            let path = self.root.join(&relative);
            std::fs::create_dir_all(path.parent().ok_or("shard path has no parent")?)?;
            let file = OpenOptions::new().write(true).create_new(true).open(path)?;
            self.current = Some(OpenShard {
                relative,
                writer: BufWriter::new(file),
                hasher: blake3::Hasher::new(),
                records: 0,
                jsonl_bytes: 0,
            });
        }

        let encoded = serde_json::to_vec(record)?;
        let shard = self.current.as_mut().expect("opened above");
        shard.writer.write_all(&encoded)?;
        shard.writer.write_all(b"\n")?;
        shard.hasher.update(&encoded);
        shard.hasher.update(b"\n");
        shard.records += 1;
        shard.jsonl_bytes += encoded.len() as u64 + 1;
        self.records += 1;
        self.jsonl_bytes += encoded.len() as u64 + 1;
        update_foundation_fingerprint(&mut self.fingerprint, record)?;
        Ok(())
    }

    fn finish(mut self) -> Result<FoundationSplitManifest, Box<dyn Error>> {
        self.finish_current()?;
        Ok(FoundationSplitManifest {
            records: self.records,
            jsonl_bytes: self.jsonl_bytes,
            fingerprint: format!("blake3:{}", self.fingerprint.finalize().to_hex()),
            shards: self.shards,
        })
    }

    fn finish_current(&mut self) -> Result<(), Box<dyn Error>> {
        let Some(mut shard) = self.current.take() else {
            return Ok(());
        };
        shard.writer.flush()?;
        let file = shard.writer.into_inner()?;
        file.sync_all()?;
        self.shards.push(FoundationShard {
            file: shard.relative,
            file_hash: format!("blake3:{}", shard.hasher.finalize().to_hex()),
            records: shard.records,
            jsonl_bytes: shard.jsonl_bytes,
        });
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.out.exists() {
        return Err(format!(
            "{} already exists; prepared corpora are immutable",
            args.out.display()
        )
        .into());
    }
    let parent = args
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let name = args
        .out
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("--out must end in a valid UTF-8 directory name")?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(format!("stale temporary path exists: {}", temporary.display()).into());
    }
    std::fs::create_dir(&temporary)?;

    let result = prepare(&args, &temporary);
    match result {
        Ok(manifest) => {
            let manifest_path = temporary.join("foundation-corpus.json");
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&manifest_path)?;
            serde_json::to_writer_pretty(&mut file, &manifest)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            std::fs::rename(&temporary, &args.out)?;
            println!(
                "prepared foundation corpus: train={} eval={} languages={:?} -> {}",
                manifest.training.records,
                manifest.evaluation.records,
                manifest.languages,
                args.out.display()
            );
            println!(
                "decontamination: duplicate={} eval_exact={} eval_{}gram={}",
                manifest.decontamination.removed_duplicate_text,
                manifest.decontamination.removed_evaluation_exact,
                manifest.decontamination.word_ngram_size,
                manifest.decontamination.removed_evaluation_ngram
            );
            Ok(())
        }
        Err(error) => {
            // This path was freshly created above and is never user-supplied.
            let _ = std::fs::remove_dir_all(&temporary);
            Err(error)
        }
    }
}

fn prepare(args: &Args, output: &Path) -> Result<FoundationCorpusManifest, Box<dyn Error>> {
    let mut counts = CorpusCounts::default();
    let mut eval_ids = HashSet::new();
    let mut eval_texts = HashSet::new();
    let mut eval_ngrams = HashSet::new();
    let mut eval_writer =
        ShardWriter::new(output, FoundationSplit::Evaluation, args.records_per_shard);

    stream_records(&args.eval_jsonl, |record, location| {
        record.validate()?;
        validate_language(record, location, &args.allowed_languages)?;
        if !eval_ids.insert(record.id.clone()) {
            return Err(format!("duplicate evaluation id {:?} at {location}", record.id).into());
        }
        let text_hash = normalized_text_hash(&record.text);
        if !eval_texts.insert(text_hash) {
            return Err(format!(
                "duplicate normalized evaluation text at {location}; evaluation must not overweight copies"
            )
            .into());
        }
        eval_ngrams.extend(word_ngram_hashes(&record.text, args.ngram_size));
        counts.observe(FoundationSplit::Evaluation, record);
        eval_writer.write_record(record)
    })?;
    let evaluation = eval_writer.finish()?;

    let mut train_ids = HashSet::new();
    let mut train_texts = HashSet::new();
    let mut removed_duplicate_text = 0u64;
    let mut removed_evaluation_exact = 0u64;
    let mut removed_evaluation_ngram = 0u64;
    let mut train_writer =
        ShardWriter::new(output, FoundationSplit::Training, args.records_per_shard);
    stream_records(&args.train_jsonl, |record, location| {
        record.validate()?;
        validate_language(record, location, &args.allowed_languages)?;
        if eval_ids.contains(&record.id) {
            return Err(format!(
                "train/evaluation identity leakage for {:?} at {location}",
                record.id
            )
            .into());
        }
        if !train_ids.insert(record.id.clone()) {
            return Err(format!("duplicate training id {:?} at {location}", record.id).into());
        }
        let text_hash = normalized_text_hash(&record.text);
        if eval_texts.contains(&text_hash) {
            removed_evaluation_exact += 1;
            return Ok(());
        }
        if !train_texts.insert(text_hash) {
            removed_duplicate_text += 1;
            return Ok(());
        }
        if word_ngram_hashes(&record.text, args.ngram_size)
            .iter()
            .any(|hash| eval_ngrams.contains(hash))
        {
            removed_evaluation_ngram += 1;
            return Ok(());
        }
        counts.observe(FoundationSplit::Training, record);
        train_writer.write_record(record)
    })?;
    let training = train_writer.finish()?;

    let languages = std::mem::take(&mut counts.languages);
    if !args.allowed_languages.is_empty() {
        let observed: BTreeSet<_> = languages.keys().cloned().collect();
        if observed != args.allowed_languages {
            return Err(format!(
                "prepared language set {observed:?} does not equal required set {:?}",
                args.allowed_languages
            )
            .into());
        }
    }
    let sources = counts.source_summaries();
    let manifest = FoundationCorpusManifest {
        schema_version: FOUNDATION_CORPUS_MANIFEST_SCHEMA_VERSION,
        generator_version: FOUNDATION_CORPUS_GENERATOR_VERSION.into(),
        run_id: args.run_id.clone(),
        training,
        evaluation,
        languages,
        sources,
        decontamination: FoundationDecontaminationSummary {
            method: format!("nfkc_lower_whitespace_exact+word_{}gram", args.ngram_size),
            word_ngram_size: args.ngram_size,
            evaluation_texts: eval_texts.len() as u64,
            evaluation_ngrams: eval_ngrams.len() as u64,
            removed_duplicate_text,
            removed_evaluation_exact,
            removed_evaluation_ngram,
        },
    };
    manifest.verify()?;
    Ok(manifest)
}

fn validate_language(
    record: &FoundationRecord,
    location: &str,
    allowed: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    if !allowed.is_empty() && !allowed.contains(&record.language) {
        return Err(format!(
            "foundation language {:?} at {location} is outside the allowed set {allowed:?}",
            record.language
        )
        .into());
    }
    Ok(())
}

fn stream_records(
    paths: &[PathBuf],
    mut observe: impl FnMut(&FoundationRecord, &str) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    for path in paths {
        let reader = BufReader::new(File::open(path)?);
        for (line_index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let location = format!("{}:{}", path.display(), line_index + 1);
            let record: FoundationRecord = serde_json::from_str(&line)
                .map_err(|error| format!("{location} is not a FoundationRecord: {error}"))?;
            observe(&record, &location)?;
        }
    }
    Ok(())
}

fn normalized_words(text: &str) -> Vec<String> {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn normalized_text_hash(text: &str) -> [u8; 32] {
    let normalized = normalized_words(text).join(" ");
    *blake3::hash(normalized.as_bytes()).as_bytes()
}

fn word_ngram_hashes(text: &str, size: usize) -> Vec<[u8; 32]> {
    let words = normalized_words(text);
    words
        .windows(size)
        .map(|window| *blake3::hash(window.join(" ").as_bytes()).as_bytes())
        .collect()
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut train_jsonl = Vec::new();
    let mut eval_jsonl = Vec::new();
    let mut out = None;
    let mut run_id = None;
    let mut records_per_shard = 50_000u64;
    let mut ngram_size = 13usize;
    let mut allowed_languages = BTreeSet::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--train-jsonl" => train_jsonl.push(PathBuf::from(
                arguments.next().ok_or("--train-jsonl requires FILE")?,
            )),
            "--eval-jsonl" => eval_jsonl.push(PathBuf::from(
                arguments.next().ok_or("--eval-jsonl requires FILE")?,
            )),
            "--out" => out = Some(PathBuf::from(arguments.next().ok_or("--out requires DIR")?)),
            "--run-id" => run_id = Some(arguments.next().ok_or("--run-id requires ID")?),
            "--records-per-shard" => {
                records_per_shard = arguments
                    .next()
                    .ok_or("--records-per-shard requires N")?
                    .parse()?
            }
            "--ngram-size" => {
                ngram_size = arguments.next().ok_or("--ngram-size requires N")?.parse()?
            }
            "--allow-language" => {
                let language = arguments.next().ok_or("--allow-language requires CODE")?;
                if language.trim().is_empty() || !allowed_languages.insert(language) {
                    return Err("allowed language codes must be non-blank and unique".into());
                }
            }
            "--help" | "-h" => {
                println!(
                    "prepare_foundation --train-jsonl FILE... --eval-jsonl FILE... --out DIR --run-id ID [--records-per-shard 50000] [--ngram-size 13] [--allow-language CODE...]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if train_jsonl.is_empty() || eval_jsonl.is_empty() {
        return Err("at least one --train-jsonl and --eval-jsonl are required".into());
    }
    if records_per_shard == 0 || ngram_size < 2 {
        return Err("--records-per-shard must be positive and --ngram-size at least 2".into());
    }
    let run_id = run_id.ok_or("--run-id is required")?;
    if run_id.trim().is_empty() {
        return Err("--run-id must not be blank".into());
    }
    Ok(Args {
        train_jsonl,
        eval_jsonl,
        out: out.ok_or("--out is required")?,
        run_id,
        records_per_shard,
        ngram_size,
        allowed_languages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_catches_case_and_unicode_compatibility() {
        assert_eq!(
            normalized_text_hash("A  FULLWIDTH： TEST"),
            normalized_text_hash("a fullwidth: test")
        );
    }

    #[test]
    fn ngram_matching_is_order_sensitive() {
        let one = word_ngram_hashes("one two three four", 3);
        let two = word_ngram_hashes("one three two four", 3);
        assert!(one.iter().all(|hash| !two.contains(hash)));
    }
}
