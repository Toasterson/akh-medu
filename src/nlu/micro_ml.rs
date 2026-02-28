//! Micro-ML NER layer — NLU Tier 2.
//!
//! Uses an ONNX-based multilingual NER model (DistilBERT) to extract entity
//! spans, then classifies intent from entity patterns. The ML layer *augments*
//! the rule parser (Tier 1) rather than replacing it — entities discovered by
//! NER can be fed back into `parse_prose()` for a better structured parse.
//!
//! Graceful degradation: if model files are absent, `try_load()` returns `None`
//! and the pipeline silently skips this tier.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::ParseContext;
use crate::grammar::parser::{self, ParseResult};

use super::error::{NluError, NluResult};

// ── Types ──────────────────────────────────────────────────────────────────

/// BIO NER entity type (from DistilBERT multilingual NER).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NerEntityType {
    Person,
    Location,
    Organization,
    Date,
    Misc,
}

impl NerEntityType {
    /// Parse a BIO tag suffix into an entity type.
    ///
    /// Expects the part after `B-` or `I-`, e.g. `"PER"`, `"LOC"`.
    pub fn from_bio_suffix(suffix: &str) -> Option<Self> {
        match suffix {
            "PER" => Some(Self::Person),
            "LOC" => Some(Self::Location),
            "ORG" => Some(Self::Organization),
            "DATE" => Some(Self::Date),
            "MISC" => Some(Self::Misc),
            _ => None,
        }
    }
}

impl std::fmt::Display for NerEntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Person => write!(f, "PER"),
            Self::Location => write!(f, "LOC"),
            Self::Organization => write!(f, "ORG"),
            Self::Date => write!(f, "DATE"),
            Self::Misc => write!(f, "MISC"),
        }
    }
}

/// A recognized entity span in the input text.
#[derive(Debug, Clone)]
pub struct EntitySpan {
    /// The surface text of the entity.
    pub text: String,
    /// The entity type label.
    pub entity_type: NerEntityType,
    /// Start character offset in the original input.
    pub start: usize,
    /// End character offset (exclusive) in the original input.
    pub end: usize,
    /// Model confidence for this entity span.
    pub confidence: f32,
}

/// Intent classification from entity patterns.
///
/// Derived from entity types and structural cues — no hardcoded
/// natural-language keywords (the system is language-agnostic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentClass {
    /// Declarative assertion ("X is Y").
    Assertion,
    /// Information query ("What is X?").
    Query,
    /// Imperative command ("Do X").
    Command,
    /// Goal-seeking ("Find X", "Explore Y").
    Goal,
    /// Temporal reference detected (Date entity present).
    Temporal,
    /// Social context (Person or Organization entities present).
    Social,
    /// Could not classify.
    Unknown,
}

/// Augmented parse result from the ML layer.
#[derive(Debug, Clone)]
pub struct AugmentedParse {
    /// Extracted entity spans.
    pub entities: Vec<EntitySpan>,
    /// Classified intent.
    pub intent: IntentClass,
    /// Re-parsed tree (if entity substitution improved parsing).
    pub tree: Option<AbsTree>,
    /// Overall confidence.
    pub confidence: f32,
}

// ── BIO tag decoding ───────────────────────────────────────────────────────

/// A single BIO-tagged token with its character offsets.
#[derive(Debug, Clone)]
pub(crate) struct BioToken {
    /// The BIO tag: "O", "B-PER", "I-LOC", etc.
    tag: String,
    /// Start character offset in the original text.
    char_start: usize,
    /// End character offset (exclusive) in the original text.
    char_end: usize,
}

/// Merge BIO-tagged tokens into contiguous entity spans.
///
/// Rules:
/// - `B-X` starts a new entity of type X.
/// - `I-X` continues the current entity if the type matches; otherwise starts new.
/// - `O` closes any open entity.
pub(crate) fn merge_bio_spans(tokens: &[BioToken], text: &str) -> Vec<EntitySpan> {
    let mut spans = Vec::new();
    let mut current_type: Option<NerEntityType> = None;
    let mut current_start: usize = 0;
    let mut current_end: usize = 0;

    for token in tokens {
        let (prefix, suffix) = parse_bio_tag(&token.tag);

        match prefix {
            BioPrefix::B => {
                // Close previous entity if open
                if let Some(etype) = current_type.take() {
                    push_span(&mut spans, text, etype, current_start, current_end);
                }
                // Start new entity
                if let Some(etype) = suffix.and_then(NerEntityType::from_bio_suffix) {
                    current_type = Some(etype);
                    current_start = token.char_start;
                    current_end = token.char_end;
                }
            }
            BioPrefix::I => {
                if let Some(etype) = suffix.and_then(NerEntityType::from_bio_suffix) {
                    if current_type.as_ref() == Some(&etype) {
                        // Continue current entity
                        current_end = token.char_end;
                    } else {
                        // Type mismatch — close previous, start new
                        if let Some(prev) = current_type.take() {
                            push_span(&mut spans, text, prev, current_start, current_end);
                        }
                        current_type = Some(etype);
                        current_start = token.char_start;
                        current_end = token.char_end;
                    }
                }
            }
            BioPrefix::O => {
                // Close any open entity
                if let Some(etype) = current_type.take() {
                    push_span(&mut spans, text, etype, current_start, current_end);
                }
            }
        }
    }

    // Close trailing entity
    if let Some(etype) = current_type.take() {
        push_span(&mut spans, text, etype, current_start, current_end);
    }

    spans
}

fn push_span(
    spans: &mut Vec<EntitySpan>,
    text: &str,
    entity_type: NerEntityType,
    start: usize,
    end: usize,
) {
    let end = end.min(text.len());
    let start = start.min(end);
    let surface = text[start..end].trim();
    if !surface.is_empty() {
        spans.push(EntitySpan {
            text: surface.to_string(),
            entity_type,
            start,
            end,
            confidence: 1.0, // Will be refined by model logits
        });
    }
}

#[derive(Debug, PartialEq)]
enum BioPrefix {
    B,
    I,
    O,
}

fn parse_bio_tag(tag: &str) -> (BioPrefix, Option<&str>) {
    if tag == "O" {
        return (BioPrefix::O, None);
    }
    if let Some(suffix) = tag.strip_prefix("B-") {
        return (BioPrefix::B, Some(suffix));
    }
    if let Some(suffix) = tag.strip_prefix("I-") {
        return (BioPrefix::I, Some(suffix));
    }
    (BioPrefix::O, None)
}

// ── Intent classification ──────────────────────────────────────────────────

/// Classify intent from entity types and structural cues.
///
/// This is purely rule-based over entity types — no hardcoded natural-language
/// strings. The classification uses entity type distributions:
///
/// - Date entities → `Temporal`
/// - Person or Organization entities → `Social`
/// - No entities and parse failed → `Unknown`
///
/// When entities are present but don't match special types, falls back to
/// `Assertion` (the most common intent for factual statements).
pub fn classify_intent(entities: &[EntitySpan], parse_result: &ParseResult) -> IntentClass {
    // Check parse result structure first
    match parse_result {
        ParseResult::Query { .. } => return IntentClass::Query,
        ParseResult::Command(_) => return IntentClass::Command,
        ParseResult::Goal { .. } => return IntentClass::Goal,
        _ => {}
    }

    // Classify from entity types
    let has_date = entities.iter().any(|e| e.entity_type == NerEntityType::Date);
    let has_social = entities
        .iter()
        .any(|e| e.entity_type == NerEntityType::Person || e.entity_type == NerEntityType::Organization);

    if has_date {
        IntentClass::Temporal
    } else if has_social {
        IntentClass::Social
    } else if !entities.is_empty() {
        IntentClass::Assertion
    } else {
        match parse_result {
            ParseResult::Facts(_) => IntentClass::Assertion,
            _ => IntentClass::Unknown,
        }
    }
}

// ── MicroMlLayer ───────────────────────────────────────────────────────────

/// The ONNX-based NER model session with tokenizer.
pub struct MicroMlLayer {
    /// The ONNX runtime session for the NER model.
    #[cfg(feature = "nlu-ml")]
    ner_session: ort::session::Session,
    /// The HuggingFace tokenizer for subword encoding.
    #[cfg(feature = "nlu-ml")]
    tokenizer: tokenizers::Tokenizer,
    /// Map from model output index to BIO tag string.
    label_map: Vec<String>,
}

impl std::fmt::Debug for MicroMlLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicroMlLayer")
            .field("label_count", &self.label_map.len())
            .finish_non_exhaustive()
    }
}

impl MicroMlLayer {
    /// Expected model directory layout under `data_dir`:
    ///
    /// ```text
    /// models/ner/
    /// ├── model.onnx
    /// ├── tokenizer.json
    /// └── config.json   # { "id2label": { "0": "O", "1": "B-PER", ... } }
    /// ```
    const NER_SUBDIR: &'static str = "models/ner";

    /// Load the NER model from `model_dir`.
    ///
    /// Returns `NluError::ModelNotFound` if any required file is missing,
    /// or `NluError::ModelLoadFailed` if loading fails.
    #[cfg(feature = "nlu-ml")]
    pub fn load(model_dir: &Path) -> NluResult<Self> {
        let ner_dir = model_dir.join(Self::NER_SUBDIR);

        let model_path = ner_dir.join("model.onnx");
        if !model_path.exists() {
            return Err(NluError::ModelNotFound { path: model_path });
        }

        let tokenizer_path = ner_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(NluError::ModelNotFound {
                path: tokenizer_path,
            });
        }

        let config_path = ner_dir.join("config.json");
        if !config_path.exists() {
            return Err(NluError::ModelNotFound { path: config_path });
        }

        // Load label map from config.json
        let label_map = load_label_map(&config_path)?;

        // Load ONNX session
        let ner_session = ort::session::Session::builder()
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("ONNX session builder: {e}"),
            })?
            .commit_from_file(&model_path)
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("ONNX model load: {e}"),
            })?;

        // Load tokenizer
        let tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| NluError::ModelLoadFailed {
                reason: format!("Tokenizer load: {e}"),
            })?;

        Ok(Self {
            ner_session,
            tokenizer,
            label_map,
        })
    }

    /// Non-feature-gated load that always fails (used when `nlu-ml` is disabled).
    #[cfg(not(feature = "nlu-ml"))]
    pub fn load(model_dir: &Path) -> NluResult<Self> {
        let ner_dir = model_dir.join(Self::NER_SUBDIR);
        let model_path = ner_dir.join("model.onnx");
        Err(NluError::ModelNotFound { path: model_path })
    }

    /// Gracefully attempt to load models. Returns `None` if any file is missing
    /// or loading fails — the pipeline will silently skip Tier 2.
    pub fn try_load(data_dir: &Path) -> Option<Self> {
        Self::load(data_dir).ok()
    }

    /// Extract named entities from input text via the ONNX NER model.
    #[cfg(feature = "nlu-ml")]
    pub fn extract_entities(&mut self, text: &str) -> NluResult<Vec<EntitySpan>> {
        use ort::value::Tensor;

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("Tokenization failed: {e}"),
            })?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let seq_len = input_ids.len();

        // Build input tensors [1, seq_len]
        let ids_tensor = Tensor::from_array(([1, seq_len], input_ids)).map_err(|e| {
            NluError::ModelLoadFailed {
                reason: format!("Tensor creation: {e}"),
            }
        })?;
        let mask_tensor = Tensor::from_array(([1, seq_len], attention_mask)).map_err(|e| {
            NluError::ModelLoadFailed {
                reason: format!("Tensor creation: {e}"),
            }
        })?;

        // Run inference
        let outputs = self
            .ner_session
            .run(ort::inputs![ids_tensor, mask_tensor])
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("ONNX inference: {e}"),
            })?;

        // Extract logits: shape [1, seq_len, num_labels]
        let (shape, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("Logit extraction: {e}"),
            })?;
        let num_labels = if shape.len() == 3 {
            shape[2] as usize
        } else {
            self.label_map.len()
        };

        // Argmax per token → BIO tag
        let offsets = encoding.get_offsets();
        let mut bio_tokens = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Skip special tokens ([CLS], [SEP])
            let (start, end) = offsets[i];
            if start == 0 && end == 0 && i > 0 {
                continue;
            }

            // Index into flat logits: [batch=0, token=i, label=j]
            let base = i * num_labels;
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_labels {
                if let Some(&val) = logits_data.get(base + j) {
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }
            }

            let tag = self
                .label_map
                .get(max_idx)
                .cloned()
                .unwrap_or_else(|| "O".to_string());

            bio_tokens.push(BioToken {
                tag,
                char_start: start,
                char_end: end,
            });
        }

        Ok(merge_bio_spans(&bio_tokens, text))
    }

    /// Non-feature-gated stub that returns an empty entity list.
    #[cfg(not(feature = "nlu-ml"))]
    pub fn extract_entities(&mut self, _text: &str) -> NluResult<Vec<EntitySpan>> {
        Ok(Vec::new())
    }

    /// Augment a parse with NER-extracted entities and intent classification.
    ///
    /// 1. Extract entity spans via NER model
    /// 2. Classify intent from entity types + initial parse result
    /// 3. Re-run `parse_prose()` with entity labels substituted as grounded tokens
    /// 4. If re-parse produces a structured result (not Freeform), return it
    /// 5. Otherwise, return AugmentedParse with entities but `tree: None`
    pub fn augment_parse(&mut self, text: &str, ctx: &ParseContext) -> NluResult<AugmentedParse> {
        let entities = self.extract_entities(text)?;

        // Get the initial parse result for intent classification
        let initial_parse = parser::parse_prose(text, ctx);
        let intent = classify_intent(&entities, &initial_parse);

        // If entities found, try re-parsing with entity labels substituted
        let tree = if !entities.is_empty() {
            let enriched = substitute_entities(text, &entities);
            let re_parse = parser::parse_prose(&enriched, ctx);
            match re_parse {
                ParseResult::Facts(facts) if !facts.is_empty() => {
                    if facts.len() == 1 {
                        Some(facts.into_iter().next().unwrap())
                    } else {
                        Some(AbsTree::and(facts))
                    }
                }
                ParseResult::Query { tree, .. } => Some(tree),
                _ => None,
            }
        } else {
            None
        };

        let confidence = if tree.is_some() { 0.80 } else { 0.50 };

        Ok(AugmentedParse {
            entities,
            intent,
            tree,
            confidence,
        })
    }

    /// Access the label map (for testing/inspection).
    pub fn label_map(&self) -> &[String] {
        &self.label_map
    }
}

/// Substitute entity spans in text with typed labels for re-parsing.
///
/// E.g., "John went to Paris" → "Person went to Location"
/// This gives the rule parser better structural cues.
fn substitute_entities(text: &str, entities: &[EntitySpan]) -> String {
    if entities.is_empty() {
        return text.to_string();
    }

    // Sort entities by start offset (descending) to replace from back to front
    let mut sorted: Vec<&EntitySpan> = entities.iter().collect();
    sorted.sort_by(|a, b| b.start.cmp(&a.start));

    let mut result = text.to_string();
    for entity in sorted {
        let end = entity.end.min(result.len());
        let start = entity.start.min(end);
        let replacement = format!("{}", entity.entity_type);
        result.replace_range(start..end, &replacement);
    }

    result
}

/// Load the id2label map from a HuggingFace config.json.
///
/// Expected format: `{ "id2label": { "0": "O", "1": "B-PER", ... }, ... }`
#[cfg(feature = "nlu-ml")]
fn load_label_map(config_path: &Path) -> NluResult<Vec<String>> {
    let content = std::fs::read_to_string(config_path).map_err(|e| NluError::ModelLoadFailed {
        reason: format!("Cannot read config.json: {e}"),
    })?;

    let config: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("Invalid config.json: {e}"),
        })?;

    let id2label = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .ok_or_else(|| NluError::ModelLoadFailed {
            reason: "config.json missing 'id2label' map".to_string(),
        })?;

    let max_idx = id2label
        .keys()
        .filter_map(|k| k.parse::<usize>().ok())
        .max()
        .unwrap_or(0);

    let mut labels = vec!["O".to_string(); max_idx + 1];
    for (k, v) in id2label {
        if let (Ok(idx), Some(label)) = (k.parse::<usize>(), v.as_str()) {
            if idx < labels.len() {
                labels[idx] = label.to_string();
            }
        }
    }

    Ok(labels)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    // ── NerEntityType tests ────────────────────────────────────────────

    #[test]
    fn ner_entity_type_serde_roundtrip() {
        for etype in [
            NerEntityType::Person,
            NerEntityType::Location,
            NerEntityType::Organization,
            NerEntityType::Date,
            NerEntityType::Misc,
        ] {
            let json = serde_json::to_string(&etype).unwrap();
            let restored: NerEntityType = serde_json::from_str(&json).unwrap();
            assert_eq!(etype, restored);
        }
    }

    #[test]
    fn ner_entity_type_from_bio_suffix() {
        assert_eq!(
            NerEntityType::from_bio_suffix("PER"),
            Some(NerEntityType::Person)
        );
        assert_eq!(
            NerEntityType::from_bio_suffix("LOC"),
            Some(NerEntityType::Location)
        );
        assert_eq!(
            NerEntityType::from_bio_suffix("ORG"),
            Some(NerEntityType::Organization)
        );
        assert_eq!(
            NerEntityType::from_bio_suffix("DATE"),
            Some(NerEntityType::Date)
        );
        assert_eq!(
            NerEntityType::from_bio_suffix("MISC"),
            Some(NerEntityType::Misc)
        );
        assert_eq!(NerEntityType::from_bio_suffix("UNKNOWN"), None);
    }

    // ── BIO tag merging tests ──────────────────────────────────────────

    #[test]
    fn bio_merge_single_entity() {
        let text = "John Smith went home";
        let tokens = vec![
            BioToken { tag: "B-PER".into(), char_start: 0, char_end: 4 },
            BioToken { tag: "I-PER".into(), char_start: 5, char_end: 10 },
            BioToken { tag: "O".into(), char_start: 11, char_end: 15 },
            BioToken { tag: "O".into(), char_start: 16, char_end: 20 },
        ];

        let spans = merge_bio_spans(&tokens, text);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "John Smith");
        assert_eq!(spans[0].entity_type, NerEntityType::Person);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 10);
    }

    #[test]
    fn bio_merge_multiple_entities() {
        let text = "John visited Paris on Monday";
        let tokens = vec![
            BioToken { tag: "B-PER".into(), char_start: 0, char_end: 4 },
            BioToken { tag: "O".into(), char_start: 5, char_end: 12 },
            BioToken { tag: "B-LOC".into(), char_start: 13, char_end: 18 },
            BioToken { tag: "O".into(), char_start: 19, char_end: 21 },
            BioToken { tag: "B-DATE".into(), char_start: 22, char_end: 28 },
        ];

        let spans = merge_bio_spans(&tokens, text);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, "John");
        assert_eq!(spans[0].entity_type, NerEntityType::Person);
        assert_eq!(spans[1].text, "Paris");
        assert_eq!(spans[1].entity_type, NerEntityType::Location);
        assert_eq!(spans[2].text, "Monday");
        assert_eq!(spans[2].entity_type, NerEntityType::Date);
    }

    #[test]
    fn bio_merge_all_o_tokens() {
        let text = "nothing special here";
        let tokens = vec![
            BioToken { tag: "O".into(), char_start: 0, char_end: 7 },
            BioToken { tag: "O".into(), char_start: 8, char_end: 15 },
            BioToken { tag: "O".into(), char_start: 16, char_end: 20 },
        ];

        let spans = merge_bio_spans(&tokens, text);
        assert!(spans.is_empty());
    }

    #[test]
    fn bio_merge_type_mismatch_splits() {
        // I-LOC after B-PER should start a new entity
        let text = "John Paris";
        let tokens = vec![
            BioToken { tag: "B-PER".into(), char_start: 0, char_end: 4 },
            BioToken { tag: "I-LOC".into(), char_start: 5, char_end: 10 },
        ];

        let spans = merge_bio_spans(&tokens, text);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].entity_type, NerEntityType::Person);
        assert_eq!(spans[1].entity_type, NerEntityType::Location);
    }

    #[test]
    fn bio_merge_trailing_entity() {
        let text = "hello World";
        let tokens = vec![
            BioToken { tag: "O".into(), char_start: 0, char_end: 5 },
            BioToken { tag: "B-LOC".into(), char_start: 6, char_end: 11 },
        ];

        let spans = merge_bio_spans(&tokens, text);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "World");
    }

    // ── Intent classification tests ────────────────────────────────────

    #[test]
    fn intent_temporal_from_date_entity() {
        let entities = vec![EntitySpan {
            text: "Monday".into(),
            entity_type: NerEntityType::Date,
            start: 0,
            end: 6,
            confidence: 0.9,
        }];
        let parse = ParseResult::Freeform {
            text: "on Monday".into(),
            partial: vec![],
        };
        assert_eq!(classify_intent(&entities, &parse), IntentClass::Temporal);
    }

    #[test]
    fn intent_social_from_person() {
        let entities = vec![EntitySpan {
            text: "Alice".into(),
            entity_type: NerEntityType::Person,
            start: 0,
            end: 5,
            confidence: 0.9,
        }];
        let parse = ParseResult::Freeform {
            text: "Alice".into(),
            partial: vec![],
        };
        assert_eq!(classify_intent(&entities, &parse), IntentClass::Social);
    }

    #[test]
    fn intent_social_from_org() {
        let entities = vec![EntitySpan {
            text: "ACME".into(),
            entity_type: NerEntityType::Organization,
            start: 0,
            end: 4,
            confidence: 0.9,
        }];
        let parse = ParseResult::Freeform {
            text: "ACME".into(),
            partial: vec![],
        };
        assert_eq!(classify_intent(&entities, &parse), IntentClass::Social);
    }

    #[test]
    fn intent_assertion_from_misc_entity() {
        let entities = vec![EntitySpan {
            text: "Rust".into(),
            entity_type: NerEntityType::Misc,
            start: 0,
            end: 4,
            confidence: 0.9,
        }];
        let parse = ParseResult::Freeform {
            text: "Rust".into(),
            partial: vec![],
        };
        assert_eq!(classify_intent(&entities, &parse), IntentClass::Assertion);
    }

    #[test]
    fn intent_query_from_parse_result() {
        let entities = vec![];
        let parse = ParseResult::Query {
            subject: "dogs".into(),
            tree: AbsTree::entity("dogs"),
        };
        assert_eq!(classify_intent(&entities, &parse), IntentClass::Query);
    }

    #[test]
    fn intent_unknown_when_empty() {
        let entities = vec![];
        let parse = ParseResult::Freeform {
            text: "hmm".into(),
            partial: vec![],
        };
        assert_eq!(classify_intent(&entities, &parse), IntentClass::Unknown);
    }

    // ── AugmentedParse tests ───────────────────────────────────────────

    #[test]
    fn augmented_parse_with_tree() {
        let tree = AbsTree::triple(
            AbsTree::entity("dogs"),
            AbsTree::relation("are"),
            AbsTree::entity("mammals"),
        );
        let aug = AugmentedParse {
            entities: vec![],
            intent: IntentClass::Assertion,
            tree: Some(tree.clone()),
            confidence: 0.85,
        };
        assert!(aug.tree.is_some());
        assert_eq!(aug.confidence, 0.85);
    }

    #[test]
    fn augmented_parse_without_tree() {
        let aug = AugmentedParse {
            entities: vec![EntitySpan {
                text: "John".into(),
                entity_type: NerEntityType::Person,
                start: 0,
                end: 4,
                confidence: 0.9,
            }],
            intent: IntentClass::Social,
            tree: None,
            confidence: 0.50,
        };
        assert!(aug.tree.is_none());
        assert_eq!(aug.entities.len(), 1);
    }

    // ── Entity substitution tests ──────────────────────────────────────

    #[test]
    fn substitute_entities_replaces_spans() {
        let text = "John went to Paris";
        let entities = vec![
            EntitySpan {
                text: "John".into(),
                entity_type: NerEntityType::Person,
                start: 0,
                end: 4,
                confidence: 0.9,
            },
            EntitySpan {
                text: "Paris".into(),
                entity_type: NerEntityType::Location,
                start: 13,
                end: 18,
                confidence: 0.9,
            },
        ];
        let result = substitute_entities(text, &entities);
        assert_eq!(result, "PER went to LOC");
    }

    #[test]
    fn substitute_entities_empty() {
        let text = "hello world";
        let result = substitute_entities(text, &[]);
        assert_eq!(result, "hello world");
    }

    // ── Graceful degradation tests ─────────────────────────────────────

    #[test]
    fn try_load_returns_none_for_missing_dir() {
        let nonexistent = PathBuf::from("/tmp/akh-medu-nonexistent-test-dir");
        assert!(MicroMlLayer::try_load(&nonexistent).is_none());
    }

    #[test]
    fn load_returns_model_not_found_error() {
        let nonexistent = PathBuf::from("/tmp/akh-medu-nonexistent-test-dir");
        let err = MicroMlLayer::load(&nonexistent).unwrap_err();
        assert!(
            matches!(err, NluError::ModelNotFound { .. }),
            "Expected ModelNotFound, got: {err:?}"
        );
    }
}
