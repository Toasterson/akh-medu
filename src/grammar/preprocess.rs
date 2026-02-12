//! Pre-processor output types and mapping functions for the Eleutherios integration.
//!
//! Provides serializable types that match Eleutherios's extraction pipeline schema:
//! entities, claims, and structured pre-processor output. The grammar parser produces
//! `AbsTree` structures which are mapped into these types for downstream consumption.

use serde::{Deserialize, Serialize};

use super::abs::AbsTree;
use super::detect::{detect_language, detect_per_sentence};
use super::entity_resolution::EntityResolver;
use super::lexer::Language;
use super::parser::{parse_prose, ParseResult};
use super::concrete::ParseContext;

/// A text chunk to pre-process (input from Eleutherios).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// Optional chunk identifier from the upstream pipeline.
    #[serde(default)]
    pub id: Option<String>,
    /// The raw text content.
    pub text: String,
    /// Optional language hint (BCP 47). If absent, auto-detect.
    #[serde(default)]
    pub language: Option<String>,
}

/// An entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Surface form as it appeared in text.
    pub name: String,
    /// Entity type classification.
    pub entity_type: String,
    /// Canonical (resolved) name.
    pub canonical_name: String,
    /// Extraction confidence (0.0–1.0).
    pub confidence: f32,
    /// Known aliases (cross-lingual or variant forms).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// BCP 47 language code of the source text.
    pub source_language: String,
}

/// A claim (relation) extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedClaim {
    /// The original sentence or clause.
    pub claim_text: String,
    /// Claim type classification.
    pub claim_type: String,
    /// Extraction confidence (0.0–1.0).
    pub confidence: f32,
    /// Subject entity label.
    pub subject: String,
    /// Canonical predicate label (e.g., "is-a", "causes").
    pub predicate: String,
    /// Object entity label.
    pub object: String,
    /// BCP 47 language code of the source text.
    pub source_language: String,
}

/// Full pre-processor output for a single text chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreProcessorOutput {
    /// Chunk identifier from the input, if provided.
    #[serde(default)]
    pub chunk_id: Option<String>,
    /// Detected source language (BCP 47).
    pub source_language: String,
    /// Language detection confidence.
    pub detected_language_confidence: f32,
    /// Entities extracted from the chunk.
    pub entities: Vec<ExtractedEntity>,
    /// Claims (relations) extracted from the chunk.
    pub claims: Vec<ExtractedClaim>,
    /// Full abstract syntax trees for consumers that want the raw parse.
    pub abs_trees: Vec<AbsTree>,
}

/// Batch request for the pre-processor HTTP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreProcessRequest {
    pub chunks: Vec<TextChunk>,
}

/// Batch response from the pre-processor HTTP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreProcessResponse {
    pub results: Vec<PreProcessorOutput>,
    pub processing_time_ms: u64,
}

/// Map a canonical predicate label to a claim type.
pub fn predicate_to_claim_type(predicate: &str) -> &'static str {
    match predicate {
        "is-a" | "has-a" | "contains" | "implements" | "defines" => "FACTUAL",
        "causes" => "CAUSAL",
        "located-in" => "SPATIAL",
        "similar-to" => "RELATIONAL",
        "part-of" | "composed-of" => "STRUCTURAL",
        "depends-on" => "DEPENDENCY",
        _ => "OTHER",
    }
}

/// Infer entity type from context (predicate + position).
fn infer_entity_type(predicate: &str, is_subject: bool) -> &'static str {
    match predicate {
        "located-in" if !is_subject => "PLACE",
        "is-a" if !is_subject => "CONCEPT",
        _ => "CONCEPT",
    }
}

/// Pre-process a single text chunk into structured output.
///
/// Uses the grammar parser with auto-detection or an explicit language hint.
pub fn preprocess_chunk(chunk: &TextChunk, ctx: &ParseContext) -> PreProcessorOutput {
    let lang_hint = chunk
        .language
        .as_deref()
        .and_then(Language::from_code);

    let detection = detect_language(&chunk.text);
    let effective_lang = lang_hint.unwrap_or(detection.language);
    let lang_code = effective_lang.bcp47().to_string();

    // Build a parse context with the detected language
    let parse_ctx = ParseContext {
        registry: ctx.registry,
        ops: ctx.ops,
        item_memory: ctx.item_memory,
        lexicon: None,
        language: effective_lang,
    };

    let result = parse_prose(&chunk.text, &parse_ctx);

    let mut entities = Vec::new();
    let mut claims = Vec::new();
    let mut abs_trees = Vec::new();

    match result {
        ParseResult::Facts(facts) => {
            for fact in &facts {
                extract_from_tree(fact, &chunk.text, &lang_code, &mut entities, &mut claims);
                abs_trees.push(fact.clone());
            }
        }
        ParseResult::Freeform { partial, text } => {
            if partial.is_empty() {
                // Try sentence splitting
                for sentence in text.split('.').map(str::trim).filter(|s| !s.is_empty()) {
                    let sub_result = parse_prose(sentence, &parse_ctx);
                    if let ParseResult::Facts(facts) = sub_result {
                        for fact in &facts {
                            extract_from_tree(fact, sentence, &lang_code, &mut entities, &mut claims);
                            abs_trees.push(fact.clone());
                        }
                    }
                }
            } else {
                for tree in &partial {
                    extract_from_tree(tree, &chunk.text, &lang_code, &mut entities, &mut claims);
                    abs_trees.push(tree.clone());
                }
            }
        }
        _ => {}
    }

    // Run entity resolution (cross-lingual alias matching + dedup)
    let resolver = EntityResolver::new();
    resolver.resolve_entities(&mut entities);

    PreProcessorOutput {
        chunk_id: chunk.id.clone(),
        source_language: lang_code,
        detected_language_confidence: detection.confidence,
        entities,
        claims,
        abs_trees,
    }
}

/// Pre-process a mixed-language corpus (splits by sentence, detects each).
pub fn preprocess_mixed_corpus(text: &str, ctx: &ParseContext) -> Vec<PreProcessorOutput> {
    detect_per_sentence(text)
        .into_iter()
        .map(|(sentence, detection)| {
            let chunk = TextChunk {
                id: None,
                text: sentence,
                language: Some(detection.language.bcp47().to_string()),
            };
            preprocess_chunk(&chunk, ctx)
        })
        .collect()
}

/// Pre-process a batch of chunks.
pub fn preprocess_batch(chunks: &[TextChunk], ctx: &ParseContext) -> Vec<PreProcessorOutput> {
    chunks.iter().map(|chunk| preprocess_chunk(chunk, ctx)).collect()
}

/// Extract entities and claims from an AbsTree node.
fn extract_from_tree(
    tree: &AbsTree,
    claim_text: &str,
    lang_code: &str,
    entities: &mut Vec<ExtractedEntity>,
    claims: &mut Vec<ExtractedClaim>,
) {
    match tree {
        AbsTree::Triple {
            subject,
            predicate,
            object,
        } => {
            let subj_label = subject.label().unwrap_or("?").to_string();
            let pred_label = predicate.label().unwrap_or("?").to_string();
            let obj_label = object.label().unwrap_or("?").to_string();

            let claim_type = predicate_to_claim_type(&pred_label);

            entities.push(ExtractedEntity {
                name: subj_label.clone(),
                entity_type: infer_entity_type(&pred_label, true).to_string(),
                canonical_name: subj_label.clone(),
                confidence: 0.90,
                aliases: vec![],
                source_language: lang_code.to_string(),
            });

            entities.push(ExtractedEntity {
                name: obj_label.clone(),
                entity_type: infer_entity_type(&pred_label, false).to_string(),
                canonical_name: obj_label.clone(),
                confidence: 0.90,
                aliases: vec![],
                source_language: lang_code.to_string(),
            });

            claims.push(ExtractedClaim {
                claim_text: claim_text.to_string(),
                claim_type: claim_type.to_string(),
                confidence: 0.90,
                subject: subj_label,
                predicate: pred_label,
                object: obj_label,
                source_language: lang_code.to_string(),
            });
        }
        AbsTree::WithConfidence { inner, confidence } => {
            // Recurse but with adjusted confidence
            extract_from_tree(inner, claim_text, lang_code, entities, claims);
            // Adjust the last claim's confidence if we just added one
            if let Some(last_claim) = claims.last_mut() {
                last_claim.confidence = *confidence;
            }
            // Adjust entity confidences
            let entity_count = entities.len();
            if entity_count >= 2 {
                entities[entity_count - 1].confidence = *confidence;
                entities[entity_count - 2].confidence = *confidence;
            }
        }
        AbsTree::Conjunction { items, .. } => {
            for item in items {
                extract_from_tree(item, claim_text, lang_code, entities, claims);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_output_roundtrips_json() {
        let output = PreProcessorOutput {
            chunk_id: Some("test-1".to_string()),
            source_language: "en".to_string(),
            detected_language_confidence: 0.85,
            entities: vec![ExtractedEntity {
                name: "Dog".to_string(),
                entity_type: "CONCEPT".to_string(),
                canonical_name: "Dog".to_string(),
                confidence: 0.90,
                aliases: vec![],
                source_language: "en".to_string(),
            }],
            claims: vec![ExtractedClaim {
                claim_text: "Dogs are mammals".to_string(),
                claim_type: "FACTUAL".to_string(),
                confidence: 0.90,
                subject: "Dogs".to_string(),
                predicate: "is-a".to_string(),
                object: "mammals".to_string(),
                source_language: "en".to_string(),
            }],
            abs_trees: vec![],
        };

        let json = serde_json::to_string(&output).unwrap();
        let roundtrip: PreProcessorOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.chunk_id.as_deref(), Some("test-1"));
        assert_eq!(roundtrip.entities.len(), 1);
        assert_eq!(roundtrip.claims.len(), 1);
    }

    #[test]
    fn preprocess_english_text() {
        let chunk = TextChunk {
            id: Some("1".to_string()),
            text: "Dogs are mammals".to_string(),
            language: None,
        };
        let ctx = ParseContext::default();
        let output = preprocess_chunk(&chunk, &ctx);

        assert!(!output.claims.is_empty(), "should extract at least one claim");
        assert_eq!(output.source_language, "en");
        assert!(!output.entities.is_empty(), "should extract entities");
    }

    #[test]
    fn preprocess_russian_text() {
        let chunk = TextChunk {
            id: Some("2".to_string()),
            text: "Собака является млекопитающим".to_string(),
            language: None,
        };
        let ctx = ParseContext::default();
        let output = preprocess_chunk(&chunk, &ctx);

        assert_eq!(output.source_language, "ru");
        assert!(!output.claims.is_empty(), "should extract claim from Russian text");
        if let Some(claim) = output.claims.first() {
            assert_eq!(claim.predicate, "is-a");
            assert_eq!(claim.source_language, "ru");
        }
    }

    #[test]
    fn preprocess_french_text() {
        let chunk = TextChunk {
            id: None,
            text: "Le chien est un mammifère".to_string(),
            language: Some("fr".to_string()),
        };
        let ctx = ParseContext::default();
        let output = preprocess_chunk(&chunk, &ctx);

        assert_eq!(output.source_language, "fr");
        if let Some(claim) = output.claims.first() {
            assert_eq!(claim.predicate, "is-a");
        }
    }

    #[test]
    fn predicate_to_claim_type_mapping() {
        assert_eq!(predicate_to_claim_type("is-a"), "FACTUAL");
        assert_eq!(predicate_to_claim_type("has-a"), "FACTUAL");
        assert_eq!(predicate_to_claim_type("contains"), "FACTUAL");
        assert_eq!(predicate_to_claim_type("causes"), "CAUSAL");
        assert_eq!(predicate_to_claim_type("located-in"), "SPATIAL");
        assert_eq!(predicate_to_claim_type("similar-to"), "RELATIONAL");
        assert_eq!(predicate_to_claim_type("part-of"), "STRUCTURAL");
        assert_eq!(predicate_to_claim_type("composed-of"), "STRUCTURAL");
        assert_eq!(predicate_to_claim_type("depends-on"), "DEPENDENCY");
        assert_eq!(predicate_to_claim_type("unknown"), "OTHER");
    }

    #[test]
    fn preprocess_batch_multiple_chunks() {
        let chunks = vec![
            TextChunk {
                id: Some("1".to_string()),
                text: "Dogs are mammals".to_string(),
                language: None,
            },
            TextChunk {
                id: Some("2".to_string()),
                text: "Собака является млекопитающим".to_string(),
                language: None,
            },
        ];
        let ctx = ParseContext::default();
        let results = preprocess_batch(&chunks, &ctx);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].source_language, "en");
        assert_eq!(results[1].source_language, "ru");
    }
}
