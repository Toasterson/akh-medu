//! Pre-processor output types and mapping functions for the Eleutherios integration.
//!
//! Provides serializable types that match Eleutherios's extraction pipeline schema:
//! entities, claims, and structured pre-processor output. The grammar parser produces
//! `AbsTree` structures which are mapped into these types for downstream consumption.

use serde::{Deserialize, Serialize};

use super::abs::AbsTree;
use super::concrete::ParseContext;
use super::detect::{detect_language, detect_per_sentence};
use super::entity_resolution::EntityResolver;
use super::lexer::Language;
use super::parser::{ParseResult, parse_prose};

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
/// Accepts an optional `EntityResolver` for cross-lingual resolution with
/// learned equivalences. If `None`, creates a default (empty) resolver.
pub fn preprocess_chunk(chunk: &TextChunk, ctx: &ParseContext) -> PreProcessorOutput {
    preprocess_chunk_with_resolver(chunk, ctx, &EntityResolver::new())
}

/// Pre-process a single text chunk with an explicit entity resolver.
pub fn preprocess_chunk_with_resolver(
    chunk: &TextChunk,
    ctx: &ParseContext,
    resolver: &EntityResolver,
) -> PreProcessorOutput {
    let lang_hint = chunk.language.as_deref().and_then(Language::from_code);

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
                            extract_from_tree(
                                fact,
                                sentence,
                                &lang_code,
                                &mut entities,
                                &mut claims,
                            );
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
    chunks
        .iter()
        .map(|chunk| preprocess_chunk(chunk, ctx))
        .collect()
}

/// Pre-process a batch of chunks with an explicit entity resolver.
pub fn preprocess_batch_with_resolver(
    chunks: &[TextChunk],
    ctx: &ParseContext,
    resolver: &EntityResolver,
) -> Vec<PreProcessorOutput> {
    chunks
        .iter()
        .map(|chunk| preprocess_chunk_with_resolver(chunk, ctx, resolver))
        .collect()
}

/// Pre-process a batch and then run co-occurrence learning on the results.
///
/// Returns the preprocessor outputs and the number of new equivalences
/// discovered from parallel chunk alignment.
pub fn preprocess_batch_with_learning(
    chunks: &[TextChunk],
    ctx: &ParseContext,
    resolver: &mut EntityResolver,
) -> (Vec<PreProcessorOutput>, usize) {
    let outputs = preprocess_batch_with_resolver(chunks, ctx, resolver);
    let discovered = resolver.learn_from_parallel_chunks(&outputs);
    (outputs, discovered)
}

/// Pre-process a single text chunk with library context enrichment.
///
/// Runs base preprocessing with the entity resolver, then searches the engine's
/// library (paragraphs ingested as `para:*` symbols) for similar content. Entities
/// found in matching library paragraphs boost confidence of extracted entities or
/// are added as supplementary entities.
pub fn preprocess_chunk_with_library(
    chunk: &TextChunk,
    ctx: &ParseContext,
    resolver: &EntityResolver,
    engine: &crate::engine::Engine,
) -> PreProcessorOutput {
    let mut output = preprocess_chunk_with_resolver(chunk, ctx, resolver);

    // Encode chunk text as a query vector
    let query_vec = match crate::vsa::grounding::encode_text_as_vector(
        &chunk.text,
        engine,
        engine.ops(),
        engine.item_memory(),
    ) {
        Ok(v) => v,
        Err(_) => return output,
    };

    // Search for similar library paragraphs
    let neighbors = match engine.search_similar(&query_vec, 15) {
        Ok(n) => n,
        Err(_) => return output,
    };

    // Filter to para:* symbols and take top 3
    let para_matches: Vec<_> = neighbors
        .iter()
        .filter(|n| {
            engine
                .get_symbol_meta(n.symbol_id)
                .is_ok_and(|m| m.label.starts_with("para:"))
        })
        .take(3)
        .collect();

    // Collect library entities from matching paragraphs
    let mut library_entities: Vec<(String, f32)> = Vec::new();
    for para in &para_matches {
        let triples = engine.triples_from(para.symbol_id);
        for triple in &triples {
            let obj_label = match engine.get_symbol_meta(triple.object) {
                Ok(m) => m.label,
                Err(_) => continue,
            };
            // Skip structural labels
            if obj_label.starts_with("para:")
                || obj_label.starts_with("ch:")
                || obj_label.starts_with("sec:")
                || obj_label.parse::<u64>().is_ok()
            {
                continue;
            }
            library_entities.push((obj_label, para.similarity));
        }
    }

    // Confidence boost: if library entity matches an extracted entity
    for entity in output.entities.iter_mut() {
        for (lib_label, _sim) in &library_entities {
            if entity.canonical_name.to_lowercase() == lib_label.to_lowercase() {
                entity.confidence = (entity.confidence + 0.05).min(1.0);
                break;
            }
        }
    }

    // Supplementary entities: library entities not in extracted set with high similarity
    let existing_names: std::collections::HashSet<String> = output
        .entities
        .iter()
        .map(|e| e.canonical_name.to_lowercase())
        .collect();

    for (lib_label, sim) in &library_entities {
        if *sim > 0.7 && !existing_names.contains(&lib_label.to_lowercase()) {
            output.entities.push(ExtractedEntity {
                name: lib_label.clone(),
                entity_type: "CONCEPT".to_string(),
                canonical_name: lib_label.clone(),
                confidence: 0.6,
                aliases: vec![],
                source_language: output.source_language.clone(),
            });
        }
    }

    output
}

/// Pre-process a batch of chunks with library context enrichment.
pub fn preprocess_batch_with_library(
    chunks: &[TextChunk],
    ctx: &ParseContext,
    resolver: &EntityResolver,
    engine: &crate::engine::Engine,
) -> Vec<PreProcessorOutput> {
    chunks
        .iter()
        .map(|chunk| preprocess_chunk_with_library(chunk, ctx, resolver, engine))
        .collect()
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

        assert!(
            !output.claims.is_empty(),
            "should extract at least one claim"
        );
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
        assert!(
            !output.claims.is_empty(),
            "should extract claim from Russian text"
        );
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
    fn preprocess_chunk_with_library_enriches_output() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        // Create an engine and ingest a fact so the KG has content
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        // Ingest prose to create KG content
        let _ = engine.ingest_prose("Dogs are mammals");

        let chunk = TextChunk {
            id: Some("test-lib".to_string()),
            text: "Dogs are mammals".to_string(),
            language: None,
        };

        let ctx = ParseContext::with_engine(engine.registry(), engine.ops(), engine.item_memory());
        let resolver = engine.entity_resolver();

        // Run the library-enriched preprocessor
        let output = preprocess_chunk_with_library(&chunk, &ctx, &resolver, &engine);

        // Should at minimum produce the same entities as the base preprocessor
        assert!(!output.entities.is_empty(), "should extract entities");
        assert_eq!(output.source_language, "en");
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
