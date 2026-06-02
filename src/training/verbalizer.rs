//! Triple verbalization pipeline for training data generation.
//!
//! Converts KG triples into diverse natural language training pairs using
//! the grammar framework's linearization. Generates multiple surface forms
//! per triple (template variation) and negative samples (corrupted triples)
//! for contrastive training.

use crate::grammar::abs::AbsTree;
use crate::skills::LabelTriple;

// ---------------------------------------------------------------------------
// TripleVerbalizer
// ---------------------------------------------------------------------------

/// Converts KG triples into natural language training pairs.
///
/// Uses template-based variation to generate diverse surface forms from
/// a single triple, maximizing training data from limited KG content.
pub struct TripleVerbalizer {
    /// Whether to generate negative (corrupted) samples.
    pub generate_negatives: bool,
    /// Maximum template variations per triple.
    pub max_variations: usize,
}

impl Default for TripleVerbalizer {
    fn default() -> Self {
        Self {
            generate_negatives: true,
            max_variations: 4,
        }
    }
}

/// A verbalized training pair.
#[derive(Debug, Clone)]
pub struct VerbalizedPair {
    /// Natural language text.
    pub text: String,
    /// AbsTree JSON for this verbalization.
    pub abstree_json: String,
    /// Whether this is a positive (true) or corrupted negative (false) example.
    pub positive: bool,
}

impl TripleVerbalizer {
    /// Verbalize a single triple into multiple training pairs.
    ///
    /// Produces template variations and optionally negative samples.
    pub fn verbalize(&self, triple: &LabelTriple) -> Vec<VerbalizedPair> {
        let mut pairs = Vec::new();

        let abstree = AbsTree::triple(
            AbsTree::entity(&triple.subject),
            AbsTree::relation(&triple.predicate),
            AbsTree::entity(&triple.object),
        );
        let abstree_json = serde_json::to_string(&abstree).unwrap_or_default();

        // Generate template variations.
        let templates = self.template_variations(
            &triple.subject,
            &triple.predicate,
            &triple.object,
        );

        for (i, text) in templates.into_iter().enumerate() {
            if i >= self.max_variations {
                break;
            }
            pairs.push(VerbalizedPair {
                text,
                abstree_json: abstree_json.clone(),
                positive: true,
            });
        }

        // Generate negative samples by corruption.
        if self.generate_negatives {
            let negatives = self.corrupt_triple(triple);
            for neg in negatives {
                let neg_tree = AbsTree::triple(
                    AbsTree::entity(&neg.subject),
                    AbsTree::relation(&neg.predicate),
                    AbsTree::entity(&neg.object),
                );
                let neg_json = serde_json::to_string(&neg_tree).unwrap_or_default();
                let text = format!("{} {} {}", neg.subject, neg.predicate, neg.object);
                pairs.push(VerbalizedPair {
                    text,
                    abstree_json: neg_json,
                    positive: false,
                });
            }
        }

        pairs
    }

    /// Verbalize a batch of triples.
    pub fn verbalize_batch(&self, triples: &[LabelTriple]) -> Vec<VerbalizedPair> {
        triples.iter().flat_map(|t| self.verbalize(t)).collect()
    }

    /// Generate template variations for a subject-predicate-object triple.
    fn template_variations(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Vec<String> {
        let pred_lower = predicate.to_lowercase();
        let mut variations = Vec::with_capacity(self.max_variations);

        // Variation 1: Direct statement
        variations.push(format!("{subject} {predicate} {object}"));

        // Variation 2: Predicate-specific templates
        match pred_lower.as_str() {
            "is-a" | "is a" | "isa" => {
                variations.push(format!("{subject} is a type of {object}"));
                variations.push(format!("{subject} is an instance of {object}"));
                variations.push(format!("{subject} belongs to the category {object}"));
            }
            "has" | "has-a" | "has a" => {
                variations.push(format!("{subject} possesses {object}"));
                variations.push(format!("{object} is a property of {subject}"));
                variations.push(format!("{subject} is characterized by {object}"));
            }
            "part-of" | "part of" => {
                variations.push(format!("{subject} is a component of {object}"));
                variations.push(format!("{object} contains {subject}"));
                variations.push(format!("{subject} belongs to {object}"));
            }
            "causes" | "leads-to" => {
                variations.push(format!("{subject} leads to {object}"));
                variations.push(format!("{object} is caused by {subject}"));
                variations.push(format!("{subject} results in {object}"));
            }
            "used-for" | "used for" => {
                variations.push(format!("{subject} is used for {object}"));
                variations.push(format!("{object} requires {subject}"));
                variations.push(format!("You can use {subject} to {object}"));
            }
            _ => {
                // Generic variations for unknown predicates
                variations.push(format!("The {predicate} of {subject} is {object}"));
                variations.push(format!("{subject}'s {predicate} is {object}"));
            }
        }

        variations
    }

    /// Generate corrupted (negative) versions of a triple.
    ///
    /// Corruption strategies:
    /// 1. Swap subject and object
    /// 2. Replace predicate with a generic wrong one
    fn corrupt_triple(&self, triple: &LabelTriple) -> Vec<LabelTriple> {
        let mut negatives = Vec::new();

        // Strategy 1: Swap subject ↔ object
        negatives.push(LabelTriple {
            subject: triple.object.clone(),
            predicate: triple.predicate.clone(),
            object: triple.subject.clone(),
            confidence: 0.0,
        });

        // Strategy 2: Wrong predicate
        let wrong_pred = match triple.predicate.to_lowercase().as_str() {
            "is-a" => "part-of",
            "has" => "is-a",
            "part-of" => "causes",
            _ => "unrelated-to",
        };
        negatives.push(LabelTriple {
            subject: triple.subject.clone(),
            predicate: wrong_pred.to_string(),
            object: triple.object.clone(),
            confidence: 0.0,
        });

        negatives
    }

    /// Summary statistics for a verbalization run.
    pub fn stats(&self, triples: &[LabelTriple]) -> VerbalizationStats {
        let pairs = self.verbalize_batch(triples);
        let positive = pairs.iter().filter(|p| p.positive).count();
        let negative = pairs.len() - positive;
        VerbalizationStats {
            input_triples: triples.len(),
            positive_pairs: positive,
            negative_pairs: negative,
            total_pairs: pairs.len(),
        }
    }
}

/// Summary of a verbalization run.
#[derive(Debug, Clone)]
pub struct VerbalizationStats {
    pub input_triples: usize,
    pub positive_pairs: usize,
    pub negative_pairs: usize,
    pub total_pairs: usize,
}

impl std::fmt::Display for VerbalizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} triples → {} pairs ({} positive, {} negative)",
            self.input_triples, self.total_pairs, self.positive_pairs, self.negative_pairs
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triple(s: &str, p: &str, o: &str) -> LabelTriple {
        LabelTriple {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            confidence: 1.0,
        }
    }

    #[test]
    fn verbalize_isa_triple() {
        let verbalizer = TripleVerbalizer::default();
        let triple = sample_triple("dog", "is-a", "mammal");
        let pairs = verbalizer.verbalize(&triple);

        // Should have positive + negative pairs.
        assert!(!pairs.is_empty());
        let positives: Vec<_> = pairs.iter().filter(|p| p.positive).collect();
        let negatives: Vec<_> = pairs.iter().filter(|p| !p.positive).collect();
        assert!(!positives.is_empty());
        assert!(!negatives.is_empty());

        // Check template variation content.
        assert!(positives.iter().any(|p| p.text.contains("type of")));
        assert!(positives.iter().any(|p| p.text == "dog is-a mammal"));
    }

    #[test]
    fn verbalize_generic_predicate() {
        let verbalizer = TripleVerbalizer::default();
        let triple = sample_triple("rust", "compiles-to", "machine code");
        let pairs = verbalizer.verbalize(&triple);
        let positives: Vec<_> = pairs.iter().filter(|p| p.positive).collect();
        assert!(positives.len() >= 2); // Direct + generic templates
    }

    #[test]
    fn negative_samples_swap_subject_object() {
        let verbalizer = TripleVerbalizer::default();
        let triple = sample_triple("dog", "is-a", "mammal");
        let pairs = verbalizer.verbalize(&triple);
        let negatives: Vec<_> = pairs.iter().filter(|p| !p.positive).collect();

        // One negative should have swapped subject/object.
        assert!(negatives.iter().any(|n| n.text.starts_with("mammal")));
    }

    #[test]
    fn no_negatives_when_disabled() {
        let verbalizer = TripleVerbalizer {
            generate_negatives: false,
            ..Default::default()
        };
        let triple = sample_triple("a", "b", "c");
        let pairs = verbalizer.verbalize(&triple);
        assert!(pairs.iter().all(|p| p.positive));
    }

    #[test]
    fn batch_verbalization() {
        let verbalizer = TripleVerbalizer::default();
        let triples = vec![
            sample_triple("dog", "is-a", "mammal"),
            sample_triple("cat", "has", "whiskers"),
        ];
        let pairs = verbalizer.verbalize_batch(&triples);
        assert!(pairs.len() > 4); // Multiple variations per triple + negatives
    }

    #[test]
    fn stats_report() {
        let verbalizer = TripleVerbalizer::default();
        let triples = vec![sample_triple("x", "is-a", "y")];
        let stats = verbalizer.stats(&triples);
        assert_eq!(stats.input_triples, 1);
        assert!(stats.positive_pairs > 0);
        assert!(stats.negative_pairs > 0);
        assert_eq!(stats.total_pairs, stats.positive_pairs + stats.negative_pairs);
    }

    #[test]
    fn abstree_json_is_valid() {
        let verbalizer = TripleVerbalizer::default();
        let triple = sample_triple("sun", "is-a", "star");
        let pairs = verbalizer.verbalize(&triple);
        for pair in &pairs {
            let parsed: Result<AbsTree, _> = serde_json::from_str(&pair.abstree_json);
            assert!(parsed.is_ok(), "invalid AbsTree JSON: {}", pair.abstree_json);
        }
    }
}
