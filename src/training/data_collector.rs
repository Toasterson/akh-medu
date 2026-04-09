//! Training data collection pipeline.
//!
//! Logs every successful NLU parse, grammar linearization, and knowledge
//! extraction as training pairs. Storage is in redb with category-keyed tables.
//! No extra dependencies — uses redb (already in deps) and serde_json.
//!
//! The collector runs from day one but does not trigger training until
//! sufficient data accumulates (cold-start thresholds).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::skills::LabelTriple;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for training data collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// Minimum NLU pairs before training can start (default: 500).
    pub nlu_threshold: usize,
    /// Minimum NLG pairs before NLG training can start (default: 200).
    pub nlg_threshold: usize,
    /// Maximum pairs to store per category before pruning oldest (default: 50_000).
    pub max_pairs_per_category: usize,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            nlu_threshold: 500,
            nlg_threshold: 200,
            max_pairs_per_category: 50_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Training pair types
// ---------------------------------------------------------------------------

/// Category of training pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrainingPairKind {
    /// NLU: natural text → AbsTree JSON.
    Nlu,
    /// NLG: triples → natural text.
    Nlg,
    /// Elicitation: prompt → extracted triples (positive or negative).
    Elicitation,
}

impl TrainingPairKind {
    fn table_prefix(&self) -> &'static str {
        match self {
            Self::Nlu => "nlu",
            Self::Nlg => "nlg",
            Self::Elicitation => "elic",
        }
    }
}

/// A single training pair stored in the collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPair {
    /// Category.
    pub kind: TrainingPairKind,
    /// Input text (NLU: user input; NLG: triple descriptions; Elicitation: prompt).
    pub input: String,
    /// Output text (NLU: AbsTree JSON; NLG: natural text; Elicitation: triple JSON).
    pub output: String,
    /// Confidence from the source tier.
    pub confidence: f32,
    /// Which NLU tier or grammar produced this pair.
    pub source: String,
    /// Unix timestamp of when this pair was collected.
    pub timestamp: u64,
    /// Whether this is a positive (true) or negative (false) example.
    pub positive: bool,
}

// ---------------------------------------------------------------------------
// Redb table
// ---------------------------------------------------------------------------

const TRAINING_TABLE: redb::TableDefinition<u64, &[u8]> =
    redb::TableDefinition::new("training_pairs");

const COUNTER_TABLE: redb::TableDefinition<&str, u64> =
    redb::TableDefinition::new("training_counters");

// ---------------------------------------------------------------------------
// TrainingDataCollector
// ---------------------------------------------------------------------------

/// Collects and stores training pairs for future fine-tuning.
///
/// Each successful NLU parse, grammar linearization, or knowledge extraction
/// is logged as a training pair. The collector checks thresholds to determine
/// when enough data has accumulated to start training.
pub struct TrainingDataCollector {
    db: redb::Database,
    config: CollectorConfig,
    /// Monotonically increasing pair ID.
    next_id: u64,
}

impl TrainingDataCollector {
    /// Open or create a training data store at the given path.
    pub fn open(path: &Path, config: CollectorConfig) -> Result<Self, redb::DatabaseError> {
        let db = redb::Database::create(path)?;

        // Initialize tables.
        {
            let txn = db.begin_write()?;
            {
                let _table = txn.open_table(TRAINING_TABLE)?;
                let _counters = txn.open_table(COUNTER_TABLE)?;
            }
            txn.commit()?;
        }

        // Read current counter state.
        let next_id = {
            let txn = db.begin_read()?;
            let table = txn.open_table(TRAINING_TABLE)?;
            table.len()?
        };

        Ok(Self {
            db,
            config,
            next_id,
        })
    }

    /// Record a training pair.
    pub fn record(&mut self, pair: TrainingPair) -> Result<(), redb::DatabaseError> {
        let id = self.next_id;
        self.next_id += 1;

        let bytes = serde_json::to_vec(&pair).unwrap_or_default();
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(TRAINING_TABLE)?;
            table.insert(id, bytes.as_slice())?;

            // Update counter for this category.
            let mut counters = txn.open_table(COUNTER_TABLE)?;
            let key = pair.kind.table_prefix();
            let current = counters.get(key)?.map(|v| v.value()).unwrap_or(0);
            counters.insert(key, current + 1)?;
        }
        txn.commit()?;

        Ok(())
    }

    /// Record an NLU training pair from a successful parse.
    pub fn record_nlu(
        &mut self,
        input: &str,
        abstree_json: &str,
        source_tier: u8,
        confidence: f32,
    ) -> Result<(), redb::DatabaseError> {
        self.record(TrainingPair {
            kind: TrainingPairKind::Nlu,
            input: input.to_string(),
            output: abstree_json.to_string(),
            confidence,
            source: format!("tier{source_tier}"),
            timestamp: now_secs(),
            positive: true,
        })
    }

    /// Record an NLG training pair from a grammar linearization.
    pub fn record_nlg(
        &mut self,
        triples_desc: &str,
        output_text: &str,
        grammar: &str,
        confidence: f32,
    ) -> Result<(), redb::DatabaseError> {
        self.record(TrainingPair {
            kind: TrainingPairKind::Nlg,
            input: triples_desc.to_string(),
            output: output_text.to_string(),
            confidence,
            source: grammar.to_string(),
            timestamp: now_secs(),
            positive: true,
        })
    }

    /// Record an elicitation training pair (positive or negative).
    pub fn record_elicitation(
        &mut self,
        prompt: &str,
        triples_json: &str,
        accepted: bool,
    ) -> Result<(), redb::DatabaseError> {
        self.record(TrainingPair {
            kind: TrainingPairKind::Elicitation,
            input: prompt.to_string(),
            output: triples_json.to_string(),
            confidence: if accepted { 1.0 } else { 0.0 },
            source: "elicitation".to_string(),
            timestamp: now_secs(),
            positive: accepted,
        })
    }

    /// Count of pairs per category.
    pub fn counts(&self) -> Result<TrainingCounts, redb::DatabaseError> {
        let txn = self.db.begin_read()?;
        let counters = txn.open_table(COUNTER_TABLE)?;

        let nlu = counters.get("nlu")?.map(|v| v.value()).unwrap_or(0) as usize;
        let nlg = counters.get("nlg")?.map(|v| v.value()).unwrap_or(0) as usize;
        let elicitation = counters
            .get("elic")?
            .map(|v| v.value())
            .unwrap_or(0) as usize;

        Ok(TrainingCounts {
            nlu,
            nlg,
            elicitation,
        })
    }

    /// Whether enough NLU data has accumulated to start NLU training.
    pub fn nlu_ready(&self) -> Result<bool, redb::DatabaseError> {
        let counts = self.counts()?;
        Ok(counts.nlu >= self.config.nlu_threshold)
    }

    /// Whether enough NLG data has accumulated to start NLG training.
    pub fn nlg_ready(&self) -> Result<bool, redb::DatabaseError> {
        let counts = self.counts()?;
        Ok(counts.nlg >= self.config.nlg_threshold)
    }

    /// Iterate all pairs of a given kind (for training batch construction).
    pub fn pairs_of_kind(
        &self,
        kind: TrainingPairKind,
    ) -> Result<Vec<TrainingPair>, redb::DatabaseError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(TRAINING_TABLE)?;
        let mut pairs = Vec::new();

        for entry in table.iter()? {
            let (_, value) = entry?;
            if let Ok(pair) = serde_json::from_slice::<TrainingPair>(value.value()) {
                if pair.kind == kind {
                    pairs.push(pair);
                }
            }
        }

        Ok(pairs)
    }

    /// Total number of stored pairs.
    pub fn total_pairs(&self) -> Result<u64, redb::DatabaseError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(TRAINING_TABLE)?;
        table.len()
    }

    /// Access the configuration.
    pub fn config(&self) -> &CollectorConfig {
        &self.config
    }
}

impl std::fmt::Debug for TrainingDataCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingDataCollector")
            .field("config", &self.config)
            .field("next_id", &self.next_id)
            .finish_non_exhaustive()
    }
}

/// Counts of training pairs per category.
#[derive(Debug, Clone)]
pub struct TrainingCounts {
    pub nlu: usize,
    pub nlg: usize,
    pub elicitation: usize,
}

impl std::fmt::Display for TrainingCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NLU: {}, NLG: {}, Elicitation: {}",
            self.nlu, self.nlg, self.elicitation
        )
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_collector() -> TrainingDataCollector {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("training.redb");
        let collector =
            TrainingDataCollector::open(&path, CollectorConfig::default()).unwrap();
        // Leak the dir so it doesn't get cleaned up while we use the DB.
        std::mem::forget(dir);
        collector
    }

    #[test]
    fn record_and_count_nlu() {
        let mut collector = temp_collector();
        collector
            .record_nlu("dogs are mammals", r#"{"Triple":{}}"#, 1, 0.85)
            .unwrap();
        collector
            .record_nlu("cats are animals", r#"{"Triple":{}}"#, 1, 0.80)
            .unwrap();

        let counts = collector.counts().unwrap();
        assert_eq!(counts.nlu, 2);
        assert_eq!(counts.nlg, 0);
    }

    #[test]
    fn record_nlg_and_elicitation() {
        let mut collector = temp_collector();
        collector
            .record_nlg("dog is-a mammal", "Dogs are mammals.", "narrative", 0.9)
            .unwrap();
        collector
            .record_elicitation("What do you know about dogs?", "[]", true)
            .unwrap();
        collector
            .record_elicitation("Bad prompt", "[]", false)
            .unwrap();

        let counts = collector.counts().unwrap();
        assert_eq!(counts.nlg, 1);
        assert_eq!(counts.elicitation, 2);
    }

    #[test]
    fn threshold_check() {
        let config = CollectorConfig {
            nlu_threshold: 3,
            nlg_threshold: 2,
            max_pairs_per_category: 100,
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("training.redb");
        let mut collector = TrainingDataCollector::open(&path, config).unwrap();

        assert!(!collector.nlu_ready().unwrap());
        for i in 0..3 {
            collector
                .record_nlu(&format!("input {i}"), "{}", 1, 0.8)
                .unwrap();
        }
        assert!(collector.nlu_ready().unwrap());
        assert!(!collector.nlg_ready().unwrap());
    }

    #[test]
    fn pairs_of_kind_filters() {
        let mut collector = temp_collector();
        collector
            .record_nlu("hello", "{}", 1, 0.8)
            .unwrap();
        collector
            .record_nlg("triple", "text", "narrative", 0.9)
            .unwrap();

        let nlu_pairs = collector.pairs_of_kind(TrainingPairKind::Nlu).unwrap();
        assert_eq!(nlu_pairs.len(), 1);
        assert_eq!(nlu_pairs[0].input, "hello");

        let nlg_pairs = collector.pairs_of_kind(TrainingPairKind::Nlg).unwrap();
        assert_eq!(nlg_pairs.len(), 1);
    }

    #[test]
    fn total_pairs() {
        let mut collector = temp_collector();
        assert_eq!(collector.total_pairs().unwrap(), 0);
        collector.record_nlu("a", "{}", 1, 0.8).unwrap();
        collector.record_nlg("b", "c", "x", 0.9).unwrap();
        assert_eq!(collector.total_pairs().unwrap(), 2);
    }
}
