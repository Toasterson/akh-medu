//! Training data collection and live fine-tuning infrastructure.
//!
//! This module provides:
//!
//! - [`TrainingDataCollector`] — logs every successful NLU parse, grammar
//!   linearization, and knowledge extraction as training pairs in redb
//! - [`TripleVerbalizer`] — converts KG triples into diverse natural language
//!   training pairs using the grammar framework
//! - (Future) Burn training loop for LoRA fine-tuning

pub mod data_collector;
pub mod trainer;
pub mod verbalizer;
pub mod verifier;

pub use data_collector::{CollectorConfig, TrainingDataCollector, TrainingPair, TrainingPairKind};
pub use trainer::{BridgeTrainingSession, TrainerConfig, TrainingResult};
pub use verbalizer::TripleVerbalizer;
pub use verifier::{TrainingVerifier, VerificationResult};
