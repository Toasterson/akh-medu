//! Training data collection and LoRA fine-tuning infrastructure.
//!
//! This module provides:
//!
//! - [`TrainingDataCollector`] — logs every successful NLU parse, grammar
//!   linearization, and knowledge extraction as training pairs in redb
//! - [`TripleVerbalizer`] — converts KG triples into diverse natural language
//!   training pairs using the grammar framework
//! - [`LoraRegistry`] — manages per-agent LoRA adapters with < 100 ms hot-swap
//! - [`QaGenerator`] — generates seed QA pairs for each Ennead agent persona
//! - [`AdapterBenchmark`] — evaluates adapter quality against agent-specific eval sets

pub mod benchmark;
pub mod data_collector;
pub mod lora;
pub mod qa_generator;
pub mod trainer;
pub mod verbalizer;
pub mod verifier;

pub use benchmark::{AdapterBenchmark, AgentBenchmarkResult, BenchmarkConfig, BenchmarkError};
pub use data_collector::{
    CollectorConfig, CollectorError, CollectorResult, TrainingDataCollector, TrainingPair,
    TrainingPairKind,
};
pub use lora::{AgentId, LoraAdapterMeta, LoraConfig, LoraError, LoraRegistry, LoadedAdapter};
pub use qa_generator::{AgentPersonaProfile, QaCategory, QaGenerator, QaPair};
pub use trainer::{BridgeTrainingSession, TrainerConfig, TrainingResult};
pub use verbalizer::TripleVerbalizer;
pub use verifier::{TrainingVerifier, VerificationResult};
