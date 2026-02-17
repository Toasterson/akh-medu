//! Agent layer: autonomous reasoning with deliberate memory, goals, tools, and OODA loop.
//!
//! The agent wraps an `Arc<Engine>` and adds:
//! - **Working memory** (ephemeral per-session scratch)
//! - **Episodic memory** (persistent long-term, stored in the knowledge graph)
//! - **Goals** (represented as KG entities with well-known predicates)
//! - **Tools** (compile-time trait impls with runtime registration)
//! - **OODA loop** (Observe → Orient → Decide → Act cycle)
//! - **Consolidation** (deliberate reasoning about what to remember)

pub mod agent;
pub mod chat;
pub mod cli_tool;
#[cfg(feature = "daemon")]
pub mod daemon;
pub mod error;
pub mod goal;
pub mod idle;
pub mod memory;
pub mod nlp;
pub mod ooda;
pub mod plan;
pub mod reflect;
pub mod semantic_enrichment;
pub mod synthesize;
pub mod synthesize_abs;
pub mod tool;
pub mod tool_manifest;
pub mod tool_semantics;
pub mod tools;
pub mod trigger;
#[cfg(feature = "wasm-tools")]
pub mod wasm_runtime;

pub use agent::{Agent, AgentConfig, AgentPredicates};
pub use chat::{Conversation, Participant, ParticipantSource, discover_ssh_fingerprint};
#[cfg(feature = "daemon")]
pub use daemon::{AgentDaemon, DaemonConfig};
pub use error::{AgentError, AgentResult};
pub use goal::{Goal, GoalStatus};
pub use idle::{IdleScheduler, IdleTaskResult};
pub use memory::{
    ConsolidationConfig, ConsolidationResult, EpisodicEntry, WorkingMemory, WorkingMemoryEntry,
    WorkingMemoryKind,
};
pub use nlp::{QuestionWord, UserIntent, classify_intent};
pub use ooda::{ActionResult, Decision, GoalProgress, Observation, OodaCycleResult, Orientation};
pub use plan::{Plan, PlanStatus, PlanStep, StepStatus};
pub use reflect::{Adjustment, ReflectionConfig, ReflectionResult};
pub use semantic_enrichment::{EnrichmentResult, SemanticPredicates};
pub use synthesize::NarrativeSummary;
pub use tool::{Tool, ToolInput, ToolOutput, ToolRegistry, ToolSignature};
pub use tool_manifest::{Capability, DangerInfo, DangerLevel, ToolManifest, ToolSource};
pub use trigger::{Trigger, TriggerAction, TriggerCondition, TriggerStore};
