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
pub mod decomposition;
pub mod drives;
pub mod error;
pub mod goal;
pub mod goal_generation;
pub mod idle;
pub mod library_learn;
pub mod memory;
pub mod nlp;
pub mod ooda;
pub mod plan;
pub mod priority_reasoning;
pub mod project;
pub mod reflect;
pub mod semantic_enrichment;
pub mod synthesize;
pub mod synthesize_abs;
pub mod tool;
pub mod tool_manifest;
pub mod tool_semantics;
pub mod tools;
pub mod trigger;
pub mod watch;
#[cfg(feature = "wasm-tools")]
pub mod wasm_runtime;

pub use agent::{Agent, AgentConfig, AgentPredicates};
pub use chat::{Conversation, Participant, ParticipantSource, discover_ssh_fingerprint};
#[cfg(feature = "daemon")]
pub use daemon::{AgentDaemon, DaemonConfig};
pub use decomposition::{
    DecompositionMethod, DecompositionOutput, DecompositionStrategy, DependencyEdge, MethodRegistry,
    TaskNode, TaskNodeKind, TaskTree,
};
pub use drives::{DriveKind, DriveSystem};
pub use error::{AgentError, AgentResult};
pub use goal::{Goal, GoalSource, GoalStatus};
pub use goal_generation::{GoalGenerationConfig, GoalGenerationResult, GoalProposal};
pub use idle::{IdleScheduler, IdleTaskResult};
pub use library_learn::{LibraryLearner, LibraryLearningResult};
pub use memory::{
    ConsolidationConfig, ConsolidationResult, EpisodicEntry, SessionSummary, WorkingMemory,
    WorkingMemoryEntry, WorkingMemoryKind,
};
pub use nlp::{QuestionWord, UserIntent, classify_intent};
pub use ooda::{
    ActionResult, Decision, DecisionImpasse, GoalProgress, ImpasseKind, Observation,
    OodaCycleResult, Orientation,
};
pub use plan::{Plan, PlanStatus, PlanStep, StepStatus};
pub use priority_reasoning::{Audience, PriorityArgument, PriorityVerdict, Value};
pub use project::{Agenda, Project, ProjectAssignment, ProjectPredicates, ProjectStatus};
pub use reflect::{Adjustment, ReflectionConfig, ReflectionResult};
pub use semantic_enrichment::{EnrichmentResult, SemanticPredicates};
pub use synthesize::NarrativeSummary;
pub use tool::{Tool, ToolInput, ToolOutput, ToolRegistry, ToolSignature};
pub use tool_manifest::{Capability, DangerInfo, DangerLevel, ToolManifest, ToolSource};
pub use trigger::{Trigger, TriggerAction, TriggerCondition, TriggerStore};
pub use watch::{
    Discrepancy, Expectation, TriplePattern, Watch, WatchAction, WatchCondition, WatchFiring,
    WorldSnapshot,
};
