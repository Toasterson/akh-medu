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
pub mod error;
pub mod goal;
pub mod llm;
pub mod memory;
pub mod nlp;
pub mod ooda;
pub mod plan;
pub mod reflect;
pub mod tool;
pub mod tool_semantics;
pub mod tools;

pub use agent::{Agent, AgentConfig, AgentPredicates};
pub use chat::Conversation;
pub use error::{AgentError, AgentResult};
pub use goal::{Goal, GoalStatus};
pub use llm::{LlmError, OllamaClient, OllamaConfig};
pub use memory::{
    ConsolidationConfig, ConsolidationResult, EpisodicEntry, WorkingMemory, WorkingMemoryEntry,
    WorkingMemoryKind,
};
pub use nlp::{classify_intent, UserIntent};
pub use ooda::{
    ActionResult, Decision, GoalProgress, Observation, OodaCycleResult, Orientation,
};
pub use plan::{Plan, PlanStatus, PlanStep, StepStatus};
pub use reflect::{Adjustment, ReflectionConfig, ReflectionResult};
pub use tool::{Tool, ToolInput, ToolOutput, ToolRegistry, ToolSignature};
