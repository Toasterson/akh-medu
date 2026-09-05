//! Built-in tools for the agent: KG query, KG mutate, memory recall, reason,
//! similarity search, file I/O, HTTP fetch, shell exec,
//! user interaction, infer rules, gap analysis,
//! code ingest, doc gen, pattern mine, audit log,
//! and MCP client for remote tool access.

pub mod agent_management;
pub mod audit_log;
pub mod code_gen;
pub mod code_ingest;
pub mod code_predicates;
pub mod compile_feedback;
pub mod core_memory;
pub mod delegate;
pub mod diagnostics;
pub mod doc_gen;
pub mod edit;
pub mod entity_schema;
pub mod file_io;
pub mod gap_analysis;
pub mod glob;
pub mod grep;
pub mod http_fetch;
pub mod infer_rules;
pub mod inference_config;
pub mod kg_delete;
pub mod kg_mutate;
pub mod kg_query;
pub mod mcp;
pub mod memory_recall;
pub mod memory_write;
pub mod openclaw_discover;
pub mod pattern_mine;
pub mod reason;
pub mod shell_exec;
pub mod similarity_search;
pub mod skill_author;
pub mod structured_task;
pub mod study_note;
pub mod task_list;
pub mod tool_query;
pub mod user_interact;
pub mod web_search;
// Dispatch-trap triage reads `[upstreams.anima]` via `crate::config::AkhomedConfig`
// and calls out through `crate::upstream_client::UpstreamMcpClient`, both of
// which are themselves `#[cfg(feature = "daemon")]` in `src/lib.rs`. Gate the
// whole tool the same way rather than sprinkling cfgs inside the file.
#[cfg(feature = "daemon")]
pub mod reactor_manage;
#[cfg(feature = "daemon")]
pub mod triage_dispatch_traps;
pub mod trigger_manage;

pub use agent_management::{AgentListTool, AgentMessageTool, AgentRetireTool, AgentSpawnTool};
pub use audit_log::AuditLogTool;
pub use code_gen::CodeGenTool;
pub use code_ingest::CodeIngestTool;
pub use code_predicates::CodePredicates;
pub use compile_feedback::CompileFeedbackTool;
pub use core_memory::{
    ArchivalSearchTool, CoreMemoryAppendTool, CoreMemoryReplaceTool, QueryGraphTool,
};
pub use delegate::DelegateTool;
pub use diagnostics::DiagnosticsTool;
pub use doc_gen::DocGenTool;
pub use edit::EditTool;
pub use entity_schema::{EntitySchemaTool, EntityTypeFilterTool, EntityTypesTool};
pub use file_io::FileIoTool;
pub use gap_analysis::GapAnalysisTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use http_fetch::HttpFetchTool;
pub use infer_rules::InferRulesTool;
pub use inference_config::InferenceConfigTool;
pub use kg_delete::KgDeleteTool;
pub use kg_mutate::KgMutateTool;
pub use kg_query::KgQueryTool;
pub use mcp::{McpCallTool, McpClient, McpDiscoverTool, McpError, McpServerConfig};
pub use memory_recall::MemoryRecallTool;
pub use memory_write::MemoryWriteTool;
pub use openclaw_discover::OpenClawDiscoverTool;
pub use pattern_mine::PatternMineTool;
#[cfg(feature = "daemon")]
pub use reactor_manage::ReactorManageTool;
pub use reason::ReasonTool;
pub use shell_exec::ShellExecTool;
pub use similarity_search::SimilaritySearchTool;
pub use skill_author::SkillAuthorTool;
pub use structured_task::{
    StructuredTaskCompleteTool, StructuredTaskCreateTool, StructuredTaskListTool,
    StructuredTaskUpdateTool,
};
pub use study_note::{
    StudyNoteAppendTool, StudyNoteCatalogTool, StudyNoteListTool, StudyNoteReadTool,
    StudyNoteReviseTool,
};
pub use task_list::{TaskItem, TaskListTool, TaskStatus};
pub use tool_query::ToolQueryTool;
#[cfg(feature = "daemon")]
pub use triage_dispatch_traps::TriageDispatchTrapsTool;
pub use trigger_manage::TriggerManageTool;
pub use user_interact::UserInteractTool;
pub use web_search::WebSearchTool;
