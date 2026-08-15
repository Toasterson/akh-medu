//! Built-in tools for the agent: KG query, KG mutate, memory recall, reason,
//! similarity search, file I/O, HTTP fetch, shell exec,
//! user interaction, infer rules, gap analysis,
//! code ingest, doc gen, pattern mine, audit log,
//! and MCP client for remote tool access.

pub mod audit_log;
pub mod entity_schema;
pub mod code_gen;
pub mod core_memory;
pub mod delegate;
pub mod code_ingest;
pub mod code_predicates;
pub mod compile_feedback;
pub mod diagnostics;
pub mod doc_gen;
pub mod edit;
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
pub mod memory_recall;
pub mod memory_write;
pub mod openclaw_discover;
pub mod pattern_mine;
pub mod skill_author;
pub mod reason;
pub mod shell_exec;
pub mod similarity_search;
pub mod structured_task;
pub mod task_list;
pub mod user_interact;
pub mod web_search;
pub mod agent_management;
pub mod tool_query;
pub mod mcp;
pub mod triage_dispatch_traps;
pub mod trigger_manage;

pub use code_gen::CodeGenTool;
pub use code_ingest::CodeIngestTool;
pub use core_memory::{ArchivalSearchTool, CoreMemoryAppendTool, CoreMemoryReplaceTool, QueryGraphTool};
pub use code_predicates::CodePredicates;
pub use compile_feedback::CompileFeedbackTool;
pub use diagnostics::DiagnosticsTool;
pub use doc_gen::DocGenTool;
pub use edit::EditTool;
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
pub use memory_recall::MemoryRecallTool;
pub use memory_write::MemoryWriteTool;
pub use openclaw_discover::OpenClawDiscoverTool;
pub use pattern_mine::PatternMineTool;
pub use skill_author::SkillAuthorTool;
pub use reason::ReasonTool;
pub use shell_exec::ShellExecTool;
pub use similarity_search::SimilaritySearchTool;
pub use structured_task::{
    StructuredTaskCompleteTool, StructuredTaskCreateTool, StructuredTaskListTool,
    StructuredTaskUpdateTool,
};
pub use task_list::{TaskItem, TaskListTool, TaskStatus};
pub use user_interact::UserInteractTool;
pub use web_search::WebSearchTool;
pub use agent_management::{AgentListTool, AgentMessageTool, AgentRetireTool, AgentSpawnTool};
pub use tool_query::ToolQueryTool;
pub use trigger_manage::TriggerManageTool;
pub use audit_log::AuditLogTool;
pub use entity_schema::{EntitySchemaTool, EntityTypesTool, EntityTypeFilterTool};
pub use delegate::DelegateTool;
pub use mcp::{McpCallTool, McpClient, McpDiscoverTool, McpError, McpServerConfig};
