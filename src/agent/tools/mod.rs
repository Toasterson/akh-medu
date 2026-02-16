//! Built-in tools for the agent: KG query, KG mutate, memory recall, reason,
//! similarity search, library search, file I/O, HTTP fetch, shell exec,
//! user interaction, infer rules, gap analysis, CSV ingest, text ingest,
//! code ingest, content ingest, doc gen.

pub mod code_ingest;
pub mod code_predicates;
pub mod content_ingest;
pub mod csv_ingest;
pub mod doc_gen;
pub mod file_io;
pub mod gap_analysis;
pub mod http_fetch;
pub mod infer_rules;
pub mod kg_mutate;
pub mod kg_query;
pub mod library_search;
pub mod memory_recall;
pub mod reason;
pub mod shell_exec;
pub mod similarity_search;
pub mod text_ingest;
pub mod user_interact;
pub mod agent_management;
pub mod trigger_manage;

pub use code_ingest::CodeIngestTool;
pub use code_predicates::CodePredicates;
pub use content_ingest::ContentIngestTool;
pub use csv_ingest::CsvIngestTool;
pub use doc_gen::DocGenTool;
pub use file_io::FileIoTool;
pub use gap_analysis::GapAnalysisTool;
pub use http_fetch::HttpFetchTool;
pub use infer_rules::InferRulesTool;
pub use kg_mutate::KgMutateTool;
pub use kg_query::KgQueryTool;
pub use library_search::LibrarySearchTool;
pub use memory_recall::MemoryRecallTool;
pub use reason::ReasonTool;
pub use shell_exec::ShellExecTool;
pub use similarity_search::SimilaritySearchTool;
pub use text_ingest::TextIngestTool;
pub use user_interact::UserInteractTool;
pub use agent_management::{AgentListTool, AgentMessageTool, AgentRetireTool, AgentSpawnTool};
pub use trigger_manage::TriggerManageTool;
