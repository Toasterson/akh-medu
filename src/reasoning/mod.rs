//! Reasoning bridges — higher-level reasoning pipelines built on core engines.
//!
//! This module wires together Phase 17 (Dempster-Shafer), Phase 18 (ACH +
//! source reliability), and the knowledge graph into composable analysis
//! functions callable from MCP tools or autonomous agents.

pub mod ach_for_drawers;
