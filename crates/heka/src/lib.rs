//! Heka — personal memory daemon.
//!
//! Stores timestamped "drawers" (memory entries) with MCP tool access.
//! The `heka_contradict` tool is the flagship feature: given two drawer IDs,
//! it runs Phase 18 ACH + Phase 17 Dempster-Shafer to determine which claim
//! is better supported, returning Dempster-Shafer confidence intervals and
//! a deciding-evidence chain.
//!
//! Anima WI: `019dd0c7-2557-7e40-874d-d3c99af2e239` (Phase 42i)

pub mod drawer;
pub mod error;
pub mod mcp;
