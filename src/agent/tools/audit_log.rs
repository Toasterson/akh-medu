//! LogTool: allows the agent to voluntarily write messages to the audit ledger.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::audit::{AuditEntry, AuditKind, LogLevel};
use crate::engine::Engine;

/// Tool that writes agent-authored messages to the audit ledger.
pub struct AuditLogTool;

impl Tool for AuditLogTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "audit_log".into(),
            description: "Write a message to the audit ledger for permanent record-keeping.".into(),
            parameters: vec![
                ToolParam {
                    name: "message".into(),
                    description: "The message to record in the audit log.".into(),
                    required: true,
                },
                ToolParam {
                    name: "level".into(),
                    description: "Log level: info (default), warn, or debug.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let message = input.require("message", "audit_log")?;
        let level = match input.get("level").unwrap_or("info") {
            "warn" | "warning" => LogLevel::Warn,
            "debug" => LogLevel::Debug,
            _ => LogLevel::Info,
        };

        let ledger = match engine.audit_ledger() {
            Some(l) => l,
            None => {
                return Ok(ToolOutput::err(
                    "Audit ledger unavailable — no persistence configured.",
                ));
            }
        };

        let mut entry = AuditEntry::new(
            AuditKind::AgentLog {
                level,
                message: message.to_string(),
            },
            "default",
        );

        match ledger.append(&mut entry) {
            Ok(id) => Ok(ToolOutput::ok(format!(
                "Logged to audit ledger (id: {id}).",
            ))),
            Err(e) => Ok(ToolOutput::err(format!(
                "Failed to write audit log: {e}",
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "audit_log".into(),
            description: "Write a message to the audit ledger for permanent record-keeping.".into(),
            parameters: vec![
                ToolParamSchema::required("message", "The message to record."),
                ToolParamSchema::optional("level", "Log level: info (default), warn, or debug."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::WriteKg]),
                description: "Appends to the audit ledger (write-only, no read).".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
