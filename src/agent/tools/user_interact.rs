//! User interaction tool: ask the user a question and read their response.
//!
//! Prints a prompt to stdout and reads a line from stdin. This tool is
//! intended for use in the REPL or interactive modes — it will block
//! waiting for user input.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

/// Ask the user a question and incorporate their answer.
pub struct UserInteractTool;

impl Tool for UserInteractTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "user_interact".into(),
            description: "Ask the user a question and read their response from stdin.".into(),
            parameters: vec![ToolParam {
                name: "question".into(),
                description: "Question to ask the user.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let question = input.require("question", "user_interact")?;

        // Print the question.
        println!("[Agent asks]: {question}");
        print!("[Your answer]: ");

        use std::io::Write;
        std::io::stdout().flush().ok();

        let mut answer = String::new();
        match std::io::stdin().read_line(&mut answer) {
            Ok(0) => {
                // EOF — no input available.
                Ok(ToolOutput::ok(
                    "No user input available (stdin closed).",
                ))
            }
            Ok(_) => {
                let trimmed = answer.trim().to_string();
                if trimmed.is_empty() {
                    Ok(ToolOutput::ok("User provided empty response."))
                } else {
                    Ok(ToolOutput::ok(format!(
                        "User responded: \"{trimmed}\""
                    )))
                }
            }
            Err(e) => Ok(ToolOutput::err(format!(
                "Failed to read user input: {e}"
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "user_interact".into(),
            description: "Prompts the user for input — no side effects beyond I/O.".into(),
            parameters: vec![
                ToolParamSchema::required("question", "Question to ask the user."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::UserInteraction]),
                description: "Prompts the user for input — no side effects beyond I/O.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
