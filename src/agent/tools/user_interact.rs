//! User interaction tool: ask the user a question and read their response.
//!
//! Prints a prompt to stdout and reads a line from stdin. This tool is
//! intended for use in the REPL or interactive modes — it will block
//! waiting for user input.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;

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
}
