//! File I/O tool: read and write files from the agent's scratch space.
//!
//! Reads are allowed from any path. Writes are restricted to paths under
//! the engine's data directory (or a `scratch/` subdirectory within it)
//! to prevent the agent from writing to arbitrary locations.

use std::path::PathBuf;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;

/// Read and write files for agent scratch / data purposes.
pub struct FileIoTool {
    /// Directory the agent is allowed to write into.
    /// If `None`, writes are disabled.
    scratch_dir: Option<PathBuf>,
}

impl FileIoTool {
    /// Create a new FileIoTool. If `scratch_dir` is Some, the agent can write
    /// files there. Pass the engine's data_dir or a subdirectory of it.
    pub fn new(scratch_dir: Option<PathBuf>) -> Self {
        Self { scratch_dir }
    }
}

impl Tool for FileIoTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "file_io".into(),
            description: "Read or write files. Writes are restricted to the scratch directory."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "action".into(),
                    description: "Action: 'read' or 'write'.".into(),
                    required: true,
                },
                ToolParam {
                    name: "path".into(),
                    description: "File path (absolute or relative to scratch dir).".into(),
                    required: true,
                },
                ToolParam {
                    name: "content".into(),
                    description: "Content to write (required for 'write' action).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let action = input.require("action", "file_io")?;
        let path_str = input.require("path", "file_io")?;

        match action {
            "read" => {
                let path = PathBuf::from(path_str);
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        let truncated = if content.len() > 4096 {
                            format!("{}... [truncated at 4096 bytes, total: {}]", &content[..4096], content.len())
                        } else {
                            content.clone()
                        };
                        Ok(ToolOutput::ok(format!(
                            "Read {} bytes from \"{}\":\n{}",
                            content.len(),
                            path.display(),
                            truncated,
                        )))
                    }
                    Err(e) => Ok(ToolOutput::err(format!(
                        "Failed to read \"{}\": {e}",
                        path.display()
                    ))),
                }
            }
            "write" => {
                let content = input.require("content", "file_io")?;
                let scratch = match &self.scratch_dir {
                    Some(dir) => dir,
                    None => {
                        return Ok(ToolOutput::err(
                            "Writes disabled: no scratch directory configured. \
                             Use --data-dir to enable file writes.",
                        ));
                    }
                };

                // Resolve path relative to scratch dir.
                let path = if PathBuf::from(path_str).is_absolute() {
                    PathBuf::from(path_str)
                } else {
                    scratch.join(path_str)
                };

                // Security: ensure the resolved path is within the scratch dir.
                let canonical_scratch = scratch
                    .canonicalize()
                    .unwrap_or_else(|_| scratch.to_path_buf());
                let canonical_path = path.parent().and_then(|p| p.canonicalize().ok());

                let within_scratch = canonical_path
                    .as_ref()
                    .is_some_and(|cp| cp.starts_with(&canonical_scratch));

                // Also allow if path is directly in scratch dir (file may not exist yet).
                let parent_is_scratch = path
                    .parent()
                    .is_some_and(|p| p == scratch.as_path() || p.starts_with(scratch.as_path()));

                if !within_scratch && !parent_is_scratch {
                    return Ok(ToolOutput::err(format!(
                        "Write denied: path \"{}\" is outside scratch directory \"{}\".",
                        path.display(),
                        scratch.display(),
                    )));
                }

                // Ensure parent directory exists.
                if let Some(parent) = path.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return Ok(ToolOutput::err(format!(
                            "Failed to create directory \"{}\": {e}",
                            parent.display()
                        )));
                    }
                }

                match std::fs::write(&path, content) {
                    Ok(()) => Ok(ToolOutput::ok(format!(
                        "Wrote {} bytes to \"{}\".",
                        content.len(),
                        path.display(),
                    ))),
                    Err(e) => Ok(ToolOutput::err(format!(
                        "Failed to write \"{}\": {e}",
                        path.display()
                    ))),
                }
            }
            other => Ok(ToolOutput::err(format!(
                "Unknown file_io action: \"{other}\". Use 'read' or 'write'."
            ))),
        }
    }
}
