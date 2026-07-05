//! File I/O tool: read and write files through the session's [`Backend`]
//! (ADR 082).
//!
//! Both reads and writes are confined to the akh's sandbox roots. The previous
//! implementation read from *any* path and only loosely checked writes; this
//! tool now delegates all path resolution and containment to the injected
//! [`Backend`] (whose file verbs carry the [`Filesystem`] contract), so the
//! sandbox policy lives in one audited place.

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::backend::{Backend, InProcessBackend};
use crate::engine::Engine;
use crate::fs::{Filesystem, FsError};
use std::collections::HashSet;

/// Read and write files through the session's [`Backend`].
pub struct FileIoTool {
    /// The backend the tool operates through. A backend wrapping a
    /// [`crate::fs::LocalFs::denied`] filesystem disables all access
    /// (fail-closed).
    fs: Arc<dyn Backend>,
}

impl FileIoTool {
    /// Create a FileIoTool backed by the given sandboxed filesystem, executed
    /// in-process (the historic behavior, verbatim).
    pub fn new(fs: Arc<dyn Filesystem>) -> Self {
        Self::with_backend(Arc::new(InProcessBackend::new(fs)))
    }

    /// Create a FileIoTool routed through an explicit backend — the seam
    /// where 48b/48c plug in privilege-dropped or remote targets.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self { fs: backend }
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

        let path = PathBuf::from(path_str);
        match action {
            "read" => match self.fs.read_to_string(&path) {
                Ok(content) => {
                    let truncated = if content.len() > 4096 {
                        format!(
                            "{}... [truncated at 4096 bytes, total: {}]",
                            &content[..4096],
                            content.len()
                        )
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
                Err(FsError::OutsideSandbox { .. }) => Ok(ToolOutput::err(format!(
                    "Read denied: \"{}\" is outside the akh's read sandbox.",
                    path.display()
                ))),
                Err(e) => Ok(ToolOutput::err(format!(
                    "Failed to read \"{}\": {e}",
                    path.display()
                ))),
            },
            "write" => {
                let content = input.require("content", "file_io")?;
                match self.fs.write(&path, content) {
                    Ok(()) => Ok(ToolOutput::ok(format!(
                        "Wrote {} bytes to \"{}\".",
                        content.len(),
                        path.display(),
                    ))),
                    Err(FsError::WritesDisabled) => Ok(ToolOutput::err(
                        "Writes disabled: no writable sandbox configured. \
                         Run the akh with a data directory to enable file writes.",
                    )),
                    Err(e @ FsError::OutsideSandbox { .. }) => Ok(ToolOutput::err(format!(
                        "Write denied: {e} (path \"{}\" is outside the write sandbox).",
                        path.display()
                    ))),
                    Err(e @ FsError::Traversal { .. }) => {
                        Ok(ToolOutput::err(format!("Write denied: {e}.")))
                    }
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

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "file_io".into(),
            description: "Reads and writes files — filesystem side effects.".into(),
            parameters: vec![
                ToolParamSchema::required("action", "Action: 'read' or 'write'."),
                ToolParamSchema::required(
                    "path",
                    "File path (absolute or relative to scratch dir).",
                ),
                ToolParamSchema::optional(
                    "content",
                    "Content to write (required for 'write' action).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([
                    Capability::ReadFilesystem,
                    Capability::WriteFilesystem,
                ]),
                description: "Reads and writes files — filesystem side effects.".into(),
                shadow_triggers: vec!["write".into(), "delete".into(), "overwrite".into()],
            },
            source: ToolSource::Native,
        }
    }
}
