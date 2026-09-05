//! Edit tool: exact-string find-and-replace in a file through a sandboxed
//! [`Filesystem`].
//!
//! Mirrors Claude Code's `Edit`: read a file, locate an exact `old_string`,
//! and replace it with `new_string`. By default the match must be unique —
//! the edit fails if `old_string` occurs more than once unless `replace_all`
//! is set, which forces every occurrence to be replaced. All reads and writes
//! are confined to the akh's sandbox roots via the injected [`Filesystem`], so
//! containment policy lives in one audited place (a [`crate::fs::LocalFs::denied`]
//! filesystem fails closed).

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::backend::{Backend, InProcessBackend};
use crate::engine::Engine;
use crate::fs::Filesystem;
use std::collections::HashSet;

/// Exact-string find-and-replace in a file, through the session's
/// [`Backend`] (ADR 082), whose file verbs carry the sandboxed
/// [`Filesystem`] contract.
pub struct EditTool {
    /// The backend the tool reads and writes through. A backend wrapping a
    /// [`crate::fs::LocalFs::denied`] filesystem disables all access
    /// (fail-closed).
    fs: Arc<dyn Backend>,
}

impl EditTool {
    /// Create an `EditTool` backed by the given sandboxed filesystem,
    /// executed in-process (the historic behavior, verbatim).
    pub fn new(fs: Arc<dyn Filesystem>) -> Self {
        Self::with_backend(Arc::new(InProcessBackend::new(fs)))
    }

    /// Create an `EditTool` routed through an explicit backend — the seam
    /// where 48b/48c plug in privilege-dropped or remote targets.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self { fs: backend }
    }
}

impl Tool for EditTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "edit".into(),
            description: "Exact-string find-and-replace in a file. Replaces a unique \
                          occurrence of old_string with new_string; set replace_all to \
                          replace every occurrence."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "path".into(),
                    description: "File path to edit (absolute or relative to scratch dir).".into(),
                    required: true,
                },
                ToolParam {
                    name: "old_string".into(),
                    description: "Exact text to find. Must be unique unless replace_all is set."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "new_string".into(),
                    description: "Replacement text.".into(),
                    required: true,
                },
                ToolParam {
                    name: "replace_all".into(),
                    description: "Replace every occurrence instead of requiring uniqueness \
                                  (default: false)."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path_str = input.require("path", "edit")?;
        let old_string = input.require("old_string", "edit")?;
        let new_string = input.require("new_string", "edit")?;
        let replace_all = input.get_bool("replace_all").unwrap_or(false);

        if old_string.is_empty() {
            return Ok(ToolOutput::err(
                "old_string must not be empty; use file_io write to create a file",
            ));
        }
        if old_string == new_string {
            return Ok(ToolOutput::err("old_string and new_string are identical"));
        }

        let path = PathBuf::from(path_str);

        // Read the current contents through the sandbox.
        let content = match self.fs.read_to_string(&path) {
            Ok(content) => content,
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "Failed to read \"{}\": {e}",
                    path.display()
                )));
            }
        };

        // Count non-overlapping, left-to-right occurrences — the same semantics
        // `replacen`/`replace` use, so the reported count matches what is replaced.
        let count = content.matches(old_string).count();
        if count == 0 {
            return Ok(ToolOutput::err(format!(
                "old_string not found in {}",
                path.display()
            )));
        }
        if count > 1 && !replace_all {
            return Ok(ToolOutput::err(format!(
                "old_string is not unique ({count} occurrences); set replace_all=true or \
                 include more surrounding context to make it unique"
            )));
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            // Exactly one occurrence here (count == 1), so replacen(.., 1) replaces it.
            content.replacen(old_string, new_string, 1)
        };
        let replaced = if replace_all { count } else { 1 };
        let bytes = new_content.len();

        // Write the result back through the sandbox.
        match self.fs.write(&path, &new_content) {
            Ok(()) => Ok(ToolOutput::ok(format!(
                "Edited {}: replaced {} occurrence(s), {} bytes written",
                path.display(),
                replaced,
                bytes
            ))),
            Err(e) => Ok(ToolOutput::err(format!(
                "Failed to write \"{}\": {e}",
                path.display()
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "edit".into(),
            description: "Exact-string find-and-replace in a file — filesystem side effects."
                .into(),
            parameters: vec![
                ToolParamSchema::required(
                    "path",
                    "File path to edit (absolute or relative to scratch dir).",
                ),
                ToolParamSchema::required(
                    "old_string",
                    "Exact text to find. Must be unique unless replace_all is set.",
                ),
                ToolParamSchema::required("new_string", "Replacement text."),
                ToolParamSchema::optional(
                    "replace_all",
                    "Replace every occurrence instead of requiring uniqueness (default: false).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([
                    Capability::ReadFilesystem,
                    Capability::WriteFilesystem,
                ]),
                description: "Reads a file and overwrites it with a find-and-replace edit — \
                              filesystem side effects."
                    .into(),
                shadow_triggers: vec!["write".into(), "edit".into(), "overwrite".into()],
            },
            source: ToolSource::Native,
        }
    }

    fn locus(&self) -> Option<crate::backend::ExecLocus> {
        Some(self.fs.locus())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::fs::LocalFs;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    /// Build an `EditTool` over a fresh temp sandbox, returning the tool and the
    /// kept-alive `TempDir` so the directory outlives the test body.
    fn tool_in_tempdir() -> (EditTool, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let fs: Arc<dyn Filesystem> = Arc::new(LocalFs::sandbox(dir.path()));
        (EditTool::new(fs), dir)
    }

    #[test]
    fn edit_replaces_unique() {
        let engine = test_engine();
        let (tool, dir) = tool_in_tempdir();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello world").unwrap();

        let input = ToolInput::new()
            .with_param("path", file.to_string_lossy().to_string())
            .with_param("old_string", "world")
            .with_param("new_string", "akh");
        let out = tool.execute(&engine, input).unwrap();

        assert!(out.success, "expected success: {}", out.result);
        assert!(
            out.result.contains("replaced 1 occurrence(s)"),
            "{}",
            out.result
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello akh");
    }

    #[test]
    fn edit_rejects_non_unique_without_replace_all() {
        let engine = test_engine();
        let (tool, dir) = tool_in_tempdir();
        let file = dir.path().join("b.txt");
        std::fs::write(&file, "a a a").unwrap();

        let input = ToolInput::new()
            .with_param("path", file.to_string_lossy().to_string())
            .with_param("old_string", "a")
            .with_param("new_string", "b");
        let out = tool.execute(&engine, input).unwrap();

        assert!(!out.success, "expected failure for non-unique match");
        assert!(out.result.contains("not unique"), "{}", out.result);
        assert!(out.result.contains("3 occurrences"), "{}", out.result);
        // File must be untouched.
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "a a a");
    }

    #[test]
    fn edit_replace_all() {
        let engine = test_engine();
        let (tool, dir) = tool_in_tempdir();
        let file = dir.path().join("c.txt");
        std::fs::write(&file, "a a a").unwrap();

        let input = ToolInput::new()
            .with_param("path", file.to_string_lossy().to_string())
            .with_param("old_string", "a")
            .with_param("new_string", "b")
            .with_value("replace_all", true);
        let out = tool.execute(&engine, input).unwrap();

        assert!(out.success, "expected success: {}", out.result);
        assert!(
            out.result.contains("replaced 3 occurrence(s)"),
            "{}",
            out.result
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "b b b");
    }

    #[test]
    fn edit_errors_when_not_found() {
        let engine = test_engine();
        let (tool, dir) = tool_in_tempdir();
        let file = dir.path().join("d.txt");
        std::fs::write(&file, "hello").unwrap();

        let input = ToolInput::new()
            .with_param("path", file.to_string_lossy().to_string())
            .with_param("old_string", "absent")
            .with_param("new_string", "x");
        let out = tool.execute(&engine, input).unwrap();

        assert!(!out.success, "expected failure when old_string is absent");
        assert!(out.result.contains("not found"), "{}", out.result);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello");
    }

    #[test]
    fn edit_rejects_empty_old_string() {
        let engine = test_engine();
        let (tool, dir) = tool_in_tempdir();
        let file = dir.path().join("e.txt");
        std::fs::write(&file, "hello").unwrap();

        let input = ToolInput::new()
            .with_param("path", file.to_string_lossy().to_string())
            .with_param("old_string", "")
            .with_param("new_string", "x");
        let out = tool.execute(&engine, input).unwrap();

        assert!(!out.success);
        assert!(out.result.contains("must not be empty"), "{}", out.result);
    }

    #[test]
    fn edit_rejects_identical_strings() {
        let engine = test_engine();
        let (tool, dir) = tool_in_tempdir();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "hello").unwrap();

        let input = ToolInput::new()
            .with_param("path", file.to_string_lossy().to_string())
            .with_param("old_string", "hello")
            .with_param("new_string", "hello");
        let out = tool.execute(&engine, input).unwrap();

        assert!(!out.success);
        assert!(out.result.contains("identical"), "{}", out.result);
    }

    #[test]
    fn edit_missing_required_param_errors() {
        let engine = test_engine();
        let (tool, _dir) = tool_in_tempdir();
        // No `path` provided → hard AgentError via input.require.
        let input = ToolInput::new()
            .with_param("old_string", "a")
            .with_param("new_string", "b");
        let result = tool.execute(&engine, input);
        assert!(result.is_err(), "missing required param must be Err");
    }
}
