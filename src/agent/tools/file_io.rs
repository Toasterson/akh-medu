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
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
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
                ToolParam {
                    name: "offset".into(),
                    description: "For 'read': 1-based start line (optional, defaults to 1).".into(),
                    required: false,
                },
                ToolParam {
                    name: "limit".into(),
                    description:
                        "For 'read': number of lines to read (optional, defaults to whole file)."
                            .into(),
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
            "read" => {
                // Parse optional offset and limit parameters
                let offset: Option<usize> = input
                    .get("offset")
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n > 0);
                let limit: Option<usize> = input.get("limit").and_then(|s| s.parse().ok());

                match self.fs.read_to_string(&path) {
                    Ok(content) => {
                        let lines: Vec<&str> = content.lines().collect();
                        let total_lines = lines.len();

                        // Calculate line range (1-based offset, inclusive)
                        let start_line = offset.unwrap_or(1).saturating_sub(1);
                        let end_line = if let Some(lim) = limit {
                            (start_line + lim).min(total_lines)
                        } else {
                            total_lines
                        };

                        if start_line >= total_lines {
                            return Ok(ToolOutput::err(format!(
                                "Offset {} is beyond the file length ({} lines).",
                                offset.unwrap_or(1),
                                total_lines
                            )));
                        }

                        // Extract the requested lines
                        let selected_lines = &lines[start_line..end_line];
                        let output = selected_lines.join("\n");

                        // Char-safe truncation at 8KB (doubled from 4KB to be more useful)
                        const MAX_BYTES: usize = 8192;
                        let (display_content, was_truncated, truncation_point) =
                            if output.len() > MAX_BYTES {
                                // Find the last valid char boundary before MAX_BYTES
                                let mut trunc_at = MAX_BYTES;
                                while trunc_at > 0 && !output.is_char_boundary(trunc_at) {
                                    trunc_at -= 1;
                                }
                                // Count how many complete lines fit
                                let truncated_str = &output[..trunc_at];
                                let displayed_lines = truncated_str.lines().count();
                                (truncated_str, true, start_line + displayed_lines)
                            } else {
                                (output.as_str(), false, end_line)
                            };

                        let mut result = format!(
                            "Read {} bytes from \"{}\" (lines {}-{} of {}):\n",
                            content.len(),
                            path.display(),
                            start_line + 1,
                            end_line,
                            total_lines
                        );

                        // Add line numbers to output
                        for (idx, line) in display_content.lines().enumerate() {
                            result.push_str(&format!("{:6} {}\n", start_line + idx + 1, line));
                        }

                        if was_truncated {
                            result.push_str(&format!(
                                "\n... [output truncated at {} KB, showing up to line {}]\n",
                                MAX_BYTES / 1024,
                                truncation_point
                            ));
                            result.push_str(&format!(
                                "To read more, use: offset={}, limit=<lines>\n",
                                truncation_point + 1
                            ));
                        } else if end_line < total_lines {
                            result.push_str(&format!(
                                "\n[Showing {} of {} lines. To read more, use: offset={}, limit=<lines>]\n",
                                end_line - start_line,
                                total_lines,
                                end_line + 1
                            ));
                        }

                        Ok(ToolOutput::ok(result))
                    }
                    Err(FsError::OutsideSandbox { .. }) => Ok(ToolOutput::err(format!(
                        "Read denied: \"{}\" is outside the akh's read sandbox.",
                        path.display()
                    ))),
                    Err(e) => Ok(ToolOutput::err(format!(
                        "Failed to read \"{}\": {e}",
                        path.display()
                    ))),
                }
            }
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
                ToolParamSchema::optional(
                    "offset",
                    "For 'read': 1-based start line (optional, defaults to 1).",
                ),
                ToolParamSchema::optional(
                    "limit",
                    "For 'read': number of lines to read (optional, defaults to whole file).",
                ),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
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

    fn locus(&self) -> Option<crate::backend::ExecLocus> {
        Some(self.fs.locus())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;
    use crate::engine::{Engine, EngineConfig};
    use crate::fs::LocalFs;
    use crate::vsa::Dimension;
    use tempfile::TempDir;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn setup_test_fs() -> (TempDir, Arc<dyn Filesystem>) {
        let temp = TempDir::new().unwrap();
        let fs = LocalFs::new(
            vec![temp.path().to_path_buf()],
            vec![temp.path().to_path_buf()],
        );
        (temp, Arc::new(fs))
    }

    #[test]
    fn test_read_with_offset_and_limit() {
        let (temp, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs.clone());
        let engine = test_engine();

        // Create a test file with 100 lines
        let test_path = temp.path().join("test.txt");
        let mut content = String::new();
        for i in 1..=100 {
            content.push_str(&format!("Line {}\n", i));
        }
        std::fs::write(&test_path, &content).unwrap();

        // Test reading lines 740-769 (mimicking the canary failure case)
        // Since we only have 100 lines, test reading lines 50-79
        let input = ToolInput::new()
            .with_param("action", "read")
            .with_param("path", test_path.to_str().unwrap())
            .with_param("offset", "50")
            .with_param("limit", "30");

        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success);

        let output = &result.result;
        // Should contain line 50
        assert!(output.contains("    50 Line 50"));
        // Should contain line 79
        assert!(output.contains("    79 Line 79"));
        // Should NOT contain line 49
        assert!(!output.contains("    49 Line 49"));
        // Should NOT contain line 80
        assert!(!output.contains("    80 Line 80"));
        // Should indicate lines 50-79 of 100 (30 lines: 50 through 79)
        assert!(output.contains("lines 50-79 of 100"));
    }

    #[test]
    fn test_read_with_multibyte_utf8_at_truncation_boundary() {
        let (temp, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs.clone());
        let engine = test_engine();

        // Create a file with multi-byte UTF-8 characters
        let test_path = temp.path().join("utf8.txt");
        let mut content = String::new();
        // Fill to approach the 8KB truncation limit with multi-byte chars
        for i in 0..300 {
            content.push_str(&format!(
                "Line {} with emoji 🎉 and unicode ñ ü ö {}\n",
                i,
                "x".repeat(20)
            ));
        }
        std::fs::write(&test_path, &content).unwrap();

        let input = ToolInput::new()
            .with_param("action", "read")
            .with_param("path", test_path.to_str().unwrap());

        // This should not panic even if a multi-byte character straddles the 8KB boundary
        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success);
        // If truncated, should have a helpful message
        if result.result.contains("truncated") {
            assert!(result.result.contains("offset="));
            assert!(result.result.contains("limit="));
        }
    }

    #[test]
    fn test_read_whole_file_when_no_offset_limit() {
        let (temp, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs.clone());
        let engine = test_engine();

        let test_path = temp.path().join("small.txt");
        let content = "Line 1\nLine 2\nLine 3\n";
        std::fs::write(&test_path, content).unwrap();

        let input = ToolInput::new()
            .with_param("action", "read")
            .with_param("path", test_path.to_str().unwrap());

        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success);

        let output = &result.result;
        assert!(output.contains("     1 Line 1"));
        assert!(output.contains("     2 Line 2"));
        assert!(output.contains("     3 Line 3"));
        assert!(output.contains("lines 1-3 of 3"));
    }

    #[test]
    fn test_read_offset_beyond_file_length() {
        let (temp, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs.clone());
        let engine = test_engine();

        let test_path = temp.path().join("short.txt");
        let content = "Line 1\nLine 2\n";
        std::fs::write(&test_path, content).unwrap();

        let input = ToolInput::new()
            .with_param("action", "read")
            .with_param("path", test_path.to_str().unwrap())
            .with_param("offset", "100");

        let result = tool.execute(&engine, input).unwrap();
        assert!(!result.success);
        assert!(result.result.contains("beyond the file length"));
    }

    #[test]
    fn test_truncation_message_includes_next_offset() {
        let (temp, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs.clone());
        let engine = test_engine();

        // Create a file large enough to trigger truncation
        let test_path = temp.path().join("large.txt");
        let mut content = String::new();
        for i in 0..500 {
            content.push_str(&format!("Line {} {}\n", i, "x".repeat(100)));
        }
        std::fs::write(&test_path, &content).unwrap();

        let input = ToolInput::new()
            .with_param("action", "read")
            .with_param("path", test_path.to_str().unwrap());

        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success);

        // Should be truncated and provide guidance
        assert!(result.result.contains("truncated"));
        assert!(result.result.contains("offset="));
        assert!(result.result.contains("limit="));
    }

    #[test]
    fn test_manifest_includes_new_parameters() {
        let (_, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs);
        let manifest = tool.manifest();

        // Check that offset and limit are in the manifest
        let param_names: Vec<&str> = manifest
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert!(param_names.contains(&"offset"));
        assert!(param_names.contains(&"limit"));

        // Find the offset parameter and check it's optional
        let offset_param = manifest
            .parameters
            .iter()
            .find(|p| p.name == "offset")
            .unwrap();
        assert!(!offset_param.required);
        assert!(offset_param.description.contains("1-based"));
        assert!(offset_param.description.contains("start line"));

        // Find the limit parameter and check it's optional
        let limit_param = manifest
            .parameters
            .iter()
            .find(|p| p.name == "limit")
            .unwrap();
        assert!(!limit_param.required);
        assert!(limit_param.description.contains("number of lines"));

        // Verify the manifest can be serialized (this is what gets sent to the LLM)
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("offset"));
        assert!(json.contains("limit"));
        assert!(json.contains("1-based"));
    }

    #[test]
    fn test_signature_includes_new_parameters() {
        let (_, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs);
        let signature = tool.signature();

        let param_names: Vec<&str> = signature
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert!(param_names.contains(&"offset"));
        assert!(param_names.contains(&"limit"));
    }

    #[test]
    fn test_canary_case_50kb_file_line_749() {
        // Reproduces the exact failure case from WI-255:
        // A 50KB file where line 749 (OPERATING_CONVENTIONS) is at ~30KB offset
        let (temp, fs) = setup_test_fs();
        let tool = FileIoTool::new(fs.clone());
        let engine = test_engine();

        let test_path = temp.path().join("large_source.rs");
        let mut content = String::new();

        // Create content similar to the real file - each line ~65 bytes
        for i in 1..=800 {
            if i == 749 {
                content
                    .push_str("const OPERATING_CONVENTIONS: &str = \"conventions text here\";\n");
            } else {
                content.push_str(&format!(
                    "// Line {} - some typical source code content here, padding to realistic size\n",
                    i
                ));
            }
        }

        // Verify we created a file large enough to trigger the old bug
        assert!(
            content.len() > 40000,
            "Test file should be >40KB to trigger truncation bug"
        );
        std::fs::write(&test_path, &content).unwrap();

        // This is what the agent tried: read lines around 749
        let input = ToolInput::new()
            .with_param("action", "read")
            .with_param("path", test_path.to_str().unwrap())
            .with_param("offset", "740")
            .with_param("limit", "30");

        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success, "Read should succeed");

        let output = &result.result;
        // The critical assertion: line 749 must be present
        assert!(
            output.contains("749 const OPERATING_CONVENTIONS"),
            "Line 749 must be readable even though it's 30KB into the file. Output: {}",
            output
        );

        // Should show lines 740-769
        assert!(output.contains("   740 "));
        assert!(output.contains("   769 "));
        assert!(!output.contains("   739 "));
        assert!(!output.contains("   770 "));
    }
}
