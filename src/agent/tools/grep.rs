//! Grep tool: search file contents by regex through a sandboxed [`Filesystem`].
//!
//! Walks a base directory (confined to the akh's sandbox roots) and matches a
//! caller-supplied regex against each line of each text file, mirroring Claude
//! Code's `Grep`. An optional `glob` filename filter narrows the files scanned,
//! and `output_mode` selects between listing matching files, printing matching
//! lines, or counting matches per file. Binary files (those containing a NUL
//! byte) and oversized files (>1 MB) are skipped gracefully, and all path
//! resolution / containment is delegated to the injected [`Filesystem`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::RegexBuilder;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::backend::{Backend, InProcessBackend};
use crate::engine::Engine;
use crate::fs::Filesystem;
use std::collections::HashSet;

/// Maximum file size to scan (1 MB). Larger files are skipped.
const MAX_FILE_SIZE: usize = 1024 * 1024;

/// Maximum number of files to scan in a single invocation.
const MAX_FILES_SCANNED: usize = 5_000;

/// Maximum number of content lines emitted in `content` mode.
const MAX_CONTENT_LINES: usize = 200;

/// Maximum directory recursion depth. Bounds the walk even when a subtree
/// contains no files (so the file cap never trips) — e.g. a deeply nested
/// chain. Mirrors the Glob tool's guard.
const MAX_DEPTH: usize = 64;

/// Search file contents by regex through the session's [`Backend`]
/// (ADR 082), whose file verbs carry the sandboxed [`Filesystem`] contract.
pub struct GrepTool {
    /// The backend the tool reads through. A backend wrapping a
    /// [`crate::fs::LocalFs::denied`] filesystem disables all access
    /// (fail-closed).
    fs: Arc<dyn Backend>,
}

impl GrepTool {
    /// Create a GrepTool backed by the given sandboxed filesystem, read
    /// in-process (the historic behavior, verbatim).
    pub fn new(fs: Arc<dyn Filesystem>) -> Self {
        Self::with_backend(Arc::new(InProcessBackend::new(fs)))
    }

    /// Create a GrepTool routed through an explicit backend — the seam where
    /// 48b/48c plug in privilege-dropped or remote targets.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self { fs: backend }
    }

    /// Resolve the base directory: explicit `path`, else the first read root,
    /// else the current directory.
    fn base_dir(&self, requested: Option<&str>) -> PathBuf {
        if let Some(p) = requested {
            return PathBuf::from(p);
        }
        if let Some(root) = self.fs.read_roots().first() {
            return root.clone();
        }
        PathBuf::from(".")
    }

    /// Recursively collect files under `dir` (skipping directories that fail to
    /// list), capping the total at [`MAX_FILES_SCANNED`]. Directories are
    /// detected by a successful `read_dir`; anything else is treated as a file.
    fn collect_files(&self, dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if out.len() >= MAX_FILES_SCANNED || depth > MAX_DEPTH {
            return;
        }
        let entries = match self.fs.read_dir(dir) {
            Ok(entries) => entries,
            // Not a directory (or unreadable): treat the path itself as a file.
            Err(_) => {
                out.push(dir.to_path_buf());
                return;
            }
        };
        let mut entries = entries;
        entries.sort();
        for entry in entries {
            if out.len() >= MAX_FILES_SCANNED {
                return;
            }
            // Do not follow symlinks (matches find/ripgrep defaults): this
            // prevents in-root symlink cycles from recursing forever (stack
            // overflow) and stops a symlinked leaf from being listed as a file.
            if self.fs.is_symlink(&entry) {
                continue;
            }
            // A successful read_dir means it's a directory → recurse; otherwise
            // it's a leaf file.
            if self.fs.read_dir(&entry).is_ok() {
                self.collect_files(&entry, depth + 1, out);
            } else {
                out.push(entry);
            }
        }
    }
}

/// Translate a simple glob pattern (`*`, `?`, `**`) into an anchored regex that
/// matches a *filename* (not a full path). Mirrors the Glob tool's translation.
fn glob_to_regex(glob: &str) -> String {
    let mut re = String::with_capacity(glob.len() * 2 + 4);
    re.push('^');
    let mut chars = glob.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if chars.peek() == Some(&'*') {
                    // `**` — match across any characters (incl. path separators).
                    chars.next();
                    re.push_str(".*");
                } else {
                    // `*` — match any run of non-separator characters.
                    re.push_str("[^/]*");
                }
            }
            '?' => re.push_str("[^/]"),
            // Escape regex metacharacters that may appear literally in a glob.
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                re.push('\\');
                re.push(c);
            }
            other => re.push(other),
        }
    }
    re.push('$');
    re
}

/// Whether the file at `path` matches the optional compiled glob filter. The
/// filter is applied to the file name only (the final path component).
fn matches_glob(path: &Path, glob_re: &Option<regex::Regex>) -> bool {
    match glob_re {
        None => true,
        Some(re) => path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|name| re.is_match(name))
            .unwrap_or(false),
    }
}

/// Render a path relative to `base` if possible, else its full display form.
fn rel_display(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

impl Tool for GrepTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "grep".into(),
            description: "Search file contents by regex (like Grep). Walks a base directory and \
                          matches a regex per line; supports a glob filename filter and three \
                          output modes."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "pattern".into(),
                    description: "Regular expression to search for (regex crate syntax).".into(),
                    required: true,
                },
                ToolParam {
                    name: "path".into(),
                    description: "Base directory to search (default: first read root, else '.')."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "glob".into(),
                    description: "Filename glob filter, e.g. '*.rs' (matches the file name only)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "case_insensitive".into(),
                    description: "Case-insensitive matching (default: false).".into(),
                    required: false,
                },
                ToolParam {
                    name: "output_mode".into(),
                    description:
                        "'files_with_matches' (default) | 'content' | 'count'.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let pattern = input.require("pattern", "grep")?;
        let case_insensitive = input.get_bool("case_insensitive").unwrap_or(false);
        let output_mode = input.get("output_mode").unwrap_or("files_with_matches");

        // Build the search regex; an invalid pattern is an ordinary failure.
        let regex = match RegexBuilder::new(pattern)
            .case_insensitive(case_insensitive)
            .build()
        {
            Ok(re) => re,
            Err(e) => return Ok(ToolOutput::err(format!("invalid regex: {e}"))),
        };

        // Build the optional glob filter regex.
        let glob_re: Option<regex::Regex> = match input.get("glob") {
            Some(g) if !g.trim().is_empty() => match regex::Regex::new(&glob_to_regex(g)) {
                Ok(re) => Some(re),
                Err(e) => return Ok(ToolOutput::err(format!("invalid glob \"{g}\": {e}"))),
            },
            _ => None,
        };

        let base = self.base_dir(input.get("path"));

        // The base must be readable (and within the sandbox). If it isn't, surface
        // a clear ordinary failure rather than silently returning "No matches".
        if !self.fs.exists(&base) {
            return Ok(ToolOutput::err(format!(
                "path \"{}\" does not exist or is outside the read sandbox.",
                base.display()
            )));
        }

        // Collect candidate files (the walk skips directories it cannot list).
        let mut files = Vec::new();
        self.collect_files(&base, 0, &mut files);

        // Per-file match accumulation.
        let mut matched_files: Vec<String> = Vec::new();
        let mut counts: Vec<(String, usize)> = Vec::new();
        let mut content_lines: Vec<String> = Vec::new();
        let mut content_truncated = false;
        let mut scanned = 0usize;

        for file in &files {
            if !matches_glob(file, &glob_re) {
                continue;
            }
            // Skip oversized files BEFORE reading them, so a multi-GB file is
            // never materialized in memory. An unreadable size (permission,
            // vanished) skips the file too.
            match self.fs.file_size(file) {
                Ok(size) if size as usize > MAX_FILE_SIZE => continue,
                Ok(_) => {}
                Err(_) => continue,
            }
            // Read the file; skip ones that error (permission, vanished, …).
            let contents = match self.fs.read_to_string(file) {
                Ok(c) => c,
                Err(_) => continue,
            };
            // Skip binary (NUL-containing) files gracefully.
            if contents.as_bytes().contains(&0) {
                continue;
            }
            scanned += 1;

            let rel = rel_display(file, &base);
            let mut file_count = 0usize;
            for (idx, line) in contents.lines().enumerate() {
                if regex.is_match(line) {
                    file_count += 1;
                    if output_mode == "content" && !content_truncated {
                        if content_lines.len() >= MAX_CONTENT_LINES {
                            content_truncated = true;
                        } else {
                            content_lines.push(format!("{rel}:{}:{line}", idx + 1));
                        }
                    }
                }
            }

            if file_count > 0 {
                matched_files.push(rel.clone());
                counts.push((rel, file_count));
            }
        }

        let result = match output_mode {
            "content" => {
                if content_lines.is_empty() {
                    "No matches".to_string()
                } else if content_truncated {
                    format!(
                        "{}\n... [truncated at {} lines]",
                        content_lines.join("\n"),
                        MAX_CONTENT_LINES
                    )
                } else {
                    content_lines.join("\n")
                }
            }
            "count" => {
                if counts.is_empty() {
                    "No matches".to_string()
                } else {
                    counts.sort();
                    counts
                        .iter()
                        .map(|(f, n)| format!("{f}: {n}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            "files_with_matches" => {
                if matched_files.is_empty() {
                    "No matches".to_string()
                } else {
                    matched_files.sort();
                    matched_files.dedup();
                    matched_files.join("\n")
                }
            }
            other => {
                return Ok(ToolOutput::err(format!(
                    "unknown output_mode: \"{other}\". Use 'files_with_matches', 'content', or 'count'."
                )));
            }
        };

        let _ = scanned;
        Ok(ToolOutput::ok(result))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "grep".into(),
            description: "Searches file contents by regex within the read sandbox.".into(),
            parameters: vec![
                ToolParamSchema::required(
                    "pattern",
                    "Regular expression to search for (regex crate syntax).",
                ),
                ToolParamSchema::optional(
                    "path",
                    "Base directory to search (default: first read root, else '.').",
                ),
                ToolParamSchema::optional(
                    "glob",
                    "Filename glob filter, e.g. '*.rs' (matches the file name only).",
                ),
                ToolParamSchema::optional(
                    "case_insensitive",
                    "Case-insensitive matching (default: false).",
                ),
                ToolParamSchema::optional(
                    "output_mode",
                    "'files_with_matches' (default) | 'content' | 'count'.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::ReadFilesystem]),
                description: "Reads file contents within the read sandbox to find regex matches."
                    .into(),
                shadow_triggers: vec!["grep".into(), "search".into(), "find".into()],
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
    use crate::engine::{Engine, EngineConfig};
    use crate::fs::LocalFs;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    /// Build a GrepTool over a fresh tempdir, returning the tool and the root.
    fn setup() -> (GrepTool, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        // Seed a small file tree.
        std::fs::write(root.join("alpha.rs"), "fn alpha() {}\nlet x = 1;\n").unwrap();
        std::fs::write(
            root.join("beta.txt"),
            "hello world\nGoodbye World\nanother line\n",
        )
        .unwrap();
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(
            root.join("nested/gamma.rs"),
            "fn gamma() {}\n// world here\n",
        )
        .unwrap();
        let tool = GrepTool::new(Arc::new(LocalFs::sandbox(&root)));
        (tool, dir)
    }

    #[cfg(unix)]
    #[test]
    fn grep_does_not_follow_symlink_cycle() {
        // An in-root symlink cycle (loop -> root) used to recurse forever and
        // overflow the stack (the file cap never trips on a dir-only cycle).
        // The walk must skip the symlink and still find the real file.
        let (tool, dir) = setup();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "alpha")
            .with_param("output_mode", "files_with_matches");
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success, "result: {}", out.result);
        assert!(
            out.result.contains("alpha.rs"),
            "should still find the real file: {}",
            out.result
        );
    }

    #[test]
    fn grep_finds_in_file() {
        let (tool, _dir) = setup();
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "world")
            .with_param("output_mode", "content");
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success, "result: {}", out.result);
        // Case-sensitive: only the lowercase "world" lines match.
        assert!(out.result.contains("beta.txt:1:hello world"));
        assert!(out.result.contains("gamma.rs:2:// world here"));
        // The capital-W "Goodbye World" line must NOT match (case-sensitive).
        assert!(!out.result.contains("Goodbye World"));
    }

    #[test]
    fn grep_files_with_matches_mode() {
        let (tool, _dir) = setup();
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "fn ")
            .with_param("glob", "*.rs");
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success, "result: {}", out.result);
        // Both .rs files contain "fn "; the .txt file is excluded by the glob.
        let lines: Vec<&str> = out.result.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 files, got: {}", out.result);
        assert!(out.result.contains("alpha.rs"));
        assert!(out.result.contains("gamma.rs"));
        assert!(!out.result.contains("beta.txt"));
    }

    #[test]
    fn grep_invalid_regex_errors() {
        let (tool, _dir) = setup();
        let engine = test_engine();
        // An unclosed group is an invalid regex.
        let input = ToolInput::new().with_param("pattern", "fn (");
        let out = tool.execute(&engine, input).unwrap();
        assert!(!out.success);
        assert!(
            out.result.starts_with("invalid regex:"),
            "result: {}",
            out.result
        );
    }

    #[test]
    fn grep_case_insensitive() {
        let (tool, _dir) = setup();
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "world")
            .with_value("case_insensitive", true)
            .with_param("output_mode", "count");
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success, "result: {}", out.result);
        // beta.txt now matches both "hello world" and "Goodbye World" → 2.
        assert!(
            out.result.contains("beta.txt: 2"),
            "result: {}",
            out.result
        );
        // gamma.rs matches "// world here" → 1.
        assert!(
            out.result.contains("gamma.rs: 1"),
            "result: {}",
            out.result
        );
    }

    #[test]
    fn grep_no_matches() {
        let (tool, _dir) = setup();
        let engine = test_engine();
        let input = ToolInput::new().with_param("pattern", "nonexistent_token_xyz");
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success);
        assert_eq!(out.result, "No matches");
    }
}
