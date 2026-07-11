//! Glob tool: find files matching a glob pattern through a sandboxed [`Filesystem`].
//!
//! Mirrors Claude Code's `Glob`: given a glob `pattern` (e.g. `"**/*.rs"` or
//! `"src/**/*.toml"`) and an optional base `path`, it recursively walks the base
//! directory through the injected [`Filesystem`] (which enforces sandbox
//! containment) and returns the paths of files whose base-relative path matches
//! the pattern.
//!
//! The walk goes exclusively through [`Filesystem::read_dir`], so every entry is
//! sandbox-checked; entries that fail to resolve (symlink escapes, races) are
//! skipped rather than aborting the walk. The glob is translated to an anchored
//! [`regex::Regex`] supporting `**` (any number of path segments, including
//! zero), `*` (any run of non-`/` characters), `?` (a single non-`/` character),
//! and literal segments with all other regex metacharacters escaped.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::Regex;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::backend::{Backend, InProcessBackend};
use crate::engine::Engine;
use crate::fs::Filesystem;

/// Maximum number of matches returned; beyond this the result notes truncation.
const MAX_MATCHES: usize = 1000;

/// Maximum directory depth walked, guarding against pathological/looping trees.
const MAX_DEPTH: usize = 64;

/// Find files matching a glob pattern through the session's [`Backend`]
/// (ADR 082), whose file verbs carry the sandboxed [`Filesystem`] contract.
pub struct GlobTool {
    /// The backend the walk is confined to. A backend wrapping a
    /// [`crate::fs::LocalFs::denied`] filesystem yields no readable roots
    /// (fail-closed).
    fs: Arc<dyn Backend>,
}

impl GlobTool {
    /// Create a GlobTool backed by the given sandboxed filesystem, walked
    /// in-process (the historic behavior, verbatim).
    pub fn new(fs: Arc<dyn Filesystem>) -> Self {
        Self::with_backend(Arc::new(InProcessBackend::new(fs)))
    }

    /// Create a GlobTool routed through an explicit backend — the seam where
    /// 48b/48c plug in privilege-dropped or remote targets.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self { fs: backend }
    }

    /// Resolve the base directory: an explicit `path` if given, otherwise the
    /// first read root, otherwise `"."`.
    fn base_dir(&self, path: Option<&str>) -> PathBuf {
        match path {
            Some(p) if !p.trim().is_empty() => PathBuf::from(p),
            _ => self
                .fs
                .read_roots()
                .first()
                .cloned()
                .unwrap_or_else(|| PathBuf::from(".")),
        }
    }
}

impl Tool for GlobTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "glob".into(),
            description: "Find files matching a glob pattern (e.g. \"**/*.rs\" or \
                          \"src/**/*.toml\") under a base directory."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "pattern".into(),
                    description: "Glob pattern: ** (any dirs), * (any chars except /), \
                                  ? (one char except /), plus literals."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "path".into(),
                    description: "Base directory to search (default: first read root, or \".\")."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let pattern = input.require("pattern", "glob")?;
        let base = self.base_dir(input.get("path"));

        // Translate the glob to an anchored regex; an invalid pattern is an
        // ordinary failure, not a hard error.
        let regex = match compile_glob(pattern) {
            Ok(re) => re,
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "Invalid glob pattern \"{pattern}\": {e}"
                )));
            }
        };

        // The base directory must be readable within the sandbox.
        if !self.fs.exists(&base) {
            return Ok(ToolOutput::err(format!(
                "Base directory \"{}\" does not exist or is outside the read sandbox.",
                base.display()
            )));
        }

        let mut matches: Vec<String> = Vec::new();
        let mut truncated = false;
        // The relative path is accumulated during traversal (prefix-free), so a
        // base that differs from the canonical form `read_dir` returns (e.g. the
        // macOS `/var`→`/private/var` symlink) does not break prefix stripping.
        self.walk(&base, "", 0, &regex, &mut matches, &mut truncated);

        matches.sort();

        if matches.is_empty() {
            return Ok(ToolOutput::ok(format!("No files matched {pattern}")));
        }

        let mut body = matches.join("\n");
        if truncated {
            body.push_str(&format!(
                "\n... [truncated at {MAX_MATCHES} matches]"
            ));
        }
        Ok(ToolOutput::ok(body))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "glob".into(),
            description: "Lists filesystem paths matching a glob — read-only directory traversal."
                .into(),
            parameters: vec![
                ToolParamSchema::required(
                    "pattern",
                    "Glob pattern (e.g. \"**/*.rs\", \"src/**/*.toml\").",
                ),
                ToolParamSchema::optional(
                    "path",
                    "Base directory to search (default: first read root, or \".\").",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::ReadFilesystem]),
                description: "Lists filesystem paths matching a glob — read-only traversal.".into(),
                shadow_triggers: vec!["glob".into(), "find".into(), "list".into()],
            },
            source: ToolSource::Native,
        }
    }

    fn locus(&self) -> Option<crate::backend::ExecLocus> {
        Some(self.fs.locus())
    }
}

impl GlobTool {
    /// Recursively walk `dir`, testing each file's path (relative to the walk
    /// root, accumulated in `rel_prefix`) against `regex`. Sandbox-denied or
    /// unreadable subtrees are skipped rather than aborting the whole walk.
    fn walk(
        &self,
        dir: &Path,
        rel_prefix: &str,
        depth: usize,
        regex: &Regex,
        matches: &mut Vec<String>,
        truncated: &mut bool,
    ) {
        if *truncated || depth > MAX_DEPTH {
            return;
        }
        let mut entries = match self.fs.read_dir(dir) {
            Ok(entries) => entries,
            // Unreadable/denied subtree — skip rather than abort the whole walk.
            Err(_) => return,
        };
        // Sort for deterministic traversal (final results are sorted anyway,
        // but this keeps truncation stable across runs).
        entries.sort();
        for entry in entries {
            if *truncated {
                return;
            }
            // Do not follow symlinks (matches find/ripgrep defaults): a
            // symlinked directory pointing back into the tree would otherwise
            // inflate results with bogus duplicate paths up to MAX_DEPTH, and a
            // symlinked leaf pointing outside the sandbox would be reported as a
            // match (path-existence disclosure).
            if self.fs.is_symlink(&entry) {
                continue;
            }
            // Build the relative path from the file name, not by stripping a
            // base prefix (which the canonicalizing sandbox can defeat). Skip
            // names that are not valid UTF-8 (a lossy name cannot be re-opened).
            let name = match entry.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let rel = if rel_prefix.is_empty() {
                name
            } else {
                format!("{rel_prefix}/{name}")
            };
            if entry.is_dir() {
                self.walk(&entry, &rel, depth + 1, regex, matches, truncated);
            } else if entry.is_file() && regex.is_match(&rel) {
                if matches.len() >= MAX_MATCHES {
                    *truncated = true;
                    return;
                }
                matches.push(rel);
            }
        }
    }
}

/// Translate a glob pattern into an anchored [`Regex`] matching a `/`-separated
/// relative path.
///
/// Supported syntax:
/// - `**` — any number of path segments, including zero (matches across `/`).
/// - `*` — any run of characters except `/`.
/// - `?` — a single character except `/`.
/// - any other character — matched literally (regex metacharacters escaped).
fn compile_glob(pattern: &str) -> Result<Regex, regex::Error> {
    let mut re = String::with_capacity(pattern.len() * 2 + 2);
    re.push('^');

    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    // `**` — consume the doubled star.
                    i += 1;
                    // Collapse an immediately following `/` so that `**/` also
                    // matches zero leading segments (e.g. `**/*.rs` matches
                    // `b.rs` at the root as well as `a/b.rs`).
                    if i + 1 < chars.len() && chars[i + 1] == '/' {
                        i += 1;
                        re.push_str("(?:.*/)?");
                    } else {
                        re.push_str(".*");
                    }
                } else {
                    // Single `*` — any run of non-separator characters.
                    re.push_str("[^/]*");
                }
            }
            '?' => re.push_str("[^/]"),
            '/' => re.push('/'),
            other => re.push_str(&regex::escape(&other.to_string())),
        }
        i += 1;
    }

    re.push('$');
    Regex::new(&re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;
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

    fn tool_for(root: &Path) -> GlobTool {
        GlobTool::new(Arc::new(LocalFs::sandbox(root)))
    }

    #[cfg(unix)]
    #[test]
    fn glob_does_not_follow_symlinks() {
        // A symlink pointing back into the tree must not be followed: otherwise
        // results inflate with bogus loop/loop/... duplicate paths (and can
        // crowd out real matches under the cap).
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.rs"), "fn a() {}").unwrap();
        std::os::unix::fs::symlink(root, root.join("loop")).unwrap();

        let tool = tool_for(root);
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "**/*.rs")
            .with_param("path", root.to_string_lossy().to_string());
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success, "result: {}", out.result);
        assert!(out.result.contains("a.rs"), "should find a.rs: {}", out.result);
        assert!(
            !out.result.contains("loop/"),
            "must not follow the symlink into loop/...: {}",
            out.result
        );
    }

    #[test]
    fn glob_matches_extension() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/b.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("c.txt"), "text").unwrap();

        let tool = tool_for(root);
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "**/*.rs")
            .with_param("path", root.to_string_lossy().to_string());
        let out = tool.execute(&engine, input).unwrap();

        assert!(out.success, "result: {}", out.result);
        assert!(
            out.result.contains("a/b.rs"),
            "expected a/b.rs in: {}",
            out.result
        );
        assert!(
            !out.result.contains("c.txt"),
            "should not match c.txt: {}",
            out.result
        );
    }

    #[test]
    fn glob_no_match_returns_message() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("only.txt"), "text").unwrap();

        let tool = tool_for(root);
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "**/*.rs")
            .with_param("path", root.to_string_lossy().to_string());
        let out = tool.execute(&engine, input).unwrap();

        assert!(out.success, "no-match is an ordinary ok result");
        assert!(
            out.result.contains("No files matched"),
            "expected no-match message, got: {}",
            out.result
        );
    }

    #[test]
    fn glob_double_star_nested() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("src/deep/nested")).unwrap();
        std::fs::write(root.join("src/deep/nested/x.toml"), "a = 1").unwrap();
        std::fs::write(root.join("src/top.toml"), "b = 2").unwrap();
        std::fs::write(root.join("src/deep/nested/y.rs"), "fn y() {}").unwrap();

        let tool = tool_for(root);
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("pattern", "src/**/*.toml")
            .with_param("path", root.to_string_lossy().to_string());
        let out = tool.execute(&engine, input).unwrap();

        assert!(out.success, "result: {}", out.result);
        assert!(
            out.result.contains("src/deep/nested/x.toml"),
            "expected deeply nested toml in: {}",
            out.result
        );
        assert!(
            out.result.contains("src/top.toml"),
            "** should also match zero intermediate dirs: {}",
            out.result
        );
        assert!(
            !out.result.contains("y.rs"),
            "should not match the .rs file: {}",
            out.result
        );
    }

    #[test]
    fn glob_default_base_is_first_read_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("hello.rs"), "fn h() {}").unwrap();

        let tool = tool_for(root);
        let engine = test_engine();
        // No `path` param → falls back to the first read root.
        let input = ToolInput::new().with_param("pattern", "*.rs");
        let out = tool.execute(&engine, input).unwrap();

        assert!(out.success, "result: {}", out.result);
        assert!(
            out.result.contains("hello.rs"),
            "expected hello.rs using default base, got: {}",
            out.result
        );
    }

    #[test]
    fn glob_single_star_does_not_cross_slash() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("top.rs"), "fn t() {}").unwrap();
        std::fs::write(root.join("sub/inner.rs"), "fn i() {}").unwrap();

        let tool = tool_for(root);
        let engine = test_engine();
        // `*.rs` (single star) matches only the top level.
        let input = ToolInput::new()
            .with_param("pattern", "*.rs")
            .with_param("path", root.to_string_lossy().to_string());
        let out = tool.execute(&engine, input).unwrap();

        assert!(out.success, "result: {}", out.result);
        assert!(out.result.contains("top.rs"));
        assert!(
            !out.result.contains("inner.rs"),
            "single * must not cross /: {}",
            out.result
        );
    }
}
