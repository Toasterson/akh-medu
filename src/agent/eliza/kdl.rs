//! ELIZA-style conversational script engine with hot-reload support (PR #178).
//!
//! Scripts are defined in KDL format (`.kdl` files) and can be edited and
//! reloaded without restarting akhomed. Each file may contain one or more
//! top-level `script` nodes. Pattern matching uses Rust regex; response
//! templates reference capture groups as `{0}`, `{1}`, …
//!
//! This module is gated behind the `eliza` cargo feature (which pulls in the
//! `knuffel` KDL parser and the `notify` filesystem watcher). It coexists with
//! the always-available TOML-script core engine in [`super::engine`] /
//! [`super::script`]; the two use disjoint type names.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use miette::Diagnostic;
use regex::Regex;
use thiserror::Error;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug, Error, Diagnostic)]
pub enum ElizaError {
    #[error("KDL parse error in {path}: {message}")]
    #[diagnostic(
        code(akh::eliza::parse),
        help("Fix the KDL syntax in the script file and save again.")
    )]
    Parse { path: String, message: String },

    #[error("invalid regex pattern \"{pattern}\" in script \"{script}\": {message}")]
    #[diagnostic(
        code(akh::eliza::regex),
        help("ELIZA patterns use Rust regex syntax. See https://docs.rs/regex for reference.")
    )]
    InvalidPattern {
        script: String,
        pattern: String,
        message: String,
    },

    #[error("template directory not found: {path}")]
    #[diagnostic(
        code(akh::eliza::no_dir),
        help("Create the directory or configure the correct templates path.")
    )]
    NoTemplateDir { path: String },

    #[error("IO error reading {path}: {source}")]
    #[diagnostic(code(akh::eliza::io))]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[cfg(feature = "daemon")]
    #[error("filesystem watcher error: {message}")]
    #[diagnostic(
        code(akh::eliza::watcher),
        help(
            "Check inotify/kqueue resource limits. \
             On Linux, try: sysctl fs.inotify.max_user_watches=524288"
        )
    )]
    Watcher { message: String },
}

pub type ElizaResult<T> = Result<T, ElizaError>;

// ── KDL data types (parsed by knuffel) ───────────────────────────────────────

/// Top-level `script` node in a `.kdl` file.
#[derive(Debug, knuffel::Decode)]
pub struct ScriptNode {
    /// Unique script name, used as the lookup key.
    #[knuffel(property)]
    pub name: String,
    /// Display name for the persona; defaults to `name`.
    #[knuffel(property)]
    pub persona: Option<String>,
    /// Dispatch priority — higher wins when multiple scripts match (default 50).
    #[knuffel(property)]
    pub priority: Option<u32>,
    /// Ordered list of rules evaluated top-to-bottom.
    #[knuffel(children(name = "rule"))]
    pub rules: Vec<RuleNode>,
}

/// A single `rule` node within a `script`.
#[derive(Debug, knuffel::Decode)]
pub struct RuleNode {
    /// Rust regex pattern. Absent means "match anything" (fallback rule).
    #[knuffel(property)]
    pub pattern: Option<String>,
    /// Response templates for this rule, cycled round-robin.
    #[knuffel(children(name = "response"))]
    pub responses: Vec<ResponseNode>,
}

/// A single `response` node — holds one template string.
#[derive(Debug, knuffel::Decode)]
pub struct ResponseNode {
    /// Template string; `{N}` is replaced by capture group N.
    #[knuffel(argument)]
    pub template: String,
}

// ── Compiled rule ─────────────────────────────────────────────────────────────

/// A rule compiled from a [`RuleNode`] with the pattern pre-compiled to a `Regex`.
#[derive(Debug)]
pub struct CompiledRule {
    /// Pre-compiled pattern; `None` means "match anything".
    pub pattern: Option<Regex>,
    /// Response templates, rotated round-robin on each match.
    pub responses: Vec<String>,
    /// Index of the next response to emit.
    next: AtomicUsize,
}

impl CompiledRule {
    fn compile(node: &RuleNode, script_name: &str) -> ElizaResult<Self> {
        let pattern = match &node.pattern {
            Some(raw) => {
                let re = Regex::new(raw).map_err(|e| ElizaError::InvalidPattern {
                    script: script_name.to_string(),
                    pattern: raw.clone(),
                    message: e.to_string(),
                })?;
                Some(re)
            }
            None => None,
        };
        Ok(Self {
            pattern,
            responses: node.responses.iter().map(|r| r.template.clone()).collect(),
            next: AtomicUsize::new(0),
        })
    }

    /// Try to match `input`. Returns a response with `{N}` substitutions applied,
    /// or `None` if the pattern does not match.
    pub fn apply(&self, input: &str) -> Option<String> {
        match &self.pattern {
            Some(re) => {
                let caps = re.captures(input)?;
                let template = self.pick_response()?;
                Some(substitute_captures(template, &caps))
            }
            None => self.pick_response().map(|t| t.to_string()),
        }
    }

    fn pick_response(&self) -> Option<&str> {
        if self.responses.is_empty() {
            return None;
        }
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.responses.len();
        Some(&self.responses[idx])
    }
}

/// Replace `{N}` placeholders in `template` with the corresponding regex capture group.
fn substitute_captures(template: &str, caps: &regex::Captures<'_>) -> String {
    let mut out = template.to_string();
    for i in 0..caps.len() {
        if let Some(m) = caps.get(i) {
            out = out.replace(&format!("{{{i}}}"), m.as_str());
        }
    }
    out
}

// ── Compiled script ───────────────────────────────────────────────────────────

/// A script compiled from a [`ScriptNode`] with all patterns pre-compiled.
#[derive(Debug)]
pub struct CompiledScript {
    pub name: String,
    pub persona: String,
    pub priority: u32,
    rules: Vec<CompiledRule>,
}

impl CompiledScript {
    /// Compile a [`ScriptNode`] parsed from KDL.
    pub fn compile(node: ScriptNode) -> ElizaResult<Self> {
        let rules = node
            .rules
            .iter()
            .map(|r| CompiledRule::compile(r, &node.name))
            .collect::<ElizaResult<Vec<_>>>()?;
        Ok(Self {
            persona: node.persona.clone().unwrap_or_else(|| node.name.clone()),
            priority: node.priority.unwrap_or(50),
            rules,
            name: node.name,
        })
    }

    /// Return the first matching rule's response, or `None` if nothing matches.
    pub fn match_input(&self, input: &str) -> Option<String> {
        self.rules.iter().find_map(|r| r.apply(input))
    }
}

// ── Script bank ───────────────────────────────────────────────────────────────

/// A collection of [`CompiledScript`]s, keyed by script name.
///
/// Tracks which file each script came from so the hot-reload watcher can
/// atomically replace per-file scripts when a `.kdl` file changes.
#[derive(Debug, Default)]
pub struct ScriptBank {
    scripts: HashMap<String, CompiledScript>,
    /// file path → list of script names loaded from that file.
    file_sources: HashMap<PathBuf, Vec<String>>,
}

impl ScriptBank {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert all scripts from `path`; removes any previously loaded from the same path.
    pub fn replace_from_file(&mut self, path: &Path, scripts: Vec<CompiledScript>) {
        // Drop stale scripts from this file.
        if let Some(old_names) = self.file_sources.remove(path) {
            for name in old_names {
                self.scripts.remove(&name);
            }
        }
        // Insert new scripts.
        let new_names: Vec<String> = scripts.iter().map(|s| s.name.clone()).collect();
        for script in scripts {
            self.scripts.insert(script.name.clone(), script);
        }
        self.file_sources.insert(path.to_path_buf(), new_names);
    }

    /// Remove all scripts associated with `path`.
    pub fn remove_file(&mut self, path: &Path) {
        if let Some(names) = self.file_sources.remove(path) {
            for name in names {
                self.scripts.remove(&name);
            }
        }
    }

    /// Match input against all scripts, returning the highest-priority match.
    pub fn match_input(&self, input: &str) -> Option<String> {
        let mut scripts: Vec<&CompiledScript> = self.scripts.values().collect();
        scripts.sort_by_key(|s| std::cmp::Reverse(s.priority));
        scripts.iter().find_map(|s| s.match_input(input))
    }

    /// Names of all loaded scripts.
    pub fn script_names(&self) -> Vec<&str> {
        self.scripts.keys().map(String::as_str).collect()
    }

    /// Number of loaded scripts.
    pub fn len(&self) -> usize {
        self.scripts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scripts.is_empty()
    }
}

// ── Script engine ─────────────────────────────────────────────────────────────

/// Thread-safe ELIZA engine backed by an `Arc<RwLock<ScriptBank>>`.
///
/// Clone the engine cheaply to share across threads — the hot-reload task
/// mutates the inner bank while readers call [`ScriptEngine::match_input`].
#[derive(Clone, Debug)]
pub struct ScriptEngine {
    bank: Arc<RwLock<ScriptBank>>,
}

impl ScriptEngine {
    pub fn new(bank: ScriptBank) -> Self {
        Self {
            bank: Arc::new(RwLock::new(bank)),
        }
    }

    /// Match user input against loaded scripts. Returns `None` if no rule fires.
    pub fn match_input(&self, input: &str) -> Option<String> {
        self.bank.read().ok()?.match_input(input)
    }

    /// Expose the inner bank `Arc` for the hot-reload watcher.
    pub fn bank_arc(&self) -> Arc<RwLock<ScriptBank>> {
        Arc::clone(&self.bank)
    }

    /// Names of all currently loaded scripts.
    pub fn script_names(&self) -> Vec<String> {
        self.bank
            .read()
            .ok()
            .map(|b| b.script_names().into_iter().map(String::from).collect())
            .unwrap_or_default()
    }
}

// ── Parsing ───────────────────────────────────────────────────────────────────

/// Parse a KDL source string into compiled scripts.
///
/// A single `.kdl` file may declare multiple top-level `script` nodes.
/// On parse failure the error is returned and the caller decides whether
/// to log-and-skip or propagate.
pub fn parse_scripts(path: &Path, source: &str) -> ElizaResult<Vec<CompiledScript>> {
    let file_name = path.display().to_string();
    let nodes: Vec<ScriptNode> =
        knuffel::parse(&file_name, source).map_err(|e| ElizaError::Parse {
            path: file_name.clone(),
            message: format!("{e}"),
        })?;
    nodes.into_iter().map(CompiledScript::compile).collect()
}

/// Load all `.kdl` files from `dir` into a [`ScriptBank`].
///
/// Files that fail to parse emit a `tracing::warn!` and are skipped so a
/// single invalid script does not block the others from loading.
pub fn load_directory(dir: &Path) -> ElizaResult<ScriptBank> {
    if !dir.is_dir() {
        return Err(ElizaError::NoTemplateDir {
            path: dir.display().to_string(),
        });
    }

    let entries = std::fs::read_dir(dir).map_err(|e| ElizaError::Io {
        path: dir.display().to_string(),
        source: e,
    })?;

    let mut bank = ScriptBank::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("kdl") {
            continue;
        }
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("eliza: cannot read {}: {e}", path.display());
                continue;
            }
        };
        match parse_scripts(&path, &source) {
            Ok(scripts) => bank.replace_from_file(&path, scripts),
            Err(e) => tracing::warn!("eliza: skipping {}: {e}", path.display()),
        }
    }

    Ok(bank)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rule(pattern: Option<&str>, responses: &[&str]) -> RuleNode {
        RuleNode {
            pattern: pattern.map(String::from),
            responses: responses
                .iter()
                .map(|t| ResponseNode {
                    template: t.to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn test_fallback_rule_matches_anything() {
        let rule = CompiledRule::compile(&make_rule(None, &["Tell me more."]), "test").unwrap();
        assert_eq!(rule.apply("anything at all"), Some("Tell me more.".into()));
    }

    #[test]
    fn test_pattern_match_and_capture_substitution() {
        let rule = CompiledRule::compile(
            &make_rule(Some(r"(?i)I feel (.+)"), &["Why do you feel {1}?"]),
            "test",
        )
        .unwrap();
        assert_eq!(
            rule.apply("I feel anxious"),
            Some("Why do you feel anxious?".into())
        );
        assert!(rule.apply("The weather is nice").is_none());
    }

    #[test]
    fn test_round_robin_responses() {
        let rule = CompiledRule::compile(
            &make_rule(None, &["Response A", "Response B", "Response C"]),
            "test",
        )
        .unwrap();
        assert_eq!(rule.apply("x").unwrap(), "Response A");
        assert_eq!(rule.apply("x").unwrap(), "Response B");
        assert_eq!(rule.apply("x").unwrap(), "Response C");
        assert_eq!(rule.apply("x").unwrap(), "Response A"); // wraps
    }

    #[test]
    fn test_script_rule_priority() {
        let specific = RuleNode {
            pattern: Some(r"(?i)I need (.+)".into()),
            responses: vec![ResponseNode {
                template: "specific: {1}".into(),
            }],
        };
        let fallback = RuleNode {
            pattern: None,
            responses: vec![ResponseNode {
                template: "fallback".into(),
            }],
        };
        let node = ScriptNode {
            name: "test".into(),
            persona: None,
            priority: None,
            rules: vec![specific, fallback],
        };
        let script = CompiledScript::compile(node).unwrap();
        assert_eq!(
            script.match_input("I need coffee"),
            Some("specific: coffee".into())
        );
        assert_eq!(script.match_input("Hello"), Some("fallback".into()));
    }

    #[test]
    fn test_parse_scripts_from_kdl() {
        let kdl = r#"
script name="test" persona="Tester" priority=10 {
    rule pattern="(?i)hello" {
        response "Hi!"
    }
    rule {
        response "Tell me more."
    }
}
"#;
        let path = Path::new("test.kdl");
        let scripts = parse_scripts(path, kdl).unwrap();
        assert_eq!(scripts.len(), 1);
        assert_eq!(scripts[0].name, "test");
        assert_eq!(scripts[0].match_input("hello"), Some("Hi!".into()));
        assert_eq!(
            scripts[0].match_input("anything"),
            Some("Tell me more.".into())
        );
    }

    #[test]
    fn test_invalid_kdl_returns_error() {
        let bad_kdl = "this is not valid kdl {{{";
        let result = parse_scripts(Path::new("bad.kdl"), bad_kdl);
        assert!(result.is_err());
    }

    #[test]
    fn test_script_bank_replace_from_file() {
        let mut bank = ScriptBank::new();
        let path = PathBuf::from("test.kdl");

        let script1 = CompiledScript {
            name: "alpha".into(),
            persona: "Alpha".into(),
            priority: 50,
            rules: vec![],
        };
        bank.replace_from_file(&path, vec![script1]);
        assert_eq!(bank.len(), 1);

        let script2 = CompiledScript {
            name: "beta".into(),
            persona: "Beta".into(),
            priority: 50,
            rules: vec![],
        };
        bank.replace_from_file(&path, vec![script2]);
        assert_eq!(bank.len(), 1); // alpha was replaced
        assert!(bank.script_names().contains(&"beta"));
    }
}
