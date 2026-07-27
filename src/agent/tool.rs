//! Tool system: trait-based tools with runtime registration.
//!
//! Tools are the agent's interface to the engine's capabilities.
//! Each tool implements the [`Tool`] trait and is registered in a [`ToolRegistry`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::audit::{AuditEntry, AuditKind, AuditLedger};
use crate::engine::Engine;

use super::error::{AgentError, AgentResult};
use super::tool_manifest::ToolManifest;

/// Description of a tool's interface.
#[derive(Debug, Clone)]
pub struct ToolSignature {
    /// Unique name of the tool.
    pub name: String,
    /// What this tool does.
    pub description: String,
    /// Parameters the tool accepts.
    pub parameters: Vec<ToolParam>,
}

/// A single parameter in a tool's signature.
#[derive(Debug, Clone)]
pub struct ToolParam {
    /// Parameter name.
    pub name: String,
    /// What this parameter controls.
    pub description: String,
    /// Whether this parameter must be provided.
    pub required: bool,
}

/// Input to a tool execution.
///
/// Values are [`serde_json::Value`], so a parameter can be a string, number,
/// bool, array, or nested object. The string-based API ([`with_param`](Self::with_param),
/// [`get`](Self::get), [`require`](Self::require)) is preserved verbatim —
/// `with_param` stores a JSON string and `get`/`require` return the borrowed
/// `&str` of a JSON-string value — so the existing ~285 call sites and the flat
/// `{string:string}` WASM/CLI wire contracts are unchanged. Typed/structured
/// accessors ([`with_value`](Self::with_value), [`get_bool`](Self::get_bool),
/// [`get_i64`](Self::get_i64), [`get_str_list`](Self::get_str_list),
/// [`get_value`](Self::get_value), …) are additive, for tools whose inputs are
/// genuinely structured (e.g. the task-list tool's array of task objects).
#[derive(Debug, Clone, Default)]
pub struct ToolInput {
    /// Named parameters, each a JSON value.
    pub params: serde_json::Map<String, serde_json::Value>,
}

impl ToolInput {
    /// Create a new empty input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a string parameter (back-compatible: stores a JSON string).
    pub fn with_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.params
            .insert(name.into(), serde_json::Value::String(value.into()));
        self
    }

    /// Add a parameter with an arbitrary JSON value (number, bool, array,
    /// nested object). Use this for structured tool inputs.
    pub fn with_value(
        mut self,
        name: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.params.insert(name.into(), value.into());
        self
    }

    /// Get a parameter as a string slice.
    ///
    /// Returns `Some` only when the value is a JSON string — preserving the
    /// historical string-map semantics for the existing `get`/`require` callers.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.params.get(name).and_then(|v| v.as_str())
    }

    /// Get a required string parameter, returning an error if missing.
    pub fn require(&self, name: &str, tool_name: &str) -> AgentResult<&str> {
        self.get(name).ok_or(AgentError::ToolExecution {
            tool_name: tool_name.into(),
            message: format!("missing required parameter: {name}"),
        })
    }

    /// Get the raw JSON value for a parameter.
    pub fn get_value(&self, name: &str) -> Option<&serde_json::Value> {
        self.params.get(name)
    }

    /// Get a boolean parameter, accepting either a JSON bool or the legacy
    /// string encodings (`"true"`/`"1"`/`"yes"` / `"false"`/`"0"`/`"no"`,
    /// case-insensitive). Returns `None` for any other shape.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.params.get(name)? {
            serde_json::Value::Bool(b) => Some(*b),
            serde_json::Value::String(s) => {
                let s = s.trim();
                if s.eq_ignore_ascii_case("true") || s == "1" || s.eq_ignore_ascii_case("yes") {
                    Some(true)
                } else if s.eq_ignore_ascii_case("false")
                    || s == "0"
                    || s.eq_ignore_ascii_case("no")
                {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get an integer parameter, accepting a JSON number or a numeric string.
    pub fn get_i64(&self, name: &str) -> Option<i64> {
        match self.params.get(name)? {
            serde_json::Value::Number(n) => n.as_i64(),
            serde_json::Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    /// Get an unsigned integer parameter, accepting a JSON number or string.
    pub fn get_u64(&self, name: &str) -> Option<u64> {
        match self.params.get(name)? {
            serde_json::Value::Number(n) => n.as_u64(),
            serde_json::Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    /// Get a float parameter, accepting a JSON number or numeric string.
    pub fn get_f64(&self, name: &str) -> Option<f64> {
        match self.params.get(name)? {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    /// Get a list of strings, accepting either a JSON string array or a single
    /// comma-separated string (the legacy encoding used by `tags`,
    /// `query_symbols`, `subjects`). Entries are trimmed and empties dropped;
    /// returns an empty vec when the parameter is absent.
    pub fn get_str_list(&self, name: &str) -> Vec<String> {
        match self.params.get(name) {
            Some(serde_json::Value::Array(items)) => items
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            Some(serde_json::Value::String(s)) => s
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect(),
            _ => Vec::new(),
        }
    }
}

impl From<HashMap<String, String>> for ToolInput {
    /// Build an input from a flat string map (e.g. persisted trigger params),
    /// wrapping each value as a JSON string.
    fn from(map: HashMap<String, String>) -> Self {
        let params = map
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();
        Self { params }
    }
}

/// Output from a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// Whether the tool succeeded.
    pub success: bool,
    /// Human-readable result summary.
    pub result: String,
    /// Symbols involved in the operation (for WM linking).
    pub symbols_involved: Vec<crate::symbol::SymbolId>,
}

impl ToolOutput {
    /// Create a successful output.
    pub fn ok(result: impl Into<String>) -> Self {
        Self {
            success: true,
            result: result.into(),
            symbols_involved: Vec::new(),
        }
    }

    /// Create a successful output with associated symbols.
    pub fn ok_with_symbols(
        result: impl Into<String>,
        symbols: Vec<crate::symbol::SymbolId>,
    ) -> Self {
        Self {
            success: true,
            result: result.into(),
            symbols_involved: symbols,
        }
    }

    /// Create a failed output.
    pub fn err(result: impl Into<String>) -> Self {
        Self {
            success: false,
            result: result.into(),
            symbols_involved: Vec::new(),
        }
    }
}

/// A tool the agent can execute.
pub trait Tool: Send + Sync {
    /// Describe this tool's interface.
    fn signature(&self) -> ToolSignature;

    /// Execute the tool with the given input against the engine.
    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput>;

    /// Return structured manifest describing this tool's danger metadata,
    /// capabilities, and source.
    fn manifest(&self) -> ToolManifest;

    /// Whether this tool is advertised in the LLM's agentic tool catalog
    /// (ADR 076 §1). Defaults to `true`. A tool overrides this to `false` when
    /// the model reasons better without it — e.g. a raw-text ingester the LLM
    /// should bypass by reading documents with `file_io` and emitting typed
    /// structured data to the typed writer tools (`memory_write`, `kg_mutate`)
    /// instead. Such tools stay registered (and dispatchable) for
    /// operator/CLI/daemon-ingest contexts — only the catalog hides them.
    /// No built-in currently overrides this; the seam remains for that pattern
    /// (the former raw-text ingesters that used it were removed outright).
    fn advertise_to_llm(&self) -> bool {
        true
    }

    /// The [`crate::backend::ExecLocus`] of the `Backend` this tool executed
    /// through, if any (ADR 082 §7 / Phase 48f audit envelope). `None` for
    /// tools with no backend at all (most of them — KG queries, planning,
    /// etc.); the five executor-facing tools (`shell_exec`, `file_io`,
    /// `edit`, `glob`, `grep`) override this to report the locus of their
    /// injected `Backend` so [`ToolRegistry::execute`]'s audit entry can
    /// carry it. Deliberately *not* threaded through `Engine`/`ToolRegistry`
    /// as a separate lookup — the backend is already private state on these
    /// five tool structs, and this one default-`None` method is the surgical
    /// way to surface it without a wider refactor.
    fn locus(&self) -> Option<crate::backend::ExecLocus> {
        None
    }
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    manifests: HashMap<String, ToolManifest>,
    /// Names of tools suppressed from the LLM catalog (`advertise_to_llm()` is
    /// `false`). They stay registered and dispatchable; only the catalog hides
    /// them. Recorded at registration so the catalog filter is a cheap lookup.
    unadvertised: std::collections::HashSet<String>,
    audit_ledger: Option<Arc<AuditLedger>>,
    /// Permission policy consulted before every dispatch. Defaults to
    /// [`PermissionMode::Bypass`] (allow all) so a standalone akh is unaffected.
    policy: super::permission::PermissionPolicy,
    /// Optional approver for [`PermissionDecision::Ask`] tools (e.g. an ACP
    /// client bridge). With none attached, `Ask` fails closed.
    asker: Option<Arc<dyn super::permission::PermissionAsker>>,
    /// Optional execution tracker for emitting ToolCall events over ACP.
    tracker: Option<Arc<super::tool_execution_tracker::ToolExecutionTracker>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            manifests: HashMap::new(),
            unadvertised: std::collections::HashSet::new(),
            audit_ledger: None,
            policy: super::permission::PermissionPolicy::default(),
            asker: None,
            tracker: None,
        }
    }

    /// Attach an audit ledger for automatic tool invocation logging.
    pub fn set_audit_ledger(&mut self, ledger: Arc<AuditLedger>) {
        self.audit_ledger = Some(ledger);
    }

    /// Attach a tool execution tracker for emitting ToolCall events (for ACP).
    pub fn set_tracker(&mut self, tracker: Arc<super::tool_execution_tracker::ToolExecutionTracker>) {
        self.tracker = Some(tracker);
    }

    /// Remove any attached tracker.
    pub fn clear_tracker(&mut self) {
        self.tracker = None;
    }

    /// Set the permission mode gating tool execution.
    pub fn set_permission_mode(&mut self, mode: super::permission::PermissionMode) {
        self.policy = super::permission::PermissionPolicy::new(mode);
    }

    /// The active permission mode.
    pub fn permission_mode(&self) -> super::permission::PermissionMode {
        self.policy.mode
    }

    /// Attach an approver that resolves `Ask` decisions (e.g. an ACP
    /// `session/request_permission` bridge).
    pub fn set_permission_asker(&mut self, asker: Arc<dyn super::permission::PermissionAsker>) {
        self.asker = Some(asker);
    }

    /// Remove any attached approver. `Ask` decisions then fail closed (deny).
    pub fn clear_permission_asker(&mut self) {
        self.asker = None;
    }

    /// Register a tool. Caches its manifest on registration.
    /// If a tool with the same name exists, it is replaced.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let manifest = tool.manifest();
        let name = manifest.name.clone();
        if tool.advertise_to_llm() {
            self.unadvertised.remove(&name);
        } else {
            self.unadvertised.insert(name.clone());
        }
        self.manifests.insert(name.clone(), manifest);
        self.tools.insert(name, tool);
    }

    /// Remove a previously-registered tool by name. Returns `true` if a tool was
    /// removed. Used to tear down session-scoped tools (e.g. MCP servers pushed
    /// over ACP at `session/new`) after a prompt turn completes.
    pub fn remove(&mut self, name: &str) -> bool {
        self.manifests.remove(name);
        self.unadvertised.remove(name);
        self.tools.remove(name).is_some()
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// Get a tool's manifest by name.
    pub fn manifest(&self, name: &str) -> Option<&ToolManifest> {
        self.manifests.get(name)
    }

    /// List all registered tool manifests.
    pub fn list_manifests(&self) -> Vec<&ToolManifest> {
        self.manifests.values().collect()
    }

    /// Whether a tool is advertised to the LLM catalog (ADR 076 §1). Unknown
    /// names are treated as advertised (the default).
    pub fn is_advertised(&self, name: &str) -> bool {
        !self.unadvertised.contains(name)
    }

    /// List manifests for tools advertised to the LLM catalog — i.e. excluding
    /// those whose `advertise_to_llm()` returned `false`.
    pub fn list_advertised_manifests(&self) -> Vec<&ToolManifest> {
        self.manifests
            .values()
            .filter(|m| !self.unadvertised.contains(&m.name))
            .collect()
    }

    /// List all registered tool signatures.
    pub fn list(&self) -> Vec<ToolSignature> {
        self.tools.values().map(|t| t.signature()).collect()
    }

    /// Would executing `name` right now stop at an **Ask** permission prompt?
    ///
    /// Used by the agentic dispatch loop's repair-retry (ADR 089 task A): a
    /// retry re-runs the whole permission gate, so an `Ask`-gated tool that was
    /// approved once and then failed structurally would prompt the user a
    /// *second* time (and double-log the audit ledger) for one logical call.
    /// The retry path skips such tools instead — the first approved failure
    /// stands, exactly as it would without repair.
    pub fn would_ask(&self, name: &str) -> bool {
        self.manifests.get(name).is_some_and(|m| {
            matches!(
                self.policy.decide(m),
                super::permission::PermissionDecision::Ask(_)
            )
        })
    }

    /// Execute a tool by name.
    ///
    /// If an audit ledger is attached, records a `ToolInvocation` audit entry
    /// with timing, truncated params/output, and success status.
    /// Audit failures never block the tool call.
    #[tracing::instrument(skip(self, input, engine), fields(tool = %name))]
    pub fn execute(
        &self,
        name: &str,
        input: ToolInput,
        engine: &Engine,
    ) -> AgentResult<ToolOutput> {
        let tool = self
            .get(name)
            .ok_or_else(|| AgentError::ToolNotFound { name: name.into() })?;

        // Permission gate (no-op under the default Bypass mode). A denied or
        // unapproved tool returns a failed ToolOutput rather than Err, so the
        // OODA loop's tool-error isolation is preserved.
        if let Some(manifest) = self.manifests.get(name) {
            use super::permission::PermissionDecision;
            match self.policy.decide(manifest) {
                PermissionDecision::Allow => {}
                PermissionDecision::Deny(reason) => {
                    return Ok(ToolOutput::err(format!("permission denied: {reason}")));
                }
                PermissionDecision::Ask(reason) => {
                    let summary = truncate_summary(&format!("{:?}", input.params), 200);
                    let approved = self
                        .asker
                        .as_ref()
                        .map(|a| a.ask(manifest, &summary))
                        .unwrap_or(false);
                    if !approved {
                        return Ok(ToolOutput::err(format!(
                            "permission required but not granted: {reason}"
                        )));
                    }
                }
            }
        }

        // Start tracking this tool execution for ACP ToolCall events.
        let guard = self.tracker.as_ref().map(|tracker| {
            let title = build_tool_title(name, &input);
            let locations = extract_tool_locations(name, &input);
            tracker.start_tool(name, title, locations)
        });

        let start = std::time::Instant::now();
        let result = tool.execute(engine, input.clone());
        let duration_ms = start.elapsed().as_millis() as u64;

        // Complete tracking (success or failure) if we started one.
        if let Some(guard) = guard {
            let (success, output, diff) = match &result {
                Ok(out) => {
                    let diff = extract_diff(name, &input, out);
                    (out.success, Some(truncate_summary(&out.result, 200)), diff)
                }
                Err(e) => (false, Some(truncate_summary(&e.to_string(), 200)), None),
            };
            guard.complete(success, output, diff);
        }

        // Best-effort audit logging — never fail the tool call.
        if let Some(ref ledger) = self.audit_ledger {
            let params_summary = truncate_summary(&format!("{:?}", input.params), 200);
            let (success, output_summary) = match &result {
                Ok(output) => (output.success, truncate_summary(&output.result, 200)),
                Err(e) => (false, truncate_summary(&e.to_string(), 200)),
            };
            // ADR 082 §7 audit envelope: which Backend (and where/as whom)
            // ran this tool, if it's one of the backend-backed executor
            // tools. `exit_code` stays `None` — `ToolOutput` has no exit-code
            // field today (only `shell_exec` even has one to report), and
            // adding one is out of scope for this surgical wiring.
            let envelope = tool
                .locus()
                .map(|locus| crate::audit::ExecEnvelope::from_locus(&locus, None))
                .unwrap_or_default();

            let mut entry = AuditEntry::new(
                AuditKind::ToolInvocation {
                    tool_name: name.to_string(),
                    params_summary,
                    success,
                    output_summary,
                    duration_ms,
                    envelope,
                },
                "default",
            );
            let _ = ledger.append(&mut entry);
        }

        result
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Truncate a string for audit summaries, appending "..." if cut.
///
/// Truncation lands on a UTF-8 char boundary: audit summaries include tool
/// output (web/file content) and params that can place a multibyte character
/// across byte `max_len`, where a naive `&s[..max_len]` would panic — and this
/// runs in the registry's audit wrapper around *every* tool call.
fn truncate_summary(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Build a concise title for a tool execution (for ToolCall events).
fn build_tool_title(tool_name: &str, input: &ToolInput) -> String {
    match tool_name {
        "shell_exec" => {
            input.get("command").map(|c| truncate_summary(c, 80)).unwrap_or_else(|| "shell".to_string())
        }
        "file_io" => {
            let path = input.get("path").unwrap_or("?");
            let op = input.get("operation").unwrap_or("?");
            format!("{op} {path}")
        }
        "edit" => {
            let path = input.get("path").unwrap_or("?");
            format!("edit {path}")
        }
        "web_fetch" => {
            let url = input.get("url").unwrap_or("?");
            format!("fetch {url}")
        }
        "seshat_search" | "search_library" => {
            input.get("query").map(|q| truncate_summary(q, 80)).unwrap_or_else(|| "search".to_string())
        }
        "kg_query" | "query_symbols" => {
            input.get("pattern").or_else(|| input.get("query"))
                .map(|q| truncate_summary(q, 80))
                .unwrap_or_else(|| "query".to_string())
        }
        _ => {
            // Generic fallback: show first param value or just the tool name.
            input.params.values().next()
                .and_then(|v| v.as_str())
                .map(|s| truncate_summary(s, 80))
                .unwrap_or_else(|| tool_name.to_string())
        }
    }
}

/// Extract location information for tools that operate on files/paths.
fn extract_tool_locations(tool_name: &str, input: &ToolInput) -> Vec<String> {
    match tool_name {
        "file_io" | "edit" | "glob" | "grep" => {
            if let Some(path) = input.get("path") {
                vec![path.to_string()]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

/// Extract diff information for edit operations.
fn extract_diff(tool_name: &str, _input: &ToolInput, output: &ToolOutput) -> Option<String> {
    if tool_name == "edit" && output.success {
        // The edit tool's output is typically the diff.
        // For now, we don't parse it further — just truncate.
        if output.result.contains("@@") || output.result.contains("---") {
            Some(truncate_summary(&output.result, 500))
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::agent::tool_manifest::{Capability, DangerInfo, DangerLevel, ToolSource};

    #[test]
    fn truncate_summary_handles_multibyte_at_boundary() {
        // 100 × "あ" = 300 bytes; byte 200 lands inside a 3-byte char.
        // A naive &s[..200] would panic — this must not.
        let s = "あ".repeat(100);
        assert!(!s.is_char_boundary(200));
        let out = truncate_summary(&s, 200);
        assert!(out.ends_with("..."));
        // Truncated body is valid UTF-8 and shorter than the cap.
        let body = out.trim_end_matches("...");
        assert!(body.len() <= 200 && body.is_char_boundary(body.len()));
        // ASCII shorter than the cap is returned unchanged.
        assert_eq!(truncate_summary("hello", 200), "hello");
    }

    #[test]
    fn permission_gate_denies_under_plan_mode() {
        use crate::agent::permission::PermissionMode;
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        struct WriteTool;
        impl Tool for WriteTool {
            fn signature(&self) -> ToolSignature {
                ToolSignature {
                    name: "w".into(),
                    description: "w".into(),
                    parameters: vec![],
                }
            }
            fn execute(&self, _e: &Engine, _i: ToolInput) -> AgentResult<ToolOutput> {
                Ok(ToolOutput::ok("did write"))
            }
            fn manifest(&self) -> ToolManifest {
                ToolManifest {
                    name: "w".into(),
                    description: "w".into(),
                    parameters: vec![],
                    danger: DangerInfo {
                        level: DangerLevel::Dangerous,
                        capabilities: HashSet::from([Capability::WriteFilesystem]),
                        description: "w".into(),
                        shadow_triggers: vec![],
                    },
                    source: ToolSource::Native,
                }
            }
        }

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        // Default (Bypass) → the write tool runs.
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(WriteTool));
        assert!(
            reg.execute("w", ToolInput::new(), &engine).unwrap().success,
            "bypass mode should allow the write tool"
        );

        // Plan mode → denied (failed ToolOutput, not Err — OODA isolation kept).
        reg.set_permission_mode(PermissionMode::Plan);
        let out = reg.execute("w", ToolInput::new(), &engine).unwrap();
        assert!(!out.success, "plan mode should deny the write tool");
        assert!(out.result.contains("permission denied"), "result: {}", out.result);

        // would_ask (ADR 089 task A retry gate): a side-effecting tool under
        // Standard mode would stop at an Ask prompt — the repair-retry must
        // skip it rather than re-prompt for one logical call. Under Bypass
        // (and for unknown names) it never asks.
        reg.set_permission_mode(PermissionMode::Standard);
        assert!(reg.would_ask("w"), "Standard mode should Ask for a side-effecting tool");
        reg.set_permission_mode(PermissionMode::Bypass);
        assert!(!reg.would_ask("w"), "Bypass mode never asks");
        assert!(!reg.would_ask("no_such_tool"), "unknown names never ask");
    }

    struct DummyTool;
    impl Tool for DummyTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "dummy".into(),
                description: "A test tool".into(),
                parameters: vec![],
            }
        }
        fn execute(&self, _engine: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
            Ok(ToolOutput::ok("dummy result"))
        }
        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: "dummy".into(),
                description: "A test tool".into(),
                parameters: vec![],
                danger: DangerInfo {
                    level: DangerLevel::Safe,
                    capabilities: HashSet::from([Capability::ReadKg]),
                    description: "Safe test tool".into(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    #[test]
    fn register_and_list() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        assert_eq!(reg.len(), 1);
        let sigs = reg.list();
        assert_eq!(sigs[0].name, "dummy");
    }

    #[test]
    fn manifest_cached_on_register() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        let m = reg.manifest("dummy").unwrap();
        assert_eq!(m.name, "dummy");
        assert_eq!(m.danger.level, DangerLevel::Safe);
    }

    #[test]
    fn list_manifests() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        let manifests = reg.list_manifests();
        assert_eq!(manifests.len(), 1);
    }

    #[test]
    fn manifest_missing_returns_none() {
        let reg = ToolRegistry::new();
        assert!(reg.manifest("nonexistent").is_none());
    }

    #[test]
    fn get_missing_tool() {
        let reg = ToolRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn tool_input_builder() {
        let input = ToolInput::new()
            .with_param("symbol", "Sun")
            .with_param("direction", "from");
        assert_eq!(input.get("symbol"), Some("Sun"));
        assert_eq!(input.get("direction"), Some("from"));
        assert_eq!(input.get("missing"), None);
    }
}
