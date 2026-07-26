//! The built-in fake tool catalog and the whole-tool train/holdout split.

use crate::error::{GrugError, GrugResult};
use crate::phrasing::{PhrasingBank, PhrasingRegistry, PhrasingTemplate};
use crate::schema::{ParamSpec, ParamType, ToolCall, ToolSchema};

/// The default number of tools reserved for the whole-tool holdout split out
/// of `ToolCatalog::built_in()`'s tools. Roughly 15% of the ~50-tool catalog.
pub const DEFAULT_HOLDOUT_COUNT: usize = 8;

/// A named collection of [`ToolSchema`]s, each paired with a
/// [`crate::phrasing::PhrasingBank`] of natural-language templates.
///
/// This is the unit both holdout axes operate on:
/// - `split_at` partitions the catalog into two *disjoint* sub-catalogs by
///   tool identity (whole-tool holdout, axis A) — a holdout tool's name,
///   description, params, and phrasings never appear anywhere in the
///   training stream.
/// - Within a train sub-catalog, each tool's own `PhrasingBank` further
///   splits into train/holdout phrasings (held-out-phrasing holdout, axis B)
///   — see `dataset.rs`.
#[derive(Debug, Clone, Default)]
pub struct ToolCatalog {
    tools: Vec<ToolSchema>,
    phrasings: PhrasingRegistry,
}

impl ToolCatalog {
    pub fn new(tools: Vec<ToolSchema>, phrasings: PhrasingRegistry) -> Self {
        Self { tools, phrasings }
    }

    pub fn push(&mut self, tool: ToolSchema, bank: PhrasingBank) {
        self.tools.push(tool);
        self.phrasings.insert(bank);
    }

    pub fn tools(&self) -> &[ToolSchema] {
        &self.tools
    }

    pub fn phrasings(&self) -> &PhrasingRegistry {
        &self.phrasings
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Split into `(train, holdout)` sub-catalogs, disjoint by tool name.
    /// Each sub-catalog carries only the phrasing banks for the tools it
    /// contains.
    ///
    /// The split is a deterministic function of catalog order (the last
    /// `holdout_count` tools become holdout) — determinism here matters more
    /// than randomness, since `Generator` already derives its own randomness
    /// from the seed, and a stable split makes holdout tool identity
    /// reproducible across runs for the same catalog.
    pub fn split_at(&self, holdout_count: usize) -> GrugResult<(ToolCatalog, ToolCatalog)> {
        if self.tools.is_empty() {
            return Err(GrugError::EmptyCatalog);
        }
        if holdout_count == 0 || holdout_count >= self.tools.len() {
            return Err(GrugError::DegenerateSplit {
                holdout: holdout_count,
                total: self.tools.len(),
            });
        }
        let split_point = self.tools.len() - holdout_count;
        let train_tools = self.tools[..split_point].to_vec();
        let holdout_tools = self.tools[split_point..].to_vec();

        let train_phrasings = self
            .phrasings
            .restricted_to(train_tools.iter().map(|t| t.name.as_str()));
        let holdout_phrasings = self
            .phrasings
            .restricted_to(holdout_tools.iter().map(|t| t.name.as_str()));

        Ok((
            ToolCatalog::new(train_tools, train_phrasings),
            ToolCatalog::new(holdout_tools, holdout_phrasings),
        ))
    }

    /// The built-in catalog: 40-60 varied fake tools spanning many everyday
    /// domains, deliberately varied in param *shape* (enums, optionals, 1
    /// param, 5+ params, mixed types) so schema generalization is a real
    /// test, each with a bank of 6-10 phrasing templates.
    pub fn built_in() -> Self {
        let mut tools = Vec::new();
        let mut phrasings = PhrasingRegistry::new();
        let mut add = |schema: ToolSchema, templates: Vec<PhrasingTemplate>| {
            let bank = PhrasingBank::new_default(schema.name.clone(), templates)
                .expect("built-in tools always have >= 2 phrasing templates");
            phrasings.insert(bank);
            tools.push(schema);
        };

        for (schema, templates) in crate::catalog_tools::all_tools() {
            add(schema, templates);
        }

        ToolCatalog::new(tools, phrasings)
    }

    /// A procedurally-generated catalog of `n_tools` tools, deterministic in
    /// `seed`: the new default *shape* of the data for scaling past the
    /// hand-authored ~56-tool catalog (see `crate::procedural`). Each tool
    /// gets [`crate::procedural::DEFAULT_TEMPLATES_PER_TOOL`] phrasing
    /// templates, split train/holdout exactly like `built_in()`'s tools.
    pub fn procedural(seed: u64, n_tools: usize) -> Self {
        crate::procedural::build_catalog(seed, n_tools)
    }

    /// A procedurally-generated catalog with an explicit templates-per-tool
    /// count, for callers that want a denser or thinner phrasing bank than
    /// the default.
    pub fn procedural_with_templates(seed: u64, n_tools: usize, templates_per_tool: usize) -> Self {
        crate::procedural::build_catalog_with_templates(seed, n_tools, templates_per_tool)
    }

    /// A procedurally-generated catalog under an explicit
    /// [`SignalMode`](crate::semantic::SignalMode) (Phase 51 P3d):
    /// `SignalMode::Name` is byte-identical to [`ToolCatalog::procedural`],
    /// while `Description`/`Paraphrase` move the retrieval signal off the tool
    /// name (glossed descriptions + request phrasings).
    pub fn procedural_signal(seed: u64, n_tools: usize, mode: crate::semantic::SignalMode) -> Self {
        crate::procedural::build_catalog_signal(
            seed,
            n_tools,
            crate::procedural::DEFAULT_TEMPLATES_PER_TOOL,
            mode,
        )
    }
}

/// Fuzzed-value accessors shared by every tool's phrasing templates in
/// `catalog_tools`. Small, generic helpers so each template body stays a
/// one-line `format!`.
pub(crate) fn get_str(call: &ToolCall, key: &str) -> Option<String> {
    call.arguments_map()
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

pub(crate) fn get_str_or(call: &ToolCall, key: &str, default: &str) -> String {
    get_str(call, key).unwrap_or_else(|| default.to_string())
}

pub(crate) fn get_num(call: &ToolCall, key: &str) -> Option<f64> {
    call.arguments_map().get(key).and_then(|v| v.as_f64())
}

pub(crate) fn get_num_or(call: &ToolCall, key: &str, default: f64) -> f64 {
    get_num(call, key).unwrap_or(default)
}

pub(crate) fn get_bool_or(call: &ToolCall, key: &str, default: bool) -> bool {
    call.arguments_map()
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(default)
}

pub(crate) fn param(name: &str, ty: ParamType, required: bool) -> ParamSpec {
    ParamSpec::new(name, ty, required)
}

pub(crate) fn enum_of(variants: &[&str]) -> ParamType {
    ParamType::Enum(variants.iter().map(|s| s.to_string()).collect())
}
