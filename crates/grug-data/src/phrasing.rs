//! Per-tool phrasing templates and the train/holdout **phrasing** split.
//!
//! This is the piece that makes the held-out-phrasing eval axis honest. A
//! [`PhrasingTemplate`] is a pure function from a fuzzed [`ToolCall`]'s
//! argument values to a natural-language `user_request` string — it is
//! deliberately *not* a closure, but a plain function pointer keyed by a
//! stable `template_id`, so templates are enumerable, countable, and
//! splittable the same way [`crate::catalog::ToolCatalog::split_at`] splits
//! tools: by identity, not by generated example.
//!
//! Each tool owns a small bank (aim 6-10) of these templates. [`PhrasingBank`]
//! partitions that bank into disjoint train/holdout halves *once*, at
//! construction, from the template's position in the bank — exactly
//! mirroring the whole-tool split's "last N are holdout" rule. Because the
//! split operates on the template list itself (not on any generated
//! `Example`), it is structurally impossible for a holdout-phrasing template
//! to be selected while rendering a training example: the training generator
//! never even holds a reference to the holdout half of the bank.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::error::{GrugError, GrugResult};
use crate::schema::ToolCall;

/// A single natural-language phrasing template for one tool.
///
/// `render` takes the tool's fuzzed [`ToolCall`] and produces the
/// user-facing request string that should elicit exactly that call. Templates
/// reference the call's argument values (via [`ToolCall::arguments_map`]) so
/// the rendered text and the ground-truth call agree by construction — the
/// same "label is the source of truth" discipline as `generator::fuzz_value`.
#[derive(Clone, Copy)]
pub struct PhrasingTemplate {
    /// Stable identity within a tool's bank; also its position in
    /// [`PhrasingBank::templates`] before splitting. Used only for
    /// diagnostics/tests, not for rendering.
    pub template_id: usize,
    render: fn(&ToolCall) -> String,
}

impl PhrasingTemplate {
    pub const fn new(template_id: usize, render: fn(&ToolCall) -> String) -> Self {
        Self {
            template_id,
            render,
        }
    }

    pub fn render(&self, call: &ToolCall) -> String {
        (self.render)(call)
    }
}

impl std::fmt::Debug for PhrasingTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhrasingTemplate")
            .field("template_id", &self.template_id)
            .finish_non_exhaustive()
    }
}

/// The default fraction of each tool's phrasing bank reserved for the
/// held-out-phrasing eval axis. ~30%, matching the plan's "~70/30" target.
pub const DEFAULT_HOLDOUT_PHRASING_FRACTION: f64 = 0.3;

/// One tool's full phrasing bank, already split into disjoint train/holdout
/// halves.
///
/// The split point is `templates.len() - holdout_count`, the same
/// "last-N-are-holdout" rule `ToolCatalog::split_at` uses — deterministic in
/// template *order*, independent of any RNG, so which wordings are held out
/// is a fixed property of the catalog definition, not of a seed.
#[derive(Debug, Clone)]
pub struct PhrasingBank {
    tool_name: String,
    train: Vec<PhrasingTemplate>,
    holdout: Vec<PhrasingTemplate>,
}

impl PhrasingBank {
    /// Build a bank for `tool_name` from `templates`, reserving the last
    /// `ceil(templates.len() * holdout_fraction)` (at least 1, at most
    /// `len - 1`) templates as holdout phrasings.
    pub fn new(
        tool_name: impl Into<String>,
        templates: Vec<PhrasingTemplate>,
        holdout_fraction: f64,
    ) -> GrugResult<Self> {
        let tool_name = tool_name.into();
        let total = templates.len();
        if total < 2 {
            return Err(GrugError::InsufficientPhrasings {
                tool: tool_name,
                count: total,
            });
        }
        let holdout_count = ((total as f64 * holdout_fraction).ceil() as usize).clamp(1, total - 1);
        let split_point = total - holdout_count;

        let mut templates = templates;
        let holdout = templates.split_off(split_point);
        let train = templates;

        Ok(Self {
            tool_name,
            train,
            holdout,
        })
    }

    /// Build a bank using [`DEFAULT_HOLDOUT_PHRASING_FRACTION`].
    pub fn new_default(
        tool_name: impl Into<String>,
        templates: Vec<PhrasingTemplate>,
    ) -> GrugResult<Self> {
        Self::new(tool_name, templates, DEFAULT_HOLDOUT_PHRASING_FRACTION)
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Train-phrasing templates only. The holdout half is never reachable
    /// through this method, by construction.
    pub fn train_templates(&self) -> &[PhrasingTemplate] {
        &self.train
    }

    /// Holdout-phrasing templates only.
    pub fn holdout_templates(&self) -> &[PhrasingTemplate] {
        &self.holdout
    }

    pub fn train_count(&self) -> usize {
        self.train.len()
    }

    pub fn holdout_count(&self) -> usize {
        self.holdout.len()
    }

    /// Render a request using a uniformly-chosen **train** phrasing.
    pub fn render_train(&self, rng: &mut StdRng, call: &ToolCall) -> String {
        render_with(&self.train, rng, call, &self.tool_name)
    }

    /// Render a request using a uniformly-chosen **holdout** phrasing.
    pub fn render_holdout(&self, rng: &mut StdRng, call: &ToolCall) -> String {
        render_with(&self.holdout, rng, call, &self.tool_name)
    }
}

fn render_with(
    templates: &[PhrasingTemplate],
    rng: &mut StdRng,
    call: &ToolCall,
    tool_name: &str,
) -> String {
    match templates.choose(rng) {
        Some(t) => t.render(call),
        None => format!("Please call {tool_name}."),
    }
}

/// A name-keyed collection of every tool's [`PhrasingBank`], owned alongside
/// a [`crate::catalog::ToolCatalog`].
///
/// Kept as its own map (rather than folded into `ToolSchema`) so the schema
/// type stays a plain serializable data description with no function
/// pointers in it — `ToolSchema` is what gets shown to the model in-context
/// and round-tripped through `Example`'s JSONL; phrasing templates are purely
/// a generator-side concern.
#[derive(Debug, Clone, Default)]
pub struct PhrasingRegistry {
    banks: std::collections::HashMap<String, PhrasingBank>,
}

impl PhrasingRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, bank: PhrasingBank) {
        self.banks.insert(bank.tool_name().to_string(), bank);
    }

    pub fn get(&self, tool_name: &str) -> GrugResult<&PhrasingBank> {
        self.banks
            .get(tool_name)
            .ok_or_else(|| GrugError::MissingPhrasingBank {
                tool: tool_name.to_string(),
            })
    }

    /// Restrict this registry to only the banks whose tool name is in
    /// `names` — used to build a registry scoped to a train or holdout
    /// sub-catalog after `ToolCatalog::split_at`.
    pub fn restricted_to(&self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let mut out = Self::new();
        for name in names {
            if let Some(bank) = self.banks.get(name.as_ref()) {
                out.banks.insert(name.as_ref().to_string(), bank.clone());
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.banks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }
}
