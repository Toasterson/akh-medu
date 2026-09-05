//! Procedural tool + phrasing generation.
//!
//! `crate::catalog_tools` hand-authors ~56 tools, each with hand-written
//! phrasing templates. That is enough to prove JSON *syntax* is learnable
//! (P0/P1a), but with so few tools at ~500 examples each, a model can pass
//! the eval by memorizing (tool name → training-set call), never touching
//! its schema-reading skill: the whole-tool holdout axis (axis A) collapsed
//! (schema=0, diversity=1/128) and arg-correctness plateaued at ~0.11.
//!
//! This module generates **N** tools (hundreds to thousands) deterministically
//! from a seed, so that:
//!
//! - Memorizing per-tool call shapes is infeasible — with 1000 tools and 30k
//!   examples, each tool is seen ~30 times, not ~500.
//! - Every generated phrasing template embeds each required argument's fuzzed
//!   value *verbatim* in the rendered `user_request` text (via
//!   [`crate::phrasing::TemplatePart::Arg`]), so argument-copying is
//!   learnable by construction — the model is never asked to invent a value
//!   it wasn't given.
//! - The two existing holdout axes (whole-tool, held-out-phrasing) still
//!   apply unchanged: this module only produces `(ToolSchema, PhrasingBank)`
//!   pairs, same as `catalog_tools::all_tools()`, so `ToolCatalog::split_at`
//!   and `PhrasingBank`'s train/holdout split need no changes at all.
//!
//! # Name scheme
//!
//! Tool names are `verb_noun` (e.g. `fetch_ledger`, `archive_beacon`),
//! deterministically drawn from two word banks — semi-readable (so phrasing
//! templates can refer to the action/object naturally) but combinatorially
//! large enough (verbs × nouns × a numeric suffix once the bank is exhausted)
//! to support thousands of unique names without collision.
//!
//! # Param shapes
//!
//! Each tool gets 1-5 params, procedurally named (`<adjective_or_role>_<kind>`
//! e.g. `target_id`, `max_count`), with a realistic mix of
//! `String`/`Integer`/`Number`/`Boolean`/`Enum` types and required/optional
//! flags, so the type-distribution variety the hand-authored catalog has is
//! preserved at scale.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::catalog::{ToolCatalog, param};
use crate::phrasing::{PhrasingBank, PhrasingRegistry, PhrasingTemplate, TemplatePart};
use crate::schema::{ParamType, ToolSchema};
use crate::semantic::{SignalMode, describe_tool, glossed_phrase};

/// Default number of NL phrasing templates generated per procedural tool.
/// Within the plan's "6-10 templates" target; `9` splits cleanly under the
/// default ~30% holdout fraction (3 holdout / 6 train).
pub const DEFAULT_TEMPLATES_PER_TOOL: usize = 9;

pub(crate) const VERBS: &[&str] = &[
    "get",
    "fetch",
    "list",
    "create",
    "update",
    "delete",
    "archive",
    "restore",
    "search",
    "schedule",
    "cancel",
    "start",
    "stop",
    "pause",
    "resume",
    "sync",
    "export",
    "import",
    "validate",
    "check",
    "reset",
    "rotate",
    "merge",
    "split",
    "clone",
    "publish",
    "unpublish",
    "lock",
    "unlock",
    "tag",
    "untag",
    "assign",
    "unassign",
    "notify",
    "escalate",
    "resolve",
    "close",
    "reopen",
    "approve",
    "reject",
    "queue",
    "dequeue",
    "compress",
    "decompress",
    "encrypt",
    "decrypt",
    "index",
    "reindex",
    "compile",
    "deploy",
];

pub(crate) const NOUNS: &[&str] = &[
    "ledger",
    "beacon",
    "manifest",
    "cluster",
    "shard",
    "widget",
    "module",
    "profile",
    "session",
    "invoice",
    "ticket",
    "snapshot",
    "pipeline",
    "endpoint",
    "artifact",
    "record",
    "workspace",
    "channel",
    "queue",
    "policy",
    "template",
    "dataset",
    "credential",
    "webhook",
    "schedule",
    "report",
    "dashboard",
    "device",
    "sensor",
    "package",
    "release",
    "backup",
    "cache",
    "route",
    "zone",
    "region",
    "cluster_node",
    "token",
    "certificate",
    "subscription",
    "playlist",
    "thread",
    "annotation",
    "bookmark",
    "label",
    "tag",
    "attachment",
    "form",
    "survey",
    "campaign",
    "contact",
];

/// Bank of procedural param "role" names (paired with a type-appropriate
/// suffix below) so param names read as plausible identifiers rather than
/// `param_0`, `param_1`, ... .
const PARAM_ROLES: &[&str] = &[
    "target",
    "source",
    "owner",
    "requester",
    "primary",
    "secondary",
    "parent",
    "origin",
    "destination",
    "region",
    "priority",
    "category",
    "status",
    "mode",
    "kind",
    "scope",
    "filter",
    "query",
    "batch",
    "session",
    "customer",
    "account",
    "project",
    "team",
];

const ENUM_VARIANT_SETS: &[&[&str]] = &[
    &["low", "medium", "high", "critical"],
    &["draft", "active", "archived"],
    &["pending", "approved", "rejected"],
    &["daily", "weekly", "monthly"],
    &["read", "write", "admin"],
    &["north", "south", "east", "west"],
    &["small", "medium", "large"],
];

/// Word banks used to fuzz procedural string arg values so their rendered
/// text is stable/plausible-looking, mirroring `generator::fuzz_string`'s
/// role but owned here so [`build_catalog`] can pre-render templates that
/// reference the same value bank the runtime `Generator` will later fuzz
/// from — the actual per-example value always comes from the live
/// `ToolCall` at render time, this bank only informs *identifier* text used
/// in tool/param naming, not argument values.
struct ProceduralNamer {
    rng: StdRng,
    used_names: std::collections::HashSet<String>,
}

impl ProceduralNamer {
    fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            used_names: std::collections::HashSet::new(),
        }
    }

    /// Draw a unique `verb_noun` tool name, falling back to a numeric suffix
    /// once the `verb x noun` combination space is exhausted (at 50 verbs x
    /// 50 nouns = 2500 base combos, this only matters for very large N).
    fn next_tool_name(&mut self) -> String {
        for _ in 0..64 {
            let verb = VERBS.choose(&mut self.rng).expect("VERBS non-empty");
            let noun = NOUNS.choose(&mut self.rng).expect("NOUNS non-empty");
            let name = format!("{verb}_{noun}");
            if self.used_names.insert(name.clone()) {
                return name;
            }
        }
        // Exhausted the collision-free space in 64 tries; disambiguate with
        // a monotonic counter suffix so uniqueness is still guaranteed.
        loop {
            let verb = VERBS.choose(&mut self.rng).expect("VERBS non-empty");
            let noun = NOUNS.choose(&mut self.rng).expect("NOUNS non-empty");
            let n = self.used_names.len();
            let name = format!("{verb}_{noun}_{n}");
            if self.used_names.insert(name.clone()) {
                return name;
            }
        }
    }
}

/// Draw a plausible param type + a matching "kind" suffix for its name.
fn fuzz_param_type_and_suffix(rng: &mut StdRng) -> (ParamType, &'static str) {
    match rng.gen_range(0..5) {
        0 => (ParamType::String, "name"),
        1 => (ParamType::Integer, "count"),
        2 => (ParamType::Number, "score"),
        3 => (ParamType::Boolean, "flag"),
        _ => {
            let variants = ENUM_VARIANT_SETS
                .choose(rng)
                .expect("ENUM_VARIANT_SETS non-empty");
            (
                ParamType::Enum(variants.iter().map(|s| s.to_string()).collect()),
                "level",
            )
        }
    }
}

/// Build one procedural tool's [`ToolSchema`], with 1-5 params, a
/// required/optional mix, and varied types.
fn build_tool_schema(rng: &mut StdRng, name: String, mode: SignalMode) -> ToolSchema {
    let n_params = rng.gen_range(1..=5);
    let mut params = Vec::with_capacity(n_params);
    let mut used_param_names = std::collections::HashSet::new();

    for i in 0..n_params {
        let role = PARAM_ROLES.choose(rng).expect("PARAM_ROLES non-empty");
        let (ty, suffix) = fuzz_param_type_and_suffix(rng);
        let mut pname = format!("{role}_{suffix}");
        // Disambiguate within this tool if the same role+kind combo repeats.
        if !used_param_names.insert(pname.clone()) {
            pname = format!("{role}_{suffix}_{i}");
            used_param_names.insert(pname.clone());
        }
        // First param is always required (every tool needs at least one
        // grounded, copyable argument); remaining params are required ~60%
        // of the time so the required/optional mix stays realistic.
        let required = i == 0 || rng.gen_bool(0.6);
        params.push(param(&pname, ty, required));
    }

    let description = describe_tool(&name, n_params, mode);
    ToolSchema::new(name, description, params)
}

/// Sentence "shapes" used to assemble a phrasing template's [`TemplatePart`]
/// sequence: each shape lists literal-text fragments interleaved with which
/// param slot (by index into the tool's param list) fills the gap. Having
/// several distinct *shapes* (not just word-substitution on one shape) is
/// what keeps phrasing non-trivial across templates — word order and
/// surrounding text both vary.
///
/// Every shape references **every param index 0..n_params** at least once
/// somewhere across the full template bank (enforced by
/// `templates_covering_all_params`), so held-out-phrasing eval still tests
/// copying every argument, not just the first.
type Shape = fn(&str, &[String]) -> Vec<TemplatePart>;

fn shape_imperative_lead(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "Please <verb_phrase> for <p0>[, with <p1>][ and <p2>]..."
    let mut parts = vec![TemplatePart::Literal(format!("Please {verb_phrase} for "))];
    append_param_list(&mut parts, param_names);
    parts.push(TemplatePart::Literal(".".to_string()));
    parts
}

fn shape_question_lead(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "Can you <verb_phrase> for <p0>[ using <p1>]...?"
    let mut parts = vec![TemplatePart::Literal(format!("Can you {verb_phrase} for "))];
    append_param_list(&mut parts, param_names);
    parts.push(TemplatePart::Literal("?".to_string()));
    parts
}

fn shape_value_first(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "For <p0>, I need to <verb_phrase>[, referencing <p1>]..."
    let mut parts = Vec::new();
    if let Some(first) = param_names.first() {
        parts.push(TemplatePart::Literal("For ".to_string()));
        parts.push(TemplatePart::Arg(first.clone()));
        parts.push(TemplatePart::Literal(format!(", I need to {verb_phrase}")));
        if param_names.len() > 1 {
            parts.push(TemplatePart::Literal(", referencing ".to_string()));
            append_param_list_from(&mut parts, &param_names[1..]);
        }
        parts.push(TemplatePart::Literal(".".to_string()));
    } else {
        parts.push(TemplatePart::Literal(format!("I need to {verb_phrase}.")));
    }
    parts
}

fn shape_casual(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "Hey, <verb_phrase> real quick — <p0>[, <p1>]..."
    let mut parts = vec![TemplatePart::Literal(format!(
        "Hey, {verb_phrase} real quick \u{2014} "
    ))];
    append_param_list(&mut parts, param_names);
    parts.push(TemplatePart::Literal(", if you can.".to_string()));
    parts
}

fn shape_formal(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "I would like to <verb_phrase>. The relevant detail(s): <p0>[, <p1>]..."
    let mut parts = vec![TemplatePart::Literal(format!(
        "I would like to {verb_phrase}. The relevant detail(s): "
    ))];
    append_param_list(&mut parts, param_names);
    parts.push(TemplatePart::Literal(".".to_string()));
    parts
}

fn shape_urgent(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "Urgent: <verb_phrase> now. <p0>[ / <p1>]..."
    let mut parts = vec![TemplatePart::Literal(format!(
        "Urgent: {verb_phrase} now. "
    ))];
    append_param_list(&mut parts, param_names);
    parts.push(TemplatePart::Literal(".".to_string()));
    parts
}

fn shape_trailing_context(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "<p0>[ and <p1>]... — go ahead and <verb_phrase>."
    let mut parts = Vec::new();
    append_param_list(&mut parts, param_names);
    parts.push(TemplatePart::Literal(format!(
        " \u{2014} go ahead and {verb_phrase}."
    )));
    parts
}

fn shape_conditional(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "If possible, <verb_phrase> using <p0>[, and <p1>]..."
    let mut parts = vec![TemplatePart::Literal(format!(
        "If possible, {verb_phrase} using "
    ))];
    append_param_list(&mut parts, param_names);
    parts.push(TemplatePart::Literal(" today.".to_string()));
    parts
}

fn shape_terse(verb_phrase: &str, param_names: &[String]) -> Vec<TemplatePart> {
    // "<verb_phrase>: <p0>[, <p1>]..."
    let mut parts = vec![TemplatePart::Literal(format!("{verb_phrase}: "))];
    append_param_list(&mut parts, param_names);
    parts
}

const SHAPES: &[Shape] = &[
    shape_imperative_lead,
    shape_question_lead,
    shape_value_first,
    shape_casual,
    shape_formal,
    shape_urgent,
    shape_trailing_context,
    shape_conditional,
    shape_terse,
];

/// Append `", <name>"`-joined `Arg` parts (with a leading literal lead-in
/// word varied by position) for every param name, comma/and-joined so a
/// multi-param tool's request reads as one clause.
fn append_param_list(parts: &mut Vec<TemplatePart>, param_names: &[String]) {
    append_param_list_from(parts, param_names);
}

fn append_param_list_from(parts: &mut Vec<TemplatePart>, param_names: &[String]) {
    for (i, name) in param_names.iter().enumerate() {
        if i > 0 {
            let sep = if i == param_names.len() - 1 && param_names.len() > 1 {
                ", and "
            } else {
                ", "
            };
            parts.push(TemplatePart::Literal(sep.to_string()));
        }
        parts.push(TemplatePart::Arg(name.clone()));
    }
}

/// Build `n_templates` phrasing templates for `tool`, cycling through
/// [`SHAPES`] (with a per-tool rotation offset so consecutive tools don't
/// share the exact same shape-order) so template bodies vary in sentence
/// structure, not just literal wording.
///
/// Every template includes **every** param name (via `append_param_list`
/// over the tool's full param name list) so the "value appears verbatim"
/// guarantee holds for every required arg regardless of which template is
/// drawn — this is the load-bearing invariant checked by the
/// `arg_value_copyable_across_generated_examples` test.
fn build_templates(
    tool: &ToolSchema,
    n_templates: usize,
    rotation: usize,
    mode: SignalMode,
) -> Vec<PhrasingTemplate> {
    let param_names: Vec<String> = tool.params.iter().map(|p| p.name.clone()).collect();
    let verb_phrase = glossed_phrase(&tool.name, mode);

    (0..n_templates)
        .map(|i| {
            let shape = SHAPES[(i + rotation) % SHAPES.len()];
            let parts = shape(&verb_phrase, &param_names);
            PhrasingTemplate::from_parts(i, parts)
        })
        .collect()
}

/// Build a procedural [`ToolCatalog`] of exactly `n_tools` tools,
/// deterministically from `seed`: same seed + same `n_tools` always produces
/// byte-identical tool schemas and phrasing banks.
///
/// Each tool gets [`DEFAULT_TEMPLATES_PER_TOOL`] phrasing templates, split
/// train/holdout by [`PhrasingBank::new_default`] exactly as the
/// hand-authored catalog does.
pub fn build_catalog(seed: u64, n_tools: usize) -> ToolCatalog {
    build_catalog_with_templates(seed, n_tools, DEFAULT_TEMPLATES_PER_TOOL)
}

/// Like [`build_catalog`], but with an explicit templates-per-tool count
/// (exposed for tests that want to probe the minimum bank size, and for
/// callers that want a denser/thinner phrasing bank than the default).
pub fn build_catalog_with_templates(
    seed: u64,
    n_tools: usize,
    templates_per_tool: usize,
) -> ToolCatalog {
    build_catalog_signal(seed, n_tools, templates_per_tool, SignalMode::Name)
}

/// Like [`build_catalog_with_templates`], but under an explicit [`SignalMode`]
/// (Phase 51 P3d): `SignalMode::Name` reproduces the current corpus
/// byte-identically (the RNG streams and param shapes are untouched — only the
/// description string and the request verb-phrase change), while `Description`
/// (D1) and `Paraphrase` (D2) move the retrieval signal off the tool name via
/// the [`crate::semantic`] gloss lexicon.
pub fn build_catalog_signal(
    seed: u64,
    n_tools: usize,
    templates_per_tool: usize,
    mode: SignalMode,
) -> ToolCatalog {
    assert!(n_tools > 0, "procedural catalog needs at least 1 tool");
    assert!(
        templates_per_tool >= 2,
        "PhrasingBank requires >= 2 templates to split train/holdout"
    );

    let mut namer = ProceduralNamer::new(seed);
    // Separate RNG stream (derived, not reused) for param-shape fuzzing, so
    // changing `templates_per_tool` never perturbs which names/shapes get
    // drawn, and vice versa.
    let mut shape_rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A_1234_9876);

    let mut tools = Vec::with_capacity(n_tools);
    let mut phrasings = PhrasingRegistry::new();

    for i in 0..n_tools {
        let name = namer.next_tool_name();
        let schema = build_tool_schema(&mut shape_rng, name, mode);
        let templates = build_templates(&schema, templates_per_tool, i, mode);
        let bank = PhrasingBank::new_default(schema.name.clone(), templates)
            .expect("build_templates always returns >= 2 templates");
        phrasings.insert(bank);
        tools.push(schema);
    }

    ToolCatalog::new(tools, phrasings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ToolCall;
    use std::collections::HashSet;

    #[test]
    fn deterministic_same_seed_same_catalog() {
        let a = build_catalog(4242, 200);
        let b = build_catalog(4242, 200);
        assert_eq!(a.len(), b.len());
        for (ta, tb) in a.tools().iter().zip(b.tools().iter()) {
            assert_eq!(ta, tb);
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let a = build_catalog(1, 100);
        let b = build_catalog(2, 100);
        let names_a: HashSet<&str> = a.tools().iter().map(|t| t.name.as_str()).collect();
        let names_b: HashSet<&str> = b.tools().iter().map(|t| t.name.as_str()).collect();
        assert_ne!(names_a, names_b);
    }

    #[test]
    fn all_tool_names_unique() {
        let catalog = build_catalog(7, 1500);
        let names: HashSet<&str> = catalog.tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), catalog.len(), "tool names must be unique");
    }

    #[test]
    fn every_tool_has_1_to_5_params_and_a_phrasing_bank() {
        let catalog = build_catalog(9, 300);
        for tool in catalog.tools() {
            assert!(
                (1..=5).contains(&tool.params.len()),
                "{} has {} params",
                tool.name,
                tool.params.len()
            );
            let bank = catalog.phrasings().get(&tool.name).unwrap();
            assert!(bank.train_count() >= 1);
            assert!(bank.holdout_count() >= 1);
        }
    }

    #[test]
    fn required_arg_values_appear_verbatim_in_every_rendered_template() {
        let catalog = build_catalog(123, 50);
        let mut rng = StdRng::seed_from_u64(999);
        for tool in catalog.tools() {
            let bank = catalog.phrasings().get(&tool.name).unwrap();
            // Build a concrete call with every param present (required and
            // optional) so we can check every template embeds every value.
            let mut args = serde_json::Map::new();
            for p in &tool.params {
                let v = crate::generator::fuzz_value(&mut rng, &p.ty);
                args.insert(p.name.clone(), v);
            }
            let call = ToolCall::new(tool.name.clone(), serde_json::Value::Object(args));

            for t in bank
                .train_templates()
                .iter()
                .chain(bank.holdout_templates())
            {
                let rendered = t.render(&call);
                for p in tool.required_params() {
                    let value_text =
                        crate::phrasing::value_as_text(call.arguments_map().get(&p.name).unwrap());
                    assert!(
                        rendered.contains(&value_text),
                        "tool {} template {} did not embed required arg {}={} verbatim: {rendered:?}",
                        tool.name,
                        t.template_id,
                        p.name,
                        value_text
                    );
                }
            }
        }
    }
}
