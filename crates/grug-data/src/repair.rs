//! KG-backed call repair (Phase 51 P3b, ADR 089 mechanism 2).
//!
//! The neural model (System-1) proposes a tool call; this module is the
//! symbolic grounding + repair step (System-2) that snaps that proposal onto
//! the *authoritative schema* before it executes. The four zero-shot negatives
//! (capacity, copy, copy-supervision, format) established that a tiny model
//! can't be *trusted* to emit an exactly-schema-valid call on its own — so we
//! don't ask it to. It proposes; the schema corrects.
//!
//! Here the "KG" is the offered [`ToolSchema`] set (the procedural catalog as a
//! stand-in for the real capability registry). Repair:
//!
//! 1. **Ground the tool** — pick the offered tool that best explains the
//!    proposal (name similarity blended with how many of the emitted argument
//!    names match that tool's parameters).
//! 2. **Ground each argument name** — map every emitted arg onto the nearest
//!    parameter of the grounded tool (above a similarity floor), so
//!    `region` → `region_name` is repaired, not failed. Unmatched args drop.
//! 3. **Coerce values** — re-type each kept value to its parameter's declared
//!    type (a bare `"4"` for an `Integer` becomes `4`), so a value that was
//!    right-but-mistyped stops counting as wrong.
//!
//! The result is **schema-consistent by construction** for the grounded tool.
//! What repair deliberately does *not* do (v1): invent values for required
//! params the model never produced — that is generation, not repair, and is
//! the retrieval/gap-token layer's job (ADR 089 mechanisms 1 & 3). Similarity
//! is edit-distance here; the real KG will use SymbolId / VSA similarity.

use serde_json::{Map, Value};

use crate::schema::{ParamType, ToolCall, ToolSchema};
use crate::similarity::normalized_similarity;

/// Minimum normalized name similarity for an emitted argument to be grounded
/// onto a schema parameter (below this it is treated as unknown and dropped).
const PARAM_MATCH_FLOOR: f64 = 0.34;

/// What repair did, for measuring/attribution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RepairStats {
    /// The grounded tool name differed from the model's emitted name.
    pub tool_changed: bool,
    /// Number of emitted args whose name was snapped to a different schema param.
    pub args_renamed: usize,
    /// Number of emitted args dropped (no schema param within the floor).
    pub args_dropped: usize,
    /// Number of values re-typed by coercion.
    pub values_coerced: usize,
}

/// Repair a proposed [`ToolCall`] against the offered schemas, returning the
/// grounded/repaired call and [`RepairStats`]. See the module docs.
pub fn repair_call(call: &ToolCall, offered: &[ToolSchema]) -> (ToolCall, RepairStats) {
    let mut stats = RepairStats::default();
    let Some(tool) = ground_tool(call, offered) else {
        // No tools offered at all: nothing to ground against.
        return (call.clone(), stats);
    };
    stats.tool_changed = tool.name != call.name;

    let mut args = Map::new();
    let mut used: Vec<String> = Vec::new();
    for (arg_name, value) in call.arguments_map() {
        match ground_param(&arg_name, tool, &used) {
            Some(pname) => {
                if pname != arg_name {
                    stats.args_renamed += 1;
                }
                let spec_ty = tool.param(&pname).map(|p| &p.ty);
                let (coerced, did) = coerce_value(&value, spec_ty);
                if did {
                    stats.values_coerced += 1;
                }
                args.insert(pname.clone(), coerced);
                used.push(pname);
            }
            None => stats.args_dropped += 1,
        }
    }

    (ToolCall::new(tool.name.clone(), Value::Object(args)), stats)
}

/// Pick the offered tool that best explains `call`: a blend of name similarity
/// and the fraction of emitted argument names that fuzzy-match the tool's
/// parameters (so a good-params/wrong-name proposal still grounds correctly).
fn ground_tool<'a>(call: &ToolCall, offered: &'a [ToolSchema]) -> Option<&'a ToolSchema> {
    // Exact name match wins outright.
    if let Some(t) = offered.iter().find(|t| t.name == call.name) {
        return Some(t);
    }
    let emitted: Vec<String> = call.arguments_map().keys().cloned().collect();
    offered
        .iter()
        .map(|t| {
            let name_sim = normalized_similarity(&call.name, &t.name);
            let param_overlap = if emitted.is_empty() {
                0.0
            } else {
                let matched = emitted
                    .iter()
                    .filter(|e| {
                        t.params
                            .iter()
                            .any(|p| normalized_similarity(e, &p.name) >= PARAM_MATCH_FLOOR)
                    })
                    .count();
                matched as f64 / emitted.len() as f64
            };
            (t, 0.6 * name_sim + 0.4 * param_overlap)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(t, _)| t)
}

/// Ground one emitted argument name onto the nearest not-yet-used parameter of
/// `tool` whose similarity clears [`PARAM_MATCH_FLOOR`].
fn ground_param(arg_name: &str, tool: &ToolSchema, used: &[String]) -> Option<String> {
    let is_used = |n: &str| used.iter().any(|u| u == n);
    // Exact match wins if free.
    if tool.params.iter().any(|p| p.name == arg_name) && !is_used(arg_name) {
        return Some(arg_name.to_string());
    }
    tool.params
        .iter()
        .filter(|p| !is_used(&p.name))
        .map(|p| (p, normalized_similarity(arg_name, &p.name)))
        .filter(|(_, s)| *s >= PARAM_MATCH_FLOOR)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(p, _)| p.name.clone())
}

/// Re-type a value to its parameter's declared type where possible. Returns
/// `(value, changed)`. A string that parses as the target scalar is converted;
/// anything already matching, or unconvertible, is returned unchanged.
fn coerce_value(v: &Value, ty: Option<&ParamType>) -> (Value, bool) {
    let Some(ty) = ty else {
        return (v.clone(), false);
    };
    if ty.matches(v) {
        return (v.clone(), false);
    }
    let Some(s) = v.as_str() else {
        return (v.clone(), false);
    };
    let s = s.trim();
    match ty {
        ParamType::Integer => s.parse::<i64>().ok().map(|n| (Value::from(n), true)),
        ParamType::Number => s
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(|n| (Value::Number(n), true)),
        ParamType::Boolean => match s {
            "true" => Some((Value::Bool(true), true)),
            "false" => Some((Value::Bool(false), true)),
            _ => None,
        },
        _ => None,
    }
    .unwrap_or_else(|| (v.clone(), false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ParamSpec, ParamType, ToolCall, ToolSchema};

    fn tool(name: &str, params: &[(&str, ParamType)]) -> ToolSchema {
        ToolSchema::new(
            name,
            "d",
            params
                .iter()
                .map(|(n, t)| ParamSpec::new(*n, t.clone(), true))
                .collect(),
        )
    }

    #[test]
    fn repairs_a_near_miss_param_name() {
        let tools = vec![tool("get_weather", &[("region_name", ParamType::String)])];
        // Model emitted "region" instead of "region_name".
        let call = ToolCall::new("get_weather", serde_json::json!({"region": "berlin"}));
        let (fixed, stats) = repair_call(&call, &tools);
        assert_eq!(fixed.arguments["region_name"], serde_json::json!("berlin"));
        assert!(stats.args_renamed >= 1);
    }

    #[test]
    fn coerces_a_stringified_integer() {
        let tools = vec![tool("book_table", &[("party_size", ParamType::Integer)])];
        // Value came back as a bare string (e.g. from a wrong-tool parse).
        let call = ToolCall::new("book_table", serde_json::json!({"party_size": "4"}));
        let (fixed, stats) = repair_call(&call, &tools);
        assert_eq!(fixed.arguments["party_size"], serde_json::json!(4));
        assert!(stats.values_coerced >= 1);
    }

    #[test]
    fn grounds_wrong_tool_name_to_nearest_offered() {
        let tools = vec![
            tool("send_email", &[("to", ParamType::String)]),
            tool("book_table", &[("party_size", ParamType::Integer)]),
        ];
        // Model named an unoffered near-miss of send_email, with its param.
        let call = ToolCall::new("send_emails", serde_json::json!({"to": "a@b.com"}));
        let (fixed, stats) = repair_call(&call, &tools);
        assert_eq!(fixed.name, "send_email");
        assert!(stats.tool_changed);
        assert_eq!(fixed.arguments["to"], serde_json::json!("a@b.com"));
    }

    #[test]
    fn drops_an_unmatched_arg() {
        let tools = vec![tool("get_weather", &[("region_name", ParamType::String)])];
        let call = ToolCall::new(
            "get_weather",
            serde_json::json!({"region_name": "berlin", "wholly_unrelated_xyz": "9"}),
        );
        let (fixed, stats) = repair_call(&call, &tools);
        assert!(fixed.arguments.get("wholly_unrelated_xyz").is_none());
        assert!(stats.args_dropped >= 1);
    }

    #[test]
    fn exact_call_is_unchanged() {
        let tools = vec![tool("get_weather", &[("region_name", ParamType::String)])];
        let call = ToolCall::new("get_weather", serde_json::json!({"region_name": "berlin"}));
        let (fixed, stats) = repair_call(&call, &tools);
        assert_eq!(fixed.name, "get_weather");
        assert_eq!(fixed.arguments["region_name"], serde_json::json!("berlin"));
        assert_eq!(stats, RepairStats::default());
    }
}
