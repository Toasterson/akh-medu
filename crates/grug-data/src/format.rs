//! Boundary target formats (Phase 51 P2, ADR 088).
//!
//! The P1a-4 arc showed that adding *model* machinery (capacity, a copy
//! mechanism, copy supervision) never moved zero-shot tool-calling. The
//! literature on small models explains why: for a tiny model, JSON's braces,
//! quotes, per-field keys and type punctuation consume a *material* part of the
//! generation budget — the format itself is a tax the task can't afford. So the
//! boundary representation is swappable here, and P2 tests a minimal
//! **High-Grug v0** format against JSON on an otherwise-identical pipeline.
//!
//! Both the *offered schema* (prompt side) and the *emitted call* (target
//! side) are rendered through this module, and model output is parsed back
//! through it for scoring — so a format is defined in exactly one place.
//!
//! # High-Grug v0 (the [`TargetFormat::Grug`] variant)
//!
//! Flat, pipe-delimited, decode-safe ASCII — no nesting, no quoting, no
//! balancing:
//!
//! ```text
//! schema (one line per tool): tool <name> | <pname> <type> <req> | ...
//! call (target):              call <name> | <argname> <value> | ...
//! ```
//!
//! Structure is carried by the literal words `tool`/`call` and the ` | `
//! separator (each a single token); an argument segment splits once on its
//! first space into `(name, value)`, so values may contain spaces without
//! quoting. The parse is deterministic and total. These are ordinary tokens
//! (not tokenizer specials), so they survive `decode` and need no tokenizer
//! change — keeping the JSON-vs-Grug A/B a pure format swap.

use serde_json::{Map, Value};

use crate::schema::{ParamType, ToolCall, ToolSchema};

/// Which boundary representation to render/parse tool schemas and calls in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFormat {
    /// The original `{"arguments":{...},"name":"..."}` JSON (P0–P1a-4).
    Json,
    /// High-Grug v0: flat pipe-delimited ASCII (Phase 51 P2, ADR 088).
    Grug,
}

impl TargetFormat {
    /// Parse a short CLI/env label (`json` / `grug`), case-insensitive.
    pub fn parse_label(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => Some(TargetFormat::Json),
            "grug" => Some(TargetFormat::Grug),
            _ => None,
        }
    }
}

/// Render the offered tools as the prompt-side schema in `fmt`.
pub fn render_tools(tools: &[ToolSchema], fmt: TargetFormat) -> String {
    match fmt {
        TargetFormat::Json => serde_json::to_string(tools).unwrap_or_default(),
        TargetFormat::Grug => tools.iter().map(render_tool_grug).collect::<Vec<_>>().join("\n"),
    }
}

fn type_tag(ty: &ParamType) -> &'static str {
    match ty {
        ParamType::String => "str",
        ParamType::Integer => "int",
        ParamType::Number => "num",
        ParamType::Boolean => "bool",
        ParamType::Enum(_) => "enum",
    }
}

fn render_tool_grug(tool: &ToolSchema) -> String {
    let mut out = format!("tool {}", tool.name);
    for p in &tool.params {
        out.push_str(" | ");
        out.push_str(&p.name);
        out.push(' ');
        out.push_str(type_tag(&p.ty));
        out.push(' ');
        out.push_str(if p.required { "req" } else { "opt" });
        if let ParamType::Enum(variants) = &p.ty {
            for v in variants {
                out.push(' ');
                out.push_str(v);
            }
        }
    }
    out
}

/// Render a value as a bare Grug token (no quotes). Objects/arrays/null are not
/// expected for these flat tool args; they fall back to a compact JSON string.
fn render_value_grug(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

/// The bare High-Grug abstention marker (Phase 51 P3c, ADR 089 mechanism 3):
/// a single literal word, mirroring `call`/`tool` — decode-safe plain ASCII,
/// no tokenizer change. Emitted when no offered tool fits the request.
pub const GRUG_GAP_MARKER: &str = "gap";

/// Render a [`ToolCall`] as the target string in `fmt`.
///
/// The abstention sentinel ([`ToolCall::is_gap`]) renders to a compact,
/// call-distinct gap marker in each format (`gap` for Grug, `{"gap":true}` for
/// JSON) rather than to a literal call to a tool named `__gap__` — so a model
/// trained on these targets learns "emit gap" as its own output shape.
pub fn render_call(call: &ToolCall, fmt: TargetFormat) -> String {
    if call.is_gap() {
        return match fmt {
            TargetFormat::Json => "{\"gap\":true}".to_string(),
            TargetFormat::Grug => GRUG_GAP_MARKER.to_string(),
        };
    }
    match fmt {
        TargetFormat::Json => serde_json::json!({
            "name": call.name,
            "arguments": call.arguments,
        })
        .to_string(),
        TargetFormat::Grug => {
            let mut out = format!("call {}", call.name);
            // Sorted keys so the target is deterministic (matches the JSON
            // path's BTreeMap ordering, keeping the two formats comparable).
            for (k, v) in call.arguments_map() {
                out.push_str(" | ");
                out.push_str(&k);
                out.push(' ');
                out.push_str(&render_value_grug(&v));
            }
            out
        }
    }
}

/// Coerce a bare Grug value string to a typed JSON value using the parameter's
/// declared type (so type-checking and value-comparison see the same shapes
/// the JSON path produces). Unknown/mismatched values fall back to a JSON
/// string, which the schema type-check will then correctly flag.
fn coerce_value(s: &str, ty: Option<&ParamType>) -> Value {
    let s = s.trim();
    match ty {
        Some(ParamType::Integer) => s
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        Some(ParamType::Number) => s
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .unwrap_or_else(|| Value::String(s.to_string())),
        Some(ParamType::Boolean) => match s {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(s.to_string()),
        },
        _ => Value::String(s.to_string()),
    }
}

/// Parse a model output string back into a [`ToolCall`] under `fmt`, using
/// `offered` to type-coerce Grug argument values. Returns `None` if the output
/// does not parse into a call shape at all (the format's "invalid output"
/// signal, analogous to invalid JSON).
pub fn parse_call(raw: &str, fmt: TargetFormat, offered: &[ToolSchema]) -> Option<ToolCall> {
    match fmt {
        TargetFormat::Json => {
            let value: Value = serde_json::from_str(raw.trim()).ok()?;
            let map = value.as_object()?;
            let name = map.get("name")?.as_str()?.to_string();
            let arguments = match map.get("arguments") {
                Some(a) if a.is_object() => a.clone(),
                None => Value::Object(Default::default()),
                _ => return None,
            };
            Some(ToolCall::new(name, arguments))
        }
        TargetFormat::Grug => parse_call_grug(raw, offered),
    }
}

/// The three mutually-exclusive shapes a model's raw output can take at the
/// boundary (Phase 51 P3c, ADR 089): a concrete tool [`Call`](ParsedOutput::Call),
/// an explicit [`Abstain`](ParsedOutput::Abstain) (the gap token — "no offered
/// tool fits"), or [`Unparseable`](ParsedOutput::Unparseable) (neither shape
/// recognised). Distinguishing *abstain* from *unparseable* is the whole point:
/// the organ routes an abstain into retrieval/synthesis (ADR 089 mechanism 3),
/// but an unparseable output is a plain neural failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedOutput {
    Call(ToolCall),
    Abstain,
    Unparseable,
}

/// Parse a model output string under `fmt`, classifying it as a call, an
/// abstention (gap token), or unparseable. Abstention is checked **before** the
/// call parse so a bare gap marker is never mis-read as a degenerate call.
///
/// - **Grug**: abstain iff the first whitespace-delimited token of the trimmed
///   output is [`GRUG_GAP_MARKER`] (`gap`). Real calls always begin with the
///   literal `call`, so this is unambiguous.
/// - **JSON**: abstain iff the output parses to an object carrying a truthy
///   `"gap"` and no `"name"` (so a real call object is never swallowed).
pub fn parse_output(raw: &str, fmt: TargetFormat, offered: &[ToolSchema]) -> ParsedOutput {
    let trimmed = raw.trim();
    match fmt {
        TargetFormat::Grug => {
            if trimmed.split_whitespace().next() == Some(GRUG_GAP_MARKER) {
                return ParsedOutput::Abstain;
            }
        }
        TargetFormat::Json => {
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed) {
                let gap = map.get("gap").is_some_and(|v| v.as_bool() == Some(true));
                if gap && !map.contains_key("name") {
                    return ParsedOutput::Abstain;
                }
            }
        }
    }
    match parse_call(trimmed, fmt, offered) {
        Some(call) => ParsedOutput::Call(call),
        None => ParsedOutput::Unparseable,
    }
}

fn parse_call_grug(raw: &str, offered: &[ToolSchema]) -> Option<ToolCall> {
    let raw = raw.trim();
    // Segments delimited by " | "; the first holds `call <name>`.
    let mut segments = raw.split(" | ");
    let head = segments.next()?.trim();
    // Head is `call <name>` — be lenient about a missing/extra leading word.
    let name = head
        .strip_prefix("call ")
        .unwrap_or(head)
        .split_whitespace()
        .next()?
        .to_string();

    let schema = offered.iter().find(|t| t.name == name);
    let mut args = Map::new();
    for seg in segments {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        let (arg_name, value_str) = match seg.split_once(' ') {
            Some((n, v)) => (n.trim().to_string(), v.trim()),
            // A bare arg name with no value: record an empty string so the
            // schema check can flag it rather than silently dropping it.
            None => (seg.to_string(), ""),
        };
        let ty = schema.and_then(|s| s.param(&arg_name)).map(|p| &p.ty);
        args.insert(arg_name, coerce_value(value_str, ty));
    }

    Some(ToolCall::new(name, Value::Object(args)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Example, ParamSpec, ParamType, ToolCall, ToolSchema};
    use crate::Dataset;

    fn tool(name: &str, params: &[(&str, ParamType)]) -> ToolSchema {
        ToolSchema::new(
            name,
            "d",
            params.iter().map(|(n, t)| ParamSpec::new(*n, t.clone(), true)).collect(),
        )
    }

    #[test]
    fn grug_call_round_trips_through_parse() {
        let tools = vec![tool(
            "book_table",
            &[("party_size", ParamType::Integer), ("time", ParamType::String)],
        )];
        let call = ToolCall::new("book_table", serde_json::json!({"party_size": 4, "time": "20:00"}));
        let rendered = render_call(&call, TargetFormat::Grug);
        assert_eq!(rendered, "call book_table | party_size 4 | time 20:00");
        let parsed = parse_call(&rendered, TargetFormat::Grug, &tools).unwrap();
        assert_eq!(parsed.name, "book_table");
        assert_eq!(parsed.arguments["party_size"], serde_json::json!(4));
        assert_eq!(parsed.arguments["time"], serde_json::json!("20:00"));
    }

    #[test]
    fn grug_value_may_contain_spaces() {
        let tools = vec![tool("note", &[("text", ParamType::String)])];
        let call = ToolCall::new("note", serde_json::json!({"text": "hello there world"}));
        let rendered = render_call(&call, TargetFormat::Grug);
        let parsed = parse_call(&rendered, TargetFormat::Grug, &tools).unwrap();
        assert_eq!(parsed.arguments["text"], serde_json::json!("hello there world"));
    }

    #[test]
    fn grug_schema_renders_types_and_reqs() {
        let tools = vec![ToolSchema::new(
            "x",
            "d",
            vec![
                ParamSpec::new("a", ParamType::String, true),
                ParamSpec::new("b", ParamType::Enum(vec!["low".into(), "high".into()]), false),
            ],
        )];
        let s = render_tools(&tools, TargetFormat::Grug);
        assert_eq!(s, "tool x | a str req | b enum opt low high");
    }

    #[test]
    fn json_path_matches_expected_call_json_string() {
        // The Json variant must reproduce Example::expected_call_as_json_string
        // exactly, so switching formats doesn't perturb the JSON baseline.
        let ex = Example {
            tools: vec![tool("x", &[("b", ParamType::Integer), ("a", ParamType::String)])],
            user_request: "r".into(),
            expected_call: ToolCall::new("x", serde_json::json!({"a": "hi", "b": 3})),
            mock_response: String::new(),
        };
        assert_eq!(render_call(&ex.expected_call, TargetFormat::Json), ex.expected_call_as_json_string());
    }

    #[test]
    fn gap_renders_and_parses_in_both_formats() {
        let gap = ToolCall::gap();
        let tools = vec![tool("book_table", &[("time", ParamType::String)])];

        // Grug: bare `gap`.
        assert_eq!(render_call(&gap, TargetFormat::Grug), "gap");
        assert_eq!(
            parse_output("gap", TargetFormat::Grug, &tools),
            ParsedOutput::Abstain
        );
        // JSON: `{"gap":true}`.
        assert_eq!(render_call(&gap, TargetFormat::Json), "{\"gap\":true}");
        assert_eq!(
            parse_output("{\"gap\":true}", TargetFormat::Json, &tools),
            ParsedOutput::Abstain
        );
    }

    #[test]
    fn a_real_call_is_never_read_as_abstain() {
        let tools = vec![tool("book_table", &[("time", ParamType::String)])];
        let call = ToolCall::new("book_table", serde_json::json!({"time": "20:00"}));
        for fmt in [TargetFormat::Grug, TargetFormat::Json] {
            let rendered = render_call(&call, fmt);
            match parse_output(&rendered, fmt, &tools) {
                ParsedOutput::Call(c) => assert_eq!(c.name, "book_table"),
                other => panic!("{fmt:?} call mis-parsed as {other:?}"),
            }
        }
    }

    #[test]
    fn garbage_output_is_unparseable_not_abstain() {
        let tools = vec![tool("book_table", &[("time", ParamType::String)])];
        // A Grug call head is `call <name>`; leading junk with no `gap` word and
        // no recoverable name is unparseable, and specifically NOT an abstain.
        assert_eq!(
            parse_output("{ not json at all", TargetFormat::Json, &tools),
            ParsedOutput::Unparseable
        );
    }

    /// Grug render→parse recovers the ground-truth call across many real
    /// generator examples (the format is total + lossless for our data).
    #[test]
    fn grug_round_trips_real_generator_calls() {
        let dataset = Dataset::new_sized(42, 200, 0.15).unwrap();
        let mut checked = 0;
        for ex in dataset.train_examples(300).unwrap() {
            let rendered = render_call(&ex.expected_call, TargetFormat::Grug);
            let parsed = parse_call(&rendered, TargetFormat::Grug, &ex.tools)
                .unwrap_or_else(|| panic!("failed to parse: {rendered}"));
            assert_eq!(parsed.name, ex.expected_call.name, "name mismatch on {rendered}");
            // Every expected arg is recovered with a type-correct value.
            for (k, v) in ex.expected_call.arguments_map() {
                assert_eq!(parsed.arguments.get(&k), Some(&v), "arg {k} mismatch on {rendered}");
            }
            checked += 1;
        }
        assert!(checked > 100);
    }
}
