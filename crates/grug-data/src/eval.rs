//! The eval scorer: turns a model's raw output string into a typed
//! [`EvalOutcome`], and aggregates a batch of outcomes into an [`EvalReport`]
//! plus an anti-collapse diversity alarm.
//!
//! Each scoring level is gated on the previous one succeeding — a model that
//! emits malformed JSON cannot be "schema-valid but argument-wrong"; it just
//! fails at the JSON level. This ordering is what makes the report a useful
//! curriculum signal: it tells you *which* skill is missing.

use std::collections::HashMap;

use serde_json::Value;

use crate::schema::{Example, ParamType, ToolCall};

/// Why argument correctness failed, when it did. Kept distinct from "schema
/// invalid" — this only fires once JSON parsing *and* schema adherence have
/// already passed, so it is purely about matching ground truth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgMismatch {
    /// A required (or in-ground-truth) argument value differs from expected.
    ValueDiffers { param: String },
    /// The model made a different argument-inclusion choice than ground
    /// truth for an optional param (present when expected absent, or vice
    /// versa).
    PresenceDiffers { param: String },
}

/// Why schema validation failed, when it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaViolation {
    /// The parsed JSON was not a `{"name": ..., "arguments": {...}}`-shaped
    /// object at all (e.g. missing `name`, or `arguments` not an object).
    NotAToolCallShape,
    /// `name` is not one of the tools offered in this example.
    UnknownTool { name: String },
    /// A required parameter for the (valid, offered) tool is missing.
    MissingRequired { param: String },
    /// An argument's value does not match its declared `ParamType`.
    WrongType { param: String },
    /// An argument name is not declared on the tool's schema at all.
    UnknownArg { param: String },
}

/// The gated, level-by-level outcome of scoring one model output against its
/// [`Example`]'s ground truth.
///
/// Levels are checked in order and each is only reached if the previous
/// passed:
/// 1. `parsed_json` — did the raw output parse as JSON at all?
/// 2. `schema` — `Some(Ok(call))` if schema-valid (and gives back the parsed
///    [`ToolCall`]); `Some(Err(violations))` if JSON parsed but schema
///    checks failed; `None` if we never got past JSON parsing.
/// 3. `arg_correctness` — `Some(Ok(()))` if arguments matched ground truth;
///    `Some(Err(mismatches))` if schema-valid but arguments were wrong;
///    `None` if we never got past schema validation.
#[derive(Debug, Clone)]
pub struct EvalOutcome {
    pub parsed_json: bool,
    pub schema: Option<Result<ToolCall, Vec<SchemaViolation>>>,
    pub arg_correctness: Option<Result<(), Vec<ArgMismatch>>>,
}

impl EvalOutcome {
    pub fn valid_json(&self) -> bool {
        self.parsed_json
    }

    pub fn schema_valid(&self) -> bool {
        matches!(self.schema, Some(Ok(_)))
    }

    pub fn arg_correct(&self) -> bool {
        matches!(self.arg_correctness, Some(Ok(())))
    }

    /// The tool name the model actually invoked, if JSON parsed far enough to
    /// tell — used by the anti-collapse diversity metric, which needs a
    /// tool-name signal even from schema-invalid outputs (a model that always
    /// emits `{"name": "get_weather", ...}` regardless of the prompt is
    /// collapsing, whether or not its arguments happen to validate).
    pub fn attempted_tool_name(&self) -> Option<&str> {
        match &self.schema {
            Some(Ok(call)) => Some(call.name.as_str()),
            _ => None,
        }
    }
}

/// Score one raw model output string against `example`'s ground truth.
pub fn score_output(raw_output: &str, example: &Example) -> EvalOutcome {
    let parsed: Result<Value, _> = serde_json::from_str(raw_output.trim());
    let Ok(value) = parsed else {
        return EvalOutcome {
            parsed_json: false,
            schema: None,
            arg_correctness: None,
        };
    };

    let schema_result = validate_schema(&value, example);
    let schema_ok = schema_result.as_ref().ok().cloned();

    let arg_result = schema_ok
        .as_ref()
        .map(|call| check_arg_correctness(call, &example.expected_call, example.expected_schema()));

    EvalOutcome {
        parsed_json: true,
        schema: Some(schema_result),
        arg_correctness: arg_result,
    }
}

/// Score one raw model output against `example` under an explicit
/// [`TargetFormat`] (Phase 51 P2). `TargetFormat::Json` is byte-identical to
/// [`score_output`]; `TargetFormat::Grug` parses the flat pipe format via
/// [`crate::format::parse_call`] and validates the resulting call the same way.
pub fn score_output_fmt(raw_output: &str, example: &Example, fmt: crate::format::TargetFormat) -> EvalOutcome {
    match fmt {
        crate::format::TargetFormat::Json => score_output(raw_output, example),
        crate::format::TargetFormat::Grug => {
            let Some(call) = crate::format::parse_call(raw_output, fmt, &example.tools) else {
                return EvalOutcome {
                    parsed_json: false,
                    schema: None,
                    arg_correctness: None,
                };
            };
            let schema_result = validate_call(&call, example);
            let schema_ok = schema_result.as_ref().ok().cloned();
            let arg_result = schema_ok
                .as_ref()
                .map(|c| check_arg_correctness(c, &example.expected_call, example.expected_schema()));
            EvalOutcome {
                parsed_json: true,
                schema: Some(schema_result),
                arg_correctness: arg_result,
            }
        }
    }
}

/// Score one raw model output under `fmt`, then apply the **KG-backed repair
/// pass** ([`crate::repair::repair_call`]) before validating (Phase 51 P3b, ADR
/// 089). Parsing failure is still `parsed=false`; otherwise the parsed call is
/// grounded/repaired against the offered schemas and *that* call is scored — so
/// the raw-vs-repaired delta is exactly what the symbolic layer recovers.
pub fn score_output_repaired_fmt(
    raw_output: &str,
    example: &Example,
    fmt: crate::format::TargetFormat,
) -> EvalOutcome {
    let Some(call) = crate::format::parse_call(raw_output, fmt, &example.tools) else {
        return EvalOutcome {
            parsed_json: false,
            schema: None,
            arg_correctness: None,
        };
    };
    let (repaired, _stats) = crate::repair::repair_call(&call, &example.tools);
    let schema_result = validate_call(&repaired, example);
    let schema_ok = schema_result.as_ref().ok().cloned();
    let arg_result = schema_ok
        .as_ref()
        .map(|c| check_arg_correctness(c, &example.expected_call, example.expected_schema()));
    EvalOutcome {
        parsed_json: true,
        schema: Some(schema_result),
        arg_correctness: arg_result,
    }
}

/// Validate an already-parsed [`ToolCall`] against the schemas offered in
/// `example` — the format-agnostic half of [`validate_schema`], shared by the
/// Grug scoring path.
fn validate_call(call: &ToolCall, example: &Example) -> Result<ToolCall, Vec<SchemaViolation>> {
    let name = call.name.clone();
    let Some(schema) = example.tools.iter().find(|t| t.name == name) else {
        return Err(vec![SchemaViolation::UnknownTool { name }]);
    };

    let mut violations = Vec::new();
    let args_map = call.arguments_map();

    for required in schema.required_params() {
        if !args_map.contains_key(&required.name) {
            violations.push(SchemaViolation::MissingRequired {
                param: required.name.clone(),
            });
        }
    }
    for (arg_name, arg_value) in &args_map {
        match schema.param(arg_name) {
            None => violations.push(SchemaViolation::UnknownArg { param: arg_name.clone() }),
            Some(spec) => {
                if !spec.ty.matches(arg_value) {
                    violations.push(SchemaViolation::WrongType { param: arg_name.clone() });
                }
            }
        }
    }

    if violations.is_empty() {
        Ok(ToolCall::new(name, call.arguments.clone()))
    } else {
        Err(violations)
    }
}

/// Parse `value` as a tool call shape and validate it against the tool
/// schemas offered in `example`.
fn validate_schema(value: &Value, example: &Example) -> Result<ToolCall, Vec<SchemaViolation>> {
    let obj = value.as_object();
    let (name, arguments) = match obj {
        Some(map) => {
            let name = map.get("name").and_then(Value::as_str);
            let args = map.get("arguments");
            match (name, args) {
                (Some(n), Some(a)) if a.is_object() => (n.to_string(), a.clone()),
                (Some(n), None) => (n.to_string(), Value::Object(Default::default())),
                _ => return Err(vec![SchemaViolation::NotAToolCallShape]),
            }
        }
        None => return Err(vec![SchemaViolation::NotAToolCallShape]),
    };

    let Some(schema) = example.tools.iter().find(|t| t.name == name) else {
        return Err(vec![SchemaViolation::UnknownTool { name }]);
    };

    let mut violations = Vec::new();
    let args_map = arguments.as_object().cloned().unwrap_or_default();

    for required in schema.required_params() {
        if !args_map.contains_key(&required.name) {
            violations.push(SchemaViolation::MissingRequired {
                param: required.name.clone(),
            });
        }
    }

    for (arg_name, arg_value) in &args_map {
        match schema.param(arg_name) {
            None => violations.push(SchemaViolation::UnknownArg {
                param: arg_name.clone(),
            }),
            Some(spec) => {
                if !spec.ty.matches(arg_value) {
                    violations.push(SchemaViolation::WrongType {
                        param: arg_name.clone(),
                    });
                }
            }
        }
    }

    if violations.is_empty() {
        Ok(ToolCall::new(name, arguments))
    } else {
        Err(violations)
    }
}

/// Compare a schema-valid `call` against `expected`, normalizing values
/// before comparison (see module docs for the normalization rules).
fn check_arg_correctness(
    call: &ToolCall,
    expected: &ToolCall,
    schema: Option<&crate::schema::ToolSchema>,
) -> Result<(), Vec<ArgMismatch>> {
    let mut mismatches = Vec::new();
    if call.name != expected.name {
        // Should not happen if schema validation passed against the same
        // example, but keep the check honest/total.
        mismatches.push(ArgMismatch::ValueDiffers {
            param: "name".to_string(),
        });
        return Err(mismatches);
    }

    let got = call.arguments_map();
    let want = expected.arguments_map();
    let all_params: std::collections::BTreeSet<&String> = got.keys().chain(want.keys()).collect();

    for param in all_params {
        let ty = schema.and_then(|s| s.param(param)).map(|p| &p.ty);
        match (got.get(param), want.get(param)) {
            (Some(g), Some(w)) => {
                if !values_equal_normalized(g, w, ty) {
                    mismatches.push(ArgMismatch::ValueDiffers {
                        param: param.clone(),
                    });
                }
            }
            (None, None) => {}
            _ => mismatches.push(ArgMismatch::PresenceDiffers {
                param: param.clone(),
            }),
        }
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches)
    }
}

/// Normalization rules for argument-correctness comparison:
/// - Strings: exact match (case-sensitive). Fuzzed strings are drawn from a
///   fixed bank, so case/whitespace drift is a real mistake, not noise.
/// - Numbers (`Integer`/`Number`/untyped): compared as `f64` with a small
///   epsilon, so `3` vs `3.0` is not a false mismatch.
/// - Booleans: exact match.
/// - Enum: exact string match against the declared variant.
fn values_equal_normalized(a: &Value, b: &Value, ty: Option<&ParamType>) -> bool {
    match ty {
        Some(ParamType::Integer) | Some(ParamType::Number) => {
            match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => (x - y).abs() < 1e-6,
                _ => a == b,
            }
        }
        Some(ParamType::Boolean) => a == b,
        Some(ParamType::String) | Some(ParamType::Enum(_)) => {
            a.as_str().zip(b.as_str()).map(|(x, y)| x == y).unwrap_or(a == b)
        }
        None => {
            // Untyped fallback (e.g. an unknown extra arg on both sides):
            // try numeric-tolerant compare first, else exact.
            match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => (x - y).abs() < 1e-6,
                _ => a == b,
            }
        }
    }
}

/// Aggregate rates over a batch of `(raw_output, example)` pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalReport {
    pub valid_json_rate: f64,
    pub schema_adherence_rate: f64,
    pub arg_correct_rate: f64,
    pub n: usize,
}

impl EvalReport {
    /// Score and aggregate a batch in one call.
    pub fn from_outputs(pairs: &[(String, Example)]) -> Self {
        Self::from_outputs_fmt(pairs, crate::format::TargetFormat::Json)
    }

    /// Score and aggregate a batch under an explicit [`TargetFormat`] (Phase 51 P2).
    pub fn from_outputs_fmt(pairs: &[(String, Example)], fmt: crate::format::TargetFormat) -> Self {
        let outcomes: Vec<EvalOutcome> = pairs
            .iter()
            .map(|(raw, ex)| score_output_fmt(raw, ex, fmt))
            .collect();
        Self::from_outcomes(&outcomes)
    }

    /// Score and aggregate a batch with the KG-backed repair pass applied
    /// (Phase 51 P3b): outputs are grounded/repaired against the offered
    /// schemas before validation. Compare to [`EvalReport::from_outputs_fmt`]
    /// for the raw-vs-repaired lift.
    pub fn from_outputs_repaired_fmt(pairs: &[(String, Example)], fmt: crate::format::TargetFormat) -> Self {
        let outcomes: Vec<EvalOutcome> = pairs
            .iter()
            .map(|(raw, ex)| score_output_repaired_fmt(raw, ex, fmt))
            .collect();
        Self::from_outcomes(&outcomes)
    }

    /// Aggregate already-scored outcomes.
    pub fn from_outcomes(outcomes: &[EvalOutcome]) -> Self {
        let n = outcomes.len();
        if n == 0 {
            return Self {
                valid_json_rate: 0.0,
                schema_adherence_rate: 0.0,
                arg_correct_rate: 0.0,
                n: 0,
            };
        }
        let valid_json = outcomes.iter().filter(|o| o.valid_json()).count();
        let schema_valid = outcomes.iter().filter(|o| o.schema_valid()).count();
        let arg_correct = outcomes.iter().filter(|o| o.arg_correct()).count();
        Self {
            valid_json_rate: valid_json as f64 / n as f64,
            schema_adherence_rate: schema_valid as f64 / n as f64,
            arg_correct_rate: arg_correct as f64 / n as f64,
            n,
        }
    }
}

/// The gap-token confusion outcome for one output (Phase 51 P3c, ADR 089
/// mechanism 3): did the model abstain, and should it have? Derived from the
/// example's ground truth ([`Example::is_abstention`]) and the parsed output
/// shape ([`crate::format::parse_output`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbstentionOutcome {
    /// Ground truth: no offered tool fits, the model *should* emit the gap token.
    pub expected_abstain: bool,
    /// The model emitted the gap token.
    pub emitted_abstain: bool,
    /// The output was a recognised shape (a call or an abstain) rather than
    /// garbage — a should-call example that came out unparseable is a call
    /// failure, not a false abstention, and this flag keeps the two distinct.
    pub parseable: bool,
}

impl AbstentionOutcome {
    /// The abstain decision matched ground truth.
    pub fn correct(&self) -> bool {
        self.expected_abstain == self.emitted_abstain
    }
}

/// Score one raw output for the **abstention** decision under `fmt`.
pub fn score_abstention_fmt(
    raw_output: &str,
    example: &Example,
    fmt: crate::format::TargetFormat,
) -> AbstentionOutcome {
    let parsed = crate::format::parse_output(raw_output, fmt, &example.tools);
    AbstentionOutcome {
        expected_abstain: example.is_abstention(),
        emitted_abstain: matches!(parsed, crate::format::ParsedOutput::Abstain),
        parseable: !matches!(parsed, crate::format::ParsedOutput::Unparseable),
    }
}

/// The gap-token confusion matrix over a mixed batch of should-call and
/// should-abstain examples (Phase 51 P3c). The two headline metrics test ADR
/// 089's claim that abstention is an in-distribution classification the weights
/// *can* learn:
///
/// - **`abstain_recall`** — of the should-abstain examples, the fraction where
///   the model correctly emitted the gap token. High = the model learned to
///   recognise "no tool fits".
/// - **`false_abstention_rate`** — of the should-call examples, the fraction
///   where the model *wrongly* emitted the gap token. Low = abstention did not
///   cannibalise real tool-calling (the failure mode that would make the
///   mechanism useless).
///
/// `abstain_precision` (of the times it abstained, how often it was right) is
/// included as a secondary diagnostic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbstentionReport {
    pub abstain_recall: f64,
    pub false_abstention_rate: f64,
    pub abstain_precision: f64,
    pub n_abstain: usize,
    pub n_call: usize,
}

impl AbstentionReport {
    /// Score and aggregate a mixed batch of `(raw_output, example)` pairs under
    /// `fmt`. Examples are classified by their own ground truth, so the caller
    /// just concatenates a should-call batch and a should-abstain batch.
    pub fn from_outputs_fmt(pairs: &[(String, Example)], fmt: crate::format::TargetFormat) -> Self {
        let outcomes: Vec<AbstentionOutcome> = pairs
            .iter()
            .map(|(raw, ex)| score_abstention_fmt(raw, ex, fmt))
            .collect();
        Self::from_outcomes(&outcomes)
    }

    /// Aggregate already-scored abstention outcomes into the confusion-matrix
    /// rates. Empty groups yield `0.0` for their dependent rate (no division by
    /// zero); with no abstentions emitted at all, precision is `0.0`.
    pub fn from_outcomes(outcomes: &[AbstentionOutcome]) -> Self {
        let n_abstain = outcomes.iter().filter(|o| o.expected_abstain).count();
        let n_call = outcomes.iter().filter(|o| !o.expected_abstain).count();

        let true_abstain = outcomes
            .iter()
            .filter(|o| o.expected_abstain && o.emitted_abstain)
            .count();
        let false_abstain = outcomes
            .iter()
            .filter(|o| !o.expected_abstain && o.emitted_abstain)
            .count();
        let emitted_abstain = true_abstain + false_abstain;

        let ratio = |num: usize, den: usize| if den == 0 { 0.0 } else { num as f64 / den as f64 };

        Self {
            abstain_recall: ratio(true_abstain, n_abstain),
            false_abstention_rate: ratio(false_abstain, n_call),
            abstain_precision: ratio(true_abstain, emitted_abstain),
            n_abstain,
            n_call,
        }
    }
}

/// The anti-collapse alarm: a diversity/entropy score over a batch of raw
/// model outputs, independent of correctness. A tiny model that has
/// mode-collapsed onto emitting the same call for every prompt will pass
/// this alarm at (or near) zero even if its JSON happens to be well-formed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiversityReport {
    /// Fraction of outputs whose normalized call (tool name + sorted
    /// argument pairs) is distinct from the rest of the batch. 1.0 = every
    /// output is unique; near 0.0 = heavy repetition.
    pub distinct_call_ratio: f64,
    /// Shannon entropy (bits) over the distribution of *attempted* tool
    /// names in the batch, normalized to `[0, 1]` by dividing by log2(number
    /// of distinct tool names actually seen across the batch's examples).
    /// `0.0` = every output names the same tool (or none parsed far enough
    /// to have a name); `1.0` = tool names are used as evenly as the batch's
    /// example set allows.
    pub tool_name_entropy: f64,
    pub n: usize,
}

impl DiversityReport {
    /// Compute the diversity/entropy alarm over a batch of `(raw_output,
    /// example)` pairs. The `example`s are used only to determine the
    /// universe of tool names available for the entropy normalization
    /// denominator (the offered-tools union across the batch), not for
    /// correctness — this is deliberately independent of `EvalReport`.
    pub fn from_outputs(pairs: &[(String, Example)]) -> Self {
        Self::from_outputs_fmt(pairs, crate::format::TargetFormat::Json)
    }

    /// Diversity alarm under an explicit [`TargetFormat`] (Phase 51 P2).
    pub fn from_outputs_fmt(pairs: &[(String, Example)], fmt: crate::format::TargetFormat) -> Self {
        let outcomes: Vec<EvalOutcome> = pairs
            .iter()
            .map(|(raw, ex)| score_output_fmt(raw, ex, fmt))
            .collect();

        let universe_size: usize = {
            let mut names: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            for (_, ex) in pairs {
                for t in &ex.tools {
                    names.insert(t.name.as_str());
                }
            }
            names.len().max(1)
        };

        Self::from_outcomes(&outcomes, universe_size)
    }

    /// Compute directly from already-scored outcomes, given the size of the
    /// tool-name universe to normalize entropy against.
    pub fn from_outcomes(outcomes: &[EvalOutcome], universe_size: usize) -> Self {
        let n = outcomes.len();
        if n == 0 {
            return Self {
                distinct_call_ratio: 0.0,
                tool_name_entropy: 0.0,
                n: 0,
            };
        }

        // Distinct-call ratio: normalize by (tool name, sorted arg entries)
        // so key-order differences in the raw JSON don't count as diversity.
        let normalized_calls: Vec<String> = outcomes
            .iter()
            .map(|o| match &o.schema {
                Some(Ok(call)) => normalize_call_key(call),
                _ => "<unparsed>".to_string(),
            })
            .collect();
        let distinct: std::collections::HashSet<&String> = normalized_calls.iter().collect();
        let distinct_call_ratio = distinct.len() as f64 / n as f64;

        // Shannon entropy over attempted tool names (including an
        // "<unparsed>" bucket so a batch that never produces a tool name at
        // all also reads as zero-diversity rather than being excluded).
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for o in outcomes {
            let key = o.attempted_tool_name().unwrap_or("<unparsed>");
            *counts.entry(key).or_insert(0) += 1;
        }
        let total = n as f64;
        let raw_entropy: f64 = counts
            .values()
            .map(|&c| {
                let p = c as f64 / total;
                -p * p.log2()
            })
            .sum();
        let max_entropy = (universe_size.max(2) as f64).log2();
        let tool_name_entropy = if max_entropy > 0.0 {
            (raw_entropy / max_entropy).min(1.0)
        } else {
            0.0
        };

        Self {
            distinct_call_ratio,
            tool_name_entropy,
            n,
        }
    }
}

/// A normalized string key for a call: `name` plus sorted `"k=v"` argument
/// pairs, so two JSON-different-but-semantically-identical calls collapse to
/// the same diversity bucket.
fn normalize_call_key(call: &ToolCall) -> String {
    let args = call.arguments_map();
    let mut pairs: Vec<String> = args
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    pairs.sort();
    format!("{}({})", call.name, pairs.join(","))
}

#[cfg(test)]
mod abstention_tests {
    use super::*;
    use crate::format::{render_call, TargetFormat};
    use crate::schema::{Example, ParamSpec, ParamType, ToolCall, ToolSchema};

    fn tool(name: &str) -> ToolSchema {
        ToolSchema::new(name, "d", vec![ParamSpec::new("q", ParamType::String, true)])
    }

    /// A should-call example and a should-abstain example, scored together,
    /// yield a correct confusion matrix: a right call + a right gap = perfect
    /// recall, zero false-abstention.
    #[test]
    fn confusion_matrix_counts_both_directions() {
        let fmt = TargetFormat::Grug;
        let call_ex = Example {
            tools: vec![tool("book_table")],
            user_request: "r".into(),
            expected_call: ToolCall::new("book_table", serde_json::json!({"q": "x"})),
            mock_response: String::new(),
        };
        let abstain_ex = Example {
            tools: vec![tool("get_weather")],
            user_request: "r".into(),
            expected_call: ToolCall::gap(),
            mock_response: String::new(),
        };

        // Model gets both right.
        let good = vec![
            (render_call(&call_ex.expected_call, fmt), call_ex.clone()),
            (render_call(&ToolCall::gap(), fmt), abstain_ex.clone()),
        ];
        let r = AbstentionReport::from_outputs_fmt(&good, fmt);
        assert_eq!(r.n_call, 1);
        assert_eq!(r.n_abstain, 1);
        assert_eq!(r.abstain_recall, 1.0);
        assert_eq!(r.false_abstention_rate, 0.0);
        assert_eq!(r.abstain_precision, 1.0);

        // Model wrongly abstains on the call and wrongly calls on the abstain:
        // zero recall, full false-abstention, zero precision.
        let bad = vec![
            (render_call(&ToolCall::gap(), fmt), call_ex),
            (render_call(&ToolCall::new("get_weather", serde_json::json!({"q": "y"})), fmt), abstain_ex),
        ];
        let r = AbstentionReport::from_outputs_fmt(&bad, fmt);
        assert_eq!(r.abstain_recall, 0.0);
        assert_eq!(r.false_abstention_rate, 1.0);
        assert_eq!(r.abstain_precision, 0.0);
    }
}
