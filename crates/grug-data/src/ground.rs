//! Value grounding (Phase 51 P3a) — the last-mile symbolic pass that snaps a
//! neural proposal's *argument values* to the evidence in the request.
//!
//! [`crate::repair`] (ADR 089 mechanism 2) grounds the tool and param *names*
//! against the schema; it cannot touch *values* (a wrong value is still a valid
//! value). But an error-structure inspection of the retrieval-scoped registry
//! model (`inspect_registry.rs`) found the arg-correct residual is ~70% numeric
//! value errors — the model mis-copies multi-digit numbers (`637.82` → `637.39`)
//! or misaligns similar ones (`350` grabbed from `350.39`) — and, decisively,
//! **the correct value is present verbatim in the request 100% of the time**.
//! So the values can be re-grounded against the request's own value inventory.
//!
//! # What it does (bounded, honest)
//!
//! - **Numeric** (`Integer`/`Number`): extract every number in the request; any
//!   emitted numeric arg whose value is not already one of them is snapped, via
//!   a nearest-value assignment, to an *unused* request number. This fixes both
//!   corruption (snap to the closest) and misalignment (the assignment
//!   re-permutes), and never invents a value the request doesn't contain.
//! - **Enum**: if the emitted value is not the declared variant that appears in
//!   the request, and exactly one declared variant does appear, snap to it.
//! - **String / Boolean**: left untouched — the request's string bank and the
//!   two boolean tokens don't give an unambiguous re-grounding signal, so
//!   touching them would risk breaking correct values (measured: strings/bools
//!   are a minority of the residual).
//!
//! Values the model already copied correctly are never modified (they match a
//! request value), so grounding cannot regress an already-correct arg to a
//! wrong one — only ungrounded values are candidates.

use serde_json::Value;

use crate::schema::{ParamType, ToolCall, ToolSchema};

/// What the grounding pass changed, for measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroundStats {
    pub numeric_regrounded: usize,
    pub enum_regrounded: usize,
}

/// Re-ground `call`'s argument values against `request`, using `offered` to
/// find the emitted tool's schema (for per-arg types). Returns the grounded
/// call and a change count. If the tool is not offered, the call is returned
/// unchanged (nothing to type against).
pub fn ground_values(
    call: &ToolCall,
    request: &str,
    offered: &[ToolSchema],
) -> (ToolCall, GroundStats) {
    let Some(schema) = offered.iter().find(|t| t.name == call.name) else {
        return (call.clone(), GroundStats::default());
    };
    let mut stats = GroundStats::default();
    let args = call.arguments_map();

    // Request's numeric inventory (parsed once).
    let request_numbers = extract_numbers(request);
    // Which request numbers are already "claimed" by a correctly-copied numeric
    // arg, so they are not reused when re-grounding a different arg.
    let mut available: Vec<f64> = request_numbers.clone();
    let mut ungrounded_numeric: Vec<String> = Vec::new();

    // First pass: mark grounded numeric args and collect ungrounded ones.
    for (name, value) in &args {
        let Some(spec) = schema.param(name) else { continue };
        if matches!(spec.ty, ParamType::Integer | ParamType::Number) {
            if let Some(v) = value.as_f64() {
                if let Some(pos) = available.iter().position(|r| (r - v).abs() < 1e-6) {
                    available.remove(pos); // this request number is claimed
                } else {
                    ungrounded_numeric.push(name.clone());
                }
            }
        }
    }

    // Assign each ungrounded numeric arg to its nearest still-available request
    // number (greedy nearest — deterministic, and optimal enough for the small
    // per-call sets here). Process in a stable name order for determinism.
    ungrounded_numeric.sort();
    let mut new_args = args.clone();
    for name in &ungrounded_numeric {
        let Some(v) = args.get(name).and_then(Value::as_f64) else { continue };
        let Some(spec) = schema.param(name) else { continue };
        if available.is_empty() {
            break;
        }
        let (best_i, _) = available
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - v).abs().partial_cmp(&(*b - v).abs()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("available non-empty");
        let chosen = available.remove(best_i);
        new_args.insert(name.clone(), number_value(chosen, &spec.ty));
        stats.numeric_regrounded += 1;
    }

    // Enum grounding: snap to the declared variant present in the request when
    // the emitted value isn't already it and exactly one variant appears.
    for (name, value) in &args {
        let Some(spec) = schema.param(name) else { continue };
        if let ParamType::Enum(variants) = &spec.ty {
            // Snap to the single declared variant present in the request whenever
            // the emitted value isn't already it — even if the emitted value is a
            // *different valid* variant (a wrong-but-valid enum is the common
            // error, and the request names the correct one).
            let present: Vec<&String> = variants.iter().filter(|v| request_contains_word(request, v)).collect();
            if present.len() == 1 && value.as_str() != Some(present[0].as_str()) {
                new_args.insert(name.clone(), Value::String(present[0].clone()));
                stats.enum_regrounded += 1;
            }
        }
    }

    (ToolCall::new(call.name.clone(), Value::Object(new_args)), stats)
}

/// Coerce a grounded request number into the JSON shape the param type expects:
/// `Integer` → a JSON integer (the request wrote it without a fraction);
/// `Number` → a JSON float.
fn number_value(n: f64, ty: &ParamType) -> Value {
    match ty {
        ParamType::Integer => Value::from(n.round() as i64),
        _ => serde_json::Number::from_f64(n).map(Value::Number).unwrap_or(Value::Null),
    }
}

/// Extract every number appearing in `text` as an `f64`, preserving duplicates
/// (a request may legitimately mention the same value for two params). A number
/// span is a maximal run of ASCII digits with at most one embedded `.`; a
/// trailing `.` (sentence punctuation) is trimmed before parsing.
fn extract_numbers(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_digit() {
            let start = i;
            let mut seen_dot = false;
            while i < bytes.len() {
                let b = bytes[i];
                if b.is_ascii_digit() {
                    i += 1;
                } else if b == b'.' && !seen_dot {
                    // Only consume the dot if a digit follows (else it's a
                    // sentence period, not a decimal point).
                    if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                        seen_dot = true;
                        i += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if let Ok(n) = text[start..i].parse::<f64>() {
                out.push(n);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Whole-word (case-sensitive) containment of `word` in `text`, so an enum
/// variant `read` doesn't match inside `ready`.
fn request_contains_word(text: &str, word: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric()).any(|tok| tok == word)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ParamSpec, ParamType, ToolSchema};

    fn tool() -> ToolSchema {
        ToolSchema::new(
            "start_survey",
            "d",
            vec![
                ParamSpec::new("target_count", ParamType::Integer, true),
                ParamSpec::new("query_score", ParamType::Number, true),
                ParamSpec::new("mode_level", ParamType::Enum(vec!["low".into(), "high".into()]), true),
            ],
        )
    }

    #[test]
    fn extracts_numbers_including_floats_and_ignores_periods() {
        let n = extract_numbers("401, 637.82, and 350.39.");
        assert_eq!(n, vec![401.0, 637.82, 350.39]);
    }

    #[test]
    fn snaps_corrupted_number_to_nearest_request_value() {
        let offered = vec![tool()];
        let request = "for 401 and 637.82 with high";
        // query_score corrupted to 637.39 (not in request) → snap to 637.82.
        let call = ToolCall::new(
            "start_survey",
            serde_json::json!({"target_count": 401, "query_score": 637.39, "mode_level": "high"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats.numeric_regrounded, 1);
        assert_eq!(grounded.arguments["query_score"], serde_json::json!(637.82));
        assert_eq!(grounded.arguments["target_count"], serde_json::json!(401)); // already correct, untouched
    }

    #[test]
    fn assignment_fixes_two_corrupted_numbers_without_collision() {
        let offered = vec![tool()];
        let request = "for 401 and 637.82 with high";
        // Both numbers corrupted; nearest-assignment must map each to a distinct
        // request number, not both to the closest.
        let call = ToolCall::new(
            "start_survey",
            serde_json::json!({"target_count": 405, "query_score": 630.0, "mode_level": "high"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats.numeric_regrounded, 2);
        assert_eq!(grounded.arguments["target_count"], serde_json::json!(401));
        assert_eq!(grounded.arguments["query_score"], serde_json::json!(637.82));
    }

    #[test]
    fn snaps_enum_to_the_variant_in_the_request() {
        let offered = vec![tool()];
        let request = "for 401 and 637.82 with high";
        let call = ToolCall::new(
            "start_survey",
            serde_json::json!({"target_count": 401, "query_score": 637.82, "mode_level": "low"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats.enum_regrounded, 1);
        assert_eq!(grounded.arguments["mode_level"], serde_json::json!("high"));
    }

    #[test]
    fn a_fully_correct_call_is_unchanged() {
        let offered = vec![tool()];
        let request = "for 401 and 637.82 with high";
        let call = ToolCall::new(
            "start_survey",
            serde_json::json!({"target_count": 401, "query_score": 637.82, "mode_level": "high"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats, GroundStats::default());
        assert_eq!(grounded.arguments, call.arguments);
    }
}
