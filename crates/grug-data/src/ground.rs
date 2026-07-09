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
//! - **String**: the request embeds the argument's string value *verbatim*
//!   (the generator synthesizes the value first and the phrasing references it),
//!   so — exactly like numbers — the correct value is a span of the request. An
//!   emitted string that is not already a verbatim request span is snapped to
//!   the request word-window (1..=[`STR_MAX_WORDS`] words) most similar to it by
//!   normalized Levenshtein, but *only* if that similarity clears
//!   [`STR_SNAP_THRESHOLD`] — a confidently-wrong emission with no close request
//!   span is left untouched rather than snapped to noise. As with numbers,
//!   already-claimed spans aren't reused, so two string args can't collapse onto
//!   the same evidence.
//! - **Boolean**: deliberately left untouched. A boolean is a single bit and a
//!   bag-of-words request gives no unambiguous polarity signal (the value isn't
//!   a copyable span, and "enable/disable"-style cues need positional parsing we
//!   don't do here) — snapping would be a coin-flip that can *regress* a
//!   correct value, which violates this pass's do-no-harm contract. Measured:
//!   booleans are a small minority of the residual, so the miss is cheap.
//!
//! Values the model already copied correctly are never modified (they match a
//! request value/span), so grounding cannot regress an already-correct arg to a
//! wrong one — only ungrounded values are candidates.

use serde_json::Value;

use crate::schema::{ParamType, ToolCall, ToolSchema};

/// What the grounding pass changed, for measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroundStats {
    pub numeric_regrounded: usize,
    pub enum_regrounded: usize,
    pub string_regrounded: usize,
}

/// Longest request word-window (in whitespace words) considered a candidate
/// span when re-grounding a string value. The generator's string bank tops out
/// at three words (`"the new proposal"`), so 3 covers every legitimate value
/// without inviting spurious long-span matches.
const STR_MAX_WORDS: usize = 3;

/// Minimum normalized (`1 - lev/maxlen`) similarity between an ungrounded string
/// value and a request span for the snap to fire. Below this the value is left
/// unchanged — a genuinely-wrong emission with no close request evidence must
/// not be dragged onto an unrelated span. 0.5 admits corruption/truncation
/// (`"Zuric"` → `"Zurich"`, sim 0.83) while rejecting unrelated pairs.
const STR_SNAP_THRESHOLD: f64 = 0.5;

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

    // String grounding: the value is a verbatim request span. Claim the span of
    // every already-grounded string first (so two args can't snap to the same
    // evidence), then snap each ungrounded string to its nearest unclaimed span
    // above threshold. `Enum` values are strings too but are handled above — a
    // param whose declared type is `Enum` is skipped here.
    let spans = word_spans(request); // (byte_start, byte_end) of each request word
    let windows: Vec<&str> = span_windows(request, &spans);
    let mut claimed: Vec<bool> = vec![false; windows.len()];

    // First pass: mark windows claimed by strings the model already copied
    // verbatim (case-insensitive exact match), so they aren't reused. Claim the
    // first still-unclaimed matching window (a value can appear once per span).
    for (name, value) in &args {
        let Some(spec) = schema.param(name) else { continue };
        if spec.ty != ParamType::String {
            continue;
        }
        let Some(s) = value.as_str() else { continue };
        if let Some(i) = (0..windows.len()).find(|&i| !claimed[i] && windows[i].eq_ignore_ascii_case(s)) {
            claimed[i] = true;
        }
    }

    // Second pass: snap ungrounded strings to the nearest unclaimed window.
    for (name, value) in &args {
        let Some(spec) = schema.param(name) else { continue };
        if spec.ty != ParamType::String {
            continue;
        }
        let Some(s) = value.as_str() else { continue };
        // Grounded already (a window equals it, case-insensitively)?
        if windows.iter().any(|win| win.eq_ignore_ascii_case(s)) {
            continue;
        }
        // Best unclaimed window by normalized similarity.
        let mut best: Option<(usize, f64)> = None;
        for (i, win) in windows.iter().enumerate() {
            if claimed[i] {
                continue;
            }
            let sim = normalized_similarity(s, win);
            if best.is_none_or(|(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }
        if let Some((i, sim)) = best {
            if sim >= STR_SNAP_THRESHOLD && !windows[i].eq_ignore_ascii_case(s) {
                new_args.insert(name.clone(), Value::String(windows[i].to_string()));
                claimed[i] = true;
                stats.string_regrounded += 1;
            }
        }
    }

    (ToolCall::new(call.name.clone(), Value::Object(new_args)), stats)
}

/// Byte ranges of each "word" in `text`: a maximal run of ASCII alphanumeric
/// chars or apostrophes (so `grandma's` stays one word). Word boundaries are
/// char boundaries, so the ranges are safe to slice `text` with.
fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'\'';
        if is_word(bytes[i]) {
            let start = i;
            while i < bytes.len() && is_word(bytes[i]) {
                i += 1;
            }
            out.push((start, i));
        } else {
            i += 1;
        }
    }
    out
}

/// Every candidate span: for each word start, the exact request substring
/// covering 1..=[`STR_MAX_WORDS`] consecutive words. Slices are borrowed from
/// `text` so casing/apostrophes are preserved verbatim (a snapped value must
/// byte-equal the request's own span to score).
fn span_windows<'a>(text: &'a str, words: &[(usize, usize)]) -> Vec<&'a str> {
    let mut out = Vec::new();
    for i in 0..words.len() {
        let end = (i + STR_MAX_WORDS).min(words.len());
        for j in i..end {
            out.push(&text[words[i].0..words[j].1]);
        }
    }
    out
}

/// Normalized string similarity in `[0, 1]`: `1 - lev(a,b) / max(|a|,|b|)`,
/// case-insensitive. `1.0` for two empty strings.
fn normalized_similarity(a: &str, b: &str) -> f64 {
    let al: Vec<char> = a.to_ascii_lowercase().chars().collect();
    let bl: Vec<char> = b.to_ascii_lowercase().chars().collect();
    let max = al.len().max(bl.len());
    if max == 0 {
        return 1.0;
    }
    1.0 - levenshtein(&al, &bl) as f64 / max as f64
}

/// Classic two-row Levenshtein edit distance over char slices.
fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
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

    /// A tool with two string params (plus an enum), for string-grounding tests.
    fn str_tool() -> ToolSchema {
        ToolSchema::new(
            "share_doc",
            "d",
            vec![
                ParamSpec::new("document", ParamType::String, true),
                ParamSpec::new("place", ParamType::String, true),
                ParamSpec::new("mode_level", ParamType::Enum(vec!["low".into(), "high".into()]), false),
            ],
        )
    }

    #[test]
    fn snaps_corrupted_single_word_string_to_request_span() {
        let offered = vec![str_tool()];
        let request = "share the new proposal with Zurich";
        // `place` corrupted (Zur1ch) → snap to the verbatim request span "Zurich".
        let call = ToolCall::new(
            "share_doc",
            serde_json::json!({"document": "the new proposal", "place": "Zur1ch"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats.string_regrounded, 1);
        assert_eq!(grounded.arguments["place"], serde_json::json!("Zurich"));
        // The already-correct multi-word string is untouched.
        assert_eq!(grounded.arguments["document"], serde_json::json!("the new proposal"));
    }

    #[test]
    fn snaps_corrupted_multi_word_string_verbatim_with_apostrophe() {
        let offered = vec![str_tool()];
        let request = "please share grandma's recipe with Oslo";
        let call = ToolCall::new(
            "share_doc",
            serde_json::json!({"document": "grandmas recipe", "place": "Oslo"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats.string_regrounded, 1);
        // Snapped to the exact request substring, apostrophe preserved.
        assert_eq!(grounded.arguments["document"], serde_json::json!("grandma's recipe"));
    }

    #[test]
    fn leaves_confidently_wrong_string_untouched() {
        let offered = vec![str_tool()];
        let request = "share the budget review with Tokyo";
        // A value with no close request span (well below threshold vs every
        // window) must be left alone, not dragged onto noise.
        let call = ToolCall::new(
            "share_doc",
            serde_json::json!({"document": "zzzzzzzzzz", "place": "Tokyo"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats.string_regrounded, 0);
        assert_eq!(grounded.arguments["document"], serde_json::json!("zzzzzzzzzz"));
    }

    #[test]
    fn two_corrupted_strings_do_not_collapse_to_one_span() {
        let offered = vec![str_tool()];
        let request = "share Denver with Cairo";
        // Both single-word strings corrupted; each must claim a distinct span.
        let call = ToolCall::new(
            "share_doc",
            serde_json::json!({"document": "Denvr", "place": "Cair0"}),
        );
        let (grounded, stats) = ground_values(&call, request, &offered);
        assert_eq!(stats.string_regrounded, 2);
        assert_eq!(grounded.arguments["document"], serde_json::json!("Denver"));
        assert_eq!(grounded.arguments["place"], serde_json::json!("Cairo"));
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
