//! Bounding one tool result against the turn's running request size.
//!
//! Anima WI 01a0e270-314e ("Lilith C2"), ADR 116.
//!
//! [`super::manager::ContextManager::assemble`] (Phase 44) sizes the prompt
//! once, at the top of the turn. Everything a tool call returns afterwards is
//! appended to the running conversation in [`super::super::agentic_chat`] with
//! no further check — fine for the Roman akhs (cloud backends, ≥128K token
//! windows) but not for a narrow-window backend such as Lilith's self-hosted
//! vLLM (32768 tokens, ADR 116): one MCP call (`mcp__anima__list_plans`)
//! returning ~96 KB / ~30K tokens of plan bodies pushed the *next* request
//! over the hard limit, and the server rejected the whole turn outright —
//! after the tool had already run and the result was thrown away.
//!
//! [`bound_tool_result`] is the seam: called once per tool result, right
//! before it is pushed onto the conversation, it estimates how much of the
//! window is already spent (system prompt + tool schemas + conversation so
//! far, via the turn's [`TokenCounter`]) and caps the new result to what's
//! left, minus the model's output reserve and a safety margin. A truncated
//! result carries a trailing **structured** marker (a JSON object, not a
//! sentence) recording the original/kept size and, when the tool's own schema
//! shows narrowing parameters (`limit`, `cursor`, `page_token`, …), which ones
//! — so the model can retry narrower instead of re-requesting the same firehose.
//!
//! The marker is data, never prose: per the project's rule against
//! hardcoded natural-language matching, nothing here classifies free text —
//! `narrowing_params` only inspects the tool's own JSON-Schema property
//! *names*, which are a fixed technical namespace the akh itself defines
//! (ADR 076 tool-manifest parameters), not natural language.

use serde_json::{Map, Value, json};

use super::manager::TokenCounter;
use super::types::TokenBudget;

/// Chars-per-token ratio used **only** to size a tool-result truncation cut,
/// deliberately below the `len/4` heuristic [`super::manager::HeuristicTokenCounter`]
/// uses for the rest of the prompt.
///
/// BPE/SentencePiece tokenizers (Gemma's included) count structural JSON
/// punctuation (`{`, `}`, `":`, `,`, quoted short keys) far more densely than
/// English prose — a JSON-heavy tool result routinely runs under 3 chars per
/// token, sometimes closer to 2. Sizing the cut with the prose ratio would
/// under-truncate a JSON payload (keep more characters than the real token
/// budget allows) — exactly the failure mode this module exists to prevent.
/// 3 is a conservative empirical floor for JSON; combined with
/// [`SAFETY_MARGIN_NUM`]/[`SAFETY_MARGIN_DEN`] it leaves headroom for both the
/// ratio's own error and the counter's estimate error.
const JSON_CHARS_PER_TOKEN: usize = 3;

/// Extra shrink applied to the remaining-token budget before it is spent,
/// expressed as an integer numerator/denominator (no floats in a token-budget
/// computation the estimation error already dominates). Keeping only 85% of
/// the computed headroom absorbs counter drift (the heuristic counter is
/// `len/4`; real tokenizers vary token density by content) so a single
/// mis-estimated tool result still leaves the next request comfortably inside
/// the window rather than exactly on the edge.
const SAFETY_MARGIN_NUM: usize = 85;
const SAFETY_MARGIN_DEN: usize = 100;

/// Fixed character budget reserved for the trailing truncation marker itself,
/// so the marker's own bytes are always inside the cap this module promises —
/// the marker is small and bounded (tool name + six small integers + a short
/// parameter-name list), 512 chars covers it with room to spare even for a
/// long fully-qualified MCP tool name.
const RESERVED_MARKER_CHARS: usize = 512;

/// Keep this fraction of the truncated content's char budget as the *head*
/// (from the start of the result); the rest is a *tail* (from the end), which
/// is often where a JSON array's most recent / most specific entries live.
/// A tool result under the tail floor below keeps head only — a tail that
/// tiny would just be noise.
const HEAD_FRACTION_NUM: usize = 3;
const HEAD_FRACTION_DEN: usize = 4;
const MIN_USEFUL_TAIL_CHARS: usize = 256;

/// Technical JSON-Schema property-name substrings that conventionally mark a
/// parameter as narrowing/paging, matched against this repo's own tool
/// manifests (see `ollama_tool_catalog` in `agentic_chat.rs` and the
/// `ToolParamSchema` call sites across `src/agent/tools/`, e.g. `limit`,
/// `top_k`, `cursor`, `page_token`, `offset`, `query`, `filter`). This is a
/// fixed API-identifier convention, not a natural-language word list: it
/// never runs against user- or model-authored text, only against the akh's
/// own schema, which is under this repo's control.
const NARROWING_PARAM_MARKERS: &[&str] = &[
    "page_token",
    "cursor",
    "limit",
    "offset",
    "top_k",
    "max_results",
    "page_size",
    "query",
    "filter",
];

/// The result of checking one tool result against the turn's remaining
/// budget: either it fit untouched, or it was capped and [`TruncationRecord`]
/// documents the decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResultBound {
    /// The content to push as the `role:"tool"` message — either `result`
    /// unchanged, or `result` truncated plus a trailing marker object.
    pub content: String,
    /// `Some` exactly when truncation was applied.
    pub truncation: Option<TruncationRecord>,
}

/// A structured, language-neutral record of one truncation decision.
///
/// Rendered as a JSON object appended to the tool message's own content (the
/// OpenAI `role:"tool"` contract only carries a string, so this rides inside
/// it rather than as a separate field) — the model receives an object it can
/// inspect programmatically, never a sentence it has to parse, which keeps
/// this compatible with the project's language-agnostic-NLU rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationRecord {
    pub tool: String,
    pub original_bytes: usize,
    pub kept_bytes: usize,
    pub original_tokens_est: usize,
    pub kept_tokens_est: usize,
    /// Names of parameters in the tool's own schema that look like they
    /// narrow or page the result (empty when the schema offers none, or
    /// when no schema was supplied).
    pub narrowing_params: Vec<String>,
}

impl TruncationRecord {
    /// The JSON marker object, keyed under a single `_akh_`-namespaced field
    /// so it cannot collide with the tool's own result shape.
    pub fn to_marker_value(&self) -> Value {
        json!({
            "_akh_tool_result_truncated": {
                "tool": self.tool,
                "original_bytes": self.original_bytes,
                "kept_bytes": self.kept_bytes,
                "original_tokens_est": self.original_tokens_est,
                "kept_tokens_est": self.kept_tokens_est,
                "narrowing_params": self.narrowing_params,
                "reason": "context_window_budget",
            }
        })
    }
}

/// Estimate the token cost of everything already committed to this turn's
/// next request — system prompt + offered tool schemas + the conversation
/// built so far (including any tool results already appended earlier in the
/// same round, since the caller runs this once per tool call in sequence).
///
/// This mirrors what each backend's `to_openai_messages`-style translation
/// actually sends, close enough for a budgeting estimate: exact wire framing
/// (per-message role/id overhead) is not counted, which is exactly why
/// [`bound_tool_result`] applies a safety margin on top.
pub fn estimate_running_tokens(
    counter: &dyn TokenCounter,
    system_prompt: Option<&str>,
    tools: &[Value],
    conversation: &[Value],
) -> usize {
    let mut total = 0usize;
    if let Some(sp) = system_prompt
        && !sp.is_empty()
    {
        total += counter.count(sp);
    }
    if !tools.is_empty() {
        let tools_str = serde_json::to_string(tools).unwrap_or_default();
        total += counter.count(&tools_str);
    }
    for msg in conversation {
        if let Some(content) = msg.get("content").and_then(Value::as_str) {
            total += counter.count(content);
        }
        if let Some(calls) = msg.get("tool_calls").filter(|v| match v {
            Value::Array(a) => !a.is_empty(),
            _ => false,
        }) {
            let s = serde_json::to_string(calls).unwrap_or_default();
            total += counter.count(&s);
        }
    }
    total
}

/// Names of `schema`'s top-level `function.parameters.properties` (or, given
/// a bare parameters object, `properties` directly) whose key contains one of
/// [`NARROWING_PARAM_MARKERS`].
fn narrowing_params_in_schema(schema: &Value) -> Vec<String> {
    let properties: Option<&Map<String, Value>> = schema
        .get("function")
        .and_then(|f| f.get("parameters"))
        .and_then(|p| p.get("properties"))
        .and_then(Value::as_object)
        .or_else(|| {
            schema
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(Value::as_object)
        })
        .or_else(|| schema.get("properties").and_then(Value::as_object));
    let Some(properties) = properties else {
        return Vec::new();
    };
    let mut found: Vec<String> = properties
        .keys()
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            NARROWING_PARAM_MARKERS
                .iter()
                .any(|marker| lower.contains(marker))
        })
        .cloned()
        .collect();
    found.sort();
    found
}

/// Truncate `text` to at most `char_budget` bytes, keeping a
/// [`HEAD_FRACTION_NUM`]/[`HEAD_FRACTION_DEN`] head and, when there is enough
/// left over, a tail from the end — both cut at UTF-8 char boundaries so a
/// multi-byte character is never split.
fn truncate_head_tail(text: &str, char_budget: usize) -> String {
    if text.len() <= char_budget {
        return text.to_string();
    }
    if char_budget == 0 {
        return String::new();
    }
    let head_budget = char_budget * HEAD_FRACTION_NUM / HEAD_FRACTION_DEN;
    let tail_budget = char_budget.saturating_sub(head_budget);

    let head_end = floor_char_boundary(text, head_budget);
    if tail_budget < MIN_USEFUL_TAIL_CHARS || head_end >= text.len() {
        return text[..head_end].to_string();
    }
    let tail_start_raw = text.len().saturating_sub(tail_budget);
    let tail_start = ceil_char_boundary(text, tail_start_raw).max(head_end);
    if tail_start >= text.len() {
        return text[..head_end].to_string();
    }
    format!("{}\n\n{}", &text[..head_end], &text[tail_start..])
}

fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
    let max = text.len();
    if idx >= max {
        return max;
    }
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(text: &str, mut idx: usize) -> usize {
    let max = text.len();
    if idx >= max {
        return max;
    }
    while idx < max && !text.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

/// Cap one tool result against the turn's remaining token budget.
///
/// `used_tokens` is the estimate from [`estimate_running_tokens`] for
/// everything already committed to the next request (system prompt + tool
/// schemas + conversation so far, **not including** `result`). `budget` is
/// the akh's window; `counter` is the same [`TokenCounter`] the turn's
/// context assembly used, so the estimate is internally consistent.
///
/// `tool_schema` is the tool's own OpenAI-shaped function schema (looked up
/// from the offered/full catalog by name), used only to surface
/// `narrowing_params` in the truncation marker — omitted (`None`) leaves that
/// list empty, never an error, since the hint is best-effort.
pub fn bound_tool_result(
    tool_name: &str,
    tool_schema: Option<&Value>,
    result: &str,
    used_tokens: usize,
    budget: &TokenBudget,
    counter: &dyn TokenCounter,
) -> ToolResultBound {
    let available = budget.available();
    let remaining_tokens = available.saturating_sub(used_tokens);
    let safe_remaining_tokens = remaining_tokens * SAFETY_MARGIN_NUM / SAFETY_MARGIN_DEN;

    let result_tokens_est = counter.count(result);
    if result_tokens_est <= safe_remaining_tokens {
        return ToolResultBound {
            content: result.to_string(),
            truncation: None,
        };
    }

    let char_budget_total = safe_remaining_tokens.saturating_mul(JSON_CHARS_PER_TOKEN);
    let content_char_budget = char_budget_total.saturating_sub(RESERVED_MARKER_CHARS);
    let kept = truncate_head_tail(result, content_char_budget);
    let kept_tokens_est = counter.count(&kept);

    let narrowing_params = tool_schema
        .map(narrowing_params_in_schema)
        .unwrap_or_default();
    let record = TruncationRecord {
        tool: tool_name.to_string(),
        original_bytes: result.len(),
        kept_bytes: kept.len(),
        original_tokens_est: result_tokens_est,
        kept_tokens_est,
        narrowing_params,
    };
    let marker = record.to_marker_value();
    let content = format!("{kept}\n\n{marker}");

    ToolResultBound {
        content,
        truncation: Some(record),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::manager::HeuristicTokenCounter;

    fn schema_with_params(names: &[&str]) -> Value {
        let mut properties = Map::new();
        for name in names {
            properties.insert((*name).to_string(), json!({"type": "string"}));
        }
        json!({
            "type": "function",
            "function": {
                "name": "some_tool",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                }
            }
        })
    }

    #[test]
    fn small_result_passes_through_untouched() {
        let counter = HeuristicTokenCounter;
        let budget = TokenBudget::new(32_768, 1_024);
        let result = "72F, sunny";
        let bound = bound_tool_result("weather", None, result, 0, &budget, &counter);
        assert_eq!(bound.content, result);
        assert!(bound.truncation.is_none());
    }

    /// The exact live scenario (2026-09-28, Lilith akh): a 96 KB tool result
    /// against a 32768-token window must produce a request that fits.
    #[test]
    fn ninety_six_kb_result_fits_a_32768_token_window() {
        let counter = HeuristicTokenCounter;
        let window = 32_768usize;
        let budget = TokenBudget::new(window, 1_024);

        // Simulate the rest of the turn: a system prompt, ~103 tool schemas
        // (~57 KB per the live incident), and a conversation already holding
        // the user prompt plus the `list_projects` round.
        let system_prompt = "x".repeat(4_000);
        let tools: Vec<Value> = (0..103)
            .map(|i| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("mcp__anima__tool_{i}"),
                        "description": "x".repeat(500),
                        "parameters": {"type": "object", "properties": {}},
                    }
                })
            })
            .collect();
        let conversation = vec![
            json!({"role": "user", "content": "What plans exist?"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "mcp__anima__list_projects", "arguments": {}}}
            ]}),
            json!({"role": "tool", "content": "x".repeat(2_000)}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "mcp__anima__list_plans", "arguments": {}}}
            ]}),
        ];

        let used = estimate_running_tokens(&counter, Some(&system_prompt), &tools, &conversation);

        // The 96 KB tool result from the live incident.
        let tool_schema = schema_with_params(&["project_id", "page_token", "limit"]);
        let result = "y".repeat(96 * 1024);
        let bound = bound_tool_result(
            "mcp__anima__list_plans",
            Some(&tool_schema),
            &result,
            used,
            &budget,
            &counter,
        );

        assert!(bound.truncation.is_some(), "96 KB result must be truncated");
        let record = bound.truncation.unwrap();
        assert_eq!(record.original_bytes, 96 * 1024);
        assert!(record.kept_bytes < record.original_bytes);
        assert_eq!(record.narrowing_params, vec!["limit", "page_token"]);

        // The guarantee this module exists for: the *next* request (system +
        // tools + conversation-so-far + this bounded tool message) must fit
        // the window, measured with the same counter the budget was sized
        // against.
        let mut next_conversation = conversation.clone();
        next_conversation.push(json!({"role": "tool", "content": bound.content}));
        let total =
            estimate_running_tokens(&counter, Some(&system_prompt), &tools, &next_conversation);
        assert!(
            total <= budget.available(),
            "total {total} must fit available budget {}",
            budget.available()
        );
        assert!(total <= window, "total {total} must fit window {window}");
    }

    #[test]
    fn truncation_marker_is_structured_not_prose() {
        let counter = HeuristicTokenCounter;
        let budget = TokenBudget::new(2_048, 256);
        let result = "z".repeat(50_000);
        let bound = bound_tool_result("big_tool", None, &result, 0, &budget, &counter);
        let record = bound.truncation.expect("must truncate");
        let marker = record.to_marker_value();
        // Data, not English: every field is a number, a name, or a list of
        // schema identifiers — never a sentence a language-specific matcher
        // would have to parse.
        assert!(marker["_akh_tool_result_truncated"]["original_bytes"].is_number());
        assert!(marker["_akh_tool_result_truncated"]["kept_bytes"].is_number());
        assert!(marker["_akh_tool_result_truncated"]["narrowing_params"].is_array());
        assert!(bound.content.contains("_akh_tool_result_truncated"));
    }

    #[test]
    fn narrowing_params_found_from_nested_function_schema() {
        let schema = schema_with_params(&["page_token", "unrelated_field", "top_k"]);
        let found = narrowing_params_in_schema(&schema);
        assert_eq!(found, vec!["page_token", "top_k"]);
    }

    #[test]
    fn no_schema_yields_empty_narrowing_params() {
        let counter = HeuristicTokenCounter;
        let budget = TokenBudget::new(2_048, 256);
        let result = "z".repeat(50_000);
        let bound = bound_tool_result("big_tool", None, &result, 0, &budget, &counter);
        assert!(bound.truncation.unwrap().narrowing_params.is_empty());
    }

    #[test]
    fn head_tail_split_preserves_char_boundaries_on_multibyte_text() {
        let text = "é".repeat(10_000); // 2 bytes per char
        let kept = truncate_head_tail(&text, 1_001);
        assert!(kept.len() <= 1_001 + MIN_USEFUL_TAIL_CHARS + 2);
        // Must be valid UTF-8 (would panic on slicing a non-boundary).
        assert!(kept.chars().count() > 0 || kept.is_empty());
    }

    #[test]
    fn tiny_budget_still_returns_within_reason() {
        let counter = HeuristicTokenCounter;
        // Deliberately overshot: used_tokens already exceeds available, so
        // remaining_tokens saturates to 0.
        let budget = TokenBudget::new(1_000, 100);
        let result = "a".repeat(10_000);
        let bound = bound_tool_result("t", None, &result, 5_000, &budget, &counter);
        let record = bound.truncation.expect("must truncate");
        // The marker itself is the floor cost; content may be empty.
        assert!(record.kept_bytes <= RESERVED_MARKER_CHARS + result.len());
        assert!(bound.content.contains("_akh_tool_result_truncated"));
    }
}
