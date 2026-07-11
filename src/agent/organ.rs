//! The organ's tool-call passes at the **live** agent boundary (ADR 089,
//! Phase 51 P3 wired into production).
//!
//! Two mechanisms, measured in the Phase-51 lab (`crates/grug-data`) and reused
//! here rather than reimplemented:
//!
//! - **M1 — retrieval-scoped tool catalog** ([`retrieve_scoped`]): instead of
//!   offering the model every registered tool (several thousand tokens; small
//!   in-process models degenerate past ~14 tools) or blind-truncating
//!   (`AKH_AGENT_MAX_TOOLS`), offer the top-K tools *relevant to this request*,
//!   scored over each tool's name **and description** (the P3d lesson: the
//!   name alone is surface signal; the description carries the meaning — keying
//!   on both covers tools named well and tools described well). Measured in the
//!   lab: recall@8 ≈ 0.99–1.0 whenever request and tool text share vocabulary.
//!   Opt-in via `AKH_AGENT_TOOL_RETRIEVAL=<k>` because it changes what the
//!   model sees.
//!
//! - **M2 — repair of unknown tool calls** ([`repair_unknown_call`]): when the
//!   model emits a tool name the registry does not know (today a guaranteed
//!   `unknown tool` error), ground the call against the *offered* schemas —
//!   nearest tool by name similarity + param overlap, near-miss param names
//!   snapped, values coerced to declared types (`grug_data::repair`, ~20× lift
//!   on near-miss naming in the lab). **Threshold-gated at this boundary**: the
//!   lab's `ground_tool` has no similarity floor (its requests always targeted
//!   an offered tool), but in production a fully hallucinated name must *not*
//!   silently route to an arbitrary tool — repairs below
//!   [`REPAIR_NAME_FLOOR`] are rejected and the original error surfaces.
//!   Enabled by default (it only fires where the call would otherwise fail —
//!   pure downside-protection); opt out with `AKH_AGENT_TOOL_REPAIR=0`.
//!
//! Execution stays permission-gated after both passes, so neither can widen
//! what the model is *allowed* to do — only what it is shown and how its
//! near-misses are grounded.

use grug_data::{repair_call, ParamSpec, ParamType, Retriever, ToolCall, ToolSchema};
use serde_json::Value;

/// Minimum normalized name similarity (`1 - lev/maxlen`) between the emitted
/// tool name and the repaired tool's name for an M2 repair to be **accepted**.
/// Below this the emitted name is treated as a hallucination, not a near-miss,
/// and the original unknown-tool error surfaces instead of a silent reroute.
/// 0.5 admits genuine near-misses (`fetch_ledgr` → `fetch_ledger` ≈ 0.92,
/// `kg_query` → `kg_query_sparql` ≈ 0.53) while rejecting unrelated names.
const REPAIR_NAME_FLOOR: f64 = 0.5;

/// M1 — reduce an Ollama-format tool catalog to the top-`k` tools relevant to
/// `request`, ranked by lexical overlap against each tool's name (underscores
/// read as spaces) and description. Returns the catalog unchanged when it
/// already fits in `k`. Selection is deterministic (ties break by catalog
/// order, which the caller sorts by name).
pub fn retrieve_scoped(tools: &[Value], request: &str, k: usize) -> Vec<Value> {
    if tools.len() <= k {
        return tools.to_vec();
    }
    let texts: Vec<String> = tools.iter().map(retrieval_text).collect();
    let retriever = Retriever::new_over(&texts);
    retriever
        .top_k(request, k)
        .into_iter()
        .map(|i| tools[i].clone())
        .collect()
}

/// The retrieval key for one catalog entry: `name` with `_`/`.`/`-` opened to
/// spaces (so `kg_query_sparql` and `mcp.list-resources` token-match request
/// words) plus the description.
fn retrieval_text(tool: &Value) -> String {
    let f = &tool["function"];
    let name = f["name"].as_str().unwrap_or_default();
    let desc = f["description"].as_str().unwrap_or_default();
    let opened: String = name
        .chars()
        .map(|c| if c == '_' || c == '.' || c == '-' { ' ' } else { c })
        .collect();
    format!("{opened} {desc}")
}

/// A repaired tool call: what to execute instead of the unknown original, plus
/// a one-line human note for the narration/audit trail.
#[derive(Debug, Clone)]
pub struct RepairedCall {
    pub name: String,
    pub arguments: Value,
    /// e.g. `"fetch_ledgr → fetch_ledger (2 args renamed)"`.
    pub note: String,
}

/// M2 — attempt to repair a tool call whose `name` the registry does not know,
/// grounding it against the schemas the model was actually **offered**.
///
/// Returns `None` (caller lets the original unknown-tool error surface) when:
/// - repair is opted out (`AKH_AGENT_TOOL_REPAIR=0`),
/// - no offered entry parses into a schema,
/// - the best grounding's name similarity falls below [`REPAIR_NAME_FLOOR`]
///   (a hallucinated name, not a near-miss).
pub fn repair_unknown_call(name: &str, arguments: &Value, offered: &[Value]) -> Option<RepairedCall> {
    if std::env::var("AKH_AGENT_TOOL_REPAIR").is_ok_and(|v| v == "0") {
        return None;
    }
    let schemas: Vec<ToolSchema> = offered.iter().filter_map(schema_from_catalog).collect();
    if schemas.is_empty() {
        return None;
    }

    let call = ToolCall::new(name.to_string(), arguments.clone());
    let (repaired, stats) = repair_call(&call, &schemas);

    // The boundary accept-gate the lab pass deliberately lacks: only trust the
    // reroute when the emitted name genuinely resembles the repaired tool.
    if normalized_similarity(name, &repaired.name) < REPAIR_NAME_FLOOR {
        return None;
    }

    let mut notes = vec![format!("{name} → {}", repaired.name)];
    if stats.args_renamed > 0 {
        notes.push(format!("{} arg(s) renamed", stats.args_renamed));
    }
    if stats.args_dropped > 0 {
        notes.push(format!("{} arg(s) dropped", stats.args_dropped));
    }
    if stats.values_coerced > 0 {
        notes.push(format!("{} value(s) coerced", stats.values_coerced));
    }
    Some(RepairedCall {
        name: repaired.name.clone(),
        arguments: repaired.arguments,
        note: notes.join(", "),
    })
}

/// Parse one Ollama/OpenAI-format catalog entry
/// (`{"type":"function","function":{name,description,parameters}}`) into the
/// organ's [`ToolSchema`]. Returns `None` for entries without a name (nothing
/// to ground against).
fn schema_from_catalog(tool: &Value) -> Option<ToolSchema> {
    let f = tool.get("function")?;
    let name = f.get("name")?.as_str()?.to_string();
    let description = f
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let mut params = Vec::new();
    if let Some(props) = f
        .pointer("/parameters/properties")
        .and_then(Value::as_object)
    {
        let required: Vec<&str> = f
            .pointer("/parameters/required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        for (pname, spec) in props {
            let ty = param_type_from_schema(spec);
            params.push(ParamSpec::new(pname.clone(), ty, required.contains(&pname.as_str())));
        }
    }
    Some(ToolSchema::new(name, description, params))
}

/// Map a JSON-Schema property fragment onto the organ's [`ParamType`]. An
/// `enum` list of strings becomes `Enum` (so repair can validate membership);
/// unknown/absent types default to `String` (the permissive choice — string
/// coercion never fires on it).
fn param_type_from_schema(spec: &Value) -> ParamType {
    if let Some(variants) = spec.get("enum").and_then(Value::as_array) {
        let vs: Vec<String> = variants
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        if !vs.is_empty() {
            return ParamType::Enum(vs);
        }
    }
    match spec.get("type").and_then(Value::as_str) {
        Some("integer") => ParamType::Integer,
        Some("number") => ParamType::Number,
        Some("boolean") => ParamType::Boolean,
        _ => ParamType::String,
    }
}

/// Normalized (case-insensitive) name similarity in `[0, 1]`:
/// `1 - lev(a,b)/max(|a|,|b|)`. Local to this boundary — it implements the
/// accept-gate the lab pass doesn't have.
fn normalized_similarity(a: &str, b: &str) -> f64 {
    let al: Vec<char> = a.to_ascii_lowercase().chars().collect();
    let bl: Vec<char> = b.to_ascii_lowercase().chars().collect();
    let max = al.len().max(bl.len());
    if max == 0 {
        return 1.0;
    }
    1.0 - levenshtein(&al, &bl) as f64 / max as f64
}

/// Classic two-row Levenshtein edit distance.
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn catalog_entry(name: &str, desc: &str, props: Value, required: Value) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required}
            }
        })
    }

    fn sample_catalog() -> Vec<Value> {
        vec![
            catalog_entry(
                "kg_query_sparql",
                "Run a SPARQL query against the knowledge graph.",
                json!({"query": {"type": "string", "description": "the SPARQL text"}}),
                json!(["query"]),
            ),
            catalog_entry(
                "calendar_add_event",
                "Add an event to the calendar.",
                json!({
                    "title": {"type": "string", "description": "event title"},
                    "day_count": {"type": "integer", "description": "days from now"}
                }),
                json!(["title"]),
            ),
            catalog_entry(
                "send_channel_message",
                "Send a message to a communication channel.",
                json!({"channel": {"type": "string"}, "body": {"type": "string"}}),
                json!(["channel", "body"]),
            ),
        ]
    }

    #[test]
    fn retrieval_scopes_to_the_relevant_tool() {
        let tools = sample_catalog();
        let scoped = retrieve_scoped(&tools, "please add a calendar event for the review", 1);
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0]["function"]["name"], "calendar_add_event");
        // A catalog already within k is returned whole.
        assert_eq!(retrieve_scoped(&tools, "anything", 5).len(), 3);
    }

    #[test]
    fn repair_grounds_a_near_miss_name_and_params() {
        let tools = sample_catalog();
        // Near-miss tool name + near-miss param name + stringified integer.
        let repaired = repair_unknown_call(
            "calendar_add_evnt",
            &json!({"title": "standup", "day_cnt": "3"}),
            &tools,
        )
        .expect("a near-miss must repair");
        assert_eq!(repaired.name, "calendar_add_event");
        assert_eq!(repaired.arguments["title"], json!("standup"));
        // Param renamed and value coerced to the declared integer type.
        assert_eq!(repaired.arguments["day_count"], json!(3));
        assert!(repaired.note.contains("calendar_add_evnt → calendar_add_event"));
    }

    #[test]
    fn repair_rejects_a_hallucinated_name() {
        let tools = sample_catalog();
        // Nothing offered resembles this — the accept-gate must refuse rather
        // than silently reroute to an arbitrary tool.
        assert!(repair_unknown_call("launch_missiles", &json!({"target": "x"}), &tools).is_none());
    }

    #[test]
    fn repair_handles_empty_offered_set() {
        assert!(repair_unknown_call("anything", &json!({}), &[]).is_none());
    }
}
