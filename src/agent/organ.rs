//! The organ's tool-call passes at the **live** agent boundary (ADR 089,
//! Phase 51 P3 wired into production).
//!
//! Four mechanisms, measured in the Phase-51 lab (`crates/grug-data`) and
//! reused here rather than reimplemented:
//!
//! - **M1 — retrieval-scoped tool catalog** ([`retrieve_scoped`]): instead of
//!   offering the model every registered tool (several thousand tokens; small
//!   in-process models degenerate past ~14 tools) or blind-truncating
//!   (`AKH_AGENT_MAX_TOOLS`), offer the top-K tools *relevant to this request*,
//!   scored over each tool's name **and description** (the P3d lesson: the
//!   name alone is surface signal; the description carries the meaning — keying
//!   on both covers tools named well and tools described well). Primary scorer
//!   is VSA bundle-similarity when an [`Engine`] is available, lexical token
//!   overlap otherwise — see [`retrieve_scoped`]'s doc for the honest ceiling
//!   of that VSA scoring. Opt-in via `AKH_AGENT_TOOL_RETRIEVAL=<k>` because it
//!   changes what the model sees.
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
//! - **Task A — param-repair-and-retry for *known* tool names**
//!   ([`repair_known_call_args`]): when a *recognized* tool name fails
//!   execution with a structurally repairable error (a missing/mis-typed
//!   param — never a permission denial, which is a different `Ok`-channel
//!   outcome, see the caller in `agentic_chat.rs`), repair the arguments
//!   against *only that tool's own schema* — so, unlike M2, the tool name can
//!   never be rerouted here — and let the caller retry once. Enabled by
//!   default; opt out with `AKH_AGENT_PARAM_REPAIR=0`.
//!
//! - **Task B — value grounding at the live boundary**
//!   ([`ground_call_values`]): snap mis-copied argument *values* (not names)
//!   onto evidence in the request, via `grug_data::ground_values`. **Opt-in,
//!   default OFF** — see that function's doc for why.
//!
//! Execution stays permission-gated after every pass, so none of them can
//! widen what the model is *allowed* to do — only what it is shown, how its
//! near-misses are grounded, and whether a structurally-broken call gets one
//! repaired retry.

use grug_data::{
    ParamSpec, ParamType, Retriever, ToolCall, ToolSchema, ground_values, normalized_similarity_ci,
    repair_call,
};
use serde_json::Value;

use crate::engine::Engine;
use crate::vsa::grounding::encode_text_as_vector;

/// Minimum normalized name similarity (`1 - lev/maxlen`) between the emitted
/// tool name and the repaired tool's name for an M2 repair to be **accepted**.
/// Below this the emitted name is treated as a hallucination, not a near-miss,
/// and the original unknown-tool error surfaces instead of a silent reroute.
/// 0.5 admits genuine near-misses (`fetch_ledgr` → `fetch_ledger` ≈ 0.92,
/// `kg_query` → `kg_query_sparql` ≈ 0.53) while rejecting unrelated names.
const REPAIR_NAME_FLOOR: f64 = 0.5;

/// M1 — reduce an Ollama-format tool catalog to the top-`k` tools relevant to
/// `request`. Returns the catalog unchanged when it already fits in `k`.
///
/// **Primary scorer (ADR 089 task C):** when `engine` is `Some`, each catalog
/// entry is scored by VSA bundle-similarity between `request` and the entry's
/// [`retrieval_text`] (name with separators opened to spaces, plus
/// description) — reusing the same `encode_text_as_vector` +
/// `VsaOps::similarity` primitives as `agent::tool_semantics`, applied
/// directly to the catalog rather than through a `ToolProfile`/`Goal`
/// indirection this call site has no use for. This bridges shared-vocabulary
/// AND KG-grounded/synonym-listed terms — it is **not** a learned paraphrase
/// embedder, so two genuine synonyms with no shared surface and no KG link
/// still won't match (that ceiling is inherent to `encode_text_as_vector`;
/// see ADR 089's task C scout report for the full honesty note).
///
/// **Fallback:** plain lexical token overlap (`grug_data::Retriever`) when
/// `engine` is `None`, or when any VSA encoding/similarity call fails —
/// falls back *wholesale* for the call (never mixes scored and unscored
/// entries), so a caller without engine access, or a genuinely degenerate
/// encoding, still gets deterministic scoping instead of an empty catalog.
///
/// Selection is deterministic (ties break by catalog order, which the caller
/// sorts by name) in both paths.
pub fn retrieve_scoped(
    tools: &[Value],
    request: &str,
    k: usize,
    engine: Option<&Engine>,
) -> Vec<Value> {
    if tools.len() <= k {
        return tools.to_vec();
    }
    if let Some(engine) = engine
        && let Some(scoped) = retrieve_scoped_vsa(tools, request, k, engine)
    {
        return scoped;
    }
    retrieve_scoped_lexical(tools, request, k)
}

/// VSA-scored path behind [`retrieve_scoped`]. Returns `None` (never partial
/// results) the moment any encoding or similarity call fails, so the caller
/// falls back to the lexical path wholesale.
fn retrieve_scoped_vsa(
    tools: &[Value],
    request: &str,
    k: usize,
    engine: &Engine,
) -> Option<Vec<Value>> {
    let ops = engine.ops();
    let item_memory = engine.item_memory();
    let request_vec = encode_text_as_vector(request, engine, ops, item_memory).ok()?;

    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(tools.len());
    for (i, tool) in tools.iter().enumerate() {
        let tool_vec =
            encode_text_as_vector(&retrieval_text(tool), engine, ops, item_memory).ok()?;
        let sim = ops.similarity(&request_vec, &tool_vec).ok()?;
        scored.push((i, sim));
    }
    // Stable sort: ties keep their catalog order, matching the lexical path's
    // tie-break contract.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Some(
        scored
            .into_iter()
            .take(k)
            .map(|(i, _)| tools[i].clone())
            .collect(),
    )
}

/// Lexical fallback behind [`retrieve_scoped`]: token overlap via
/// `grug_data::Retriever` over each entry's [`retrieval_text`]. This was the
/// whole implementation before ADR 089 task C added VSA scoring as primary.
fn retrieve_scoped_lexical(tools: &[Value], request: &str, k: usize) -> Vec<Value> {
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
        .map(|c| {
            if c == '_' || c == '.' || c == '-' {
                ' '
            } else {
                c
            }
        })
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
pub fn repair_unknown_call(
    name: &str,
    arguments: &Value,
    offered: &[Value],
) -> Option<RepairedCall> {
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
    // Case-insensitive: a model that emits `Fetch_Ledgr` for `fetch_ledger`
    // made a casing slip, not a different tool.
    if normalized_similarity_ci(name, &repaired.name) < REPAIR_NAME_FLOOR {
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

/// Task A — attempt to repair a *known* tool call's **arguments** (param
/// renames / drops / value coercions) using **only that tool's own schema**,
/// so a structurally repairable `execute()` failure (missing/mis-typed
/// param) can be retried once instead of surfacing as a terminal error.
///
/// Unlike [`repair_unknown_call`] (M2), which grounds against the *whole*
/// offered catalog to recover from a name the registry doesn't recognize at
/// all, this pass is handed only the single schema matching `name` — so the
/// tool the call executes against can never change here, only its argument
/// shape. `full_catalog` is the **pre-retrieval-scoping** catalog (not just
/// what M1 happened to offer this round): a call can fail on a tool that
/// M1 scoped out of this round's offered set but that the registry still
/// recognizes (e.g. the model recalled it from earlier context), and this
/// pass should still be able to repair it.
///
/// Returns `None` when `name` isn't in `full_catalog` at all, or when repair
/// changed nothing — a no-op repair is not worth a retry.
pub fn repair_known_call_args(
    name: &str,
    arguments: &Value,
    full_catalog: &[Value],
) -> Option<RepairedCall> {
    let entry = full_catalog
        .iter()
        .find(|t| t["function"]["name"].as_str() == Some(name))?;
    let schema = schema_from_catalog(entry)?;

    let call = ToolCall::new(name.to_string(), arguments.clone());
    // A single-schema offered slice: `repair_call`'s tool-grounding step has
    // nothing else to reroute to, by construction.
    let (repaired, stats) = repair_call(&call, std::slice::from_ref(&schema));

    let changed = stats.args_renamed > 0 || stats.args_dropped > 0 || stats.values_coerced > 0;
    if !changed {
        return None;
    }

    let mut notes = Vec::new();
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
        name: repaired.name,
        arguments: repaired.arguments,
        note: notes.join(", "),
    })
}

/// Task B — value grounding at the live boundary: snap mis-copied argument
/// **values** (not names — see [`repair_known_call_args`] for that) onto
/// evidence in `request`, via `grug_data::ground_values`.
///
/// **Opt-in, default OFF (`AKH_AGENT_VALUE_GROUND=1`), checked here so every
/// call site gets the same default without threading the flag through.**
/// The lab premise this pass is built on — "the correct value appears
/// verbatim in the request" — held 100% on the synthetic corpus
/// (`crates/grug-data/src/ground.rs` module docs), because the generator
/// synthesizes each value and then writes it into the phrasing. That does
/// **not** universally hold in live chat: a model may legitimately *derive*
/// a value that never appeared verbatim in the prompt (a computed date, a
/// default it filled in, a value pulled from an earlier tool result or from
/// conversation history rather than this turn's `request` string alone). In
/// those cases grounding has no correct request span to snap to and — worse
/// — could confidently snap a *correct* derived value onto the nearest
/// *wrong* one, corrupting a call that would otherwise have succeeded. That
/// asymmetry (bounded upside on copy-errors, unbounded downside on
/// legitimately-derived values) is why this stays opt-in rather than
/// following M2/task-A's default-on "only fires where it would otherwise
/// fail" logic — grounding can fire on a call that was about to *succeed*.
///
/// `offered` is deliberately the same pool task A uses (the pre-scoping full
/// catalog, not just what M1 offered this round): grounding is a pure
/// value-typing pass with no name-reroute risk, so it has no reason to be
/// scoped more narrowly than the repair pass it rides alongside.
///
/// Returns `None` when the flag is unset, `name` isn't in `offered`, or
/// nothing changed.
pub fn ground_call_values(
    name: &str,
    arguments: &Value,
    request: &str,
    offered: &[Value],
) -> Option<RepairedCall> {
    if !std::env::var("AKH_AGENT_VALUE_GROUND").is_ok_and(|v| v == "1") {
        return None;
    }
    let schemas: Vec<ToolSchema> = offered.iter().filter_map(schema_from_catalog).collect();
    let call = ToolCall::new(name.to_string(), arguments.clone());
    let (grounded, stats) = ground_values(&call, request, &schemas);

    let changed =
        stats.numeric_regrounded > 0 || stats.enum_regrounded > 0 || stats.string_regrounded > 0;
    if !changed {
        return None;
    }

    let mut notes = Vec::new();
    if stats.numeric_regrounded > 0 {
        notes.push(format!(
            "{} numeric value(s) grounded",
            stats.numeric_regrounded
        ));
    }
    if stats.enum_regrounded > 0 {
        notes.push(format!("{} enum value(s) grounded", stats.enum_regrounded));
    }
    if stats.string_regrounded > 0 {
        notes.push(format!(
            "{} string value(s) grounded",
            stats.string_regrounded
        ));
    }
    Some(RepairedCall {
        name: grounded.name,
        arguments: grounded.arguments,
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
            params.push(ParamSpec::new(
                pname.clone(),
                ty,
                required.contains(&pname.as_str()),
            ));
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

    fn test_engine() -> Engine {
        Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .expect("engine construction should succeed")
    }

    #[test]
    fn retrieval_falls_back_to_lexical_without_an_engine() {
        let tools = sample_catalog();
        let scoped = retrieve_scoped(
            &tools,
            "please add a calendar event for the review",
            1,
            None,
        );
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0]["function"]["name"], "calendar_add_event");
        // A catalog already within k is returned whole, engine or not.
        assert_eq!(retrieve_scoped(&tools, "anything", 5, None).len(), 3);
    }

    #[test]
    fn retrieval_scores_via_vsa_when_engine_is_available() {
        let engine = test_engine();
        let tools = sample_catalog();
        let scoped = retrieve_scoped(
            &tools,
            "please add a calendar event for the review",
            1,
            Some(&engine),
        );
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0]["function"]["name"], "calendar_add_event");
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
        assert!(
            repaired
                .note
                .contains("calendar_add_evnt → calendar_add_event")
        );
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

    // --- Task A: repair_known_call_args ---

    #[test]
    fn known_call_repair_fixes_param_name_and_coerces_value() {
        let tools = sample_catalog();
        // Same tool name (already known/registered) but a near-miss param
        // name and a stringified integer — the name must NOT change.
        let repaired = repair_known_call_args(
            "calendar_add_event",
            &json!({"title": "standup", "day_cnt": "3"}),
            &tools,
        )
        .expect("a near-miss param must repair");
        assert_eq!(repaired.name, "calendar_add_event");
        assert_eq!(repaired.arguments["title"], json!("standup"));
        assert_eq!(repaired.arguments["day_count"], json!(3));
        assert!(repaired.note.contains("renamed"));
        assert!(repaired.note.contains("coerced"));
    }

    #[test]
    fn known_call_repair_returns_none_when_nothing_changes() {
        let tools = sample_catalog();
        assert!(
            repair_known_call_args(
                "calendar_add_event",
                &json!({"title": "standup", "day_count": 3}),
                &tools,
            )
            .is_none()
        );
    }

    #[test]
    fn known_call_repair_returns_none_for_a_name_absent_from_the_catalog() {
        assert!(
            repair_known_call_args("nonexistent_tool", &json!({}), &sample_catalog()).is_none()
        );
    }

    // --- Task B: ground_call_values (opt-in AKH_AGENT_VALUE_GROUND=1) ---
    //
    // One test function, not several: the env var is process-global and
    // cargo runs tests in threads within one binary, so parallel tests each
    // mutating the same var would race (same rationale as
    // `agentic_chat::tests::transcript_capture_gate_and_emit_turn_wal`).
    #[test]
    fn value_grounding_is_opt_in_and_off_by_default() {
        // SAFETY: edition-2024 requires explicit unsafe for env mutation;
        // this test owns AKH_AGENT_VALUE_GROUND for its duration and clears
        // it before returning.
        unsafe {
            std::env::remove_var("AKH_AGENT_VALUE_GROUND");
        }

        let tools = vec![catalog_entry(
            "start_survey",
            "Start a survey.",
            json!({
                "target_count": {"type": "integer", "description": "how many"},
                "query_score": {"type": "number", "description": "threshold"}
            }),
            json!(["target_count", "query_score"]),
        )];
        let request = "for 401 and 637.82 respondents";
        // query_score mis-copied (637.39 not in the request; 637.82 is).
        let args = json!({"target_count": 401, "query_score": 637.39});

        // Default off: the mis-copied value is left untouched even though a
        // request-evidence fix exists.
        assert!(
            ground_call_values("start_survey", &args, request, &tools).is_none(),
            "grounding must be a no-op until explicitly enabled"
        );

        unsafe {
            std::env::set_var("AKH_AGENT_VALUE_GROUND", "1");
        }
        let grounded = ground_call_values("start_survey", &args, request, &tools)
            .expect("enabled grounding must snap the corrupted numeric value");
        assert_eq!(grounded.arguments["query_score"], json!(637.82));
        assert_eq!(grounded.arguments["target_count"], json!(401)); // already correct
        assert!(grounded.note.contains("numeric"));

        unsafe {
            std::env::remove_var("AKH_AGENT_VALUE_GROUND");
        }
        assert!(
            ground_call_values("start_survey", &args, request, &tools).is_none(),
            "unset env must return to the default-off behavior"
        );
    }
}
